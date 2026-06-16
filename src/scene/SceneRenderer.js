import { RenderGraph, CANVAS } from '../render-graph/RenderGraph.js';
import { GeometryArena, interleaveStandard } from '../geometry/GeometryArena.js';
import { MultiDrawSystem } from '../culling/MultiDrawSystem.js';
import { ShadowMap } from '../lighting/ShadowMap.js';
import { Camera } from '../camera/Camera.js';
import { identity } from '../math/mat4.js';
import { lambertShader, basicShader, pointsShader, shadowDepthShader } from './sceneShaders.wgsl.js';

const OBJ_CAP = 2048;          // per-batch object capacity
const MATERIAL_STRIDE = 16;    // vec4 color (rgb + opacity)
// Max simultaneous lights in the forward lights block. MUST match the
// `array<Light, N>` size in sceneShaders.wgsl.js. Each light is 32 bytes; the
// lambert shader loops over `count` of them per fragment.
const MAX_LIGHTS = 8;

// Byte size per GPUVertexFormat, for building custom-shader vertex layouts.
const VERTEX_FORMAT_BYTES = {
  float32: 4, float32x2: 8, float32x3: 12, float32x4: 16,
  uint32: 4, sint32: 4,
};

// SceneRenderer: draws a retained Scene graph through the GPU-driven path.
//
// Objects are grouped into BATCHES by pipeline configuration (lambert opaque /
// basic opaque / basic additive / basic alpha). Each batch owns a GeometryArena
// (all its meshes' geometry packed into shared buffers), per-object world-matrix
// + material storage buffers, an AABB buffer, and a MultiDrawSystem that
// frustum-culls on the GPU and emits a compacted indexed-indirect arg array.
// Custom ShaderMaterial meshes and Points draw via their own pipelines.
//
// A mesh is registered into its batch on first sight (geometry uploaded into the
// arena, a draw-record + bounds written); subsequent frames just refresh its
// world matrix + material color. Removed meshes free their arena slice.

export class SceneRenderer {
  constructor(device, canvas, { antialias = true, pixelRatio = 1 } = {}) {
    this.device = device;
    this.canvas = canvas;
    this.pixelRatio = pixelRatio;
    this.sampleCount = antialias ? 4 : 1;

    this.context = device.getCanvasContext(canvas);
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this._buildSizedTargets();

    // Shared scene uniforms.
    this.lightsBuffer = device.resources.createBuffer({ size: 16 + MAX_LIGHTS * 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fogBuffer = device.resources.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // --- Optional shadow map (off until enableShadows() is called) ---
    // Bindings 3-5 of the scene group are ALWAYS present so the layout never
    // changes; when shadows are off they point at a 1x1 dummy depth texture and
    // a disabled uniform (shadowParams enabled=0), and shaders that don't include
    // the shadow path simply never sample them. This keeps existing pipelines and
    // examples 19-22 byte-for-byte compatible.
    this.shadowMap = null;            // ShadowMap instance when enabled
    this.shadowEnabled = false;
    // shadowParams uniform: lightViewProj (mat4) + lightDir.xyz + enabled (vec4).
    this.shadowParamsBuffer = device.resources.createBuffer({ size: 64 + 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.shadowParamsBuffer.gpuBuffer, 0, new Float32Array(20)); // all zero => enabled=0
    this._dummyShadowTex = device.resources.createTexture({
      size: [1, 1], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowSampler = device.resources.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });

    // Bind group layouts shared by the lambert/basic batch pipelines.
    this.sceneBGL = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },     // shadowParams
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },   // shadow map
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },                          // shadow comparison sampler
      ],
    });
    this.objectBGL = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    this._batches = new Map();   // key -> Batch
    this._mergeBatches = new Map(); // ShaderMaterial -> ShaderMergeBatch
    this._seenMerge = new Set();
    // (custom ShaderMaterial pipelines are cached on each material instance)
    this._points = new Map();    // mesh.id -> points GPU state
    this._registered = new WeakSet();
    this._seen = new Set();      // mesh ids seen this frame (for removal GC)

    this._meshById = new Map();
  }

  _buildSizedTargets() {
    const w = this.canvas.width, h = this.canvas.height;
    this.depthTexture = this.device.resources.createTexture({
      size: [w, h], format: 'depth24plus', sampleCount: this.sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    if (this.sampleCount > 1) {
      this.msaaColor = this.device.resources.createTexture({
        size: [w, h], format: this.format, sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    } else {
      this.msaaColor = null;
    }
  }

  setSize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.depthTexture.destroy();
    if (this.msaaColor) this.msaaColor.destroy();
    this._buildSizedTargets();
  }

  // --- Optional directional-sun shadows ---
  //
  // Off by default; apps that want their own shadow scheme just never call this.
  // `bounds` is the world-space AABB the shadow map covers (for a planet, a box
  // enclosing it, e.g. { min:[-200,-200,-200], max:[200,200,200] }). Casters are
  // every shadow-batched mesh; a material/mesh opts out with castShadow:false.
  // Receivers are lambert batches and any ShaderMaterial with receiveShadow:true
  // (e.g. the terrain shader), which sample group(0) bindings 3-5 / its own
  // shadow group.
  enableShadows({ size = 2048, bounds, direction = [0.4, -0.7, 0.35] } = {}) {
    if (!this.shadowMap) this.shadowMap = new ShadowMap(this.device, { mapSize: size });
    this.shadowEnabled = true;
    this.shadowBounds = bounds || { min: [-200, -200, -200], max: [200, 200, 200] };
    this.shadowDirection = direction;
    // A pseudo-camera whose frustum is the light's, so each batch's
    // MultiDrawSystem can cull casters against the shadow volume and the depth
    // pass draws them. projection = lightViewProj, view = identity.
    if (!this._lightCamera) this._lightCamera = new Camera(this.device);
    this._refreshShadow();
    // Existing per-camera scene bind groups reference the dummy shadow view —
    // drop them so they rebuild against the real shadow map.
    for (const batch of this._batches.values()) batch._perCamera.clear();
  }

  disableShadows() {
    this.shadowEnabled = false;
    this.device.queue.writeBuffer(this.shadowParamsBuffer.gpuBuffer, 0, new Float32Array(20)); // enabled=0
    for (const batch of this._batches.values()) batch._perCamera.clear();
  }

  /** Sets the sun's travel direction (world space) for the shadow map. */
  setShadowLight(direction) {
    this.shadowDirection = direction;
    if (this.shadowEnabled) this._refreshShadow();
  }

  // Rebuilds the light-space matrix from the current direction + bounds and
  // mirrors it into shadowParamsBuffer (lightViewProj + dir + enabled=1).
  _refreshShadow() {
    this.shadowMap.update(this.shadowDirection, this.shadowBounds);
    const data = new Float32Array(20);
    data.set(this.shadowMap.viewProjectionMatrix, 0);
    const d = this.shadowDirection;
    const len = Math.hypot(d[0], d[1], d[2]) || 1;
    data[16] = d[0] / len; data[17] = d[1] / len; data[18] = d[2] / len;
    data[19] = 1; // enabled
    this.device.queue.writeBuffer(this.shadowParamsBuffer.gpuBuffer, 0, data);
    // Drive the light camera's frustum from the shadow matrix (proj = lightVP,
    // view = identity) so batches cull casters against the shadow volume.
    this._lightCamera.setProjectionMatrix(this.shadowMap.viewProjectionMatrix);
    this._lightCamera.setViewMatrix(identity());
    this._lightCamera.update();
  }

  // --- batch management ---

  _batchKey(mat) {
    const fog = mat.fog === false ? 0 : 1;
    if (mat.kind === 'lambert') return `lambert:f${fog}`;
    if (mat.kind === 'basic') {
      const blend = mat.blending === 'additive' ? 'add' : (mat.transparent ? 'alpha' : 'opaque');
      return `basic:${blend}:dw${mat.depthWrite ? 1 : 0}:dt${mat.depthTest ? 1 : 0}:s${mat.side}:w${mat.wireframe ? 1 : 0}:f${fog}`;
    }
    return null;
  }

  _getBatch(mat, cameraBuffer) {
    const key = this._batchKey(mat);
    if (this._batches.has(key)) return this._batches.get(key);
    const batch = new Batch(this, mat, key);
    this._batches.set(key, batch);
    return batch;
  }

  _getMergeBatch(material) {
    let mb = this._mergeBatches.get(material);
    if (!mb) { mb = new ShaderMergeBatch(this, material); this._mergeBatches.set(material, mb); }
    return mb;
  }

  // Releases all GPU resources created on this renderer's device (each game
  // session creates a fresh device, so this frees the whole session's
  // resources). After dispose() the renderer must not be used again.
  dispose() {
    this.device.destroyAll();
  }

  // --- per-frame ---

  /**
   * @param {Scene} scene retained scene graph
   * @param {Camera} camera engine Camera (already update()d by caller, or we update it)
   * @param {object} [opts] { viewport:[x,y,w,h], scissor:[x,y,w,h], layerMask, clear:bool }
   */
  render(scene, camera, opts = {}) {
    const layerMask = opts.layerMask ?? 0xffffffff;
    scene.updateWorldMatrix(null);

    // Gather lights + fog from the scene.
    this._uploadLights(scene);
    this._uploadFog(scene);

    // Sync the scene → GPU buffers ONCE per frame (not per camera). A frame is
    // identified by opts.frame; if two render() calls share it (main + minimap),
    // the second reuses the already-synced state. If no frame id is given, mint a
    // fresh monotonic id each call so every call re-syncs — and, crucially, so
    // the per-frame guards downstream (e.g. shader-material uniform upload at
    // `r._uFrame !== this._syncedFrame`) actually advance. (A plain `undefined`
    // here would equal the initial `_uFrame`, silently skipping those uploads.)
    const frame = opts.frame !== undefined ? opts.frame : (this._autoFrame = (this._autoFrame ?? 0) + 1);
    const alreadySynced = frame === this._syncedFrame;
    let shaderMeshes = this._shaderMeshes;
    let pointsMeshes = this._pointsMeshes;

    if (!alreadySynced) {
      this._seen.clear();
      shaderMeshes = []; pointsMeshes = [];
      this._seenMerge.clear();
      scene.traverse((node) => {
        if (!node.geometry || !node.material || node.visible === false) return;
        if (!this._ancestorsVisible(node)) return;
        const kind = node.material.kind;
        if (kind === 'shader') {
          if (node.material.merge) {
            // Merge-batched custom shader (e.g. terrain): packed into one buffer.
            const mb = this._getMergeBatch(node.material);
            mb.sync(node);
            this._seenMerge.add(node.id);
          } else {
            shaderMeshes.push(node); // per-mesh custom shader
          }
          return;
        }
        if (kind === 'points') { pointsMeshes.push(node); return; }
        // NOTE: no layerMask gate here — layer filtering happens per-camera in
        // the GPU cull pass, so the shared sync covers all cameras' objects.
        const batch = this._getBatch(node.material);
        batch.sync(node);
        this._seen.add(node.id);
      });
      for (const batch of this._batches.values()) { batch.gc(this._seen); batch.syncRecords(); }
      for (const mb of this._mergeBatches.values()) mb.gc(this._seenMerge);
      this._shaderMeshes = shaderMeshes;
      this._pointsMeshes = pointsMeshes;
      this._syncedFrame = frame;
    }

    // Per-camera cull setup (object count + this camera's layer mask).
    for (const batch of this._batches.values()) batch.cullFor(camera, layerMask);

    const graph = new RenderGraph(this.device);
    graph.setCanvasTarget(this.context);

    // Cull passes (compute) for each batch, for THIS camera.
    for (const batch of this._batches.values()) {
      if (batch.count === 0) continue;
      const pc = batch._perCamera.get(camera);
      graph.addPass({
        name: `cull-${batch.key}`,
        writes: [pc.multi.drawArgsBuffer, pc.multi.drawCountBuffer, pc.multi.slotToObjectBuffer],
        reads: [camera.buffer, batch.worldBuffer, batch.boundsBuffer, pc.multi.recordBuffer],
        execute: (encoder) => pc.multi.build(encoder),
      });
    }

    // --- Shadow pass (optional): cull casters for the light camera, then render
    // their depth into the shadow map. Only the FIRST render() of a frame needs
    // it (the shadow map is shared across cameras), so skip when this camera
    // shares an already-synced frame's shadow build.
    if (this.shadowEnabled && this._shadowBuiltFrame !== frame) {
      this._shadowBuiltFrame = frame;
      const lc = this._lightCamera;
      const shadowBatches = [];
      for (const batch of this._batches.values()) {
        if (batch.count === 0 || batch._sampleMaterial.castShadow === false) continue;
        batch.cullFor(lc, 0xffffffff);
        shadowBatches.push(batch);
      }
      for (const batch of shadowBatches) {
        const pc = batch._perCamera.get(lc);
        graph.addPass({
          name: `shadow-cull-${batch.key}`,
          writes: [pc.multi.drawArgsBuffer, pc.multi.drawCountBuffer, pc.multi.slotToObjectBuffer],
          reads: [lc.buffer, batch.worldBuffer, batch.boundsBuffer, pc.multi.recordBuffer],
          execute: (encoder) => pc.multi.build(encoder),
        });
      }
      const shadowReads = [lc.buffer, this.shadowParamsBuffer];
      for (const batch of shadowBatches) {
        const pc = batch._perCamera.get(lc);
        shadowReads.push(pc.multi.drawArgsBuffer, pc.multi.slotToObjectBuffer, batch.worldBuffer);
      }
      // Merge batches (terrain) cast too — their vertices are world-space, drawn
      // straight from their pages in the depth pass.
      const mergeCasters = [...this._mergeBatches.values()].filter(mb => mb.material.castShadow !== false);
      graph.addPass({
        name: 'shadow-depth',
        depthStencilAttachment: { target: this.shadowMap.depthTexture, view: this.shadowMap.getView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
        writes: [this.shadowMap.depthTexture],
        reads: shadowReads,
        execute: (rp) => {
          for (const batch of shadowBatches) batch.drawShadow(rp, lc);
          for (const mb of mergeCasters) mb.drawShadow(rp);
        },
      });
    }

    // Single forward pass: shader-meshes, opaque batches, transparent batches, points.
    const msaa = this.sampleCount > 1;
    const colorAttachment = msaa
      ? { target: this.msaaColor, resolveTarget: CANVAS, clearValue: this._clearColor(scene), loadOp: opts.clear === false ? 'load' : 'clear', storeOp: 'store' }
      : { target: CANVAS, clearValue: this._clearColor(scene), loadOp: opts.clear === false ? 'load' : 'clear', storeOp: 'store' };

    // Declare reads of each batch's cull outputs so the render graph orders the
    // cull compute passes before this forward pass.
    const forwardReads = [camera.buffer, this.lightsBuffer, this.fogBuffer];
    for (const batch of this._batches.values()) {
      if (batch.count === 0) continue;
      const pc = batch._perCamera.get(camera);
      forwardReads.push(pc.multi.drawArgsBuffer, pc.multi.slotToObjectBuffer, batch.worldBuffer, batch.materialBuffer);
    }

    graph.addPass({
      name: 'scene-forward',
      colorAttachments: [colorAttachment],
      depthStencilAttachment: { target: this.depthTexture, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [this.depthTexture],
      reads: forwardReads,
      execute: (rp) => {
        if (opts.viewport) rp.setViewport(opts.viewport[0], opts.viewport[1], opts.viewport[2], opts.viewport[3], 0, 1);
        if (opts.scissor) rp.setScissorRect(opts.scissor[0], opts.scissor[1], opts.scissor[2], opts.scissor[3]);

        // Merge-batched shader meshes (terrain) — one buffer, ~1 draw call.
        for (const mb of this._mergeBatches.values()) mb.draw(rp, camera, this._syncedFrame);

        // Per-mesh custom-shader meshes.
        for (const mesh of shaderMeshes) this._drawShaderMesh(rp, mesh, camera);

        // Opaque batches, then transparent (additive/alpha) batches.
        const batches = [...this._batches.values()].filter(b => b.count > 0);
        batches.sort((a, b) => (a.transparent ? 1 : 0) - (b.transparent ? 1 : 0));
        for (const batch of batches) batch.draw(rp, camera);

        // Points last (additive-ish, no depth write).
        for (const mesh of pointsMeshes) this._drawPoints(rp, mesh, camera);
      },
    });

    graph.execute();
  }

  _ancestorsVisible(node) {
    let p = node.parent;
    while (p) { if (p.visible === false) return false; p = p.parent; }
    return true;
  }

  _clearColor(scene) {
    const c = scene.background || { r: 0, g: 0, b: 0 };
    return { r: c.r ?? 0, g: c.g ?? 0, b: c.b ?? 0, a: 1 };
  }

  _uploadLights(scene) {
    const data = new Float32Array(4 + MAX_LIGHTS * 8);
    let count = 0;
    let ambR = 0, ambG = 0, ambB = 0;
    const lights = [];
    scene.traverse((n) => {
      if (n.isAmbient) { ambR += n.color.r * n.intensity; ambG += n.color.g * n.intensity; ambB += n.color.b * n.intensity; }
      else if (n.isPointLight && lights.length < MAX_LIGHTS) lights.push({ point: true, n });
      else if (n.isDirectional && lights.length < MAX_LIGHTS) lights.push({ point: false, n });
    });
    data[0] = ambR; data[1] = ambG; data[2] = ambB; data[3] = lights.length;
    lights.forEach((L, i) => {
      const off = 4 + i * 8;
      const n = L.n;
      if (L.point) {
        data[off] = n.position.x; data[off + 1] = n.position.y; data[off + 2] = n.position.z;
        // w: 1 = quadratic distance falloff, 2 = no falloff (decay 0).
        data[off + 3] = n.decay === 0 ? 2 : 1;
      }
      else { const d = n.direction || n.position; data[off] = d.x; data[off + 1] = d.y; data[off + 2] = d.z; data[off + 3] = 0; }
      data[off + 4] = n.color.r; data[off + 5] = n.color.g; data[off + 6] = n.color.b; data[off + 7] = n.intensity;
    });
    this.device.queue.writeBuffer(this.lightsBuffer.gpuBuffer, 0, data);
  }

  _uploadFog(scene) {
    const f = scene.fog;
    const data = f
      ? new Float32Array([f.color.r, f.color.g, f.color.b, 1, f.near, f.far, 0, 0])
      : new Float32Array([0, 0, 0, 0, 0, 1, 0, 0]);
    this.device.queue.writeBuffer(this.fogBuffer.gpuBuffer, 0, data);
  }

  // --- custom ShaderMaterial (per-mesh, app-supplied WGSL) ---
  //
  // The engine builds a pipeline from the material's own WGSL + declared
  // vertex attributes, with camera at group(0) and one app-controlled uniform
  // buffer at group(1). It has no knowledge of what the shader computes — the
  // app fills the uniform bytes via material.updateUniforms(view).

  _ensureShaderPipeline(material, geometry) {
    // One pipeline per material instance (cached on the material).
    if (material._pipeline) return material._pipeline;
    const device = this.device;
    const module = device.device.createShaderModule({ code: material.wgsl });
    const cameraBGL = device.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    const uBGL = device.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    // Build vertex buffer layouts from the geometry's named attributes, in the
    // material's declared order; shaderLocation == declaration index.
    const buffers = material.attributes.map((name, loc) => {
      const attr = geometry.attributes[name];
      if (!attr) throw new Error(`ShaderMaterial: geometry missing attribute "${name}"`);
      return { arrayStride: VERTEX_FORMAT_BYTES[attr.format], attributes: [{ format: attr.format, offset: 0, shaderLocation: loc }] };
    });
    const cull = material.side === 'double' ? 'none' : (material.side === 'back' ? 'front' : 'back');
    const pipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [cameraBGL, uBGL] }),
      vertex: { module, entryPoint: 'vertexMain', buffers },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: this.format }] },
      primitive: { topology: material.topology, cullMode: material.topology === 'triangle-list' ? cull : 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: material.depthWrite, depthCompare: material.depthCompare },
      multisample: { count: this.sampleCount },
    });
    material._pipeline = { pipeline, cameraBGL, uBGL };
    return material._pipeline;
  }

  _drawShaderMesh(rp, mesh, camera) {
    const material = mesh.material;
    const p = this._ensureShaderPipeline(material, mesh.geometry);
    const r = mesh._render || (mesh._render = { cameraBGs: new Map() });
    if (!r.cameraBGs) r.cameraBGs = new Map();
    // Camera bind group is per-camera (so the minimap uses its own matrices).
    let cameraBG = r.cameraBGs.get(camera);
    if (!cameraBG) {
      cameraBG = this.device.device.createBindGroup({
        layout: p.cameraBGL, entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
      });
      r.cameraBGs.set(camera, cameraBG);
    }
    if (!r.uBuffer && material.uniformSize > 0) {
      r.uBuffer = this.device.resources.createBuffer({ size: material.uniformSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      r.uView = new Float32Array(material.uniformSize / 4);
      r.uBG = this.device.device.createBindGroup({
        layout: p.uBGL, entries: [{ binding: 0, resource: { buffer: r.uBuffer.gpuBuffer } }],
      });
    }
    // Uniforms are camera-independent — upload once per frame, not per camera.
    if (r.uBuffer && material.updateUniforms && r._uFrame !== this._syncedFrame) {
      r._uFrame = this._syncedFrame;
      material.updateUniforms(r.uView);
      this.device.queue.writeBuffer(r.uBuffer.gpuBuffer, 0, r.uView);
    }

    rp.setPipeline(p.pipeline);
    rp.setBindGroup(0, cameraBG);
    if (r.uBG) rp.setBindGroup(1, r.uBG);
    const g = mesh.geometry;
    material.attributes.forEach((name, loc) => rp.setVertexBuffer(loc, g.attributes[name].buffer.gpuBuffer));
    rp.draw(g.vertexCount);
  }

  // --- points ---

  _drawPoints(rp, mesh, camera) {
    let s = this._points.get(mesh.id);
    if (!s) { s = this._buildPoints(mesh, camera); this._points.set(mesh.id, s); }
    // Per-camera bind group (minimap uses its own matrices).
    let cameraBG = s.cameraBGs.get(camera);
    if (!cameraBG) {
      cameraBG = this.device.device.createBindGroup({ layout: this._pointsPipeline.cameraBGL, entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }] });
      s.cameraBGs.set(camera, cameraBG);
    }
    // Update NDC half-size from the material's pixel size (once per frame).
    if (s._frame !== this._syncedFrame) {
      s._frame = this._syncedFrame;
      const sizePx = mesh.material.size;
      const ndcX = sizePx / this.canvas.width, ndcY = sizePx / this.canvas.height;
      this.device.queue.writeBuffer(s.paramsBuffer.gpuBuffer, 0, new Float32Array([
        mesh.material.color.r, mesh.material.color.g, mesh.material.color.b, 0,
        ndcX, ndcY, 0, 0,
      ]));
    }
    s.cameraBG = cameraBG;
    rp.setPipeline(s.pipeline);
    rp.setBindGroup(0, s.cameraBG);
    rp.setBindGroup(1, s.bindGroup);
    rp.draw(6, s.count);
  }

  _buildPoints(mesh, camera) {
    const device = this.device;
    if (!this._pointsPipeline) {
      const module = device.device.createShaderModule({ code: pointsShader });
      const cameraBGL = device.device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
      });
      const dataBGL = device.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ],
      });
      const pipeline = device.device.createRenderPipeline({
        layout: device.device.createPipelineLayout({ bindGroupLayouts: [cameraBGL, dataBGL] }),
        vertex: { module, entryPoint: 'vertexMain' },
        fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
        multisample: { count: this.sampleCount },
      });
      this._pointsPipeline = { pipeline, cameraBGL, dataBGL };
    }
    const pp = this._pointsPipeline;
    // The mesh's geometry holds point positions in attribute 'position'.
    const posData = mesh.geometry.attributes.position.data;
    const count = posData.length / 3;
    const positionsBuffer = device.resources.createBuffer({ size: posData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(positionsBuffer.gpuBuffer, 0, posData);
    const paramsBuffer = device.resources.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bindGroup = device.device.createBindGroup({
      layout: pp.dataBGL,
      entries: [
        { binding: 0, resource: { buffer: positionsBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.gpuBuffer } },
      ],
    });
    return { pipeline: pp.pipeline, cameraBGs: new Map(), bindGroup, positionsBuffer, paramsBuffer, count };
  }
}

// A merged batch for a `merge: true` ShaderMaterial: many static, identity-
// transform meshes (e.g. terrain chunks) packed into ONE non-indexed vertex
// buffer and drawn in as few draw() calls as possible (one per contiguous live
// span — usually 1). No per-object transform, no culling, no indirect: the
// vertices are world-space and the shader reads only the material's shared
// uniform. This turns thousands of chunk draws into ~1.
class ShaderMergeBatch {
  constructor(renderer, material) {
    this.renderer = renderer;
    this.device = renderer.device;
    this.material = material;
    this.floatsPerVertex = material.attributes.reduce((s, n) => s + ATTR_FLOATS[n], 0);
    this.stride = this.floatsPerVertex * 4;

    // A list of fixed-size pages instead of one giant buffer: WebGPU caps a
    // single buffer at maxBufferSize (often 256MB). Terrain across thousands of
    // chunks can exceed that, so we page it. Each page is drawn with one draw()
    // per contiguous live span (≈ a handful of calls total).
    const maxBuf = (this.device.device.limits && this.device.device.limits.maxBufferSize) || (256 * 1024 * 1024);
    this.pageVerts = Math.floor(Math.min(maxBuf, 128 * 1024 * 1024) / this.stride);
    this.pages = []; // { buffer, head, slots:Map(id->{offset,count}), free:[], dirty:true, spanCache:[] }

    this._slots = new Map();       // mesh.id -> { page, offset, count }

    this.pipeline = null;
    this.uBuffer = null;
    this.uView = null;
    this._cameraBGs = new Map();
    this._uFrame = -1;
  }

  _newPage() {
    const page = {
      buffer: this.device.resources.createBuffer({
        size: this.pageVerts * this.stride,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      }),
      head: 0, slots: new Map(), free: [], dirty: true, spanCache: [],
    };
    this.pages.push(page);
    return page;
  }

  _ensurePipeline() {
    if (this.pipeline) return;
    const device = this.device;
    const m = this.material;
    const module = device.device.createShaderModule({ code: m.wgsl });
    this._cameraBGL = device.device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
    const uBGL = device.device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
    // Interleaved vertex layout from the material's attributes.
    let offset = 0;
    const attrs = m.attributes.map((name, loc) => {
      const a = { format: ATTR_FORMAT[name], offset, shaderLocation: loc };
      offset += ATTR_FLOATS[name] * 4;
      return a;
    });
    const cull = m.side === 'double' ? 'none' : (m.side === 'back' ? 'front' : 'back');

    // Optional shadow-receive group(2): shadowParams + shadow map + comparison
    // sampler, when the renderer has shadows on AND this material opts in
    // (material.receiveShadow). The material's WGSL must declare the matching
    // group(2) bindings (see the game's terrain shader).
    this._receivesShadow = this.renderer.shadowEnabled && m.receiveShadow === true;
    const layouts = [this._cameraBGL, uBGL];
    if (this._receivesShadow) {
      this._shadowBGL = device.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        ],
      });
      layouts.push(this._shadowBGL);
    }

    this.pipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: layouts }),
      vertex: { module, entryPoint: 'vertexMain', buffers: [{ arrayStride: this.stride, attributes: attrs }] },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: this.renderer.format }] },
      primitive: { topology: m.topology, cullMode: m.topology === 'triangle-list' ? cull : 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: m.depthWrite, depthCompare: m.depthCompare },
      multisample: { count: this.renderer.sampleCount },
    });
    if (m.uniformSize > 0) {
      this.uBuffer = device.resources.createBuffer({ size: m.uniformSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.uView = new Float32Array(m.uniformSize / 4);
      this.uBG = device.device.createBindGroup({ layout: uBGL, entries: [{ binding: 0, resource: { buffer: this.uBuffer.gpuBuffer } }] });
    }
    if (this._receivesShadow) {
      const r = this.renderer;
      this.shadowBG = device.device.createBindGroup({
        layout: this._shadowBGL,
        entries: [
          { binding: 0, resource: { buffer: r.shadowParamsBuffer.gpuBuffer } },
          { binding: 1, resource: r.shadowMap.getView() },
          { binding: 2, resource: r.shadowSampler.gpuSampler },
        ],
      });
    }
  }

  // Interleave a mesh's named attribute arrays into this batch's vertex format.
  _interleave(mesh) {
    const attrs = this.material.attributes;
    const n = mesh.geometry.attributes[attrs[0]].data.length / ATTR_FLOATS[attrs[0]];
    const out = new Float32Array(n * this.floatsPerVertex);
    let base = 0;
    for (let v = 0; v < n; v++) {
      let o = base;
      for (const name of attrs) {
        const fpv = ATTR_FLOATS[name];
        const src = mesh.geometry.attributes[name].data;
        for (let k = 0; k < fpv; k++) out[o++] = src[v * fpv + k];
      }
      base += this.floatsPerVertex;
    }
    return { data: out, count: n };
  }

  sync(mesh) {
    if (this._slots.has(mesh.id)) return;
    const { data, count } = this._interleave(mesh);
    if (count > this.pageVerts) { console.warn('ShaderMergeBatch: mesh exceeds page size'); return; }

    // Find a page with a fitting free span or tail room; else open a new page.
    let chosen = null, offset = -1;
    for (const page of this.pages) {
      for (let i = 0; i < page.free.length; i++) {
        if (page.free[i].count >= count) {
          offset = page.free[i].offset;
          if (page.free[i].count === count) page.free.splice(i, 1);
          else { page.free[i].offset += count; page.free[i].count -= count; }
          chosen = page; break;
        }
      }
      if (chosen) break;
      if (page.head + count <= this.pageVerts) { chosen = page; offset = page.head; page.head += count; break; }
    }
    if (!chosen) { chosen = this._newPage(); offset = chosen.head; chosen.head += count; }

    this.device.queue.writeBuffer(chosen.buffer.gpuBuffer, offset * this.stride, data);
    chosen.slots.set(mesh.id, { offset, count });
    chosen.dirty = true;
    this._slots.set(mesh.id, { page: chosen, offset, count });
  }

  gc(seenIds) {
    for (const [id, rec] of [...this._slots]) {
      if (!seenIds.has(id)) {
        rec.page.free.push({ offset: rec.offset, count: rec.count });
        rec.page.slots.delete(id);
        rec.page.dirty = true;
        this._slots.delete(id);
      }
    }
  }

  // Contiguous live vertex spans within a page (merging adjacent allocations).
  _pageSpans(page) {
    if (!page.dirty) return page.spanCache;
    const live = [...page.slots.values()].sort((a, b) => a.offset - b.offset);
    const spans = [];
    for (const s of live) {
      const last = spans[spans.length - 1];
      if (last && last.offset + last.count === s.offset) last.count += s.count;
      else spans.push({ offset: s.offset, count: s.count });
    }
    page.spanCache = spans;
    page.dirty = false;
    return spans;
  }

  // Depth-only render of this merge batch's vertices into the sun shadow map.
  // Terrain vertices are already world-space (identity model), so the shadow
  // vertex shader is just lightViewProj * pos. Built lazily; skipped when the
  // material opts out via castShadow:false.
  _ensureShadowPipeline() {
    if (this._shadowPipe !== undefined) return this._shadowPipe;
    if (this.material.castShadow === false) { this._shadowPipe = null; return null; }
    const device = this.device;
    const code = /* wgsl */ `
struct ShadowParams { lightViewProj: mat4x4f, lightDirEnabled: vec4f, };
@group(0) @binding(0) var<uniform> shadow: ShadowParams;
@vertex
fn vertexMain(@location(0) p: vec3f) -> @builtin(position) vec4f {
  return shadow.lightViewProj * vec4f(p, 1.0);
}`;
    const module = device.device.createShaderModule({ code });
    const bgl = device.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    });
    this._shadowPipe = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      // Only the position attribute matters; bind the full interleaved stride and
      // read location 0 (position is always attribute 0 of the merge layout).
      vertex: { module, entryPoint: 'vertexMain', buffers: [{ arrayStride: this.stride, attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }] }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this._shadowBG = device.device.createBindGroup({
      layout: bgl, entries: [{ binding: 0, resource: { buffer: this.renderer.shadowParamsBuffer.gpuBuffer } }],
    });
    return this._shadowPipe;
  }

  drawShadow(rp) {
    if (this._slots.size === 0) return;
    if (!this._ensureShadowPipeline()) return;
    rp.setPipeline(this._shadowPipe);
    rp.setBindGroup(0, this._shadowBG);
    for (const page of this.pages) {
      const spans = this._pageSpans(page);
      if (spans.length === 0) continue;
      rp.setVertexBuffer(0, page.buffer.gpuBuffer);
      for (const s of spans) rp.draw(s.count, 1, s.offset);
    }
  }

  draw(rp, camera, syncedFrame) {
    this._ensurePipeline();
    if (this._slots.size === 0) return;
    let camBG = this._cameraBGs.get(camera);
    if (!camBG) {
      camBG = this.device.device.createBindGroup({ layout: this._cameraBGL, entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }] });
      this._cameraBGs.set(camera, camBG);
    }
    if (this.uBuffer && this.material.updateUniforms && this._uFrame !== syncedFrame) {
      this._uFrame = syncedFrame;
      this.material.updateUniforms(this.uView);
      this.device.queue.writeBuffer(this.uBuffer.gpuBuffer, 0, this.uView);
    }
    rp.setPipeline(this.pipeline);
    rp.setBindGroup(0, camBG);
    if (this.uBG) rp.setBindGroup(1, this.uBG);
    if (this.shadowBG) rp.setBindGroup(2, this.shadowBG);
    for (const page of this.pages) {
      const spans = this._pageSpans(page);
      if (spans.length === 0) continue;
      rp.setVertexBuffer(0, page.buffer.gpuBuffer);
      for (const s of spans) rp.draw(s.count, 1, s.offset);
    }
  }
}

// Floats + WGSL vertex format per named attribute, for merge-batch interleaving.
const ATTR_FLOATS = { position: 3, normal: 3, uv: 2, color: 3, skyAccess: 1 };
const ATTR_FORMAT = { position: 'float32x3', normal: 'float32x3', uv: 'float32x2', color: 'float32x3', skyAccess: 'float32' };

// One pipeline-group's worth of objects: arena + per-object storage + cull.
class Batch {
  constructor(renderer, sampleMaterial, key) {
    this.renderer = renderer;
    this.key = key;
    this.device = renderer.device;
    this.transparent = key !== 'lambert' && !key.includes(':opaque:');
    this.count = 0;

    this.arena = new GeometryArena(this.device);
    this.worldBuffer = this.device.resources.createBuffer({ size: OBJ_CAP * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.materialBuffer = this.device.resources.createBuffer({ size: OBJ_CAP * MATERIAL_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.boundsBuffer = this.device.resources.createBuffer({ size: OBJ_CAP * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    this._slots = new Map();    // mesh.id -> { slot, alloc, world(Float32Array cache), matKey }
    this._freeSlots = [];
    this._nextSlot = 0;

    // Per-camera GPU state: each camera gets its own MultiDrawSystem (cull +
    // indirect args) and scene bind group, but they SHARE this batch's record /
    // world / material / bounds buffers — so a second view (e.g. the minimap)
    // culls the same objects from its own viewpoint. The pipeline + objectBG
    // are camera-independent and built once.
    this._perCamera = new Map();
    this.pipeline = null;
    this.objectBG = null;
    this._sampleMaterial = sampleMaterial;
    this.count = 0;
  }

  // Returns (creating if needed) the per-camera cull system + scene bind group.
  _forCamera(camera) {
    let pc = this._perCamera.get(camera);
    if (pc) return pc;

    // The first camera's MultiDrawSystem owns the record buffer; later cameras
    // share it (and the draw-slot layout) so they cull the same scene.
    const first = this._perCamera.values().next().value;
    const multi = new MultiDrawSystem(this.device, camera, this.worldBuffer, this.boundsBuffer, OBJ_CAP,
      first ? { recordBuffer: first.multi.recordBuffer, drawSlotBindGroupLayout: first.multi.drawSlotBindGroupLayout } : {});

    if (!this.pipeline) {
      this._multiForLayout = multi; // pipeline layout needs a drawSlotBindGroupLayout
      this.pipeline = this._buildPipeline();
      this.objectBG = this.device.device.createBindGroup({
        layout: this.renderer.objectBGL,
        entries: [
          { binding: 0, resource: { buffer: this.worldBuffer.gpuBuffer } },
          { binding: 1, resource: { buffer: this.materialBuffer.gpuBuffer } },
        ],
      });
    }

    const r = this.renderer;
    const shadowView = (r.shadowEnabled && r.shadowMap)
      ? r.shadowMap.getView()
      : r._dummyShadowTex.gpuTexture.createView();
    const sceneBG = this.device.device.createBindGroup({
      layout: r.sceneBGL,
      entries: [
        { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: r.lightsBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: r.fogBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: r.shadowParamsBuffer.gpuBuffer } },
        { binding: 4, resource: shadowView },
        { binding: 5, resource: r.shadowSampler.gpuSampler },
      ],
    });
    pc = { multi, sceneBG };
    this._perCamera.set(camera, pc);
    return pc;
  }

  // Lazily builds the depth-only pipeline used to render this batch's casters
  // into the sun shadow map. group(0)=shadowParams, group(1)=worldMatrices,
  // and (slot path only) group(2)=drawSlot. No fragment/color target.
  _shadowPipeline() {
    if (this._shadowPipe) return this._shadowPipe;
    const device = this.device;
    const byFirstInstance = this._multiForLayout.firstInstanceId;
    const module = device.device.createShaderModule({ code: shadowDepthShader(byFirstInstance) });
    // group(0) = shadowParams uniform; reuse a tiny dedicated layout.
    if (!this.renderer._shadowDepthBGL) {
      this.renderer._shadowDepthBGL = device.device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
      });
      this.renderer._shadowWorldBGL = device.device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
      });
    }
    const layouts = byFirstInstance
      ? [this.renderer._shadowDepthBGL, this.renderer._shadowWorldBGL]
      : [this.renderer._shadowDepthBGL, this.renderer._shadowWorldBGL, this._multiForLayout.drawSlotBindGroupLayout];
    this._shadowPipe = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: layouts }),
      vertex: { module, entryPoint: 'vertexMain', buffers: this.arena.vertexBufferLayouts },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this._shadowParamsBG = device.device.createBindGroup({
      layout: this.renderer._shadowDepthBGL,
      entries: [{ binding: 0, resource: { buffer: this.renderer.shadowParamsBuffer.gpuBuffer } }],
    });
    this._shadowWorldBG = device.device.createBindGroup({
      layout: this.renderer._shadowWorldBGL,
      entries: [{ binding: 0, resource: { buffer: this.worldBuffer.gpuBuffer } }],
    });
    return this._shadowPipe;
  }

  // Records this batch's casters into the active shadow (depth) render pass,
  // using `camera`'s already-built cull/draw args (the light camera).
  drawShadow(rp, camera) {
    if (this.count === 0) return;
    if (this._sampleMaterial.castShadow === false) return;
    const pc = this._perCamera.get(camera);
    if (!pc) return;
    this._shadowPipeline();
    rp.setPipeline(this._shadowPipe);
    rp.setBindGroup(0, this._shadowParamsBG);
    rp.setBindGroup(1, this._shadowWorldBG);
    this.arena.bind(rp);
    pc.multi.drawAll(rp, 2);
  }

  _buildPipeline() {
    const device = this.device;
    const mat = this._sampleMaterial;
    const isLambert = mat.kind === 'lambert';
    // Object id comes from firstInstance (instance_index) whenever the device
    // supports it (multi-draw OR indirect-first-instance) — no group(2). Only
    // the last-resort path needs the slotToObject draw-slot group.
    const byFirstInstance = this._multiForLayout.firstInstanceId;
    const fogEnabled = mat.fog !== false;
    const code = (isLambert ? lambertShader : basicShader)(byFirstInstance, fogEnabled);
    const module = device.device.createShaderModule({ code });
    const bindGroupLayouts = byFirstInstance
      ? [this.renderer.sceneBGL, this.renderer.objectBGL]
      : [this.renderer.sceneBGL, this.renderer.objectBGL, this._multiForLayout.drawSlotBindGroupLayout];

    let blend = undefined;
    if (mat.kind === 'basic' && mat.blending === 'additive') {
      blend = { color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    } else if (mat.kind === 'basic' && mat.transparent) {
      blend = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } };
    }
    const cull = mat.side === 'double' ? 'none' : (mat.side === 'back' ? 'front' : 'back');
    const depthWrite = mat.kind === 'basic' ? mat.depthWrite : true;
    const depthCompare = (mat.kind === 'basic' && mat.depthTest === false) ? 'always' : 'less';
    const topology = (mat.kind === 'basic' && mat.wireframe) ? 'line-list' : 'triangle-list';

    return device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts }),
      vertex: { module, entryPoint: 'vertexMain', buffers: this.arena.vertexBufferLayouts },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: this.renderer.format, blend }] },
      primitive: { topology, cullMode: topology === 'line-list' ? 'none' : cull },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: depthWrite, depthCompare },
      multisample: { count: this.renderer.sampleCount },
    });
  }

  /**
   * Registers a new mesh, or refreshes an existing one's transform/material
   * only when it actually changed. Called ONCE per frame per mesh (not per
   * camera) — the GPU buffers are shared across cameras. Avoiding the
   * unconditional per-frame upload is what keeps CPU cost low for the static
   * bulk (e.g. terrain chunks that never move after meshing).
   */
  sync(mesh) {
    let entry = this._slots.get(mesh.id);
    if (!entry) {
      const slot = this._freeSlots.length ? this._freeSlots.pop() : this._nextSlot++;
      const { vertexData, indexData } = interleaveStandard({
        positions: mesh.geometry.attributes.position.data,
        normals: mesh.geometry.attributes.normal.data,
        uvs: mesh.geometry.attributes.uv ? mesh.geometry.attributes.uv.data : null,
      });
      const alloc = this.arena.allocate(vertexData, indexData);
      entry = { slot, alloc, world: new Float32Array(16), matKey: '', layers: -1, frustumCulled: null };
      this._slots.set(mesh.id, entry);
      const b = computeBounds(mesh.geometry.attributes.position.data);
      this.device.queue.writeBuffer(this.boundsBuffer.gpuBuffer, slot * 32, new Float32Array([...b.min, 0, ...b.max, 0]));
      this._recordDirty = true;
    }

    // World matrix — recompute, compare, upload only if changed.
    mesh.updateWorldMatrix(mesh.parent ? mesh.parent.worldMatrix : null);
    const wm = mesh.worldMatrix, cache = entry.world;
    let changed = false;
    for (let i = 0; i < 16; i++) { if (cache[i] !== wm[i]) { changed = true; break; } }
    if (changed) { cache.set(wm); this.device.queue.writeBuffer(this.worldBuffer.gpuBuffer, entry.slot * 64, cache); }

    // Material color + opacity — upload only when changed.
    const c = mesh.material.color, op = mesh.material.opacity ?? 1;
    const matKey = `${c.r},${c.g},${c.b},${op}`;
    if (matKey !== entry.matKey) {
      entry.matKey = matKey;
      this.device.queue.writeBuffer(this.materialBuffer.gpuBuffer, entry.slot * MATERIAL_STRIDE, new Float32Array([c.r, c.g, c.b, op]));
    }

    // Record-affecting flags — mark records dirty only on actual change.
    if (mesh.layers !== entry.layers || mesh.frustumCulled !== entry.frustumCulled) {
      entry.layers = mesh.layers;
      entry.frustumCulled = mesh.frustumCulled;
      this._recordDirty = true;
    }
  }

  gc(seenIds) {
    for (const [id, entry] of [...this._slots]) {
      if (!seenIds.has(id)) {
        this.arena.free(entry.alloc.handle);
        this._freeSlots.push(entry.slot);
        this._slots.delete(id);
        this._recordDirty = true;
      }
    }
  }

  // Writes the shared draw-records (once per frame, only if membership/flags
  // changed). Record index == slot == index into world/material/bounds, since
  // the cull shader's slotToObject maps a draw slot to that record index, which
  // the vertex shader uses to fetch world + material.
  syncRecords() {
    if (!this._recordDirty) { this.count = this._nextSlot; return; }
    const owner = this._perCamera.values().next().value;
    if (!owner) return; // no camera yet
    const m = owner.multi;
    for (const slot of this._freeSlots) {
      m.setRecord(slot, { firstIndex: 0, indexCount: 0, baseVertex: 0, transformIndex: slot, layerMask: 0, flags: 0 });
    }
    for (const entry of this._slots.values()) {
      m.setRecord(entry.slot, {
        firstIndex: entry.alloc.firstIndex,
        indexCount: entry.alloc.indexCount,
        baseVertex: entry.alloc.baseVertex,
        transformIndex: entry.slot,
        layerMask: entry.layers ?? 0x1,
        flags: entry.frustumCulled === false ? 1 : 0,
      });
    }
    this._recordDirty = false;
    this.count = this._nextSlot;
  }

  // Per-camera cull setup: sets this camera's object count + layer mask.
  cullFor(camera, layerMask) {
    const pc = this._forCamera(camera);
    pc.multi.setObjectCount(this._nextSlot);
    pc.multi.setCameraLayerMask(layerMask);
    return pc;
  }

  draw(rp, camera) {
    if (this.count === 0) return;
    const pc = this._perCamera.get(camera);
    rp.setPipeline(this.pipeline);
    rp.setBindGroup(0, pc.sceneBG);
    rp.setBindGroup(1, this.objectBG);
    this.arena.bind(rp);
    pc.multi.drawAll(rp, 2);
  }
}

function computeBounds(positions) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], positions[i + a]);
      max[a] = Math.max(max[a], positions[i + a]);
    }
  }
  return { min, max };
}
