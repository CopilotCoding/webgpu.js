import { RenderGraph, CANVAS } from '../render-graph/RenderGraph.js';
import { GeometryArena, interleaveStandard } from '../geometry/GeometryArena.js';
import { MultiDrawSystem } from '../culling/MultiDrawSystem.js';
import { lambertShader, basicShader, pointsShader } from './sceneShaders.wgsl.js';

const OBJ_CAP = 2048;          // per-batch object capacity
const MATERIAL_STRIDE = 16;    // vec4 color (rgb + opacity)

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
    this.lightsBuffer = device.resources.createBuffer({ size: 16 + 4 * 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fogBuffer = device.resources.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Bind group layouts shared by the lambert/basic batch pipelines.
    this.sceneBGL = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.objectBGL = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    this._batches = new Map();   // key -> Batch
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

  // --- batch management ---

  _batchKey(mat) {
    if (mat.kind === 'lambert') return 'lambert';
    if (mat.kind === 'basic') {
      const blend = mat.blending === 'additive' ? 'add' : (mat.transparent ? 'alpha' : 'opaque');
      return `basic:${blend}:dw${mat.depthWrite ? 1 : 0}:dt${mat.depthTest ? 1 : 0}:s${mat.side}:w${mat.wireframe ? 1 : 0}`;
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

    // Reset frame bookkeeping, then collect drawables.
    this._seen.clear();
    const shaderMeshes = [];
    const pointsMeshes = [];
    scene.traverse((node) => {
      if (!node.geometry || !node.material || node.visible === false) return;
      // visibility also respects ancestors:
      if (!this._ancestorsVisible(node)) return;
      if ((node.layers & layerMask) === 0) return;
      const kind = node.material.kind;
      if (kind === 'shader') { shaderMeshes.push(node); return; }
      if (kind === 'points') { pointsMeshes.push(node); return; }
      const batch = this._getBatch(node.material);
      batch.sync(node);
      this._seen.add(node.id);
    });

    // GC meshes that disappeared from the graph.
    for (const batch of this._batches.values()) batch.gc(this._seen);

    // Update camera layer mask on each batch's cull system + counts.
    for (const batch of this._batches.values()) batch.finalize(camera, layerMask);

    const graph = new RenderGraph(this.device);
    graph.setCanvasTarget(this.context);

    // Cull passes (compute) for each batch.
    for (const batch of this._batches.values()) {
      if (batch.count === 0) continue;
      graph.addPass({
        name: `cull-${batch.key}`,
        writes: [batch.multi.drawArgsBuffer, batch.multi.drawCountBuffer, batch.multi.slotToObjectBuffer],
        reads: [camera.buffer, batch.worldBuffer, batch.boundsBuffer, batch.multi.recordBuffer],
        execute: (encoder) => batch.multi.build(encoder),
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
      forwardReads.push(batch.multi.drawArgsBuffer, batch.multi.slotToObjectBuffer, batch.worldBuffer, batch.materialBuffer);
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

        // Custom-shader meshes first (typically opaque, depth-writing).
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
    const data = new Float32Array(4 + 4 * 8);
    let count = 0;
    let ambR = 0, ambG = 0, ambB = 0;
    const lights = [];
    scene.traverse((n) => {
      if (n.isAmbient) { ambR += n.color.r * n.intensity; ambG += n.color.g * n.intensity; ambB += n.color.b * n.intensity; }
      else if (n.isPointLight && lights.length < 4) lights.push({ point: true, n });
      else if (n.isDirectional && lights.length < 4) lights.push({ point: false, n });
    });
    data[0] = ambR; data[1] = ambG; data[2] = ambB; data[3] = lights.length;
    lights.forEach((L, i) => {
      const off = 4 + i * 8;
      const n = L.n;
      if (L.point) { data[off] = n.position.x; data[off + 1] = n.position.y; data[off + 2] = n.position.z; data[off + 3] = 1; }
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
    const r = mesh._render || (mesh._render = {});
    if (!r.cameraBG) {
      r.cameraBG = this.device.device.createBindGroup({
        layout: p.cameraBGL, entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
      });
    }
    if (!r.uBuffer && material.uniformSize > 0) {
      r.uBuffer = this.device.resources.createBuffer({ size: material.uniformSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      r.uView = new Float32Array(material.uniformSize / 4);
      r.uBG = this.device.device.createBindGroup({
        layout: p.uBGL, entries: [{ binding: 0, resource: { buffer: r.uBuffer.gpuBuffer } }],
      });
    }
    if (r.uBuffer && material.updateUniforms) {
      material.updateUniforms(r.uView);
      this.device.queue.writeBuffer(r.uBuffer.gpuBuffer, 0, r.uView);
    }

    rp.setPipeline(p.pipeline);
    rp.setBindGroup(0, r.cameraBG);
    if (r.uBG) rp.setBindGroup(1, r.uBG);
    const g = mesh.geometry;
    material.attributes.forEach((name, loc) => rp.setVertexBuffer(loc, g.attributes[name].buffer.gpuBuffer));
    rp.draw(g.vertexCount);
  }

  // --- points ---

  _drawPoints(rp, mesh, camera) {
    let s = this._points.get(mesh.id);
    if (!s) { s = this._buildPoints(mesh, camera); this._points.set(mesh.id, s); }
    // Update NDC half-size from the material's pixel size each frame.
    const sizePx = mesh.material.size;
    const ndcX = sizePx / this.canvas.width, ndcY = sizePx / this.canvas.height;
    this.device.queue.writeBuffer(s.paramsBuffer.gpuBuffer, 0, new Float32Array([
      mesh.material.color.r, mesh.material.color.g, mesh.material.color.b, 0,
      ndcX, ndcY, 0, 0,
    ]));
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
    const cameraBG = device.device.createBindGroup({ layout: pp.cameraBGL, entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }] });
    const bindGroup = device.device.createBindGroup({
      layout: pp.dataBGL,
      entries: [
        { binding: 0, resource: { buffer: positionsBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer.gpuBuffer } },
      ],
    });
    return { pipeline: pp.pipeline, cameraBG, bindGroup, positionsBuffer, paramsBuffer, count };
  }
}

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

    this._slots = new Map();    // mesh.id -> { slot, alloc }
    this._freeSlots = [];
    this._nextSlot = 0;

    this.multi = null;          // built lazily once camera is known
    this.pipeline = null;
    this._sampleMaterial = sampleMaterial;
  }

  _ensureGPU(camera) {
    if (this.multi) return;
    this.multi = new MultiDrawSystem(this.device, camera, this.worldBuffer, this.boundsBuffer, OBJ_CAP);
    this.pipeline = this._buildPipeline();

    this.sceneBG = this.device.device.createBindGroup({
      layout: this.renderer.sceneBGL,
      entries: [
        { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.renderer.lightsBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.renderer.fogBuffer.gpuBuffer } },
      ],
    });
    this.objectBG = this.device.device.createBindGroup({
      layout: this.renderer.objectBGL,
      entries: [
        { binding: 0, resource: { buffer: this.worldBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.materialBuffer.gpuBuffer } },
      ],
    });
  }

  _buildPipeline() {
    const device = this.device;
    const mat = this._sampleMaterial;
    const isLambert = mat.kind === 'lambert';
    const module = device.device.createShaderModule({ code: isLambert ? lambertShader : basicShader });

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
      layout: device.device.createPipelineLayout({
        bindGroupLayouts: [this.renderer.sceneBGL, this.renderer.objectBGL, this.multi.drawSlotBindGroupLayout],
      }),
      vertex: { module, entryPoint: 'vertexMain', buffers: this.arena.vertexBufferLayouts },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: this.renderer.format, blend }] },
      primitive: { topology, cullMode: topology === 'line-list' ? 'none' : cull },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: depthWrite, depthCompare },
      multisample: { count: this.renderer.sampleCount },
    });
  }

  /** Registers a new mesh or refreshes an existing one's transform + material. */
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
      entry = { slot, alloc };
      this._slots.set(mesh.id, entry);
      // Local AABB.
      const b = computeBounds(mesh.geometry.attributes.position.data);
      this.device.queue.writeBuffer(this.boundsBuffer.gpuBuffer, slot * 32, new Float32Array([...b.min, 0, ...b.max, 0]));
      this._recordDirty = true;
      entry._needRecord = true;
    }
    // World matrix (refresh every frame — cheap, and the game animates them).
    mesh.updateWorldMatrix(mesh.parent ? mesh.parent.worldMatrix : null);
    this.device.queue.writeBuffer(this.worldBuffer.gpuBuffer, entry.slot * 64, mesh.worldMatrix);
    // Material color + opacity.
    const c = mesh.material.color;
    this.device.queue.writeBuffer(this.materialBuffer.gpuBuffer, entry.slot * MATERIAL_STRIDE,
      new Float32Array([c.r, c.g, c.b, mesh.material.opacity ?? 1]));
    entry._mesh = mesh;
    entry._frustumCulled = mesh.frustumCulled;
    entry._layers = mesh.layers;
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

  finalize(camera, layerMask) {
    this._ensureGPU(camera);
    this.multi.setCameraLayerMask(layerMask);

    // The cull shader writes slotToObject[outSlot] = recordIndex, and the
    // vertex shader uses that recordIndex to index BOTH worldMatrices and
    // materials. So record index must equal the slot we wrote world/material/
    // bounds at — we keep them identical (record index == entry.slot) rather
    // than compacting, and size the cull dispatch to cover all live slots.
    // Freed slots left with indexCount 0 in their record draw nothing.
    if (this._recordDirty) {
      // Clear records for any freed slots so stale geometry isn't drawn.
      for (const slot of this._freeSlots) {
        this.multi.setRecord(slot, { firstIndex: 0, indexCount: 0, baseVertex: 0, transformIndex: slot, layerMask: 0, flags: 0 });
      }
      for (const entry of this._slots.values()) {
        this.multi.setRecord(entry.slot, {
          firstIndex: entry.alloc.firstIndex,
          indexCount: entry.alloc.indexCount,
          baseVertex: entry.alloc.baseVertex,
          transformIndex: entry.slot,
          layerMask: entry._layers ?? 0x1,
          flags: entry._frustumCulled === false ? 1 : 0,
        });
      }
      this._recordDirty = false;
    }
    this.multi.setObjectCount(this._nextSlot);
    this.count = this._nextSlot;
  }

  draw(rp, camera) {
    if (this.count === 0) return;
    rp.setPipeline(this.pipeline);
    rp.setBindGroup(0, this.sceneBG);
    rp.setBindGroup(1, this.objectBG);
    this.arena.bind(rp);
    this.multi.drawAll(rp, 2);
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
