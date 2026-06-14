import { createDevice } from '../device/Device.js';
import { RenderGraph, CANVAS } from '../render-graph/RenderGraph.js';
import { Camera } from '../camera/Camera.js';
import { ClusterGrid } from '../lighting/ClusterGrid.js';
import { LightCulling } from '../lighting/LightCulling.js';
import { ShadowMap } from '../lighting/ShadowMap.js';
import { HiZBuffer } from '../culling/HiZBuffer.js';
import { CullingPass } from '../culling/CullingPass.js';
import { IndirectDrawSystem } from '../culling/IndirectDrawSystem.js';
import { MipGenerator } from '../textures/MipGenerator.js';
import { FullscreenPass } from '../post/FullscreenPass.js';
import {
  brightPassFragmentSource, brightPassLayoutEntries,
  blurFragmentSource, blurLayoutEntries,
  compositeFragmentSource, compositeLayoutEntries,
} from '../post/postEffects.wgsl.js';
import { GPUPicker, screenPointToRay } from '../picking/GPUPicker.js';
import { boxGeometry } from '../geometry/primitives.js';
import { Scene } from './Scene.js';
import { litShaderSource, markerShaderSource, shadowDepthShaderSource, MATERIAL_PARAMS_SIZE } from './forwardShaders.wgsl.js';
import { perspective } from '../math/mat4.js';

const HDR_FORMAT = 'rgba16float';

// The Engine composes every webgpu.js system into a single GPU-driven
// forward renderer with shadows, clustered point lights, bloom + ACES
// tonemap, and GPU picking. It does NOT hide the systems — each is exposed
// (engine.camera, engine.shadowMap, engine.lightCulling, ...) so an app can
// reach past the Engine when it needs to. The Engine just wires them
// together and runs the per-frame render graph so a typical app doesn't have
// to.
//
//   const engine = await Engine.create({ canvas });
//   const box = engine.boxGeometry();
//   engine.scene.addMesh({ geometry: box, position: [...], baseColor: [...] });
//   engine.scene.addLight({ position: [...], color: [...] });
//   engine.onUpdate = (dt) => { ...drive transforms/lights/camera... };
//   engine.start();
export class Engine {
  static async create(options) {
    const device = await createDevice();
    return new Engine(device, options);
  }

  constructor(device, {
    canvas,
    near = 0.1,
    far = 100,
    fov = Math.PI / 4,
    maxObjects = 4096,
    maxLights = 256,
    shadowMapSize = 2048,
    shadowBounds = { min: [-30, -5, -30], max: [30, 30, 30] },
    lightDirection = [0.4, -0.7, 0.35],
    textureLayers = null, // array of canvas/ImageBitmap sources; defaults to a single white pixel
    bloom = true, // opt-out of the bloom + tonemap post chain; when off the HDR scene is tonemapped straight to the canvas
  }) {
    this.device = device;
    this.canvas = canvas;
    this.bloom = bloom;
    this.near = near;
    this.far = far;
    this._fov = fov;
    this.shadowBounds = shadowBounds;
    this.lightDirection = lightDirection;
    this.onUpdate = null;

    canvas.width = canvas.clientWidth || canvas.width;
    canvas.height = canvas.clientHeight || canvas.height;
    const width = canvas.width;
    const height = canvas.height;

    this.context = device.getCanvasContext(canvas);
    this.format = navigator.gpu.getPreferredCanvasFormat();

    // --- Camera ---
    this.camera = new Camera(device);
    this.projectionMatrix = perspective(fov, width / height, near, far);
    this.camera.setProjectionMatrix(this.projectionMatrix);
    this.camera.setViewport(0, 0, width, height);

    // --- Depth + Hi-Z ---
    this.depthTexture = device.resources.createTexture({
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hiZBuffer = new HiZBuffer(device, width, height);

    // --- Clustered lighting ---
    this.clusterGrid = new ClusterGrid(device, { clusterCountX: 16, clusterCountY: 9, clusterCountZ: 24 });
    this.clusterGrid.setProjection(this.projectionMatrix, width, height, near, far);
    this.lightCulling = new LightCulling(device, this.clusterGrid, maxLights);

    // --- Shadows ---
    this.shadowMap = new ShadowMap(device, { mapSize: shadowMapSize });

    // --- Albedo texture array (mips in compute) ---
    this._buildTextureArray(textureLayers);

    // --- HDR + bloom targets ---
    this._buildPostTargets(width, height);

    // --- Scene + object storage buffers ---
    this.scene = new Scene(device, { maxObjects, maxLights });

    // --- GPU culling, indirect draw, picking over the scene's buffers ---
    this.cullingPass = new CullingPass(device, this.camera, this.scene.worldMatricesBuffer, maxObjects, this.hiZBuffer);
    this.indirectDrawSystem = new IndirectDrawSystem(device, this.cullingPass, this._sharedVertexCount ?? 0);
    this.picker = new GPUPicker(device, {
      worldBuffer: this.scene.worldMatricesBuffer,
      boundsBuffer: this.cullingPass.boundsBuffer,
      objectCount: 0,
    });

    // Keep the GPU systems' bounds + counts in sync as meshes are added.
    this.scene._onObjectAdded = (index, localBounds) => {
      this.cullingPass.setBounds(index, localBounds);
      this._syncObjectCount();
      // The indirect draw's vertexCount comes from the shared batch geometry,
      // set when the first geometry is registered.
    };

    // --- Highlight (picking) ---
    this.NO_HIGHLIGHT = 0xffffffff;
    this._highlightIndex = this.NO_HIGHLIGHT;
    this._pointer = null;

    this._buildBindGroupsAndPipelines();
    this._installPicking();

    // With bloom off, the bloom texture is never written each frame, so the
    // composite would read stale/uninitialized contents — clear it once to
    // black so "scene + bloom" reduces to just the tonemapped scene.
    if (!this.bloom) {
      const enc = device.device.createCommandEncoder();
      enc.beginRenderPass({
        colorAttachments: [{ view: this.bloomTexture.gpuTexture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      }).end();
      device.queue.submit([enc.finish()]);
    }

    this._lastTime = 0;
    this._running = false;
  }

  boxGeometry(size = [1, 1, 1]) {
    const geo = boxGeometry(this.device, size);
    this._registerGeometry(geo);
    return geo;
  }

  /** Registers the shared batch geometry (vertex count for the indirect draw). */
  _registerGeometry(geometry) {
    if (this._sharedGeometry) return;
    this._sharedGeometry = geometry;
    this._sharedVertexCount = geometry.vertexCount;
    // The indirect buffer's vertexCount slot was written 0 at construction;
    // rewrite it now that we know the geometry.
    this.device.queue.writeBuffer(
      this.indirectDrawSystem.indirectBuffer.gpuBuffer, 0,
      new Uint32Array([geometry.vertexCount]),
    );
  }

  _syncObjectCount() {
    const count = this.scene.objectCount;
    this.cullingPass.setObjectCount(count);
    this.indirectDrawSystem.setObjectCount(count);
    this.picker.setObjectCount(count);
  }

  _buildTextureArray(layers) {
    const sources = layers ?? [whitePixelCanvas()];
    const size = sources[0].width ?? 1;
    const mipCount = Math.floor(Math.log2(size)) + 1;

    this.albedoTexture = this.device.resources.createTexture({
      size: [size, size, sources.length],
      format: 'rgba8unorm',
      mipLevelCount: mipCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });

    sources.forEach((source, layer) => {
      this.device.queue.copyExternalImageToTexture(
        { source },
        { texture: this.albedoTexture.gpuTexture, origin: [0, 0, layer] },
        [size, size, 1],
      );
    });

    const mipGenerator = new MipGenerator(this.device);
    const encoder = this.device.device.createCommandEncoder();
    mipGenerator.generate(encoder, this.albedoTexture);
    this.device.queue.submit([encoder.finish()]);

    this.albedoSampler = this.device.resources.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      maxAnisotropy: 8,
    });
  }

  _buildPostTargets(width, height) {
    const make = (w, h) => this.device.resources.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrTexture = make(width, height);
    const hw = Math.max(1, width >> 1), hh = Math.max(1, height >> 1);
    this.brightTexture = make(hw, hh);
    this.blurTexture = make(hw, hh);
    this.bloomTexture = make(hw, hh);

    this.linearSampler = this.device.resources.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  _buildBindGroupsAndPipelines() {
    const device = this.device;

    this.sceneBindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    this.objectBindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.markerBindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.shadowBindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.sceneBindGroup = device.device.createBindGroup({
      layout: this.sceneBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.clusterGrid.gridInfoBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.lightCulling.clusterRangesBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: this.lightCulling.lightIndicesBuffer.gpuBuffer } },
        { binding: 4, resource: { buffer: this.lightCulling.lightsBuffer.gpuBuffer } },
        { binding: 5, resource: { buffer: this.shadowMap.buffer.gpuBuffer } },
        { binding: 6, resource: this.shadowMap.getView() },
        { binding: 7, resource: this.shadowMap.depthSampler.gpuSampler },
        { binding: 8, resource: this.albedoTexture.gpuTexture.createView({ dimension: '2d-array' }) },
        { binding: 9, resource: this.albedoSampler.gpuSampler },
      ],
    });

    this.objectBindGroup = device.device.createBindGroup({
      layout: this.objectBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.scene.worldMatricesBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.scene.materialParamsBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.indirectDrawSystem.visibleIndicesBuffer.gpuBuffer } },
      ],
    });

    this.markerBindGroup = device.device.createBindGroup({
      layout: this.markerBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.lightCulling.lightsBuffer.gpuBuffer } },
      ],
    });

    this.shadowBindGroup = device.device.createBindGroup({
      layout: this.shadowBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.shadowMap.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.scene.worldMatricesBuffer.gpuBuffer } },
      ],
    });

    // --- Post passes ---
    this.brightPass = new FullscreenPass(device, { fragmentSource: brightPassFragmentSource, bindGroupLayoutEntries: brightPassLayoutEntries, targetFormat: HDR_FORMAT });
    this.blurPass = new FullscreenPass(device, { fragmentSource: blurFragmentSource, bindGroupLayoutEntries: blurLayoutEntries, targetFormat: HDR_FORMAT });
    this.compositePass = new FullscreenPass(device, { fragmentSource: compositeFragmentSource, bindGroupLayoutEntries: compositeLayoutEntries, targetFormat: this.format });

    // Post param buffers don't depend on framebuffer size — create them once
    // and keep them; only the post bind groups (which reference the HDR/bloom
    // textures) are rebuilt on resize, in _buildPostBindGroups().
    const paramBuf = (vals) => {
      const b = device.resources.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b.gpuBuffer, 0, new Float32Array(vals));
      return b;
    };
    this._brightParams = paramBuf([1.0, 1.0, 0, 0]);
    this._blurHParams = paramBuf([1, 0, 0, 0]);
    this._blurVParams = paramBuf([0, 1, 0, 0]);

    this._buildPostBindGroups();

    // --- Pipelines (need the shared geometry's vertex layout) ---
    this._pipelinesBuilt = false;
  }

  // (Re)creates the post-process bind groups, which reference the HDR/bloom
  // textures and so must be rebuilt whenever those are recreated (resize).
  // The passes, pipelines, and param buffers are size-independent and reused.
  _buildPostBindGroups() {
    this.brightBindGroup = this.brightPass.createBindGroup([
      { binding: 0, resource: this.hdrTexture.gpuTexture.createView() },
      { binding: 1, resource: this.linearSampler.gpuSampler },
      { binding: 2, resource: { buffer: this._brightParams.gpuBuffer } },
    ]);
    this.blurHBindGroup = this.blurPass.createBindGroup([
      { binding: 0, resource: this.brightTexture.gpuTexture.createView() },
      { binding: 1, resource: this.linearSampler.gpuSampler },
      { binding: 2, resource: { buffer: this._blurHParams.gpuBuffer } },
    ]);
    this.blurVBindGroup = this.blurPass.createBindGroup([
      { binding: 0, resource: this.blurTexture.gpuTexture.createView() },
      { binding: 1, resource: this.linearSampler.gpuSampler },
      { binding: 2, resource: { buffer: this._blurVParams.gpuBuffer } },
    ]);
    this.compositeBindGroup = this.compositePass.createBindGroup([
      { binding: 0, resource: this.hdrTexture.gpuTexture.createView() },
      { binding: 1, resource: this.bloomTexture.gpuTexture.createView() },
      { binding: 2, resource: this.linearSampler.gpuSampler },
    ]);
  }

  _ensurePipelines() {
    if (this._pipelinesBuilt) return;
    const device = this.device;
    const layouts = this._sharedGeometry.vertexBufferLayouts;

    const litModule = device.device.createShaderModule({ code: litShaderSource });
    this.litPipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [this.sceneBindGroupLayout, this.objectBindGroupLayout] }),
      vertex: { module: litModule, entryPoint: 'vertexMain', buffers: layouts },
      fragment: { module: litModule, entryPoint: 'fragmentMain', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const markerModule = device.device.createShaderModule({ code: markerShaderSource });
    this.markerPipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [this.markerBindGroupLayout] }),
      vertex: { module: markerModule, entryPoint: 'vertexMain', buffers: layouts },
      fragment: { module: markerModule, entryPoint: 'fragmentMain', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const shadowModule = device.device.createShaderModule({ code: shadowDepthShaderSource });
    this.shadowPipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [this.shadowBindGroupLayout] }),
      vertex: { module: shadowModule, entryPoint: 'vertexMain', buffers: layouts },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this._pipelinesBuilt = true;
  }

  // --- Picking: hover the canvas, highlight the nearest object ---
  _installPicking() {
    this.canvas.addEventListener('pointermove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this._pointer = [e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height];
    });
    this.canvas.addEventListener('pointerleave', () => { this._pointer = null; });
  }

  /** Returns the most recent hover hit ({objectIndex, distance, point}) or null. */
  get hovered() {
    return this._hovered ?? null;
  }

  _updatePicking() {
    if (this._pointer && !this.picker.busy) {
      const ray = screenPointToRay(this.camera, this._pointer[0], this._pointer[1], this._pointer[2], this._pointer[3]);
      this.picker.pick(ray).then((hit) => {
        if (hit === undefined) return;
        this._hovered = hit;
        const newIndex = hit ? hit.objectIndex : this.NO_HIGHLIGHT;
        if (newIndex !== this._highlightIndex) this._setHighlight(newIndex);
      });
    } else if (!this._pointer && this._highlightIndex !== this.NO_HIGHLIGHT) {
      this._hovered = null;
      this._setHighlight(this.NO_HIGHLIGHT);
    }
  }

  // The highlight is rendered by adding a warm emissive on top of the hovered
  // object's own (base) emissive. The base value lives in the Scene's
  // CPU-side mirror, so restoring on un-hover keeps a glowing object glowing.
  static HIGHLIGHT_EMISSIVE = [0.6, 0.5, 0.2];

  _setHighlight(index) {
    const prev = this._highlightIndex;
    if (prev !== this.NO_HIGHLIGHT) {
      this._gpuWriteEmissive(prev, this.scene.getEmissive(prev));
    }
    this._highlightIndex = index;
    if (index !== this.NO_HIGHLIGHT) {
      const base = this.scene.getEmissive(index);
      const h = Engine.HIGHLIGHT_EMISSIVE;
      this._gpuWriteEmissive(index, [base[0] + h[0], base[1] + h[1], base[2] + h[2]]);
    }
  }

  _gpuWriteEmissive(index, emissive) {
    const offset = index * MATERIAL_PARAMS_SIZE + 32; // emissive vec3f at byte 32
    this.device.queue.writeBuffer(this.scene.materialParamsBuffer.gpuBuffer, offset, new Float32Array(emissive));
  }

  start() {
    if (this._running) return;
    this._running = true;
    this._lastTime = performance.now();
    const loop = () => {
      if (!this._running) return;
      const now = performance.now();
      const dt = (now - this._lastTime) / 1000;
      this._lastTime = now;
      this.renderFrame(dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() {
    this._running = false;
  }

  /**
   * Resizes the framebuffer-sized resources (depth, Hi-Z, HDR + bloom targets)
   * and updates the camera projection/viewport and cluster projection. Rebuilds
   * the bind groups that reference the recreated textures.
   */
  setSize(width, height) {
    if (width <= 0 || height <= 0) return;
    this.canvas.width = width;
    this.canvas.height = height;

    // Destroy the old size-dependent textures.
    this.depthTexture.destroy();
    this.hiZBuffer.destroy?.();
    this.hdrTexture.destroy();
    this.brightTexture.destroy();
    this.blurTexture.destroy();
    this.bloomTexture.destroy();

    this.depthTexture = this.device.resources.createTexture({
      size: [width, height], format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hiZBuffer = new HiZBuffer(this.device, width, height);
    this._buildPostTargets(width, height);
    if (!this.bloom) {
      const enc = this.device.device.createCommandEncoder();
      enc.beginRenderPass({ colorAttachments: [{ view: this.bloomTexture.gpuTexture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] }).end();
      this.device.queue.submit([enc.finish()]);
    }

    this.projectionMatrix = perspective(this._fov, width / height, this.near, this.far);
    this.camera.setProjectionMatrix(this.projectionMatrix);
    this.camera.setViewport(0, 0, width, height);
    this.clusterGrid.setProjection(this.projectionMatrix, width, height, this.near, this.far);

    // Rebind only what references the recreated textures: the culling pass's
    // Hi-Z binding and the post-process bind groups. The pipelines, passes,
    // param buffers, and size-independent bind groups are left intact (so a
    // resize doesn't leak a fresh set of passes/buffers each call).
    this.cullingPass.setHiZ(this.camera, this.scene.worldMatricesBuffer, this.hiZBuffer);
    this._buildPostBindGroups();
  }

  /** Stops the loop and releases every GPU resource this Engine created. */
  dispose() {
    this.stop();
    this.onUpdate = null;
    this.device.resources.destroyAll();
  }

  renderFrame(dt) {
    if (this.onUpdate) this.onUpdate(dt);
    this.camera.update();
    this._ensurePipelines();

    this.shadowMap.update(this.lightDirection, this.shadowBounds);

    // Upload lights for this frame.
    this.lightCulling.setLights(this.scene.lightData);
    this.lightCulling.setView(this.camera.viewMatrix, this.scene.lightCount);

    this._updatePicking();

    const graph = new RenderGraph(this.device);
    graph.setCanvasTarget(this.context);

    graph.addPass({
      name: 'cluster-and-cull-lights',
      writes: [this.clusterGrid.clusterBoundsBuffer, this.lightCulling.clusterRangesBuffer],
      execute: (encoder) => { this.clusterGrid.build(encoder); this.lightCulling.cull(encoder); },
    });

    graph.addPass({
      name: 'cull-and-build-draws',
      writes: [this.hiZBuffer.texture, this.cullingPass.visibilityBuffer, this.indirectDrawSystem.indirectBuffer, this.indirectDrawSystem.visibleIndicesBuffer],
      reads: [this.camera.buffer, this.scene.worldMatricesBuffer],
      execute: (encoder) => { this.hiZBuffer.build(encoder, this.depthTexture); this.cullingPass.cull(encoder); this.indirectDrawSystem.build(encoder); },
    });

    graph.addPass({
      name: 'shadow-map',
      depthStencilAttachment: { target: this.shadowMap.depthTexture, view: this.shadowMap.getView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [this.shadowMap.depthTexture],
      reads: [this.shadowMap.buffer, this.scene.worldMatricesBuffer],
      execute: (rp) => {
        rp.setPipeline(this.shadowPipeline);
        rp.setBindGroup(0, this.shadowBindGroup);
        this._bindGeometry(rp);
        rp.draw(this._sharedVertexCount, this.scene.objectCount);
      },
    });

    graph.addPass({
      name: 'forward-hdr',
      colorAttachments: [{ target: this.hdrTexture, clearValue: { r: 0.01, g: 0.01, b: 0.02, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { target: this.depthTexture, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [this.hdrTexture, this.depthTexture],
      reads: [
        this.camera.buffer, this.clusterGrid.gridInfoBuffer,
        this.lightCulling.clusterRangesBuffer, this.lightCulling.lightIndicesBuffer, this.lightCulling.lightsBuffer,
        this.shadowMap.buffer, this.shadowMap.depthTexture, this.albedoTexture,
        this.scene.worldMatricesBuffer, this.scene.materialParamsBuffer,
        this.indirectDrawSystem.visibleIndicesBuffer, this.indirectDrawSystem.indirectBuffer,
      ],
      execute: (rp) => {
        rp.setPipeline(this.litPipeline);
        rp.setBindGroup(0, this.sceneBindGroup);
        rp.setBindGroup(1, this.objectBindGroup);
        this._bindGeometry(rp);
        rp.drawIndirect(this.indirectDrawSystem.indirectBuffer.gpuBuffer, 0);

        if (this.scene.lightCount > 0) {
          rp.setPipeline(this.markerPipeline);
          rp.setBindGroup(0, this.markerBindGroup);
          rp.draw(this._sharedVertexCount, this.scene.lightCount);
        }
      },
    });

    const fs = (name, target, read, bg, pass) => graph.addPass({
      name,
      colorAttachments: [{ target, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      writes: target === CANVAS ? [] : [target],
      reads: read,
      execute: (rp) => pass.draw(rp, bg),
    });
    // Bloom is opt-out: when off, skip the bright + separable-blur passes and
    // composite the HDR scene against a zeroed bloom texture (tonemap only).
    // This is the composable-frame seam — the post chain is just passes the
    // Engine chooses to add or omit.
    if (this.bloom) {
      fs('bright-pass', this.brightTexture, [this.hdrTexture], this.brightBindGroup, this.brightPass);
      fs('bloom-blur-h', this.blurTexture, [this.brightTexture], this.blurHBindGroup, this.blurPass);
      fs('bloom-blur-v', this.bloomTexture, [this.blurTexture], this.blurVBindGroup, this.blurPass);
    }
    fs('composite', CANVAS, [this.hdrTexture, this.bloomTexture], this.compositeBindGroup, this.compositePass);

    graph.execute();
  }

  _bindGeometry(rp) {
    rp.setVertexBuffer(0, this._sharedGeometry.attributes.position.buffer.gpuBuffer);
    rp.setVertexBuffer(1, this._sharedGeometry.attributes.normal.buffer.gpuBuffer);
    rp.setVertexBuffer(2, this._sharedGeometry.attributes.uv.buffer.gpuBuffer);
  }
}

// --- helpers ---
function whitePixelCanvas() {
  const c = (typeof OffscreenCanvas !== 'undefined') ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
  c.width = 1; c.height = 1;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, 1, 1);
  return c;
}
