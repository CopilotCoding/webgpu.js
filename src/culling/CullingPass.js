import { cullingShader } from './culling.wgsl.js';

const WORKGROUP_SIZE = 64;

/**
 * Frustum + occlusion culling, run entirely on the GPU. For each object,
 * tests its world-space AABB (derived from a local AABB and the object's
 * world matrix, from TransformPropagation.worldBuffer) against the
 * camera's frustum planes and the previous frame's Hi-Z buffer, writing a
 * per-object visibility bitmask:
 *
 *   bit 0 (VISIBLE_FRUSTUM)  — AABB intersects the camera frustum
 *   bit 1 (VISIBLE_OCCLUSION) — AABB is not fully behind the Hi-Z depth
 *
 * The CPU does not iterate objects to decide visibility — it only reads
 * the resulting bitmask (here, for visualization; Stage 9's indirect draw
 * generation will consume it entirely on the GPU).
 */
export class CullingPass {
  /**
   * @param bounds either an array of { min, max } local AABBs (the count
   *   becomes the object count and capacity), or a number giving the
   *   capacity for a scene that fills bounds later via setBounds().
   */
  constructor(device, camera, worldBuffer, bounds, hiZBuffer) {
    this.device = device;

    const capacity = typeof bounds === 'number' ? bounds : bounds.length;
    this.capacity = capacity;
    this.objectCount = typeof bounds === 'number' ? 0 : bounds.length;

    this.boundsBuffer = device.resources.createBuffer({
      size: capacity * 8 * 4, // localMin (vec3+pad) + localMax (vec3+pad)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (typeof bounds !== 'number') {
      const boundsData = new Float32Array(bounds.length * 8);
      for (let i = 0; i < bounds.length; i++) {
        boundsData.set(bounds[i].min, i * 8);
        boundsData.set(bounds[i].max, i * 8 + 4);
      }
      device.queue.writeBuffer(this.boundsBuffer.gpuBuffer, 0, boundsData);
    }

    this.visibilityBuffer = device.resources.createBuffer({
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // cullParams (u32): low 31 bits = objectCount, top bit = occlusionEnabled.
    // Occlusion culling is OFF by default — see the shader header for why
    // (single-frame-lagged Hi-Z can flicker objects at depth boundaries).
    this.occlusionEnabled = false;
    this.objectCountBuffer = device.resources.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([this._packParams()]));

    const shaderModule = device.device.createShaderModule({ code: cullingShader });
    this.pipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'cull' },
    });

    this.bindGroup = device.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: worldBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.boundsBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: this.visibilityBuffer.gpuBuffer } },
        { binding: 4, resource: hiZBuffer.texture.gpuTexture.createView() },
        { binding: 5, resource: { buffer: this.objectCountBuffer.gpuBuffer } },
      ],
    });
  }

  /**
   * Rebinds the pass to a recreated Hi-Z texture (after a resize). Must be
   * given the same camera + world buffer the pass was built with.
   */
  setHiZ(camera, worldBuffer, hiZBuffer) {
    this.bindGroup = this.device.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: worldBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.boundsBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: this.visibilityBuffer.gpuBuffer } },
        { binding: 4, resource: hiZBuffer.texture.gpuTexture.createView() },
        { binding: 5, resource: { buffer: this.objectCountBuffer.gpuBuffer } },
      ],
    });
  }

  /** Writes one object's local AABB into the bounds buffer at `index`. */
  setBounds(index, localBounds) {
    const data = new Float32Array(8);
    data.set(localBounds.min, 0);
    data.set(localBounds.max, 4);
    this.device.queue.writeBuffer(this.boundsBuffer.gpuBuffer, index * 8 * 4, data);
  }

  /**
   * Updates the live object count (must be <= the capacity the pass was
   * built with). Lets a scene grow/shrink without rebuilding the pass.
   */
  // Packs (objectCount, occlusionEnabled) into one u32: low 31 bits count,
  // top bit the occlusion flag. Keeps the cull uniform a 4-byte binding.
  _packParams() {
    return (this.objectCount & 0x7fffffff) | (this.occlusionEnabled ? 0x80000000 : 0);
  }

  setObjectCount(count) {
    this.objectCount = count;
    this.device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([this._packParams()]));
  }

  /**
   * Enables/disables Hi-Z occlusion culling. OFF by default: the indirect-draw
   * path drops any object that fails occlusion against the previous frame's
   * Hi-Z, which flickers objects at depth boundaries under a moving camera.
   * Enable only where the draw path tolerates that (e.g. example 08, which uses
   * the visibility bits for coloring, not for dropping geometry) or once
   * two-phase occlusion culling is in place.
   */
  setOcclusionEnabled(enabled) {
    this.occlusionEnabled = !!enabled;
    this.device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([this._packParams()]));
  }

  cull(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.objectCount / WORKGROUP_SIZE));
    pass.end();
  }

  destroy() {
    this.boundsBuffer.destroy();
    this.visibilityBuffer.destroy();
    this.objectCountBuffer.destroy();
  }
}
