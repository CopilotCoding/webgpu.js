import { multiDrawCullShader, RECORD_SIZE } from './multiDraw.wgsl.js';

const WORKGROUP_SIZE = 64;
const DRAW_ARG_SIZE = 20; // 5 x u32: indexCount, instanceCount, firstIndex, baseVertex, firstInstance
const UNIFORM_ALIGN = 256; // min dynamic-uniform-offset alignment

/**
 * GPU-driven rendering of heterogeneous geometry. Owns:
 *  - a draw-record buffer (per object: slice + transform/layer/flags),
 *  - the indexed-indirect args buffer the GPU compacts visible draws into,
 *  - a draw-count buffer.
 *
 * Each frame: resetDraws (clear count + zero slots), cullAndCompact (frustum +
 * layer-mask test, append a DrawIndexedIndirect arg per visible object). The
 * caller then issues `capacity` drawIndexedIndirect calls (a fixed loop —
 * constant CPU cost); empty slots draw zero indices.
 *
 * Per-object world matrices and local AABBs live in storage buffers the caller
 * provides (shared with the rest of the GPU-driven scene). transformIndex in
 * each record selects the matrix; the bounds buffer is indexed by object id.
 */
export class MultiDrawSystem {
  /**
   * @param {object} device
   * @param {object} camera webgpu.js Camera (its .buffer holds frustum planes)
   * @param {object} worldMatricesBuffer storage Buffer of mat4x4f
   * @param {object} boundsBuffer storage Buffer of ObjectBounds (vec3+pad x2)
   * @param {number} capacity max objects
   * @param {object} [opts]
   * @param {object} [opts.recordBuffer] reuse an existing record buffer (lets
   *   two systems cull the SAME scene with different cameras/masks — e.g. a
   *   main view + a minimap). When given, setRecord() writes to it as usual,
   *   but typically only one owner populates records.
   */
  constructor(device, camera, worldMatricesBuffer, boundsBuffer, capacity, opts = {}) {
    this.device = device;
    this.capacity = capacity;
    this.objectCount = 0;
    this.cameraLayerMask = 0xffffffff;
    // Draw-path capability:
    //  - multiDraw: one multiDrawIndexedIndirect per batch (Chromium only). This
    //    is the ONLY path where a non-zero firstInstance in the indirect args is
    //    honored as @builtin(instance_index). A plain drawIndexedIndirect loop
    //    does NOT pick up firstInstance even with the indirect-first-instance
    //    feature enabled (observed on Firefox/wgpu — the feature gates non-zero
    //    firstInstance for direct draws, not the per-call indirect args here).
    //  - Otherwise we use the slotToObject draw-slot loop: each draw binds a
    //    per-draw dynamic uniform (slotIndex) and the vertex shader reads
    //    slotToObject[slotIndex]. Works on every device, no feature needed.
    const feats = device.device.features;
    this.multiDraw = feats.has('chromium-experimental-multi-draw-indirect');
    this.firstInstanceId = this.multiDraw;

    this.recordBuffer = opts.recordBuffer ?? device.resources.createBuffer({
      size: capacity * RECORD_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._ownsRecordBuffer = !opts.recordBuffer;
    this.drawArgsBuffer = device.resources.createBuffer({
      size: capacity * DRAW_ARG_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.drawCountBuffer = device.resources.createBuffer({
      size: 4,
      // INDIRECT so it can serve as multiDrawIndexedIndirect's GPU draw-count.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.paramsBuffer = device.resources.createBuffer({
      size: 16, // vec4u: objectCount, layerMask, _, _
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeParams();

    // Compacted slot -> object id, written by the cull pass. The render
    // pipeline reads slotToObject[slotIndex] in its vertex shader to recover
    // per-object data; slotIndex comes from a per-draw dynamic uniform offset
    // (avoids needing the indirect-first-instance feature).
    this.slotToObjectBuffer = device.resources.createBuffer({
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // One 256-byte-aligned slot per draw, holding [i] — bound at offset i*256.
    this.slotIndexBuffer = device.resources.createBuffer({
      size: capacity * UNIFORM_ALIGN,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    for (let i = 0; i < capacity; i++) {
      device.queue.writeBuffer(this.slotIndexBuffer.gpuBuffer, i * UNIFORM_ALIGN, new Uint32Array([i]));
    }

    const module = device.device.createShaderModule({ code: multiDrawCullShader });

    this.bindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // The bind group a render pipeline uses to recover per-object data:
    // binding 0 = slotToObject (storage), binding 1 = slotIndex (dynamic
    // uniform, one 256B-aligned u32 per draw). Pipelines that draw from this
    // system include this layout as one of their bind groups.
    this.drawSlotBindGroupLayout = opts.drawSlotBindGroupLayout ?? device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });

    const layout = device.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.resetPipeline = device.device.createComputePipeline({ layout, compute: { module, entryPoint: 'resetDraws' } });
    this.cullPipeline = device.device.createComputePipeline({ layout, compute: { module, entryPoint: 'cullAndCompact' } });

    this.bindGroup = device.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.recordBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: worldMatricesBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: boundsBuffer.gpuBuffer } },
        { binding: 4, resource: { buffer: this.drawArgsBuffer.gpuBuffer } },
        { binding: 5, resource: { buffer: this.drawCountBuffer.gpuBuffer } },
        { binding: 6, resource: { buffer: this.paramsBuffer.gpuBuffer } },
        { binding: 7, resource: { buffer: this.slotToObjectBuffer.gpuBuffer } },
      ],
    });

    this.drawSlotBindGroup = device.device.createBindGroup({
      layout: this.drawSlotBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.slotToObjectBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.slotIndexBuffer.gpuBuffer, size: 4 } },
      ],
    });
  }

  _writeParams() {
    this.device.queue.writeBuffer(this.paramsBuffer.gpuBuffer, 0,
      new Uint32Array([this.objectCount, this.cameraLayerMask, 0, 0]));
  }

  /**
   * Writes one object's draw record.
   * @param {number} index object slot
   * @param {object} rec { firstIndex, indexCount, baseVertex, transformIndex, layerMask=1, flags=0 }
   */
  setRecord(index, rec) {
    const data = new Uint32Array([
      rec.firstIndex, rec.indexCount, rec.baseVertex,
      rec.transformIndex ?? index,
      rec.layerMask ?? 0x1,
      rec.flags ?? 0, 0, 0,
    ]);
    this.device.queue.writeBuffer(this.recordBuffer.gpuBuffer, index * RECORD_SIZE, data);
  }

  setObjectCount(count) {
    this.objectCount = count;
    this._writeParams();
  }

  /** Sets which layers the next cull pass keeps (bitmask AND with each record). */
  setCameraLayerMask(mask) {
    this.cameraLayerMask = mask >>> 0;
    this._writeParams();
  }

  /** Records the reset + cull compute passes. */
  build(encoder) {
    const groups = Math.max(1, Math.ceil(this.capacity / WORKGROUP_SIZE));
    const reset = encoder.beginComputePass();
    reset.setPipeline(this.resetPipeline);
    reset.setBindGroup(0, this.bindGroup);
    reset.dispatchWorkgroups(groups);
    reset.end();

    const cull = encoder.beginComputePass();
    cull.setPipeline(this.cullPipeline);
    cull.setBindGroup(0, this.bindGroup);
    cull.dispatchWorkgroups(Math.max(1, Math.ceil(this.objectCount / WORKGROUP_SIZE)));
    cull.end();
  }

  /**
   * Issues the indirect draws. A fixed loop of `objectCount` drawIndexedIndirect
   * calls — constant CPU cost, no per-visible-object iteration; slots the GPU
   * left empty have indexCount 0 and draw nothing. Each draw binds the draw-slot
   * group at the dynamic offset for slot i, so the vertex shader recovers the
   * object id via slotToObject[slotIndex].
   *
   * The pipeline, the camera/object bind groups, and the arena's vertex/index
   * buffers must already be set. `slotGroupIndex` is the @group the pipeline
   * uses for this system's drawSlotBindGroupLayout.
   */
  drawAll(renderPass, slotGroupIndex) {
    if (this.multiDraw) {
      // One call drives the whole batch; count comes from drawCountBuffer.
      renderPass.multiDrawIndexedIndirect(
        this.drawArgsBuffer.gpuBuffer, 0, this.objectCount,
        this.drawCountBuffer.gpuBuffer, 0,
      );
      return;
    }
    if (this.firstInstanceId) {
      // Cheap loop: bare indirect draws, object id via firstInstance (needs
      // indirect-first-instance). No per-draw bind-group rebind. Empty slots
      // have indexCount 0 and draw nothing.
      for (let i = 0; i < this.objectCount; i++) {
        renderPass.drawIndexedIndirect(this.drawArgsBuffer.gpuBuffer, i * DRAW_ARG_SIZE);
      }
      return;
    }
    // Last-resort loop: no first-instance support — recover the id via a
    // per-draw dynamic uniform offset into slotToObject.
    for (let i = 0; i < this.objectCount; i++) {
      renderPass.setBindGroup(slotGroupIndex, this.drawSlotBindGroup, [i * UNIFORM_ALIGN]);
      renderPass.drawIndexedIndirect(this.drawArgsBuffer.gpuBuffer, i * DRAW_ARG_SIZE);
    }
  }

  destroy() {
    if (this._ownsRecordBuffer) this.recordBuffer.destroy();
    this.drawArgsBuffer.destroy();
    this.drawCountBuffer.destroy();
    this.paramsBuffer.destroy();
    this.slotToObjectBuffer.destroy();
    this.slotIndexBuffer.destroy();
  }
}
