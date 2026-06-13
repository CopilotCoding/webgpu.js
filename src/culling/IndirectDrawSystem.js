import { indirectDrawShader } from './indirectDraw.wgsl.js';

const WORKGROUP_SIZE = 64;

/**
 * Builds a GPUDrawIndirect argument buffer from CullingPass's visibility
 * bitmask: visible objects' original indices are compacted into
 * `visibleIndices`, and `indirectBuffer` ends up holding
 * { vertexCount, instanceCount, firstVertex, firstInstance } where
 * instanceCount is the number of visible objects.
 *
 * The CPU submits exactly one drawIndirect() call per batch — it never
 * iterates objects to decide what to draw. The vertex shader maps
 * instance_index -> original object index via visibleIndices.
 */
export class IndirectDrawSystem {
  constructor(device, cullingPass, vertexCount) {
    this.device = device;
    this.capacity = cullingPass.capacity ?? cullingPass.objectCount;
    this.objectCount = cullingPass.objectCount;

    this.visibleIndicesBuffer = device.resources.createBuffer({
      size: this.capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.indirectBuffer = device.resources.createBuffer({
      size: 16, // vertexCount, instanceCount, firstVertex, firstInstance
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indirectBuffer.gpuBuffer, 0, new Uint32Array([vertexCount, 0, 0, 0]));

    this.objectCountBuffer = device.resources.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([this.objectCount]));

    const shaderModule = device.device.createShaderModule({ code: indirectDrawShader });

    this.resetPipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'resetIndirectDraw' },
    });
    this.buildPipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'buildIndirectDraw' },
    });

    // `layout: 'auto'` derives each pipeline's bind group layout only from
    // the bindings its entry point actually references, so resetIndirectDraw
    // (which only touches drawArgs) gets a 1-binding layout while
    // buildIndirectDraw gets all 4 — each needs its own bind group.
    this.resetBindGroup = device.device.createBindGroup({
      layout: this.resetPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: this.indirectBuffer.gpuBuffer } },
      ],
    });
    this.buildBindGroup = device.device.createBindGroup({
      layout: this.buildPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cullingPass.visibilityBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.visibleIndicesBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.indirectBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: this.objectCountBuffer.gpuBuffer } },
      ],
    });
  }

  /** Updates the live object count (<= capacity) the build pass iterates. */
  setObjectCount(count) {
    this.objectCount = count;
    this.device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([count]));
  }

  build(encoder) {
    const resetPass = encoder.beginComputePass();
    resetPass.setPipeline(this.resetPipeline);
    resetPass.setBindGroup(0, this.resetBindGroup);
    resetPass.dispatchWorkgroups(1);
    resetPass.end();

    const buildPass = encoder.beginComputePass();
    buildPass.setPipeline(this.buildPipeline);
    buildPass.setBindGroup(0, this.buildBindGroup);
    buildPass.dispatchWorkgroups(Math.ceil(this.objectCount / WORKGROUP_SIZE));
    buildPass.end();
  }

  destroy() {
    this.visibleIndicesBuffer.destroy();
    this.indirectBuffer.destroy();
    this.objectCountBuffer.destroy();
  }
}
