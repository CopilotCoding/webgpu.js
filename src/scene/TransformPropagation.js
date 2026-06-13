import { transformPropagationShader } from './transformPropagation.wgsl.js';

const WORKGROUP_SIZE = 64;

/**
 * GPU-driven transform propagation: a flattened scene graph (see
 * flattenSceneGraph) is uploaded into storage buffers, and world matrices
 * are computed by a compute shader dispatched once per depth level. Level
 * N only reads world matrices written by level N-1, which is what makes
 * per-node parallelism within a level safe.
 *
 * The CPU never iterates the full graph at propagate time — it only
 * re-uploads local matrices for nodes it changed, then issues one compute
 * dispatch per level.
 */
export class TransformPropagation {
  constructor(device, flattened) {
    this.device = device;
    this.nodeCount = flattened.nodes.length;
    this.levels = flattened.levels;

    this.localBuffer = device.resources.createBuffer({
      size: flattened.localMatrices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.localBuffer.gpuBuffer, 0, flattened.localMatrices);

    this.parentBuffer = device.resources.createBuffer({
      size: flattened.parentIndices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.parentBuffer.gpuBuffer, 0, flattened.parentIndices);

    this.worldBuffer = device.resources.createBuffer({
      size: flattened.localMatrices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const shaderModule = device.device.createShaderModule({ code: transformPropagationShader });
    this.pipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'propagate' },
    });

    const bindGroupLayout = this.pipeline.getBindGroupLayout(0);

    // One small uniform buffer + bind group per level: each level has a
    // different (offset, count) pair, so they can't share a bind group.
    this.levelBindGroups = this.levels.map((level) => {
      const levelBuffer = device.resources.createBuffer({
        size: 8, // offset: u32, count: u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(levelBuffer.gpuBuffer, 0, new Uint32Array([level.offset, level.count]));

      const bindGroup = device.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.localBuffer.gpuBuffer } },
          { binding: 1, resource: { buffer: this.parentBuffer.gpuBuffer } },
          { binding: 2, resource: { buffer: this.worldBuffer.gpuBuffer } },
          { binding: 3, resource: { buffer: levelBuffer.gpuBuffer } },
        ],
      });

      return { bindGroup, levelBuffer, count: level.count };
    });
  }

  /** Re-uploads local matrices for nodes whose transform changed this frame. */
  updateLocalMatrices(localMatrices) {
    this.device.queue.writeBuffer(this.localBuffer.gpuBuffer, 0, localMatrices);
  }

  /** Dispatches one compute pass per depth level, in order. */
  propagate(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);

    for (const level of this.levelBindGroups) {
      pass.setBindGroup(0, level.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(level.count / WORKGROUP_SIZE));
    }

    pass.end();
  }

  destroy() {
    this.localBuffer.destroy();
    this.parentBuffer.destroy();
    this.worldBuffer.destroy();
    for (const level of this.levelBindGroups) level.levelBuffer.destroy();
  }
}
