import { lightCullingShader } from './clusters.wgsl.js';

const WORKGROUP_SIZE = 64;
const MAX_LIGHTS_PER_CLUSTER = 256; // must match clusters.wgsl.js

/**
 * Assigns point lights to clusters each frame: for every cluster, tests its
 * view-space AABB against every light's view-space sphere and writes the
 * matching light indices into a per-cluster slice of `lightIndicesBuffer`.
 * `clusterRangesBuffer` holds, per cluster, a fixed `offset` into that slice
 * and an atomic `count` of how many lights landed there (clamped to
 * MAX_LIGHTS_PER_CLUSTER when read by the fragment shader).
 *
 * Lights themselves live in `lightsBuffer` — a flat array of PointLight
 * structs (position, radius, color, intensity) uploaded by the caller.
 */
export class LightCulling {
  constructor(device, clusterGrid, maxLights) {
    this.device = device;
    this.clusterGrid = clusterGrid;
    this.maxLights = maxLights;

    // PointLight: position (vec3) + radius (f32) + color (vec3) + intensity (f32) = 32 bytes.
    this.lightsBuffer = device.resources.createBuffer({
      size: maxLights * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ClusterLightRange: offset (u32) + count (atomic<u32>) = 8 bytes.
    this.clusterRangesBuffer = device.resources.createBuffer({
      size: clusterGrid.totalClusters * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.lightIndicesBuffer = device.resources.createBuffer({
      size: clusterGrid.totalClusters * MAX_LIGHTS_PER_CLUSTER * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.viewMatrixBuffer = device.resources.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // counts: vec2u { clusterCount, lightCount }.
    this.countsBuffer = device.resources.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.countsBuffer.gpuBuffer, 0, new Uint32Array([clusterGrid.totalClusters, 0]));

    const shaderModule = device.device.createShaderModule({ code: lightCullingShader });
    this.resetPipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'resetClusterRanges' },
    });
    this.cullPipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'cullLights' },
    });

    this.resetBindGroup = device.device.createBindGroup({
      layout: this.resetPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: this.clusterRangesBuffer.gpuBuffer } },
        { binding: 5, resource: { buffer: this.countsBuffer.gpuBuffer } },
      ],
    });
    this.cullBindGroup = device.device.createBindGroup({
      layout: this.cullPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewMatrixBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: clusterGrid.clusterBoundsBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: this.lightsBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: this.clusterRangesBuffer.gpuBuffer } },
        { binding: 4, resource: { buffer: this.lightIndicesBuffer.gpuBuffer } },
        { binding: 5, resource: { buffer: this.countsBuffer.gpuBuffer } },
      ],
    });
  }

  /** Uploads the current frame's view matrix and light count. */
  setView(viewMatrix, lightCount) {
    this.device.queue.writeBuffer(this.viewMatrixBuffer.gpuBuffer, 0, viewMatrix);
    this.device.queue.writeBuffer(this.countsBuffer.gpuBuffer, 4, new Uint32Array([lightCount]));
  }

  /** Uploads the light array. `data` is a Float32Array of PointLight structs. */
  setLights(data) {
    this.device.queue.writeBuffer(this.lightsBuffer.gpuBuffer, 0, data);
  }

  /** Records the reset + cull dispatches into `encoder`. */
  cull(encoder) {
    const resetPass = encoder.beginComputePass();
    resetPass.setPipeline(this.resetPipeline);
    resetPass.setBindGroup(0, this.resetBindGroup);
    resetPass.dispatchWorkgroups(Math.ceil(this.clusterGrid.totalClusters / WORKGROUP_SIZE));
    resetPass.end();

    const cullPass = encoder.beginComputePass();
    cullPass.setPipeline(this.cullPipeline);
    cullPass.setBindGroup(0, this.cullBindGroup);
    cullPass.dispatchWorkgroups(Math.ceil(this.clusterGrid.totalClusters / WORKGROUP_SIZE));
    cullPass.end();
  }

  destroy() {
    this.lightsBuffer.destroy();
    this.clusterRangesBuffer.destroy();
    this.lightIndicesBuffer.destroy();
    this.viewMatrixBuffer.destroy();
    this.countsBuffer.destroy();
  }
}

export { MAX_LIGHTS_PER_CLUSTER };
