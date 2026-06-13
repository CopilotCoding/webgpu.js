import { clusterBoundsShader } from './clusters.wgsl.js';
import { invert } from '../math/mat4.js';

const WORKGROUP_SIZE = 4; // matches @workgroup_size(4, 4, 4) in clusters.wgsl.js

/**
 * Subdivides the camera frustum into a 3D grid of view-space AABBs
 * (clusterCountX * clusterCountY screen tiles, each split into
 * clusterCountZ exponential depth slices). Rebuilt whenever the camera's
 * projection or viewport changes — the AABBs depend only on the projection
 * matrix, near/far planes, and screen size, not on the view matrix.
 */
export class ClusterGrid {
  constructor(device, { clusterCountX = 16, clusterCountY = 9, clusterCountZ = 24 } = {}) {
    this.device = device;
    this.clusterCount = [clusterCountX, clusterCountY, clusterCountZ];
    this.totalClusters = clusterCountX * clusterCountY * clusterCountZ;

    // ClusterGridInfo: clusterCount (uvec3+pad), screenSize (vec2),
    // zNear, zFar, inverseProjection (mat4x4f) — 16 + 16 + 64 = 96 bytes.
    this.gridInfoBuffer = device.resources.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ClusterBounds: minPoint (vec3+pad) + maxPoint (vec3+pad) = 32 bytes.
    this.clusterBoundsBuffer = device.resources.createBuffer({
      size: this.totalClusters * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const shaderModule = device.device.createShaderModule({ code: clusterBoundsShader });
    this.pipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'buildClusters' },
    });

    this.bindGroup = device.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gridInfoBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.clusterBoundsBuffer.gpuBuffer } },
      ],
    });

    this._inverseProjection = new Float32Array(16);
    this._gridInfoData = new Float32Array(24); // 96 bytes / 4
  }

  /**
   * Uploads the current projection's parameters. Call whenever the
   * projection matrix, near/far planes, or canvas size change, before
   * `build()`.
   */
  setProjection(projectionMatrix, screenWidth, screenHeight, near, far) {
    invert(projectionMatrix, this._inverseProjection);

    const view = this._gridInfoData;
    const u32View = new Uint32Array(view.buffer);
    u32View[0] = this.clusterCount[0];
    u32View[1] = this.clusterCount[1];
    u32View[2] = this.clusterCount[2];
    u32View[3] = 0;
    view[4] = screenWidth;
    view[5] = screenHeight;
    view[6] = near;
    view[7] = far;
    view.set(this._inverseProjection, 8);

    this.device.queue.writeBuffer(this.gridInfoBuffer.gpuBuffer, 0, view);
  }

  /** Records the cluster AABB build into `encoder`. */
  build(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.clusterCount[0] / WORKGROUP_SIZE),
      Math.ceil(this.clusterCount[1] / WORKGROUP_SIZE),
      Math.ceil(this.clusterCount[2] / WORKGROUP_SIZE),
    );
    pass.end();
  }

  destroy() {
    this.gridInfoBuffer.destroy();
    this.clusterBoundsBuffer.destroy();
  }
}
