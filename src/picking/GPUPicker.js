import { raycastShader } from './raycast.wgsl.js';
import { invert, multiply } from '../math/mat4.js';

const WORKGROUP_SIZE = 64;
const NO_HIT = 0xffffffff;
const INDEX_MASK = 0xfff;

/**
 * GPU raycasting against object bounds. The ray is tested in a compute pass
 * against every object's world-space AABB — the same worldMatrices and
 * bounds buffers the culling pass uses, so picking always agrees with what
 * the GPU is rendering and the CPU never iterates objects.
 *
 * Note: hits are against world-space AABBs of the transformed local bounds.
 * For axis-aligned objects (no rotation) this is exact; rotated objects hit
 * slightly conservatively, like culling does.
 *
 * Usage:
 *   const picker = new GPUPicker(device, { worldBuffer, boundsBuffer, objectCount });
 *   const hit = await picker.pick(screenPointToRay(camera, x, y, w, h));
 *   // hit: { objectIndex, distance, point } or null
 */
export class GPUPicker {
  constructor(device, { worldBuffer, boundsBuffer, objectCount }) {
    if (objectCount > INDEX_MASK + 1) {
      throw new Error(`GPUPicker: objectCount ${objectCount} exceeds the ${INDEX_MASK + 1} objects a packed pick result supports`);
    }
    this.device = device;
    this.objectCount = objectCount;
    this._busy = false;

    this.rayBuffer = device.resources.createBuffer({
      size: 32, // origin vec3 + tMax + direction vec3 + pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.resultBuffer = device.resources.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.stagingBuffer = device.resources.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.objectCountBuffer = device.resources.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([objectCount]));

    const module = device.device.createShaderModule({ code: raycastShader });
    this.pipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'raycast' },
    });

    this.bindGroup = device.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.rayBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: worldBuffer.gpuBuffer } },
        { binding: 2, resource: { buffer: boundsBuffer.gpuBuffer } },
        { binding: 3, resource: { buffer: this.resultBuffer.gpuBuffer } },
        { binding: 4, resource: { buffer: this.objectCountBuffer.gpuBuffer } },
      ],
    });
  }

  /** True while a pick is awaiting its GPU readback. */
  get busy() {
    return this._busy;
  }

  /** Updates the live object count tested (<= the capacity at construction). */
  setObjectCount(count) {
    this.objectCount = count;
    this.device.queue.writeBuffer(this.objectCountBuffer.gpuBuffer, 0, new Uint32Array([count]));
  }

  /**
   * Casts the ray and resolves with the nearest hit, or null on a miss.
   * Only one pick may be in flight at a time — check `busy` before calling;
   * a call made while busy resolves to undefined (NOT null, so a skipped
   * pick can't be mistaken for a miss).
   *
   * @param {{ origin: number[], direction: number[], tMax?: number }} ray
   */
  async pick(ray) {
    if (this._busy) return undefined;
    this._busy = true;

    try {
      const rayData = new Float32Array(8);
      rayData.set(ray.origin, 0);
      rayData[3] = ray.tMax ?? 1e30;
      rayData.set(ray.direction, 4);
      this.device.queue.writeBuffer(this.rayBuffer.gpuBuffer, 0, rayData);
      this.device.queue.writeBuffer(this.resultBuffer.gpuBuffer, 0, new Uint32Array([NO_HIT]));

      const encoder = this.device.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.objectCount / WORKGROUP_SIZE));
      pass.end();
      encoder.copyBufferToBuffer(this.resultBuffer.gpuBuffer, 0, this.stagingBuffer.gpuBuffer, 0, 4);
      this.device.queue.submit([encoder.finish()]);

      await this.stagingBuffer.gpuBuffer.mapAsync(GPUMapMode.READ);
      const packed = new Uint32Array(this.stagingBuffer.gpuBuffer.getMappedRange())[0];
      this.stagingBuffer.gpuBuffer.unmap();

      if (packed === NO_HIT) return null;

      const objectIndex = packed & INDEX_MASK;
      // Recover the (mantissa-truncated) distance from the high bits.
      const distance = new Float32Array(new Uint32Array([packed & ~INDEX_MASK]).buffer)[0];
      return {
        objectIndex,
        distance,
        point: [
          ray.origin[0] + ray.direction[0] * distance,
          ray.origin[1] + ray.direction[1] * distance,
          ray.origin[2] + ray.direction[2] * distance,
        ],
      };
    } finally {
      this._busy = false;
    }
  }

  destroy() {
    this.rayBuffer.destroy();
    this.resultBuffer.destroy();
    this.stagingBuffer.destroy();
    this.objectCountBuffer.destroy();
  }
}

/**
 * Builds a world-space picking ray from a screen position, Three.js
 * Raycaster.setFromCamera-style.
 *
 * @param {{ viewMatrix: Float32Array, projectionMatrix: Float32Array }} camera
 * @param {number} x pointer x in CSS pixels relative to the canvas
 * @param {number} y pointer y in CSS pixels relative to the canvas
 * @param {number} width canvas CSS width
 * @param {number} height canvas CSS height
 * @returns {{ origin: number[], direction: number[] }}
 */
export function screenPointToRay(camera, x, y, width, height) {
  const ndcX = (x / width) * 2 - 1;
  const ndcY = 1 - (y / height) * 2;

  const viewProjection = multiply(camera.projectionMatrix, camera.viewMatrix);
  const inverse = invert(viewProjection);

  // Unproject the pointer at the near plane (z=0) and far plane (z=1).
  const near = transformPoint(inverse, [ndcX, ndcY, 0]);
  const far = transformPoint(inverse, [ndcX, ndcY, 1]);

  const direction = [far[0] - near[0], far[1] - near[1], far[2] - near[2]];
  const length = Math.hypot(direction[0], direction[1], direction[2]);

  return {
    origin: near,
    direction: [direction[0] / length, direction[1] / length, direction[2] / length],
  };
}

function transformPoint(m, p) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return [x / w, y / w, z / w];
}
