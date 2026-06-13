import { multiply } from '../math/mat4.js';

// Uniform buffer layout (std140-compatible, matches WGSL struct below):
//
// struct Camera {
//   viewMatrix: mat4x4f,       // offset 0,   64 bytes
//   projectionMatrix: mat4x4f, // offset 64,  64 bytes
//   frustumPlanes: array<vec4f, 6>, // offset 128, 96 bytes
//   viewport: vec4f,           // offset 224, 16 bytes
// }                            // total: 240 bytes
const CAMERA_BUFFER_SIZE = 240;

export class Camera {
  constructor(device) {
    this.device = device;
    this.buffer = device.resources.createBuffer({
      size: CAMERA_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.viewMatrix = new Float32Array(16);
    this.projectionMatrix = new Float32Array(16);
    this.frustumPlanes = new Float32Array(24); // 6 * vec4
    this.viewport = new Float32Array(4);
    this._viewProjectionMatrix = new Float32Array(16);
  }

  setViewMatrix(matrix) {
    this.viewMatrix.set(matrix);
  }

  setProjectionMatrix(matrix) {
    this.projectionMatrix.set(matrix);
  }

  setViewport(x, y, width, height) {
    this.viewport.set([x, y, width, height]);
  }

  /**
   * Derives world-space frustum planes from the combined view-projection
   * matrix and writes the full uniform buffer. Call once per frame after
   * updating the view/projection matrices.
   */
  update() {
    multiply(this.projectionMatrix, this.viewMatrix, this._viewProjectionMatrix);
    this._updateFrustumPlanes();

    const data = new Float32Array(CAMERA_BUFFER_SIZE / 4);
    data.set(this.viewMatrix, 0);
    data.set(this.projectionMatrix, 16);
    data.set(this.frustumPlanes, 32);
    data.set(this.viewport, 56);

    this.device.queue.writeBuffer(this.buffer.gpuBuffer, 0, data);
  }

  _updateFrustumPlanes() {
    // Extract world-space frustum planes (left, right, bottom, top, near, far)
    // from the combined view-projection matrix using the standard
    // Gribb/Hartmann method.
    const p = this._viewProjectionMatrix;
    const planes = [
      [p[3] + p[0], p[7] + p[4], p[11] + p[8], p[15] + p[12]],  // left
      [p[3] - p[0], p[7] - p[4], p[11] - p[8], p[15] - p[12]],  // right
      [p[3] + p[1], p[7] + p[5], p[11] + p[9], p[15] + p[13]],  // bottom
      [p[3] - p[1], p[7] - p[5], p[11] - p[9], p[15] - p[13]],  // top
      [p[3] + p[2], p[7] + p[6], p[11] + p[10], p[15] + p[14]], // near
      [p[3] - p[2], p[7] - p[6], p[11] - p[10], p[15] - p[14]], // far
    ];

    for (let i = 0; i < 6; i++) {
      const [a, b, c, d] = planes[i];
      const len = Math.hypot(a, b, c);
      this.frustumPlanes.set([a / len, b / len, c / len, d / len], i * 4);
    }
  }

  destroy() {
    this.buffer.destroy();
  }
}
