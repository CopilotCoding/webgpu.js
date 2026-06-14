import { Camera } from './Camera.js';
import { perspective, lookAt } from '../math/mat4.js';

// A thin, Three-style wrapper over Camera: keeps an eye position + look target
// + up vector and a perspective projection, and uploads view/projection into
// the underlying Camera's GPU uniform (which also derives frustum planes for
// GPU culling). `layers` is a bitmask a culling pass can AND against per-object
// layer masks so a camera renders only a subset (e.g. a minimap).

export class PerspectiveCamera {
  constructor(device, { fov = Math.PI / 4, aspect = 1, near = 0.1, far = 100 } = {}) {
    this.camera = new Camera(device);
    this.fov = fov;
    this.aspect = aspect;
    this.near = near;
    this.far = far;

    this.position = [0, 0, 5];
    this.target = [0, 0, 0];
    this.up = [0, 1, 0];
    this.layers = 0xffffffff;

    this.updateProjectionMatrix();
  }

  /** Underlying GPU uniform buffer (for bind groups). */
  get buffer() { return this.camera.buffer; }
  get viewMatrix() { return this.camera.viewMatrix; }
  get projectionMatrix() { return this.camera.projectionMatrix; }

  updateProjectionMatrix() {
    this.camera.setProjectionMatrix(perspective(this.fov, this.aspect, this.near, this.far));
  }

  setViewport(x, y, w, h) { this.camera.setViewport(x, y, w, h); return this; }

  lookAt(target) { this.target = target; return this; }

  /** Recomputes the view matrix from position/target/up and uploads the uniform. */
  update() {
    this.camera.setViewMatrix(lookAt(this.position, this.target, this.up));
    this.camera.update();
    return this;
  }
}
