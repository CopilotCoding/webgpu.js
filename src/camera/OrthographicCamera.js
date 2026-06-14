import { Camera } from './Camera.js';
import { orthographic, lookAt } from '../math/mat4.js';

// Three-style orthographic camera (top-down minimap, UI overlays) over the
// shared Camera GPU uniform. Same surface as PerspectiveCamera: position /
// target / up, a layer bitmask, update() uploads view + projection and derives
// frustum planes for GPU culling.

export class OrthographicCamera {
  constructor(device, { left = -1, right = 1, bottom = -1, top = 1, near = 0.1, far = 100 } = {}) {
    this.camera = new Camera(device);
    this.left = left; this.right = right;
    this.bottom = bottom; this.top = top;
    this.near = near; this.far = far;

    this.position = [0, 5, 0];
    this.target = [0, 0, 0];
    this.up = [0, 0, -1];
    this.layers = 0xffffffff;

    this.updateProjectionMatrix();
  }

  get buffer() { return this.camera.buffer; }
  get viewMatrix() { return this.camera.viewMatrix; }
  get projectionMatrix() { return this.camera.projectionMatrix; }

  updateProjectionMatrix() {
    this.camera.setProjectionMatrix(orthographic(this.left, this.right, this.bottom, this.top, this.near, this.far));
  }

  /** Sets a symmetric view box of half-size `extent`. */
  setExtent(extent) {
    this.left = -extent; this.right = extent;
    this.bottom = -extent; this.top = extent;
    this.updateProjectionMatrix();
    return this;
  }

  setViewport(x, y, w, h) { this.camera.setViewport(x, y, w, h); return this; }

  lookAt(target) { this.target = target; return this; }

  update() {
    this.camera.setViewMatrix(lookAt(arr(this.position), arr(this.target), arr(this.up)));
    this.camera.update();
    return this;
  }
}

// Accepts a [x,y,z] array or a {x,y,z} vector (e.g. Vec3).
function arr(v) { return Array.isArray(v) ? v : [v.x, v.y, v.z]; }
