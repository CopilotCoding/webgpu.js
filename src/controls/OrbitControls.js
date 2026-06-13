import { lookAt } from '../math/mat4.js';

// Orbit camera: drag to rotate around a target, wheel to zoom, right-drag to
// pan. Each frame call update() and feed the resulting view matrix to the
// camera (or let the Engine do it via engine.camera.setViewMatrix). Mirrors
// the ergonomics of Three.js OrbitControls.
export class OrbitControls {
  constructor(domElement, {
    target = [0, 0, 0],
    distance = 10,
    minDistance = 1,
    maxDistance = 200,
    azimuth = 0,          // radians around +Y
    polar = Math.PI / 4,  // radians from +Y axis
    minPolar = 0.05,
    maxPolar = Math.PI - 0.05,
    rotateSpeed = 0.005,
    zoomSpeed = 0.1,
    panSpeed = 1.0,
  } = {}) {
    this.domElement = domElement;
    this.target = [...target];
    this.distance = distance;
    this.minDistance = minDistance;
    this.maxDistance = maxDistance;
    this.azimuth = azimuth;
    this.polar = polar;
    this.minPolar = minPolar;
    this.maxPolar = maxPolar;
    this.rotateSpeed = rotateSpeed;
    this.zoomSpeed = zoomSpeed;
    this.panSpeed = panSpeed;

    this.viewMatrix = new Float32Array(16);
    this.eye = [0, 0, 0];

    this._dragging = null; // 'rotate' | 'pan'
    this._lastX = 0;
    this._lastY = 0;

    this._attach();
    this.update();
  }

  _attach() {
    const el = this.domElement;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      this._dragging = e.button === 2 ? 'pan' : 'rotate';
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', (e) => {
      this._dragging = null;
      try { el.releasePointerCapture(e.pointerId); } catch {}
    });
    el.addEventListener('pointermove', (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      if (this._dragging === 'rotate') {
        this.azimuth -= dx * this.rotateSpeed;
        this.polar = clamp(this.polar - dy * this.rotateSpeed, this.minPolar, this.maxPolar);
      } else {
        this._pan(dx, dy);
      }
    });
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = Math.exp(Math.sign(e.deltaY) * this.zoomSpeed);
      this.distance = clamp(this.distance * factor, this.minDistance, this.maxDistance);
    }, { passive: false });
  }

  _pan(dx, dy) {
    // Pan in the camera's screen plane, scaled by distance so it feels
    // consistent at any zoom.
    const forward = normalize(sub(this.target, this.eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const scale = this.distance * 0.001 * this.panSpeed;
    for (let i = 0; i < 3; i++) {
      const delta = (-dx * right[i] + dy * up[i]) * scale;
      this.target[i] += delta;
    }
  }

  /** Recomputes eye position and view matrix from the current orbit state. */
  update() {
    const sinPolar = Math.sin(this.polar);
    this.eye = [
      this.target[0] + this.distance * sinPolar * Math.sin(this.azimuth),
      this.target[1] + this.distance * Math.cos(this.polar),
      this.target[2] + this.distance * sinPolar * Math.cos(this.azimuth),
    ];
    this.viewMatrix.set(lookAt(this.eye, this.target, [0, 1, 0]));
    return this.viewMatrix;
  }
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
