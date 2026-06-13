import { lookAt } from '../math/mat4.js';

// Free-fly camera: WASD to move, Q/E down/up, drag to look around. Call
// update(dt) each frame with the frame delta in seconds and feed the view
// matrix to the camera. Yaw/pitch are tracked as Euler angles; roll is
// fixed (world-up stays up).
export class FlyControls {
  constructor(domElement, {
    position = [0, 2, 10],
    yaw = Math.PI,    // radians; PI looks toward -Z
    pitch = 0,        // radians; clamped away from straight up/down
    moveSpeed = 8,    // world units / second
    lookSpeed = 0.003,
    boostMultiplier = 4, // hold Shift
  } = {}) {
    this.domElement = domElement;
    this.position = [...position];
    this.yaw = yaw;
    this.pitch = pitch;
    this.moveSpeed = moveSpeed;
    this.lookSpeed = lookSpeed;
    this.boostMultiplier = boostMultiplier;

    this.viewMatrix = new Float32Array(16);
    this._keys = new Set();
    this._dragging = false;
    this._lastX = 0;
    this._lastY = 0;

    this._attach();
    this.update(0);
  }

  _attach() {
    const el = this.domElement;
    el.addEventListener('pointerdown', (e) => {
      this._dragging = true;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', (e) => {
      this._dragging = false;
      try { el.releasePointerCapture(e.pointerId); } catch {}
    });
    el.addEventListener('pointermove', (e) => {
      if (!this._dragging) return;
      this.yaw -= (e.clientX - this._lastX) * this.lookSpeed;
      this.pitch = clamp(this.pitch - (e.clientY - this._lastY) * this.lookSpeed, -1.5, 1.5);
      this._lastX = e.clientX;
      this._lastY = e.clientY;
    });
    // Key state on window so focus on the canvas isn't required.
    window.addEventListener('keydown', (e) => this._keys.add(e.code));
    window.addEventListener('keyup', (e) => this._keys.delete(e.code));
  }

  _forward() {
    const cp = Math.cos(this.pitch);
    return [Math.sin(this.yaw) * cp, Math.sin(this.pitch), Math.cos(this.yaw) * cp];
  }

  update(dt) {
    const forward = this._forward();
    const right = normalize(cross(forward, [0, 1, 0]));

    let speed = this.moveSpeed * dt;
    if (this._keys.has('ShiftLeft') || this._keys.has('ShiftRight')) speed *= this.boostMultiplier;

    const move = [0, 0, 0];
    const add = (v, s) => { move[0] += v[0] * s; move[1] += v[1] * s; move[2] += v[2] * s; };
    if (this._keys.has('KeyW')) add(forward, speed);
    if (this._keys.has('KeyS')) add(forward, -speed);
    if (this._keys.has('KeyD')) add(right, speed);
    if (this._keys.has('KeyA')) add(right, -speed);
    if (this._keys.has('KeyE')) move[1] += speed;
    if (this._keys.has('KeyQ')) move[1] -= speed;

    this.position[0] += move[0];
    this.position[1] += move[1];
    this.position[2] += move[2];

    const target = [
      this.position[0] + forward[0],
      this.position[1] + forward[1],
      this.position[2] + forward[2],
    ];
    this.viewMatrix.set(lookAt(this.position, target, [0, 1, 0]));
    return this.viewMatrix;
  }
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
