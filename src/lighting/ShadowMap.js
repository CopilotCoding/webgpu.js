import { multiply, identity } from '../math/mat4.js';

// A single shadow map for a directional light. The light's view-projection
// matrix is an orthographic projection that tightly bounds a given
// world-space AABB (the scene bounds), so every caster/receiver in that
// volume projects into [-1,1] x/y and [0,1] z.
//
// Depth is rendered into a depth32float 2D texture. A uniform buffer holds
// the view-projection matrix and the light direction for the lighting
// shader's shadow sample.

// Uniform buffer layout (matches ShadowMap struct in shadows.wgsl.js):
//   viewProjectionMatrix: mat4x4f, // 64 bytes
//   lightDirection: vec3f, _pad: f32 // 16 bytes
const SHADOW_BUFFER_SIZE = 64 + 16;

export class ShadowMap {
  constructor(device, { mapSize = 2048 } = {}) {
    this.device = device;
    this.mapSize = mapSize;

    this.depthTexture = device.resources.createTexture({
      size: [mapSize, mapSize],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.depthSampler = device.resources.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.buffer = device.resources.createBuffer({
      size: SHADOW_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.viewProjectionMatrix = identity();
  }

  getView() {
    return this.depthTexture.gpuTexture.createView();
  }

  /**
   * Builds an orthographic light-space view-projection matrix that tightly
   * bounds the given world-space AABB, and uploads it along with the light
   * direction.
   *
   * @param {[number,number,number]} lightDirection world-space direction the light travels (normalized)
   * @param {{min: [number,number,number], max: [number,number,number]}} sceneBounds world-space AABB to cover
   */
  update(lightDirection, sceneBounds) {
    const dir = normalize(lightDirection);
    let up = [0, 1, 0];
    if (Math.abs(dir[1]) > 0.99) up = [0, 0, 1];

    const center = [
      (sceneBounds.min[0] + sceneBounds.max[0]) / 2,
      (sceneBounds.min[1] + sceneBounds.max[1]) / 2,
      (sceneBounds.min[2] + sceneBounds.max[2]) / 2,
    ];
    const eye = [center[0] - dir[0], center[1] - dir[1], center[2] - dir[2]];
    const lightView = lookAtMatrix(eye, center, up);

    // Transform the scene AABB's 8 corners into light view space and find
    // their AABB — this is the tight orthographic bound.
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < 8; i++) {
      const corner = [
        i & 1 ? sceneBounds.max[0] : sceneBounds.min[0],
        i & 2 ? sceneBounds.max[1] : sceneBounds.min[1],
        i & 4 ? sceneBounds.max[2] : sceneBounds.min[2],
      ];
      const v = transformPoint(lightView, corner);
      minX = Math.min(minX, v[0]); maxX = Math.max(maxX, v[0]);
      minY = Math.min(minY, v[1]); maxY = Math.max(maxY, v[1]);
      minZ = Math.min(minZ, v[2]); maxZ = Math.max(maxZ, v[2]);
    }

    // View space looks down -z, so the AABB's "near"/"far" along the light's
    // view direction are -maxZ/-minZ.
    const lightProj = orthographic(minX, maxX, minY, maxY, -maxZ, -minZ);

    multiply(lightProj, lightView, this.viewProjectionMatrix);
    this._upload(dir);
  }

  _upload(dir) {
    const data = new Float32Array(SHADOW_BUFFER_SIZE / 4);
    data.set(this.viewProjectionMatrix, 0);
    data.set(dir, 16);

    this.device.queue.writeBuffer(this.buffer.gpuBuffer, 0, data);
  }

  destroy() {
    this.depthTexture.destroy();
    this.depthSampler.destroy();
    this.buffer.destroy();
  }
}

function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function lookAtMatrix(eye, target, up) {
  const zAxis = normalize(subtract(eye, target));
  const xAxis = normalize(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);

  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1,
  ]);
}

// WebGPU clip space: x,y in [-1,1], z in [0,1].
function orthographic(left, right, bottom, top, near, far) {
  const out = new Float32Array(16);
  out[0] = 2 / (right - left);
  out[5] = 2 / (top - bottom);
  // View space looks down -z, so view-z must be negated: a point at
  // view-z = -near maps to depth 0, view-z = -far maps to depth 1.
  out[10] = -1 / (far - near);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = -near / (far - near);
  out[15] = 1;
  return out;
}

function transformVec4(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

function transformPoint(m, p) {
  const [x, y, z, w] = transformVec4(m, [p[0], p[1], p[2], 1]);
  return [x / w, y / w, z / w];
}
