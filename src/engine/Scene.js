import { fromTranslationRotationScale } from '../math/mat4.js';
import { MATERIAL_PARAMS_SIZE } from './forwardShaders.wgsl.js';

// A Scene owns the per-object data the GPU-driven renderer consumes: world
// matrices, material params, and local AABBs, all in capacity-sized storage
// buffers, plus the point-light array. Objects and lights are added via
// handles whose setters write directly into the right buffer slot — no CPU
// iteration of the scene at render time.
//
// Every mesh in a Scene shares one Geometry (the box, by default). Different
// shapes belong in different Scenes/batches; this keeps the whole scene a
// single indirect draw, which is the point of the GPU-driven path.

export class Scene {
  /**
   * @param {object} options
   * @param {number} [options.maxObjects=4096] object capacity (<=4096; the
   *   pick result packs the index into 12 bits)
   * @param {number} [options.maxLights=256] point-light capacity
   */
  constructor(device, { maxObjects = 4096, maxLights = 256 } = {}) {
    this.device = device;
    this.maxObjects = maxObjects;
    this.maxLights = maxLights;

    this.objectCount = 0;
    this.lightCount = 0;

    // Per-object: world matrix (mat4x4f), material params, local bounds.
    this.worldMatricesBuffer = device.resources.createBuffer({
      size: maxObjects * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.materialParamsBuffer = device.resources.createBuffer({
      size: maxObjects * MATERIAL_PARAMS_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Local AABBs are kept CPU-side so the Engine can seed the culling pass's
    // bounds buffer per object as meshes are added.
    this.bounds = [];

    // Per-object base emissive, mirrored CPU-side: the material buffer is
    // write-only from JS, and the Engine's picking highlight needs the base
    // value to restore on un-hover.
    this.emissive = [];

    // PointLight array, CPU-side staging mirrored to the light buffer each
    // frame by the Engine.
    this.lightData = new Float32Array(maxLights * 8);

    // The Engine sets this so a mesh handle's setters can keep GPU object
    // counts in sync if a scene grows after the first frame.
    this._onObjectAdded = null;
  }

  /**
   * Adds a mesh instance.
   * @param {object} options
   * @param {import('../geometry/Geometry.js').Geometry} options.geometry shared geometry (must match the Engine's batch geometry)
   * @param {{min:number[],max:number[]}} [options.bounds] local AABB for culling/picking (defaults to geometry.computeBounds())
   * @param {number[]} [options.position=[0,0,0]]
   * @param {number[]} [options.rotation=[0,0,0,1]] quaternion
   * @param {number[]} [options.scale=[1,1,1]]
   * @param {number[]} [options.baseColor=[1,1,1]]
   * @param {number} [options.textureLayer=0]
   * @param {number[]} [options.uvScale=[1,1]]
   * @param {number[]} [options.emissive=[0,0,0]]
   * @returns {MeshHandle}
   */
  addMesh({
    geometry,
    bounds,
    position = [0, 0, 0],
    rotation = [0, 0, 0, 1],
    scale = [1, 1, 1],
    baseColor = [1, 1, 1],
    textureLayer = 0,
    uvScale = [1, 1],
    emissive = [0, 0, 0],
  }) {
    if (this.objectCount >= this.maxObjects) {
      throw new Error(`Scene: exceeded maxObjects (${this.maxObjects})`);
    }
    const index = this.objectCount++;
    const localBounds = bounds ?? geometry.computeBounds();
    this.bounds[index] = localBounds;

    this.emissive[index] = [...emissive];

    const handle = new MeshHandle(this, index);
    handle.setTransform(position, rotation, scale);
    handle.setMaterial({ baseColor, textureLayer, uvScale, emissive });

    if (this._onObjectAdded) this._onObjectAdded(index, localBounds);
    return handle;
  }

  /** Returns an object's CPU-mirrored base emissive (without highlight). */
  getEmissive(index) {
    return this.emissive[index] ?? [0, 0, 0];
  }

  /**
   * Adds a point light.
   * @returns {LightHandle}
   */
  addLight({ position = [0, 0, 0], radius = 5, color = [1, 1, 1], intensity = 1 } = {}) {
    if (this.lightCount >= this.maxLights) {
      throw new Error(`Scene: exceeded maxLights (${this.maxLights})`);
    }
    const index = this.lightCount++;
    const handle = new LightHandle(this, index);
    handle.set({ position, radius, color, intensity });
    return handle;
  }

  destroy() {
    this.worldMatricesBuffer.destroy();
    this.materialParamsBuffer.destroy();
  }
}

class MeshHandle {
  constructor(scene, index) {
    this.scene = scene;
    this.index = index;
  }

  setTransform(position, rotation = [0, 0, 0, 1], scale = [1, 1, 1]) {
    const m = fromTranslationRotationScale(position, rotation, scale);
    this.scene.device.queue.writeBuffer(this.scene.worldMatricesBuffer.gpuBuffer, this.index * 64, m);
    return this;
  }

  /** Writes the full world matrix directly (for callers driving their own math). */
  setMatrix(matrix) {
    this.scene.device.queue.writeBuffer(this.scene.worldMatricesBuffer.gpuBuffer, this.index * 64, matrix);
    return this;
  }

  setMaterial({ baseColor = [1, 1, 1], textureLayer = 0, uvScale = [1, 1], emissive = [0, 0, 0] } = {}) {
    const data = new Float32Array(MATERIAL_PARAMS_SIZE / 4);
    data.set(baseColor, 0);
    data[3] = textureLayer;
    data.set(uvScale, 4);
    data.set(emissive, 8);
    this.scene.device.queue.writeBuffer(this.scene.materialParamsBuffer.gpuBuffer, this.index * MATERIAL_PARAMS_SIZE, data);
    this.scene.emissive[this.index] = [...emissive];
    return this;
  }
}

class LightHandle {
  constructor(scene, index) {
    this.scene = scene;
    this.index = index;
  }

  set({ position, radius, color, intensity }) {
    const i = this.index * 8;
    const d = this.scene.lightData;
    if (position) { d[i] = position[0]; d[i + 1] = position[1]; d[i + 2] = position[2]; }
    if (radius !== undefined) d[i + 3] = radius;
    if (color) { d[i + 4] = color[0]; d[i + 5] = color[1]; d[i + 6] = color[2]; }
    if (intensity !== undefined) d[i + 7] = intensity;
    return this;
  }

  setPosition(x, y, z) {
    const i = this.index * 8;
    this.scene.lightData[i] = x;
    this.scene.lightData[i + 1] = y;
    this.scene.lightData[i + 2] = z;
    return this;
  }
}
