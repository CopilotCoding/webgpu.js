import { Color } from './Color.js';

// Scene-layer material descriptors. These are plain data the SceneRenderer maps
// to GPU pipelines + per-object material params — they don't own pipelines
// themselves (the renderer caches a pipeline per distinct material
// configuration). Properties are mutable so a game can animate them
// (material.color.setHex(...), material.opacity = ...).
//
// 'side': 'front' | 'back' | 'double'   (maps to cullMode)
// 'blending': 'normal' | 'additive'

export class LambertMaterial {
  constructor({ color = 0xffffff, fog = true } = {}) {
    this.kind = 'lambert';
    this.color = color instanceof Color ? color : new Color(color);
    this.fog = fog;
    this.transparent = false;
    this.opacity = 1;
  }
}

export class BasicMaterial {
  constructor({
    color = 0xffffff, opacity = 1, transparent = false,
    blending = 'normal', depthWrite = true, depthTest = true,
    side = 'front', wireframe = false, fog = true,
  } = {}) {
    this.kind = 'basic';
    this.color = color instanceof Color ? color : new Color(color);
    this.opacity = opacity;
    this.transparent = transparent;
    this.blending = blending;
    this.depthWrite = depthWrite;
    this.depthTest = depthTest;
    this.side = side;
    this.wireframe = wireframe;
    this.fog = fog;
  }
}

export class PointsMaterial {
  constructor({ color = 0xffffff, size = 1, depthWrite = false } = {}) {
    this.kind = 'points';
    this.color = color instanceof Color ? color : new Color(color);
    this.size = size;
    this.depthWrite = depthWrite;
    this.transparent = false;
    this.opacity = 1;
  }
}

// Custom-shader material (the terrain). Carries WGSL source + a uniforms object
// the renderer uploads. `uniforms` is { name: { value } } like a typical
// shader-material, packed in declaration order into one uniform buffer by the
// renderer's terrain pipeline (see SceneRenderer). vertexAttributes lists the
// per-vertex attributes the geometry must provide beyond position
// (e.g. ['color','skyAccess']).
export class ShaderMaterial {
  constructor({ uniforms = {}, wgsl = null, vertexAttributes = [], side = 'front', transparent = false }) {
    this.kind = 'shader';
    this.uniforms = uniforms;
    this.wgsl = wgsl;
    this.vertexAttributes = vertexAttributes;
    this.side = side;
    this.transparent = transparent;
    this.opacity = 1;
  }
}
