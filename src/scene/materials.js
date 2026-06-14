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

// Custom-shader material: an app brings its own WGSL and the renderer builds a
// dedicated pipeline for it, drawing each mesh that uses it per-mesh, with the
// camera at group(0) binding(0) and a single app-controlled uniform buffer at
// group(1) binding(0).
//
// This is a fully GENERAL custom-shader material — the engine has no knowledge
// of what the shader does. The app supplies:
//   - wgsl: shader source with `vertexMain`/`fragmentMain`. Camera is at
//     @group(0) @binding(0); the uniform block at @group(1) @binding(0).
//   - attributes: ordered list of the geometry attribute names to bind as
//     vertex buffers, each at the matching shaderLocation (e.g.
//     ['position','normal','color']). Their formats come from the
//     Geometry.
//   - uniformSize: byte size of the group(1) uniform buffer.
//   - updateUniforms(view): called each frame; write your bytes into the given
//     Float32Array view of the uniform buffer.
//   - cull/depthWrite/depthCompare/topology: pipeline state.
export class ShaderMaterial {
  constructor({
    wgsl,
    attributes = ['position', 'normal', 'uv'],
    uniformSize = 0,
    updateUniforms = null,
    side = 'front',
    transparent = false,
    depthWrite = true,
    depthCompare = 'less',
    topology = 'triangle-list',
  }) {
    this.kind = 'shader';
    this.wgsl = wgsl;
    this.attributes = attributes;
    this.uniformSize = uniformSize;
    this.updateUniforms = updateUniforms;
    this.side = side;
    this.transparent = transparent;
    this.depthWrite = depthWrite;
    this.depthCompare = depthCompare;
    this.topology = topology;
    this.opacity = 1;
  }
}
