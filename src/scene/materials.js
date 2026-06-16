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
  constructor({ color = 0xffffff, fog = true, castShadow = true, receiveShadow = true } = {}) {
    this.kind = 'lambert';
    this.color = color instanceof Color ? color : new Color(color);
    this.fog = fog;
    this.transparent = false;
    this.opacity = 1;
    // Lambert receives sun shadows by default (the shader always samples
    // sunShadow, which returns 1.0 when shadows are off).
    this.castShadow = castShadow;
    this.receiveShadow = receiveShadow;
  }
}

export class BasicMaterial {
  constructor({
    color = 0xffffff, opacity = 1, transparent = false,
    blending = 'normal', depthWrite = true, depthTest = true,
    side = 'front', wireframe = false, fog = true, castShadow = true,
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
    // Unlit; doesn't receive shadows. Casting defaults on but transparent/glow
    // materials (sun, atmosphere) should pass castShadow:false.
    this.castShadow = castShadow;
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
    merge = false,
    castShadow = true,
    receiveShadow = false,
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
    // Shadow participation (used when SceneRenderer.enableShadows() is active).
    // castShadow: rendered into the sun shadow map. receiveShadow: the shader
    // declares the group(2) shadow bindings and samples sunShadow().
    this.castShadow = castShadow;
    this.receiveShadow = receiveShadow;
    // merge: every mesh sharing this material has an IDENTITY transform (its
    // vertices are already world-space) and reads only this material's shared
    // uniform — so the renderer packs all of them into one geometry stream and
    // draws them in a SINGLE call. Ideal for many static chunks (terrain).
    this.merge = merge;
    this.opacity = 1;
  }
}
