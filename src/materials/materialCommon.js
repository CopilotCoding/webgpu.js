// Shared building blocks for the built-in material factories (Basic/Lambert/
// Points). These factories return a configured `Material` plus a small helper
// to make per-instance uniform data, so an app gets Three-style materials
// without hand-writing WGSL — while the underlying `Material`/pipeline path
// stays fully visible.

// The camera uniform every material's group(0) binding 0 expects. Matches
// src/camera/Camera.js's buffer layout.
export const cameraStruct = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
`;

// Linear fog applied in the fragment stage. A material with fog disabled
// passes fogParams.w = 0 so the mix is a no-op.
//
// fogParams = (color.rgb, enabled); fogRange = (near, far, _, _)
export const fogStruct = /* wgsl */ `
struct Fog {
  colorEnabled: vec4f,
  range: vec4f,
};
fn applyFog(color: vec3f, fog: Fog, viewDepth: f32) -> vec3f {
  if (fog.colorEnabled.w < 0.5) { return color; }
  let near = fog.range.x;
  let far = fog.range.y;
  let f = clamp((viewDepth - near) / max(far - near, 0.0001), 0.0, 1.0);
  return mix(color, fog.colorEnabled.rgb, f);
}
`;

// GPUBlendState presets selectable by name on the built-in materials.
export const BLEND_PRESETS = {
  // Standard src-alpha over dst.
  normal: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
  // Additive (glows: corona, lasers, atmosphere). Order-independent.
  additive: {
    color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  },
};

/** Maps a `side` option to a GPU cullMode. */
export function cullModeFor(side) {
  if (side === 'double') return 'none';
  if (side === 'back') return 'front'; // render back faces => cull front
  return 'back';
}

/**
 * Builds the fragmentTargets entry for a target format + blend option.
 * blend: 'normal' | 'additive' | null (opaque) | a raw GPUBlendState.
 */
export function fragmentTarget(format, blend) {
  if (!blend) return { format };
  const state = typeof blend === 'string' ? BLEND_PRESETS[blend] : blend;
  if (!state) throw new Error(`unknown blend preset "${blend}"`);
  return { format, blend: state };
}

/** Packs a Fog uniform (32 bytes) from { color:[r,g,b], near, far, enabled }. */
export function packFog({ color = [0, 0, 0], near = 0, far = 1, enabled = false } = {}) {
  return new Float32Array([
    color[0], color[1], color[2], enabled ? 1 : 0,
    near, far, 0, 0,
  ]);
}
