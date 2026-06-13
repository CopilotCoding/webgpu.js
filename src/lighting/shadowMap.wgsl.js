// Reusable fragment-shader snippet for directional-light shadow mapping. A
// material's shader source embeds this block (via string concatenation) to
// get the `ShadowMap` struct def and the PCF sampling function.
//
// Expected bindings (group/binding numbers are chosen by the embedding
// shader, but must match the JS-side bind group construction):
//   var<uniform> shadowMap: ShadowMap;
//   var shadowDepth: texture_depth_2d;
//   var shadowSampler: sampler_comparison;
export const shadowMapStruct = /* wgsl */ `
struct ShadowMap {
  viewProjectionMatrix: mat4x4f,
  lightDirection: vec3f,
  _pad0: f32,
};
`;

// Projects the world position into the shadow map, and returns a visibility
// factor in [0, 1] via 3x3 PCF. Returns 1.0 (fully lit) for fragments
// outside the shadow map's coverage.
export const sampleShadowFunction = /* wgsl */ `
fn sampleShadow(worldPos: vec3f, normal: vec3f) -> f32 {
  let clip = shadowMap.viewProjectionMatrix * vec4f(worldPos, 1.0);
  let ndc = clip.xyz / clip.w;

  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }

  // Slope-scaled bias: surfaces at a grazing angle to the light need a
  // larger bias to avoid acne, since the same texel covers a larger range
  // of depth along the surface. Surfaces facing the light directly need
  // very little bias.
  let sunDir = normalize(-shadowMap.lightDirection);
  let slope = clamp(1.0 - dot(normal, sunDir), 0.0, 1.0);
  let depthBias = 0.0015 + slope * 0.03;
  let compareDepth = ndc.z - depthBias;

  let uv = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
  let texSize = vec2f(textureDimensions(shadowDepth, 0));
  let texelSize = 1.0 / texSize;

  var visibility = 0.0;
  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      let offset = vec2f(f32(dx), f32(dy)) * texelSize;
      // textureSampleCompareLevel (not textureSampleCompare): the plain
      // variant computes implicit derivatives and is illegal after the
      // non-uniform early return above — WGSL's uniformity analysis rejects
      // it. The Level variant samples mip 0 with no derivatives, which is
      // exactly right for a shadow map.
      visibility += textureSampleCompareLevel(shadowDepth, shadowSampler, uv + offset, compareDepth);
    }
  }

  return visibility / 9.0;
}
`;
