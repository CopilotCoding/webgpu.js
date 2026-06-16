// WGSL for the SceneRenderer's GPU-driven batches. The vertex shader recovers
// its object id, then indexes per-object world matrices + material params.
//
// Two object-id paths (the renderer picks one at pipeline-build time):
//  - MULTI-DRAW: one multiDrawIndexedIndirect per batch; the cull pass writes
//    firstInstance = objectId, recovered via @builtin(instance_index). No
//    group(2).
//  - LOOP fallback (no multi-draw feature): one drawIndexedIndirect per slot
//    with a per-draw dynamic uniform offset; objectId = slotToObject[slotIndex]
//    at group(2).
//
// group(0): camera (0), lights (1), fog (2)
// group(1): worldMatrices (0), materialParams (1)
// group(2) (loop mode only): slotToObject (0), slotIndex dynamic uniform (1)

const commonHead = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
struct Light { posDir: vec4f, color: vec4f, };  // posDir.w: 0=dir,1=point,2=point-no-falloff
struct Lights {
  ambient: vec4f,                  // rgb + count
  lights: array<Light, 8>,  // MAX_LIGHTS — keep in sync with SceneRenderer
};
struct Fog { colorEnabled: vec4f, range: vec4f, };  // rgb+enabled ; near,far,_,_
struct MaterialParams { color: vec4f, };           // rgb + opacity

// Shadow params: directional-sun light-space matrix + dir + enabled flag.
struct ShadowParams { lightViewProj: mat4x4f, lightDirEnabled: vec4f, }; // xyz dir, w enabled

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lights;
@group(0) @binding(2) var<uniform> fog: Fog;
@group(0) @binding(3) var<uniform> shadow: ShadowParams;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<storage, read> worldMatrices: array<mat4x4f>;
@group(1) @binding(1) var<storage, read> materials: array<MaterialParams>;

fn applyFog(rgb: vec3f, viewDepth: f32) -> vec3f {
  if (fog.colorEnabled.w < 0.5) { return rgb; }
  let f = clamp((viewDepth - fog.range.x) / max(fog.range.y - fog.range.x, 0.0001), 0.0, 1.0);
  return mix(rgb, fog.colorEnabled.rgb, f);
}

// Directional-sun shadow factor in [0,1] (1 = fully lit). 3x3 PCF. Returns 1.0
// (no shadowing) when shadows are disabled. worldNormal + lightDir are used for
// a slope-scaled depth bias to fight acne on grazing surfaces.
fn sunShadow(worldPos: vec3f, worldNormal: vec3f) -> f32 {
  if (shadow.lightDirEnabled.w < 0.5) { return 1.0; }
  let clip = shadow.lightViewProj * vec4f(worldPos, 1.0);
  if (clip.w <= 0.0) { return 1.0; }
  let ndc = clip.xyz / clip.w;
  // Outside the shadow frustum -> treat as lit.
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let ndl = max(dot(normalize(worldNormal), normalize(-shadow.lightDirEnabled.xyz)), 0.0);
  let bias = clamp(0.0015 * (1.0 - ndl), 0.0003, 0.003);
  let refDepth = ndc.z - bias;
  let dim = vec2f(textureDimensions(shadowMap, 0));
  let texel = 1.0 / dim;
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum = sum + textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2f(f32(x), f32(y)) * texel, refDepth);
    }
  }
  return sum / 9.0;
}
`;

// Depth-only shadow pass: renders casters from the sun's POV into the shadow
// map. group(0) = shadowParams (for lightViewProj); group(1) = worldMatrices;
// object id via the same multi-draw/loop choice as the forward shaders.
export function shadowDepthShader(multiDraw) {
  const { extraParam, objExpr, bindings } = objIdParts(multiDraw);
  return /* wgsl */ `
struct ShadowParams { lightViewProj: mat4x4f, lightDirEnabled: vec4f, };
@group(0) @binding(0) var<uniform> shadow: ShadowParams;
@group(1) @binding(0) var<storage, read> worldMatrices: array<mat4x4f>;
${bindings}
@vertex
fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f${extraParam ? ', ' + extraParam : ''}) -> @builtin(position) vec4f {
  let obj = ${objExpr};
  return shadow.lightViewProj * worldMatrices[obj] * vec4f(p, 1.0);
}
`;
}

// Loop-mode-only bindings (group 2).
const loopBindings = /* wgsl */ `
@group(2) @binding(0) var<storage, read> slotToObject: array<u32>;
@group(2) @binding(1) var<uniform> slotIndex: u32;
`;

// Returns the WGSL fragment that yields the object id, plus the extra vertex
// param list, for the chosen mode.
function objIdParts(multiDraw) {
  if (multiDraw) {
    return { extraParam: '@builtin(instance_index) instanceIndex: u32', objExpr: 'instanceIndex', bindings: '' };
  }
  return { extraParam: '', objExpr: 'slotToObject[slotIndex]', bindings: loopBindings };
}

// Wraps a color expression in fog, or returns it raw when fog is disabled for
// this material (e.g. the sun, which must not fade into the fog/background).
function fogWrap(fogEnabled, colorExpr, depthExpr) {
  return fogEnabled ? `applyFog(${colorExpr}, ${depthExpr})` : colorExpr;
}

// Lit (Lambert) batch.
export function lambertShader(multiDraw, fogEnabled = true) {
  const { extraParam, objExpr, bindings } = objIdParts(multiDraw);
  return /* wgsl */ `
${commonHead}
${bindings}
struct VOut {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewDepth: f32,
  @location(3) @interpolate(flat) obj: u32,
};
@vertex
fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f${extraParam ? ', ' + extraParam : ''}) -> VOut {
  let obj = ${objExpr};
  let m = worldMatrices[obj];
  let world = m * vec4f(p, 1.0);
  let view = camera.viewMatrix * world;
  var o: VOut;
  o.position = camera.projectionMatrix * view;
  o.worldPos = world.xyz;
  o.normal = normalize((m * vec4f(n, 0.0)).xyz);
  o.viewDepth = -view.z;
  o.obj = obj;
  return o;
}
@fragment
fn fragmentMain(i: VOut) -> @location(0) vec4f {
  let nrm = normalize(i.normal);
  var diffuse = lighting.ambient.rgb;
  let count = u32(lighting.ambient.w);
  let shadowFactor = sunShadow(i.worldPos, nrm);
  for (var k = 0u; k < count; k = k + 1u) {
    let L = lighting.lights[k];
    var dir: vec3f; var atten = 1.0;
    if (L.posDir.w < 0.5) {
      // Directional light — shadowed by the sun shadow map.
      dir = normalize(L.posDir.xyz);
      atten = shadowFactor;
    } else {
      // Point light. posDir.w == 2 flags a no-falloff point light (decay 0);
      // == 1 uses quadratic distance attenuation. Point lights are unshadowed.
      let d = L.posDir.xyz - i.worldPos; let dist = length(d);
      dir = d / max(dist, 0.0001);
      if (L.posDir.w < 1.5) { atten = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist); }
    }
    diffuse = diffuse + L.color.rgb * (L.color.w * max(dot(nrm, dir), 0.0) * atten);
  }
  let mat = materials[i.obj];
  return vec4f(${fogWrap(fogEnabled, 'mat.color.rgb * diffuse', 'i.viewDepth')}, mat.color.a);
}
`;
}

// Unlit (Basic) batch — solid/opacity, optional fog. Blend state is set on the
// pipeline (normal vs additive), not in the shader.
export function basicShader(multiDraw, fogEnabled = true) {
  const { extraParam, objExpr, bindings } = objIdParts(multiDraw);
  return /* wgsl */ `
${commonHead}
${bindings}
struct VOut {
  @builtin(position) position: vec4f,
  @location(0) viewDepth: f32,
  @location(1) @interpolate(flat) obj: u32,
};
@vertex
fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f${extraParam ? ', ' + extraParam : ''}) -> VOut {
  let obj = ${objExpr};
  let m = worldMatrices[obj];
  let world = m * vec4f(p, 1.0);
  let view = camera.viewMatrix * world;
  var o: VOut;
  o.position = camera.projectionMatrix * view;
  o.viewDepth = -view.z;
  o.obj = obj;
  return o;
}
@fragment
fn fragmentMain(i: VOut) -> @location(0) vec4f {
  let mat = materials[i.obj];
  return vec4f(${fogWrap(fogEnabled, 'mat.color.rgb', 'i.viewDepth')}, mat.color.a);
}
`;
}

// Points: GPU-expanded camera-facing quads (no gl_PointSize).
// group(0) binding 0 camera; group(1) binding 0 positions storage, 1 params.
export const pointsShader = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
struct Params { color: vec4f, half: vec4f, };  // rgb,_ ; ndcHalfX, ndcHalfY,_,_
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<uniform> params: Params;

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(1.0,1.0),
  vec2f(-1.0,-1.0), vec2f(1.0,1.0), vec2f(-1.0,1.0),
);
struct VOut { @builtin(position) position: vec4f, };
@vertex
fn vertexMain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let b = ii * 3u;
  let world = vec3f(positions[b], positions[b+1u], positions[b+2u]);
  var clip = camera.projectionMatrix * camera.viewMatrix * vec4f(world, 1.0);
  let c = CORNERS[vi];
  clip.x = clip.x + c.x * params.half.x * clip.w;
  clip.y = clip.y + c.y * params.half.y * clip.w;
  var o: VOut; o.position = clip; return o;
}
@fragment
fn fragmentMain() -> @location(0) vec4f { return vec4f(params.color.rgb, 1.0); }
`;
