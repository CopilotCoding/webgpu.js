// WGSL for the SceneRenderer's GPU-driven batches. Each batch draws many
// objects of one pipeline configuration via MultiDrawSystem: the vertex shader
// recovers its object id from the draw-slot group (slotToObject[slotIndex]),
// then indexes per-object world matrices + material params from storage.
//
// group(0): camera (binding 0), lights (1), fog (2)
// group(1): worldMatrices (0), materialParams (1)   [per batch]
// group(2): slotToObject (0), slotIndex dynamic uniform (1)  [MultiDrawSystem]

const common = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
struct Light { posDir: vec4f, color: vec4f, };  // posDir.w: 0=dir,1=point
struct Lights {
  ambient: vec4f,                  // rgb + count
  lights: array<Light, 4>,
};
struct Fog { colorEnabled: vec4f, range: vec4f, };  // rgb+enabled ; near,far,_,_
struct MaterialParams { color: vec4f, };           // rgb + opacity

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lights;
@group(0) @binding(2) var<uniform> fog: Fog;
@group(1) @binding(0) var<storage, read> worldMatrices: array<mat4x4f>;
@group(1) @binding(1) var<storage, read> materials: array<MaterialParams>;
@group(2) @binding(0) var<storage, read> slotToObject: array<u32>;
@group(2) @binding(1) var<uniform> slotIndex: u32;

fn applyFog(rgb: vec3f, viewDepth: f32) -> vec3f {
  if (fog.colorEnabled.w < 0.5) { return rgb; }
  let f = clamp((viewDepth - fog.range.x) / max(fog.range.y - fog.range.x, 0.0001), 0.0, 1.0);
  return mix(rgb, fog.colorEnabled.rgb, f);
}
`;

// Lit (Lambert) batch.
export const lambertShader = /* wgsl */ `
${common}
struct VOut {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewDepth: f32,
  @location(3) @interpolate(flat) obj: u32,
};
@vertex
fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f) -> VOut {
  let obj = slotToObject[slotIndex];
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
  for (var k = 0u; k < count; k = k + 1u) {
    let L = lighting.lights[k];
    var dir: vec3f; var atten = 1.0;
    if (L.posDir.w < 0.5) { dir = normalize(L.posDir.xyz); }
    else {
      let d = L.posDir.xyz - i.worldPos; let dist = length(d);
      dir = d / max(dist, 0.0001);
      atten = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
    }
    diffuse = diffuse + L.color.rgb * (L.color.w * max(dot(nrm, dir), 0.0) * atten);
  }
  let mat = materials[i.obj];
  return vec4f(applyFog(mat.color.rgb * diffuse, i.viewDepth), mat.color.a);
}
`;

// Unlit (Basic) batch — solid/opacity, optional fog. Blend state is set on the
// pipeline (normal vs additive), not in the shader.
export const basicShader = /* wgsl */ `
${common}
struct VOut {
  @builtin(position) position: vec4f,
  @location(0) viewDepth: f32,
  @location(1) @interpolate(flat) obj: u32,
};
@vertex
fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f) -> VOut {
  let obj = slotToObject[slotIndex];
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
  return vec4f(applyFog(mat.color.rgb, i.viewDepth), mat.color.a);
}
`;

// Terrain shader — a direct WGSL port of the game's marching-cubes terrain
// GLSL (per-vertex color + skyAccess; sun gated by sky access and a hemisphere
// day/night term; lantern with quadratic falloff). Identity model matrix:
// terrain verts are already world-space. Custom uniforms in one block.
//
// group(0) binding 0: camera. group(1) binding 0: TerrainUniforms.
export const terrainShader = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
struct TerrainU {
  sunPosition: vec4f,      // xyz + sunIntensity in w
  lanternPosition: vec4f,  // xyz + lanternIntensity in w
  params: vec4f,           // lanternRange, ambientIntensity, fogNear, fogFar
  fogColor: vec4f,         // rgb + fogEnabled
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> u: TerrainU;

struct VOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) worldPos: vec3f,
  @location(3) skyAccess: f32,
  @location(4) viewDepth: f32,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) skyAccess: f32,
) -> VOut {
  var o: VOut;
  o.color = color;
  o.skyAccess = skyAccess;
  o.normal = normalize(normal);
  o.worldPos = position;
  let view = camera.viewMatrix * vec4f(position, 1.0);
  o.position = camera.projectionMatrix * view;
  o.viewDepth = -view.z;
  return o;
}

@fragment
fn fragmentMain(i: VOut) -> @location(0) vec4f {
  let n = normalize(i.normal);
  let sunIntensity = u.sunPosition.w;
  let lanternIntensity = u.lanternPosition.w;
  let lanternRange = u.params.x;
  let ambientIntensity = u.params.y;

  let sunDir = normalize(u.sunPosition.xyz - i.worldPos);
  let sunDot = max(0.0, dot(n, sunDir));
  let hemisphereDot = pow(max(0.0, dot(normalize(i.worldPos), sunDir)), 0.35);
  let sunContribI = sunDot * hemisphereDot * sunIntensity * i.skyAccess;
  let sunContrib = sunContribI * vec3f(1.0, 0.97, 0.88);

  let lDir = u.lanternPosition.xyz - i.worldPos;
  let lDist = length(lDir);
  let lDot = max(0.0, dot(n, normalize(lDir)));
  let lScaled = lDist / lanternRange;
  let lAtten = lanternIntensity / (1.0 + lScaled * lScaled * 0.18);
  let lFill = lAtten * 0.45;
  let lanternContrib = (lDot * lAtten + lFill) * vec3f(1.0, 0.80, 0.47);

  let light = vec3f(ambientIntensity) + sunContrib + lanternContrib;
  var rgb = i.color * light;

  if (u.fogColor.w > 0.5) {
    let f = clamp((i.viewDepth - u.params.z) / max(u.params.w - u.params.z, 0.0001), 0.0, 1.0);
    rgb = mix(rgb, u.fogColor.rgb, f);
  }
  return vec4f(rgb, 1.0);
}
`;

// Points (star field): GPU-expanded camera-facing quads (no gl_PointSize).
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
