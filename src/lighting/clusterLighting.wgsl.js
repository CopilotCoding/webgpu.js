// Reusable fragment-shader snippet for clustered light evaluation. A
// material's shader source embeds this block (via string concatenation) to
// get the `PointLight`/`ClusterLightRange` struct defs, the cluster-index
// and lighting helper functions, and the bindings they read.
//
// Expected bind group layout (group/binding numbers are chosen by the
// embedding shader, but must match the JS-side bind group construction in
// the example):
//   var<uniform> clusterGridInfo: ClusterGridInfo;
//   var<storage, read> clusterRanges: array<ClusterLightRange>;
//   var<storage, read> lightIndices: array<u32>;
//   var<storage, read> lights: array<PointLight>;
export const clusterLightingStructs = /* wgsl */ `
struct ClusterGridInfo {
  clusterCount: vec3u,
  _pad0: u32,
  screenSize: vec2f,
  zNear: f32,
  zFar: f32,
  inverseProjection: mat4x4f,
};

struct ClusterLightRange {
  offset: u32,
  count: u32,
};

struct PointLight {
  position: vec3f,
  radius: f32,
  color: vec3f,
  intensity: f32,
};

const MAX_LIGHTS_PER_CLUSTER: u32 = 256u;
`;

// Computes the cluster index for a fragment given its window-space position
// (fragCoord.xy in pixels, fragCoord.z/w view-space depth) and the cluster
// grid info. Mirrors the exponential depth slicing used to build the
// cluster AABBs in clusters.wgsl.js.
export const clusterIndexFunction = /* wgsl */ `
fn clusterIndex(fragCoord: vec4f, grid: ClusterGridInfo, viewZ: f32) -> u32 {
  let tileSize = grid.screenSize / vec2f(grid.clusterCount.xy);
  let tile = vec2u(clamp(fragCoord.xy / tileSize, vec2f(0.0), vec2f(grid.clusterCount.xy) - vec2f(1.0)));

  let depth = clamp(-viewZ, grid.zNear, grid.zFar);
  let sliceCount = f32(grid.clusterCount.z);
  let slice = u32(clamp(
    floor(log2(depth / grid.zNear) / log2(grid.zFar / grid.zNear) * sliceCount),
    0.0,
    sliceCount - 1.0,
  ));

  return tile.x + tile.y * grid.clusterCount.x + slice * grid.clusterCount.x * grid.clusterCount.y;
}
`;

// Accumulates simple Lambertian + inverse-square-falloff lighting from every
// light assigned to the fragment's cluster.
//
// naga (the WGSL implementation used by current browsers) does not support
// storage-address-space pointers as function parameters, so this function
// reads directly from the module-scope bindings named below — the embedding
// shader must declare these exact names:
//   var<storage, read> clusterRanges: array<ClusterLightRange>;
//   var<storage, read> lightIndices: array<u32>;
//   var<storage, read> lights: array<PointLight>;
export const accumulateClusterLightingFunction = /* wgsl */ `
fn accumulateClusterLighting(
  worldPos: vec3f,
  normal: vec3f,
  fragCoord: vec4f,
  grid: ClusterGridInfo,
  viewZ: f32,
) -> vec3f {
  let cluster = clusterIndex(fragCoord, grid, viewZ);
  let range = clusterRanges[cluster];
  let count = min(range.count, MAX_LIGHTS_PER_CLUSTER);

  var result = vec3f(0.0);
  for (var i = 0u; i < count; i++) {
    let light = lights[lightIndices[range.offset + i]];
    let toLight = light.position - worldPos;
    let dist = length(toLight);
    if (dist >= light.radius) {
      continue;
    }

    let attenuation = 1.0 / max(dist * dist, 0.01);
    let falloff = 1.0 - dist / light.radius;
    let ndotl = max(dot(normal, normalize(toLight)), 0.0);
    result += light.color * light.intensity * attenuation * falloff * falloff * ndotl;
  }

  return result;
}
`;
