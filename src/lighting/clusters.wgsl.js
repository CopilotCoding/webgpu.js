// Cluster grid: the view frustum is subdivided into clusterCountX *
// clusterCountY tiles in screen space, each split into clusterCountZ slices
// along view-space depth. Depth slices are exponential (each slice is
// `far/near` times deeper than the last) so that clusters stay roughly
// cube-shaped near the camera, where light density matters most.
export const clusterBoundsShader = /* wgsl */ `
struct ClusterGridInfo {
  clusterCount: vec3u,
  _pad0: u32,
  screenSize: vec2f,
  zNear: f32,
  zFar: f32,
  inverseProjection: mat4x4f,
};

struct ClusterBounds {
  minPoint: vec3f,
  _pad0: f32,
  maxPoint: vec3f,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> grid: ClusterGridInfo;
@group(0) @binding(1) var<storage, read_write> clusters: array<ClusterBounds>;

// Unprojects a screen-space point (pixels) at a given view-space depth (z,
// negative in front of the camera) back to a view-space position by
// reversing the projection matrix.
fn screenToView(screenPos: vec2f, viewZ: f32) -> vec3f {
  let ndc = vec2f(
    (screenPos.x / grid.screenSize.x) * 2.0 - 1.0,
    1.0 - (screenPos.y / grid.screenSize.y) * 2.0,
  );

  // Recover the view-space ray direction through this screen pixel at z=-1,
  // then scale it so its z component matches viewZ.
  let clip = vec4f(ndc, -1.0, 1.0);
  let viewPos = grid.inverseProjection * clip;
  let view = viewPos.xyz / viewPos.w;
  return view * (viewZ / view.z);
}

@compute @workgroup_size(4, 4, 4)
fn buildClusters(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= grid.clusterCount)) {
    return;
  }

  let tileSize = grid.screenSize / vec2f(grid.clusterCount.xy);

  let minScreen = vec2f(id.xy) * tileSize;
  let maxScreen = vec2f(id.xy + vec2u(1u, 1u)) * tileSize;

  // Exponential depth slicing: slice k spans
  // [near * (far/near)^(k/sliceCount), near * (far/near)^((k+1)/sliceCount)].
  let sliceCount = f32(grid.clusterCount.z);
  let nearK = grid.zNear * pow(grid.zFar / grid.zNear, f32(id.z) / sliceCount);
  let farK = grid.zNear * pow(grid.zFar / grid.zNear, f32(id.z + 1u) / sliceCount);

  // View space uses a right-handed convention where the camera looks down
  // -z, so depths in front of the camera are negative.
  let nearViewZ = -nearK;
  let farViewZ = -farK;

  let minNear = screenToView(minScreen, nearViewZ);
  let maxNear = screenToView(maxScreen, nearViewZ);
  let minFar = screenToView(minScreen, farViewZ);
  let maxFar = screenToView(maxScreen, farViewZ);

  let minPoint = min(min(minNear, maxNear), min(minFar, maxFar));
  let maxPoint = max(max(minNear, maxNear), max(minFar, maxFar));

  let index = id.x + id.y * grid.clusterCount.x + id.z * grid.clusterCount.x * grid.clusterCount.y;
  clusters[index].minPoint = minPoint;
  clusters[index].maxPoint = maxPoint;
}
`;

// Assigns point lights to clusters: for each cluster, tests every light's
// view-space sphere against the cluster's view-space AABB and appends
// matching light indices to a shared list. Each cluster gets a (offset,
// count) pair into that list — built with a two-pass count-then-compact
// scheme so the list has no gaps.
export const lightCullingShader = /* wgsl */ `
struct ClusterBounds {
  minPoint: vec3f,
  _pad0: f32,
  maxPoint: vec3f,
  _pad1: f32,
};

struct PointLight {
  position: vec3f,
  radius: f32,
  color: vec3f,
  intensity: f32,
};

struct ClusterLightRange {
  offset: u32,
  count: atomic<u32>,
};

@group(0) @binding(0) var<uniform> viewMatrix: mat4x4f;
@group(0) @binding(1) var<storage, read> clusters: array<ClusterBounds>;
@group(0) @binding(2) var<storage, read> lights: array<PointLight>;
@group(0) @binding(3) var<storage, read_write> clusterRanges: array<ClusterLightRange>;
@group(0) @binding(4) var<storage, read_write> lightIndices: array<u32>;
@group(0) @binding(5) var<uniform> counts: vec2u; // x: clusterCount, y: lightCount

const MAX_LIGHTS_PER_CLUSTER: u32 = 256u;

// Squared distance from a point to an AABB (0 if the point is inside).
fn sqDistPointAABB(point: vec3f, minPoint: vec3f, maxPoint: vec3f) -> f32 {
  let d = max(max(minPoint - point, vec3f(0.0)), point - maxPoint);
  return dot(d, d);
}

@compute @workgroup_size(64)
fn resetClusterRanges(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= counts.x) {
    return;
  }
  clusterRanges[id.x].offset = id.x * MAX_LIGHTS_PER_CLUSTER;
  atomicStore(&clusterRanges[id.x].count, 0u);
}

@compute @workgroup_size(64)
fn cullLights(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= counts.x) {
    return;
  }

  let bounds = clusters[id.x];
  let baseOffset = id.x * MAX_LIGHTS_PER_CLUSTER;

  for (var i = 0u; i < counts.y; i++) {
    let light = lights[i];
    let viewPos = (viewMatrix * vec4f(light.position, 1.0)).xyz;

    if (sqDistPointAABB(viewPos, bounds.minPoint, bounds.maxPoint) <= light.radius * light.radius) {
      let slot = atomicAdd(&clusterRanges[id.x].count, 1u);
      if (slot < MAX_LIGHTS_PER_CLUSTER) {
        lightIndices[baseOffset + slot] = i;
      }
    }
  }
}
`;
