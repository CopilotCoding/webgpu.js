export const cullingShader = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};

struct ObjectBounds {
  localMin: vec3f,
  localMax: vec3f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> objectBounds: array<ObjectBounds>;
@group(0) @binding(3) var<storage, read_write> visibility: array<u32>;
@group(0) @binding(4) var hiZTexture: texture_2d<f32>;
@group(0) @binding(5) var<uniform> objectCount: u32;

const VISIBLE_FRUSTUM: u32 = 1u;
const VISIBLE_OCCLUSION: u32 = 2u;

fn aabbCorners(localMin: vec3f, localMax: vec3f, worldMatrix: mat4x4f) -> array<vec3f, 8> {
  var corners: array<vec3f, 8>;
  for (var i = 0u; i < 8u; i++) {
    let x = select(localMin.x, localMax.x, (i & 1u) != 0u);
    let y = select(localMin.y, localMax.y, (i & 2u) != 0u);
    let z = select(localMin.z, localMax.z, (i & 4u) != 0u);
    let world = worldMatrix * vec4f(x, y, z, 1.0);
    corners[i] = world.xyz;
  }
  return corners;
}

// Tests a world-space AABB (given as its 8 corners) against the camera's
// 6 frustum planes. The AABB is outside if all 8 corners are on the
// negative side of any single plane.
fn testFrustum(corners: array<vec3f, 8>) -> bool {
  for (var p = 0u; p < 6u; p++) {
    let plane = camera.frustumPlanes[p];
    var allOutside = true;
    for (var i = 0u; i < 8u; i++) {
      if (dot(plane.xyz, corners[i]) + plane.w >= 0.0) {
        allOutside = false;
        break;
      }
    }
    if (allOutside) {
      return false;
    }
  }
  return true;
}

// Tests a world-space AABB against the Hi-Z buffer from the previous
// frame: projects all 8 corners to NDC, derives a screen-space bounding
// rect and the nearest (minimum) depth, picks the Hi-Z mip whose texel
// size covers that rect, and compares against the stored max depth.
fn testOcclusion(corners: array<vec3f, 8>) -> bool {
  var ndcMin = vec3f(1.0, 1.0, 1.0);
  var ndcMax = vec3f(-1.0, -1.0, -1.0);

  for (var i = 0u; i < 8u; i++) {
    let clip = camera.projectionMatrix * camera.viewMatrix * vec4f(corners[i], 1.0);
    if (clip.w <= 0.0) {
      // Behind the camera — can't be occluded by a depth comparison.
      return true;
    }
    let ndc = clip.xyz / clip.w;
    ndcMin = min(ndcMin, ndc);
    ndcMax = max(ndcMax, ndc);
  }

  // Convert the NDC AABB rect to Hi-Z texel space. NDC +Y points UP, but the
  // Hi-Z texture (copied straight from the depth attachment) has row 0 at the
  // TOP, so the Y axis must be flipped — otherwise an object low on screen would
  // sample Hi-Z texels from the top of the screen and read the wrong occluder.
  let uvA = vec2f(ndcMin.x * 0.5 + 0.5, 0.5 - ndcMin.y * 0.5);
  let uvB = vec2f(ndcMax.x * 0.5 + 0.5, 0.5 - ndcMax.y * 0.5);
  let screenMin = min(uvA, uvB) * camera.viewport.zw;
  let screenMax = max(uvA, uvB) * camera.viewport.zw;
  let sizePixels = max(screenMax - screenMin, vec2f(1.0, 1.0));

  // Cap the mip level at a small fixed value (texel size <= 4px) rather
  // than scaling it with the object's screen size. A mip level chosen from
  // the object's footprint can produce texels much larger than the
  // occluder's silhouette detail (e.g. a wall's edge) — at that coarse
  // resolution a texel can read as "fully occluded" even though a thin
  // sliver of the object is actually visible past the edge.
  let hiZSize = vec2f(textureDimensions(hiZTexture, 0));
  let mipLevel = i32(clamp(ceil(log2(max(sizePixels.x, sizePixels.y) * 0.5)), 0.0, min(2.0, f32(textureNumLevels(hiZTexture) - 1))));

  let mipSize = vec2f(textureDimensions(hiZTexture, mipLevel));
  let coordA = vec2i(clamp(screenMin / hiZSize * mipSize, vec2f(0.0), mipSize - vec2f(1.0)));
  let coordB = vec2i(clamp(screenMax / hiZSize * mipSize, vec2f(0.0), mipSize - vec2f(1.0)));
  let minCoord = min(coordA, coordB);
  let maxCoord = max(coordA, coordB);

  // Nearest depth of the AABB (closest to camera = smallest NDC z, since
  // WebGPU's depth range is [0, 1] with 0 at the near plane).
  let nearestDepth = ndcMin.z;

  // Each Hi-Z texel stores the FARTHEST (max) depth of geometry drawn into it.
  // The object is hidden only if its nearest point is behind the occluder in
  // EVERY covered texel — so take the MIN of the per-texel max-depths as the
  // shallowest occluder across the footprint. If even one covered texel is
  // shallow (e.g. the object pokes past a wall's edge into open background at
  // depth 1.0), the object stays visible. Taking max here instead would make
  // the test always pass (a texel containing the object's own prior-frame depth
  // makes max >= the object's depth), so occlusion would never fire.
  var occluderDepth = 1.0;
  for (var y = minCoord.y; y <= maxCoord.y; y++) {
    for (var x = minCoord.x; x <= maxCoord.x; x++) {
      occluderDepth = min(occluderDepth, textureLoad(hiZTexture, vec2i(x, y), mipLevel).r);
    }
  }

  // Visible if the object's nearest point is in front of (or at) the shallowest
  // occluder depth, with a tiny bias for depth-precision aliasing. The bias
  // must stay well below the depth separation between distinct occluders/occludees
  // — with a perspective projection most depth precision sits near the camera, so
  // far-apart objects can differ by only a few thousandths in NDC z. A bias as
  // large as 0.01 there would swamp the separation and disable occlusion entirely.
  const OCCLUSION_BIAS: f32 = 0.00005;
  return nearestDepth <= occluderDepth + OCCLUSION_BIAS;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= objectCount) {
    return;
  }

  let bounds = objectBounds[id.x];
  let worldMatrix = worldMatrices[id.x];
  let corners = aabbCorners(bounds.localMin, bounds.localMax, worldMatrix);

  var result = 0u;
  if (testFrustum(corners)) {
    result |= VISIBLE_FRUSTUM;
    if (testOcclusion(corners)) {
      result |= VISIBLE_OCCLUSION;
    }
  }

  visibility[id.x] = result;
}
`;
