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

// cullParams (u32): low 31 bits = object count; top bit (0x80000000) =
// occlusionEnabled. Occlusion culling against a single-frame-lagged Hi-Z can
// wrongly drop objects at depth boundaries (flicker), so it is OFF by default —
// only enable it where the draw path tolerates a frame of latency or implements
// two-phase culling. Packed into one u32 to keep this a 4-byte uniform binding.
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> objectBounds: array<ObjectBounds>;
@group(0) @binding(3) var<storage, read_write> visibility: array<u32>;
@group(0) @binding(4) var hiZTexture: texture_2d<f32>;
@group(0) @binding(5) var<uniform> cullParams: u32;

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

  // Pick the Hi-Z mip where the object's footprint spans ~2 texels per axis, the
  // standard Hi-Z choice: at this resolution the covered-texel rect is at most
  // 2x2, so the loop below reads ~4 texels (cheap and bounded). A COARSER mip
  // (e.g. a fixed 4px cap) is what made partially-occluded objects wrongly cull:
  // when a thin visible sliver pokes past the wall edge, a too-coarse texel grid
  // can miss the background texel entirely, so the reduction never sees the
  // shallow (visible) depth and the object reads as fully hidden. Choosing the
  // mip from the footprint keeps the visible sliver in its own texel.
  let hiZSize = vec2f(textureDimensions(hiZTexture, 0));
  let maxLevel = f32(textureNumLevels(hiZTexture) - 1);
  let mipLevel = i32(clamp(ceil(log2(max(sizePixels.x, sizePixels.y) * 0.5)), 0.0, maxLevel));

  let mipSize = vec2f(textureDimensions(hiZTexture, mipLevel));
  let coordA = vec2i(clamp(floor(screenMin / hiZSize * mipSize), vec2f(0.0), mipSize - vec2f(1.0)));
  let coordB = vec2i(clamp(floor(screenMax / hiZSize * mipSize), vec2f(0.0), mipSize - vec2f(1.0)));
  let minCoord = min(coordA, coordB);
  let maxCoord = max(coordA, coordB);

  // Nearest depth of the AABB (closest to camera = smallest NDC z, since
  // WebGPU's depth range is [0, 1] with 0 at the near plane).
  let nearestDepth = ndcMin.z;

  // Each Hi-Z texel stores the FARTHEST (max) depth of geometry drawn into it.
  // The object is FULLY hidden only if its nearest point is behind the farthest
  // occluder in EVERY covered texel — i.e. nearest > T_k for all k, which is
  // nearest > max_k(T_k). So reduce with MAX and keep the object visible when
  // nearest <= that max. This is what makes PARTIAL occlusion correct: a texel
  // covering the visible sliver past a wall edge holds background depth (~1.0),
  // so the max is ~1.0 >= nearest and the object stays visible. (Reducing with
  // min instead takes the shallowest occluder — the wall — so any object whose
  // footprint overlaps the wall at all reads as hidden, wrongly culling slivers.)
  var farthestOccluder = 0.0;
  for (var y = minCoord.y; y <= maxCoord.y; y++) {
    for (var x = minCoord.x; x <= maxCoord.x; x++) {
      farthestOccluder = max(farthestOccluder, textureLoad(hiZTexture, vec2i(x, y), mipLevel).r);
    }
  }

  // Visible if the object's nearest point is in front of (or at) the farthest
  // occluder across its footprint, with a tiny bias for depth-precision aliasing.
  // The bias must stay well below the depth separation between distinct
  // occluders/occludees — under perspective most depth precision sits near the
  // camera, so far-apart objects can differ by only a few thousandths in NDC z;
  // a bias as large as 0.01 there would swamp the separation.
  const OCCLUSION_BIAS: f32 = 0.00005;
  return nearestDepth <= farthestOccluder + OCCLUSION_BIAS;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) id: vec3u) {
  let objectCount = cullParams & 0x7fffffffu;
  let occlusionEnabled = (cullParams & 0x80000000u) != 0u;
  if (id.x >= objectCount) {
    return;
  }

  let bounds = objectBounds[id.x];
  let worldMatrix = worldMatrices[id.x];
  let corners = aabbCorners(bounds.localMin, bounds.localMax, worldMatrix);

  var result = 0u;
  if (testFrustum(corners)) {
    result |= VISIBLE_FRUSTUM;
    // VISIBLE_OCCLUSION gates drawing in indirectDraw.wgsl. When occlusion is
    // disabled, set it for every frustum-visible object so nothing is dropped;
    // otherwise set it only when the Hi-Z test says the object isn't hidden.
    if (!occlusionEnabled || testOcclusion(corners)) {
      result |= VISIBLE_OCCLUSION;
    }
  }

  visibility[id.x] = result;
}
`;
