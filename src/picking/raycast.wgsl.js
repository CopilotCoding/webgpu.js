// Ray/object intersection in compute. Each invocation tests one object's
// world-space AABB (derived from its local bounds and world matrix — the
// same data the culling pass uses) against the pick ray with a slab test.
//
// The nearest hit wins via atomicMin on a single packed u32:
//   high 20 bits — the hit distance t's f32 bit pattern (mantissa truncated)
//   low 12 bits  — the object index
// Positive f32s compare correctly as unsigned integers, so atomicMin keeps
// the smallest t. Truncating the mantissa costs a little distance precision
// but never changes which object is nearest beyond that precision.
// Supports up to 4096 objects per picker; NO_HIT is all ones.
export const raycastShader = /* wgsl */ `
struct Ray {
  origin: vec3f,
  tMax: f32,
  direction: vec3f,
  _pad0: f32,
};

struct ObjectBounds {
  localMin: vec3f,
  localMax: vec3f,
};

@group(0) @binding(0) var<uniform> ray: Ray;
@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> objectBounds: array<ObjectBounds>;
@group(0) @binding(3) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(4) var<uniform> objectCount: u32;

const NO_HIT: u32 = 0xFFFFFFFFu;
const INDEX_BITS: u32 = 12u;
const INDEX_MASK: u32 = 0xFFFu;

@compute @workgroup_size(64)
fn raycast(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= objectCount) {
    return;
  }

  // World-space AABB of the object's transformed local bounds.
  let bounds = objectBounds[id.x];
  let worldMatrix = worldMatrices[id.x];
  var worldMin = vec3f(1e30);
  var worldMax = vec3f(-1e30);
  for (var i = 0u; i < 8u; i++) {
    let corner = vec3f(
      select(bounds.localMin.x, bounds.localMax.x, (i & 1u) != 0u),
      select(bounds.localMin.y, bounds.localMax.y, (i & 2u) != 0u),
      select(bounds.localMin.z, bounds.localMax.z, (i & 4u) != 0u),
    );
    let world = (worldMatrix * vec4f(corner, 1.0)).xyz;
    worldMin = min(worldMin, world);
    worldMax = max(worldMax, world);
  }

  // Slab test. Division by zero yields +/-inf, which min/max handle.
  let invDir = 1.0 / ray.direction;
  let t0 = (worldMin - ray.origin) * invDir;
  let t1 = (worldMax - ray.origin) * invDir;
  let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));

  if (tFar < 0.0 || tNear > tFar || tNear > ray.tMax) {
    return;
  }

  // Ray origin inside the box hits at t = 0.
  let t = max(tNear, 0.0);

  let packed = (bitcast<u32>(t) & ~INDEX_MASK) | (id.x & INDEX_MASK);
  atomicMin(&result, packed);
}
`;
