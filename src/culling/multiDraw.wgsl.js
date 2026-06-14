// GPU culling + indexed-indirect compaction for HETEROGENEOUS geometry.
//
// Unlike the single-geometry IndirectDrawSystem (which emits ONE
// DrawIndirectArgs and an instance list), this emits an ARRAY of
// DrawIndexedIndirectArgs — one per visible object — so objects with different
// index ranges / base vertices (different meshes packed in a GeometryArena)
// can all be drawn GPU-driven. The CPU then issues a fixed number of
// drawIndexedIndirect calls (one per object slot); slots past the visible
// count have indexCount = 0 and draw nothing.
//
// Per object the CPU provides a DrawRecord:
//   { firstIndex, indexCount, baseVertex, transformIndex, layerMask, flags }
// flags bit 0 = FRUSTUM_CULL_DISABLED (always-visible: sun, atmosphere).
// The compute pass tests the object's world-space AABB against the frustum
// (unless disabled) and the camera's layer mask, then appends a packed
// DrawIndexedIndirectArgs for it and records the object's id in a compacted
// slotToObject[] at the same slot.
//
// Note we do NOT use the indirect args' firstInstance to carry the object id:
// a non-zero firstInstance in drawIndexedIndirect is a silent no-op unless the
// optional `indirect-first-instance` feature is enabled. Instead the CPU's
// fixed draw loop binds a per-draw dynamic uniform offset giving each draw its
// slot, and the vertex shader reads slotToObject[slot] to recover per-object
// data — portable on baseline WebGPU.

export const RECORD_SIZE = 32; // 8 x u32

export const multiDrawCullShader = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};

struct DrawRecord {
  firstIndex: u32,
  indexCount: u32,
  baseVertex: u32,
  transformIndex: u32,
  layerMask: u32,
  flags: u32,
  _pad0: u32,
  _pad1: u32,
};

struct ObjectBounds {
  localMin: vec3f,
  localMax: vec3f,
};

// GPUDrawIndexedIndirect: indexCount, instanceCount, firstIndex, baseVertex, firstInstance.
struct DrawIndexedArgs {
  indexCount: u32,
  instanceCount: u32,
  firstIndex: u32,
  baseVertex: i32,
  firstInstance: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> records: array<DrawRecord>;
@group(0) @binding(2) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(3) var<storage, read> objectBounds: array<ObjectBounds>;
@group(0) @binding(4) var<storage, read_write> drawArgs: array<DrawIndexedArgs>;
@group(0) @binding(5) var<storage, read_write> drawCount: atomic<u32>;
@group(0) @binding(6) var<uniform> params: vec4u; // x = objectCount, y = cameraLayerMask
@group(0) @binding(7) var<storage, read_write> slotToObject: array<u32>;

const FRUSTUM_CULL_DISABLED: u32 = 1u;

fn testFrustum(localMin: vec3f, localMax: vec3f, m: mat4x4f) -> bool {
  var corners: array<vec3f, 8>;
  for (var i = 0u; i < 8u; i++) {
    let x = select(localMin.x, localMax.x, (i & 1u) != 0u);
    let y = select(localMin.y, localMax.y, (i & 2u) != 0u);
    let z = select(localMin.z, localMax.z, (i & 4u) != 0u);
    corners[i] = (m * vec4f(x, y, z, 1.0)).xyz;
  }
  for (var p = 0u; p < 6u; p++) {
    let plane = camera.frustumPlanes[p];
    var allOutside = true;
    for (var i = 0u; i < 8u; i++) {
      if (dot(plane.xyz, corners[i]) + plane.w >= 0.0) { allOutside = false; break; }
    }
    if (allOutside) { return false; }
  }
  return true;
}

// Clears the draw count and zeroes every slot's indexCount so unfilled slots
// (past the visible count) draw nothing in the CPU's fixed-length loop.
@compute @workgroup_size(64)
fn resetDraws(@builtin(global_invocation_id) id: vec3u) {
  if (id.x == 0u) { atomicStore(&drawCount, 0u); }
  if (id.x >= params.x) { return; }
  drawArgs[id.x].indexCount = 0u;
  drawArgs[id.x].instanceCount = 0u;
}

@compute @workgroup_size(64)
fn cullAndCompact(@builtin(global_invocation_id) id: vec3u) {
  let objectCount = params.x;
  if (id.x >= objectCount) { return; }

  let rec = records[id.x];

  // Layer mask: skip objects the camera doesn't see.
  if ((rec.layerMask & params.y) == 0u) { return; }

  if ((rec.flags & FRUSTUM_CULL_DISABLED) == 0u) {
    let b = objectBounds[id.x];
    let m = worldMatrices[rec.transformIndex];
    if (!testFrustum(b.localMin, b.localMax, m)) { return; }
  }

  let slot = atomicAdd(&drawCount, 1u);
  drawArgs[slot].indexCount = rec.indexCount;
  drawArgs[slot].instanceCount = 1u;
  drawArgs[slot].firstIndex = rec.firstIndex;
  drawArgs[slot].baseVertex = i32(rec.baseVertex);
  // Object id is provided two ways so the renderer can pick a draw path:
  //  - firstInstance: read via @builtin(instance_index) under
  //    multiDrawIndexedIndirect (honors non-zero firstInstance).
  //  - slotToObject[slot]: read via a per-draw dynamic offset in the
  //    drawIndexedIndirect loop fallback (where firstInstance is a no-op).
  drawArgs[slot].firstInstance = id.x;
  slotToObject[slot] = id.x;
}
`;
