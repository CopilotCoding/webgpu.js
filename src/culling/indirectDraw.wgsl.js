// DrawIndirectArgs layout (16 bytes): vertexCount, instanceCount, firstVertex, firstInstance.
// instanceCount starts at 0 each frame and is incremented atomically as
// each visible object claims a slot in visibleIndices — this is the GPU
// building its own draw call argument from the culling results, with no
// CPU involvement.
export const indirectDrawShader = /* wgsl */ `
struct DrawIndirectArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};

@group(0) @binding(0) var<storage, read> visibility: array<u32>;
@group(0) @binding(1) var<storage, read_write> visibleIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> drawArgs: DrawIndirectArgs;
@group(0) @binding(3) var<uniform> objectCount: u32;

const VISIBLE_OCCLUSION: u32 = 2u;

// Resets instanceCount to 0. Dispatched as its own pass (1 invocation)
// before buildIndirectDraw, so the reset is guaranteed visible to every
// workgroup of the build dispatch.
@compute @workgroup_size(1)
fn resetIndirectDraw() {
  atomicStore(&drawArgs.instanceCount, 0u);
}

@compute @workgroup_size(64)
fn buildIndirectDraw(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= objectCount) {
    return;
  }

  if ((visibility[id.x] & VISIBLE_OCCLUSION) != 0u) {
    let slot = atomicAdd(&drawArgs.instanceCount, 1u);
    visibleIndices[slot] = id.x;
  }
}
`;
