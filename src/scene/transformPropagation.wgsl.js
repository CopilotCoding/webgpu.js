export const transformPropagationShader = /* wgsl */ `
struct LevelInfo {
  offset: u32,
  count: u32,
};

@group(0) @binding(0) var<storage, read> localMatrices: array<mat4x4f>;
@group(0) @binding(1) var<storage, read> parentIndices: array<i32>;
@group(0) @binding(2) var<storage, read_write> worldMatrices: array<mat4x4f>;
@group(0) @binding(3) var<uniform> level: LevelInfo;

@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= level.count) {
    return;
  }

  let index = level.offset + id.x;
  let parentIndex = parentIndices[index];

  if (parentIndex < 0) {
    worldMatrices[index] = localMatrices[index];
  } else {
    worldMatrices[index] = worldMatrices[parentIndex] * localMatrices[index];
  }
}
`;
