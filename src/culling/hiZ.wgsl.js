// Copies a depth24plus/depth32float texture into mip 0 of an r32float
// Hi-Z texture. Depth textures can't be bound as storage textures, so this
// pass moves the depth values into a format compute shaders can write to.
export const hiZCopyShader = /* wgsl */ `
@group(0) @binding(0) var depthTexture: texture_depth_2d;
@group(0) @binding(1) var hiZMip0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn copyDepth(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(depthTexture);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let depth = textureLoad(depthTexture, id.xy, 0);
  textureStore(hiZMip0, id.xy, vec4f(depth, 0.0, 0.0, 0.0));
}
`;

// Builds mip level N from mip level N-1 by taking the max depth (farthest,
// for a 0=near/1=far depth convention) of each 2x2 block. Dispatched once
// per mip level, sequentially, since each level depends on the previous.
export const hiZDownsampleShader = /* wgsl */ `
@group(0) @binding(0) var srcMip: texture_2d<f32>;
@group(0) @binding(1) var dstMip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
  let dstSize = textureDimensions(dstMip);
  if (id.x >= dstSize.x || id.y >= dstSize.y) {
    return;
  }

  let srcSize = textureDimensions(srcMip);
  let srcCoord = id.xy * 2u;

  var maxDepth = 0.0;
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let coord = min(srcCoord + vec2u(dx, dy), srcSize - vec2u(1u, 1u));
      maxDepth = max(maxDepth, textureLoad(srcMip, coord, 0).r);
    }
  }

  textureStore(dstMip, id.xy, vec4f(maxDepth, 0.0, 0.0, 0.0));
}
`;
