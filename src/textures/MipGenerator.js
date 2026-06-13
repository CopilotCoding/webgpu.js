// Compute-based mipmap generation. Each mip level is produced from the one
// above it by a 2x2 box filter running in a compute pass — no render passes,
// no blit pipelines, per CLAUDE.md ("Mipmaps are generated in compute").
//
// Works on 2D textures and 2D array textures (each layer is downsampled
// independently). The texture must be created with TEXTURE_BINDING and
// STORAGE_BINDING usage and a storage-compatible format (rgba8unorm).

const mipShaderSource = /* wgsl */ `
@group(0) @binding(0) var srcMip: texture_2d<f32>;
@group(0) @binding(1) var dstMip: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
  let dstSize = textureDimensions(dstMip);
  if (id.x >= dstSize.x || id.y >= dstSize.y) {
    return;
  }

  let srcCoord = id.xy * 2u;
  let color =
    (textureLoad(srcMip, srcCoord, 0) +
     textureLoad(srcMip, srcCoord + vec2u(1u, 0u), 0) +
     textureLoad(srcMip, srcCoord + vec2u(0u, 1u), 0) +
     textureLoad(srcMip, srcCoord + vec2u(1u, 1u), 0)) * 0.25;

  textureStore(dstMip, id.xy, color);
}
`;

export class MipGenerator {
  constructor(device) {
    this.device = device;

    this.bindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      ],
    });

    const module = device.device.createShaderModule({ code: mipShaderSource });
    this.pipeline = device.device.createComputePipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module, entryPoint: 'downsample' },
    });
  }

  /**
   * Records compute dispatches filling every mip level of the texture from
   * mip 0. For array textures, every layer is processed.
   *
   * @param {GPUCommandEncoder} encoder
   * @param {import('../resources/Texture.js').Texture} texture engine texture wrapper whose mip 0 (all layers) is already populated
   */
  generate(encoder, texture) {
    const descriptor = texture.descriptor;
    const mipLevelCount = descriptor.mipLevelCount ?? 1;
    const layerCount = descriptor.size[2] ?? 1;

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);

    for (let layer = 0; layer < layerCount; layer++) {
      for (let mip = 1; mip < mipLevelCount; mip++) {
        const bindGroup = this.device.device.createBindGroup({
          layout: this.bindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: texture.gpuTexture.createView({
                dimension: '2d',
                baseMipLevel: mip - 1,
                mipLevelCount: 1,
                baseArrayLayer: layer,
                arrayLayerCount: 1,
              }),
            },
            {
              binding: 1,
              resource: texture.gpuTexture.createView({
                dimension: '2d',
                baseMipLevel: mip,
                mipLevelCount: 1,
                baseArrayLayer: layer,
                arrayLayerCount: 1,
              }),
            },
          ],
        });

        const width = Math.max(1, descriptor.size[0] >> mip);
        const height = Math.max(1, descriptor.size[1] >> mip);

        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      }
    }

    pass.end();
  }
}
