import { hiZCopyShader, hiZDownsampleShader } from './hiZ.wgsl.js';

/**
 * Hierarchical depth (Hi-Z) pyramid built from a depth texture: mip 0 is a
 * direct copy of the depth buffer, and each subsequent mip holds the max
 * depth of the 2x2 block below it. Occlusion culling tests an object's
 * screen-space AABB against the mip level whose texel size best matches
 * the AABB's footprint.
 *
 * Built from the *previous* frame's depth texture — the current frame's
 * geometry pass hasn't run yet when culling executes.
 */
export class HiZBuffer {
  constructor(device, width, height) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.mipCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

    this.texture = device.resources.createTexture({
      size: [width, height],
      format: 'r32float',
      mipLevelCount: this.mipCount,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.mipViews = [];
    for (let level = 0; level < this.mipCount; level++) {
      this.mipViews.push(this.texture.gpuTexture.createView({ baseMipLevel: level, mipLevelCount: 1 }));
    }

    const copyShaderModule = device.device.createShaderModule({ code: hiZCopyShader });
    this.copyPipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: copyShaderModule, entryPoint: 'copyDepth' },
    });

    const downsampleShaderModule = device.device.createShaderModule({ code: hiZDownsampleShader });
    this.downsamplePipeline = device.device.createComputePipeline({
      layout: 'auto',
      compute: { module: downsampleShaderModule, entryPoint: 'downsample' },
    });
  }

  /**
   * Records the copy + mip-chain build into `encoder`. `depthTexture` is a
   * Texture wrapper (depth24plus or depth32float) from a previous frame.
   */
  build(encoder, depthTexture) {
    const copyBindGroup = this.device.device.createBindGroup({
      layout: this.copyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: depthTexture.gpuTexture.createView() },
        { binding: 1, resource: this.mipViews[0] },
      ],
    });

    const pass = encoder.beginComputePass();

    pass.setPipeline(this.copyPipeline);
    pass.setBindGroup(0, copyBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));

    pass.setPipeline(this.downsamplePipeline);
    for (let level = 1; level < this.mipCount; level++) {
      const mipWidth = Math.max(1, this.width >> level);
      const mipHeight = Math.max(1, this.height >> level);

      const bindGroup = this.device.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.mipViews[level - 1] },
          { binding: 1, resource: this.mipViews[level] },
        ],
      });

      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(mipWidth / 8), Math.ceil(mipHeight / 8));
    }

    pass.end();
  }

  destroy() {
    this.texture.destroy();
  }
}
