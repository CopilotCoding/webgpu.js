import { MaterialInstance } from './MaterialInstance.js';

/**
 * A material is a pipeline descriptor plus a typed bind group layout for
 * per-instance uniform bindings. The pipeline is compiled once (cached by
 * descriptor hash via the device's PipelineCache) and shared by every
 * MaterialInstance created from this material.
 *
 * Bind group layout convention:
 *   group 0 — camera (provided externally, shared across materials)
 *   group 1 — material instance bindings (this.bindGroupLayout)
 *
 * descriptor shape:
 * {
 *   shader: { code: '...wgsl...' },
 *   vertexBufferLayouts: [...],       // GPUVertexBufferLayout[], typically Geometry.vertexBufferLayouts
 *   fragmentTargets: [{ format }],
 *   primitive: {...},                 // optional, GPUPrimitiveState
 *   depthStencil: {...},               // optional, GPUDepthStencilState
 *   bindings: {                        // group 1, per-instance uniform bindings
 *     color: { binding: 0, visibility: GPUShaderStage.FRAGMENT, size: 16 },
 *   },
 * }
 */
export class Material {
  constructor(device, cameraBindGroupLayout, descriptor) {
    this.device = device;
    this.descriptor = descriptor;
    this.bindings = descriptor.bindings ?? {};

    this.bindGroupLayout = device.resources.createBindGroupLayout({
      entries: Object.values(this.bindings).map((binding) => ({
        binding: binding.binding,
        visibility: binding.visibility,
        buffer: { type: 'uniform' },
      })),
    });

    const shaderModule = device.device.createShaderModule({ code: descriptor.shader.code });

    const pipelineLayout = device.device.createPipelineLayout({
      bindGroupLayouts: [cameraBindGroupLayout.gpuBindGroupLayout, this.bindGroupLayout.gpuBindGroupLayout],
    });

    this.pipeline = device.pipelines.getOrCreateRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: descriptor.shader.vertexEntryPoint ?? 'vertexMain',
        buffers: descriptor.vertexBufferLayouts,
      },
      fragment: {
        module: shaderModule,
        entryPoint: descriptor.shader.fragmentEntryPoint ?? 'fragmentMain',
        targets: descriptor.fragmentTargets,
      },
      primitive: descriptor.primitive ?? { topology: 'triangle-list' },
      depthStencil: descriptor.depthStencil,
    });
  }

  createInstance(uniforms = {}) {
    return new MaterialInstance(this.device, this, uniforms);
  }
}
