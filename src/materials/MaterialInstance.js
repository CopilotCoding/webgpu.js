/**
 * Per-instance state for a Material: one uniform buffer per binding
 * declared in the material's `bindings` descriptor, plus the bind group
 * (group 1) that exposes them to the pipeline. Buffers are updated
 * explicitly via set() — there is no implicit uniform uploading.
 */
export class MaterialInstance {
  constructor(device, material, uniforms = {}) {
    this.device = device;
    this.material = material;
    this.buffers = {};

    const entries = [];
    for (const [name, binding] of Object.entries(material.bindings)) {
      const buffer = device.resources.createBuffer({
        size: binding.size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.buffers[name] = buffer;
      entries.push({ binding: binding.binding, resource: { buffer: buffer.gpuBuffer } });

      if (uniforms[name] !== undefined) {
        this.set(name, uniforms[name]);
      }
    }

    this.bindGroup = device.device.createBindGroup({
      layout: material.bindGroupLayout.gpuBindGroupLayout,
      entries,
    });
  }

  set(name, data) {
    const buffer = this.buffers[name];
    if (!buffer) {
      throw new Error(`MaterialInstance: unknown binding "${name}" — not declared in this material's bindings`);
    }
    this.device.queue.writeBuffer(buffer.gpuBuffer, 0, data);
  }

  destroy() {
    for (const buffer of Object.values(this.buffers)) {
      buffer.destroy();
    }
  }
}
