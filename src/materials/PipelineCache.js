/**
 * Caches GPURenderPipelines by a hash of their descriptor, so identical
 * pipeline descriptors (e.g. requested by multiple Materials with the same
 * shader and layout) compile only once.
 */
export class PipelineCache {
  constructor(device) {
    this.device = device;
    this.cache = new Map();
  }

  getOrCreateRenderPipeline(descriptor) {
    const key = hashDescriptor(descriptor);
    let pipeline = this.cache.get(key);
    if (!pipeline) {
      pipeline = this.device.device.createRenderPipeline(descriptor);
      this.cache.set(key, pipeline);
    }
    return pipeline;
  }
}

/**
 * Produces a stable string key for a render pipeline descriptor. Shader
 * modules and bind group layouts are GPU objects without a useful default
 * string form, so they're identified by reference (via a WeakMap of
 * assigned ids) rather than by structural content.
 */
const objectIds = new WeakMap();
let nextObjectId = 0;

function idFor(object) {
  let id = objectIds.get(object);
  if (id === undefined) {
    id = nextObjectId++;
    objectIds.set(object, id);
  }
  return id;
}

function hashDescriptor(descriptor) {
  return JSON.stringify(descriptor, (key, value) => {
    if (value instanceof GPUShaderModule || value instanceof GPUPipelineLayout || value instanceof GPUBindGroupLayout) {
      return `#${idFor(value)}`;
    }
    return value;
  });
}
