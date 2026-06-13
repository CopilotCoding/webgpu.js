import { Buffer } from './Buffer.js';
import { Texture } from './Texture.js';
import { Sampler } from './Sampler.js';
import { BindGroupLayout } from './BindGroupLayout.js';

export class ResourceManager {
  constructor(device) {
    this.device = device;
    this.resources = new Set();
  }

  _register(resource) {
    this.resources.add(resource);
    resource._onDestroy = (r) => this.resources.delete(r);
    return resource;
  }

  createBuffer(descriptor) {
    const gpuBuffer = this.device.createBuffer(descriptor);
    return this._register(new Buffer(gpuBuffer, descriptor));
  }

  createTexture(descriptor) {
    const gpuTexture = this.device.createTexture(descriptor);
    return this._register(new Texture(gpuTexture, descriptor));
  }

  createSampler(descriptor) {
    const gpuSampler = this.device.createSampler(descriptor);
    return this._register(new Sampler(gpuSampler, descriptor));
  }

  createBindGroupLayout(descriptor) {
    const gpuBindGroupLayout = this.device.createBindGroupLayout(descriptor);
    return this._register(new BindGroupLayout(gpuBindGroupLayout, descriptor));
  }

  destroyAll() {
    for (const resource of [...this.resources]) {
      resource.destroy();
    }
  }
}
