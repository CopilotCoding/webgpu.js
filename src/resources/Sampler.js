export class Sampler {
  constructor(gpuSampler, descriptor) {
    this.gpuSampler = gpuSampler;
    this.descriptor = descriptor;
    this._onDestroy = null;
  }

  // GPUSampler has no destroy() in WebGPU — destroy() here only removes
  // this resource from the ResourceManager registry.
  destroy() {
    if (this._onDestroy) this._onDestroy(this);
  }
}
