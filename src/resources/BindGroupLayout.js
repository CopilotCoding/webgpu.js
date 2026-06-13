export class BindGroupLayout {
  constructor(gpuBindGroupLayout, descriptor) {
    this.gpuBindGroupLayout = gpuBindGroupLayout;
    this.descriptor = descriptor;
    this._onDestroy = null;
  }

  // GPUBindGroupLayout has no destroy() in WebGPU — destroy() here only
  // removes this resource from the ResourceManager registry.
  destroy() {
    if (this._onDestroy) this._onDestroy(this);
  }
}
