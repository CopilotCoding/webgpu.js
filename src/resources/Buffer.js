export class Buffer {
  constructor(gpuBuffer, descriptor) {
    this.gpuBuffer = gpuBuffer;
    this.descriptor = descriptor;
    this._onDestroy = null;
  }

  destroy() {
    this.gpuBuffer.destroy();
    if (this._onDestroy) this._onDestroy(this);
  }
}
