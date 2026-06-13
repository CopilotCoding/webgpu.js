export class Texture {
  constructor(gpuTexture, descriptor) {
    this.gpuTexture = gpuTexture;
    this.descriptor = descriptor;
    this._onDestroy = null;
  }

  destroy() {
    this.gpuTexture.destroy();
    if (this._onDestroy) this._onDestroy(this);
  }
}
