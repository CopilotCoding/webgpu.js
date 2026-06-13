import { ResourceManager } from '../resources/ResourceManager.js';
import { PipelineCache } from '../materials/PipelineCache.js';

export class Device {
  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.queue = device.queue;
    this.resources = new ResourceManager(device);
    this.pipelines = new PipelineCache(this);

    this.device.lost.then((info) => {
      console.error(`GPUDevice lost: ${info.message} (reason: ${info.reason})`);
    });

    this.device.addEventListener('uncapturederror', (event) => {
      console.error(`GPUDevice uncapturederror: ${event.error.constructor.name}: ${event.error.message}`);
    });
  }

  getCanvasContext(canvas, options = {}) {
    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error("canvas.getContext('webgpu') returned null — WebGPU context could not be created");
    }

    const format = options.format ?? navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: this.device,
      format,
      alphaMode: options.alphaMode ?? 'opaque',
      ...options,
    });

    return context;
  }

  destroyAll() {
    this.resources.destroyAll();
  }
}

export async function createDevice(options = {}) {
  if (!navigator.gpu) {
    throw new Error('navigator.gpu is undefined — WebGPU is not supported in this browser');
  }

  const adapter = await navigator.gpu.requestAdapter(options.adapterOptions);
  if (!adapter) {
    throw new Error('navigator.gpu.requestAdapter() returned null — no suitable GPUAdapter is available');
  }

  const device = await adapter.requestDevice(options.deviceDescriptor);

  return new Device(adapter, device);
}
