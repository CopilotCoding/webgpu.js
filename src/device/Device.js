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

  // Opt into the indirect-draw features the GPU-driven path benefits from, when
  // the adapter has them:
  //  - indirect-first-instance (standardized; Firefox + Chrome): lets indirect
  //    draws carry a non-zero firstInstance, so the object id rides there and
  //    the per-draw bind-group rebind is eliminated.
  //  - chromium-experimental-multi-draw-indirect (Chromium only): collapses a
  //    whole batch into one multi-draw call.
  const desc = { ...(options.deviceDescriptor || {}) };
  const optional = ['indirect-first-instance', 'chromium-experimental-multi-draw-indirect'];
  const wanted = optional.filter((f) => adapter.features.has(f));
  if (wanted.length) desc.requiredFeatures = [...(desc.requiredFeatures || []), ...wanted];

  const device = await adapter.requestDevice(desc);
  console.log('[webgpu.js] indirect-first-instance:', device.features.has('indirect-first-instance') ? 'on' : 'OFF',
    '| multi-draw:', device.features.has('chromium-experimental-multi-draw-indirect') ? 'on' : 'off');

  return new Device(adapter, device);
}
