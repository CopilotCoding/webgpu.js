import { Texture } from '../resources/Texture.js';
import { Buffer } from '../resources/Buffer.js';

// Sentinel marking a color attachment that targets the canvas swap chain
// rather than a persistent Texture resource.
export const CANVAS = Symbol('canvas-output');

export class RenderGraph {
  constructor(device) {
    this.device = device;
    this.passes = [];
    this.canvasContext = null;
  }

  setCanvasTarget(context) {
    this.canvasContext = context;
  }

  addPass(descriptor) {
    this.passes.push(descriptor);
    return descriptor;
  }

  /**
   * Topologically orders passes so that any pass writing a resource runs
   * before passes that read it, then validates that each resource is
   * used in a way its declared GPUUsage flags support.
   */
  _resolve() {
    this._validateUsages();

    const order = [];
    const visited = new Set();
    const visiting = new Set();

    const writerOf = (resource) => this.passes.find((p) => (p.writes ?? []).includes(resource));

    const visit = (pass) => {
      if (visited.has(pass)) return;
      if (visiting.has(pass)) {
        throw new Error(`RenderGraph: cyclic dependency detected involving pass "${pass.name}"`);
      }
      visiting.add(pass);

      for (const resource of pass.reads ?? []) {
        const producer = writerOf(resource);
        if (producer && producer !== pass) visit(producer);
      }

      visiting.delete(pass);
      visited.add(pass);
      order.push(pass);
    };

    for (const pass of this.passes) visit(pass);

    return order;
  }

  _validateUsages() {
    for (const pass of this.passes) {
      for (const attachment of pass.colorAttachments ?? []) {
        const target = attachment.target;
        if (target === CANVAS) continue;
        if (!(target instanceof Texture)) {
          throw new Error(`RenderGraph: pass "${pass.name}" colorAttachments target must be a Texture or CANVAS`);
        }
        if (!(target.descriptor.usage & GPUTextureUsage.RENDER_ATTACHMENT)) {
          throw new Error(
            `RenderGraph: pass "${pass.name}" uses a texture as a color attachment, ` +
            `but its descriptor.usage does not include GPUTextureUsage.RENDER_ATTACHMENT`
          );
        }
      }

      const depthStencil = pass.depthStencilAttachment;
      if (depthStencil) {
        const target = depthStencil.target;
        if (!(target instanceof Texture)) {
          throw new Error(`RenderGraph: pass "${pass.name}" depthStencilAttachment target must be a Texture`);
        }
        if (!(target.descriptor.usage & GPUTextureUsage.RENDER_ATTACHMENT)) {
          throw new Error(
            `RenderGraph: pass "${pass.name}" uses a texture as a depth/stencil attachment, ` +
            `but its descriptor.usage does not include GPUTextureUsage.RENDER_ATTACHMENT`
          );
        }
      }

      for (const resource of pass.reads ?? []) {
        if (resource instanceof Texture) {
          if (!(resource.descriptor.usage & GPUTextureUsage.TEXTURE_BINDING)) {
            throw new Error(
              `RenderGraph: pass "${pass.name}" reads a texture, ` +
              `but its descriptor.usage does not include GPUTextureUsage.TEXTURE_BINDING`
            );
          }
        } else if (resource instanceof Buffer) {
          const usage = resource.descriptor.usage;
          if (!(usage & GPUBufferUsage.STORAGE) && !(usage & GPUBufferUsage.UNIFORM)) {
            throw new Error(
              `RenderGraph: pass "${pass.name}" reads a buffer, ` +
              `but its descriptor.usage does not include GPUBufferUsage.STORAGE or GPUBufferUsage.UNIFORM`
            );
          }
        }
      }

      for (const resource of pass.writes ?? []) {
        if (resource instanceof Buffer) {
          if (!(resource.descriptor.usage & GPUBufferUsage.STORAGE)) {
            throw new Error(
              `RenderGraph: pass "${pass.name}" writes a buffer, ` +
              `but its descriptor.usage does not include GPUBufferUsage.STORAGE`
            );
          }
        }
      }
    }
  }

  /**
   * Resolves pass order, records commands via each pass's execute()
   * callback, and submits the resulting command buffer to the queue.
   */
  execute() {
    const order = this._resolve();
    const encoder = this.device.device.createCommandEncoder();

    for (const pass of order) {
      if (pass.colorAttachments || pass.depthStencilAttachment) {
        const colorAttachments = (pass.colorAttachments ?? []).map((attachment) => ({
          view: this._resolveView(attachment.target),
          clearValue: attachment.clearValue,
          loadOp: attachment.loadOp ?? 'load',
          storeOp: attachment.storeOp ?? 'store',
        }));

        const renderPassDescriptor = { colorAttachments };

        if (pass.depthStencilAttachment) {
          const depthStencil = pass.depthStencilAttachment;
          renderPassDescriptor.depthStencilAttachment = {
            view: depthStencil.view ?? this._resolveView(depthStencil.target),
            depthClearValue: depthStencil.depthClearValue,
            depthLoadOp: depthStencil.depthLoadOp ?? 'load',
            depthStoreOp: depthStencil.depthStoreOp ?? 'store',
            ...(depthStencil.stencilLoadOp ? {
              stencilClearValue: depthStencil.stencilClearValue,
              stencilLoadOp: depthStencil.stencilLoadOp,
              stencilStoreOp: depthStencil.stencilStoreOp,
            } : {}),
          };
        }

        const renderPass = encoder.beginRenderPass(renderPassDescriptor);
        pass.execute(renderPass);
        renderPass.end();
      } else {
        pass.execute(encoder);
      }
    }

    this.device.queue.submit([encoder.finish()]);
  }

  _resolveView(target) {
    if (target === CANVAS) {
      if (!this.canvasContext) {
        throw new Error('RenderGraph: a pass targets CANVAS but setCanvasTarget() was never called');
      }
      return this.canvasContext.getCurrentTexture().createView();
    }
    return target.gpuTexture.createView();
  }
}
