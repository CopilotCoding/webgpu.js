import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
}

try {
  const device = await createDevice();
  const context = device.getCanvasContext(canvas);

  const depthTexture = device.resources.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  function frame() {
    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'clear',
      colorAttachments: [{
        target: CANVAS,
        clearValue: { r: 0.05, g: 0.4, b: 0.7, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        target: depthTexture,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      writes: [depthTexture],
      execute: (renderPass) => {
        // Nothing to draw yet — geometry arrives in Stage 3.
      },
    });

    graph.execute();

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
