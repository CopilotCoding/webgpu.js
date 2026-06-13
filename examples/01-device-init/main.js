import { createDevice } from '../../src/device/Device.js';

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

  function frame() {
    const encoder = device.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.4, b: 0.7, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.end();
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
