import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
}

const shaderSource = /* wgsl */ `
struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) color: vec3f,
) -> VertexOut {
  var out: VertexOut;
  out.position = vec4f(position, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

try {
  const device = await createDevice();
  const context = device.getCanvasContext(canvas);
  const format = navigator.gpu.getPreferredCanvasFormat();

  const depthTexture = device.resources.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const geometry = new Geometry(device, {
    attributes: {
      position: {
        format: 'float32x3',
        data: new Float32Array([
          0.0, 0.5, 0.0,
          -0.5, -0.5, 0.0,
          0.5, -0.5, 0.0,
        ]),
      },
      color: {
        format: 'float32x3',
        data: new Float32Array([
          1.0, 0.2, 0.2,
          0.2, 1.0, 0.2,
          0.2, 0.2, 1.0,
        ]),
      },
    },
  });

  const shaderModule = device.device.createShaderModule({ code: shaderSource });

  const pipeline = device.device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vertexMain',
      buffers: geometry.vertexBufferLayouts,
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragmentMain',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  });

  function frame() {
    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'triangle',
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
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, geometry.attributes.position.buffer.gpuBuffer);
        renderPass.setVertexBuffer(1, geometry.attributes.color.buffer.gpuBuffer);
        renderPass.draw(geometry.vertexCount);
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
