import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { perspective, lookAt, translation } from '../../src/math/mat4.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
}

const shaderSource = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};

struct Transform {
  modelMatrix: mat4x4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> transform: Transform;

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
  out.position = camera.projectionMatrix * camera.viewMatrix * transform.modelMatrix * vec4f(position, 1.0);
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

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100));
  camera.setViewMatrix(lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const shaderModule = device.device.createShaderModule({ code: shaderSource });

  const cameraBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });

  const transformBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });

  const pipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({
      bindGroupLayouts: [cameraBindGroupLayout.gpuBindGroupLayout, transformBindGroupLayout.gpuBindGroupLayout],
    }),
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

  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBindGroupLayout.gpuBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });

  // Two objects, each with their own transform uniform buffer + bind group.
  const objects = [-1.5, 1.5].map((x) => {
    const buffer = device.resources.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.device.createBindGroup({
      layout: transformBindGroupLayout.gpuBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: buffer.gpuBuffer } }],
    });
    return { x, buffer, bindGroup };
  });

  function frame() {
    camera.update();

    for (const object of objects) {
      device.queue.writeBuffer(object.buffer.gpuBuffer, 0, translation(object.x, 0, 0));
    }

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'forward',
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
      reads: [camera.buffer],
      execute: (renderPass) => {
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, cameraBindGroup);
        renderPass.setVertexBuffer(0, geometry.attributes.position.buffer.gpuBuffer);
        renderPass.setVertexBuffer(1, geometry.attributes.color.buffer.gpuBuffer);

        for (const object of objects) {
          renderPass.setBindGroup(1, object.bindGroup);
          renderPass.draw(geometry.vertexCount);
        }
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
