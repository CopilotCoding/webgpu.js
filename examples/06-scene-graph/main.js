import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { Material } from '../../src/materials/Material.js';
import { SceneNode, propagateTransforms } from '../../src/scene/SceneNode.js';
import { perspective, lookAt } from '../../src/math/mat4.js';

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
@group(1) @binding(0) var<uniform> tint: vec4f;
@group(1) @binding(1) var<uniform> transform: Transform;

struct VertexOut {
  @builtin(position) position: vec4f,
};

@vertex
fn vertexMain(@location(0) position: vec3f) -> VertexOut {
  var out: VertexOut;
  out.position = camera.projectionMatrix * camera.viewMatrix * transform.modelMatrix * vec4f(position, 1.0);
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  return tint;
}
`;

function quatFromAxisAngle(axis, angle) {
  const half = angle / 2;
  const s = Math.sin(half);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}

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
          0.0, 0.4, 0.0,
          -0.4, -0.4, 0.0,
          0.4, -0.4, 0.0,
        ]),
      },
    },
  });

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100));
  camera.setViewMatrix(lookAt([0, 0, 6], [0, 0, 0], [0, 1, 0]));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const cameraBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });

  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBindGroupLayout.gpuBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });

  const material = new Material(device, cameraBindGroupLayout, {
    shader: { code: shaderSource },
    vertexBufferLayouts: geometry.vertexBufferLayouts,
    fragmentTargets: [{ format }],
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
    bindings: {
      tint: { binding: 0, visibility: GPUShaderStage.FRAGMENT, size: 16 },
      transform: { binding: 1, visibility: GPUShaderStage.VERTEX, size: 64 },
    },
  });

  // Scene graph: a root node that spins, with two child triangles offset
  // from it. Only the root is updated each frame — propagateTransforms
  // recomputes the children's world matrices because they inherit the
  // root's dirty flag.
  const root = new SceneNode('root');

  const objects = [
    { node: new SceneNode('childA'), offset: [-1.2, 0, 0], tint: new Float32Array([1.0, 0.3, 0.3, 1.0]) },
    { node: new SceneNode('childB'), offset: [1.2, 0, 0], tint: new Float32Array([0.3, 0.6, 1.0, 1.0]) },
  ];

  for (const object of objects) {
    object.node.setPosition(...object.offset);
    root.add(object.node);
    object.instance = material.createInstance({ tint: object.tint });
  }

  let angle = 0;

  function frame() {
    angle += 0.01;
    root.setRotation(...quatFromAxisAngle([0, 0, 1], angle));
    propagateTransforms(root);

    camera.update();
    for (const object of objects) {
      object.instance.set('transform', object.node.worldMatrix);
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
        renderPass.setPipeline(material.pipeline);
        renderPass.setBindGroup(0, cameraBindGroup);
        renderPass.setVertexBuffer(0, geometry.attributes.position.buffer.gpuBuffer);

        for (const object of objects) {
          renderPass.setBindGroup(1, object.instance.bindGroup);
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
