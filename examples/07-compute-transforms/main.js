import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { Material } from '../../src/materials/Material.js';
import { SceneNode } from '../../src/scene/SceneNode.js';
import { flattenSceneGraph } from '../../src/scene/flattenSceneGraph.js';
import { TransformPropagation } from '../../src/scene/TransformPropagation.js';
import { perspective, lookAt, fromTranslationRotationScale } from '../../src/math/mat4.js';

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

  // A long thin quad (two triangles) representing one "arm segment",
  // extending from its local origin along +X.
  const geometry = new Geometry(device, {
    attributes: {
      position: {
        format: 'float32x3',
        data: new Float32Array([
          0.0, -0.08, 0.0,
          1.2, -0.08, 0.0,
          1.2, 0.08, 0.0,
          0.0, -0.08, 0.0,
          1.2, 0.08, 0.0,
          0.0, 0.08, 0.0,
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

  // 3-level chain: root -> arm -> hand. Each child is offset along +X by
  // the previous segment's length (1.2), so the chain visually extends.
  const root = new SceneNode('root');
  const arm = new SceneNode('arm');
  const hand = new SceneNode('hand');

  arm.setPosition(1.2, 0, 0);
  root.add(arm);
  hand.setPosition(1.2, 0, 0);
  arm.add(hand);

  const segments = [
    { node: root, tint: new Float32Array([1.0, 0.3, 0.3, 1.0]) },
    { node: arm, tint: new Float32Array([0.3, 1.0, 0.3, 1.0]) },
    { node: hand, tint: new Float32Array([0.3, 0.6, 1.0, 1.0]) },
  ];
  for (const segment of segments) {
    segment.instance = material.createInstance({ tint: segment.tint });
  }

  const flattened = flattenSceneGraph(root);
  const propagation = new TransformPropagation(device, flattened);

  let angle = 0;

  function frame() {
    angle += 0.01;
    root.setRotation(...quatFromAxisAngle([0, 0, 1], angle * 0.5));
    arm.setRotation(...quatFromAxisAngle([0, 0, 1], Math.sin(angle) * 0.8));
    hand.setRotation(...quatFromAxisAngle([0, 0, 1], Math.sin(angle * 2) * 1.2));

    // CPU only touches the nodes it changed: recompute their local
    // matrices and re-upload. World matrices are computed entirely on
    // the GPU by TransformPropagation.
    for (let i = 0; i < flattened.nodes.length; i++) {
      const node = flattened.nodes[i];
      fromTranslationRotationScale(node.position, node.rotation, node.scale, flattened.localMatrices.subarray(i * 16, i * 16 + 16));
    }
    propagation.updateLocalMatrices(flattened.localMatrices);

    camera.update();

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'propagate-transforms',
      writes: [propagation.worldBuffer],
      execute: (encoder) => {
        propagation.propagate(encoder);

        // Copy each node's 64-byte world matrix into its MaterialInstance's
        // transform uniform buffer.
        for (let i = 0; i < segments.length; i++) {
          encoder.copyBufferToBuffer(
            propagation.worldBuffer.gpuBuffer, i * 64,
            segments[i].instance.buffers.transform.gpuBuffer, 0,
            64,
          );
        }
      },
    });

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
      reads: [camera.buffer, propagation.worldBuffer],
      execute: (renderPass) => {
        renderPass.setPipeline(material.pipeline);
        renderPass.setBindGroup(0, cameraBindGroup);
        renderPass.setVertexBuffer(0, geometry.attributes.position.buffer.gpuBuffer);

        for (const segment of segments) {
          renderPass.setBindGroup(1, segment.instance.bindGroup);
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
