import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { Material } from '../../src/materials/Material.js';
import { SceneNode } from '../../src/scene/SceneNode.js';
import { flattenSceneGraph } from '../../src/scene/flattenSceneGraph.js';
import { TransformPropagation } from '../../src/scene/TransformPropagation.js';
import { HiZBuffer } from '../../src/culling/HiZBuffer.js';
import { CullingPass } from '../../src/culling/CullingPass.js';
import { IndirectDrawSystem } from '../../src/culling/IndirectDrawSystem.js';
import { perspective, lookAt } from '../../src/math/mat4.js';
import { createCubeData } from '../08-culling/cube.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
}

const cameraStruct = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
`;

const wallShaderSource = /* wgsl */ `
${cameraStruct}
struct Transform {
  modelMatrix: mat4x4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> tint: vec4f;
@group(1) @binding(1) var<uniform> transform: Transform;

@vertex
fn vertexMain(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return camera.projectionMatrix * camera.viewMatrix * transform.modelMatrix * vec4f(position, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return tint;
}
`;

// Cubes are drawn via a single drawIndirect() call. The instance_index
// supplied to the vertex shader only ranges over [0, instanceCount) where
// instanceCount was written by IndirectDrawSystem — it does NOT match the
// object's index in worldMatrices/visibility. visibleIndices maps the
// instance back to its original object index.
const cubeShaderSource = /* wgsl */ `
${cameraStruct}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> visibleIndices: array<u32>;

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @builtin(instance_index) instanceIndex: u32,
) -> @builtin(position) vec4f {
  let objectIndex = visibleIndices[instanceIndex];
  let worldMatrix = worldMatrices[objectIndex];
  return camera.projectionMatrix * camera.viewMatrix * worldMatrix * vec4f(position, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return vec4f(0.2, 1.0, 0.2, 1.0);
}
`;

try {
  const device = await createDevice();
  const context = device.getCanvasContext(canvas);
  const format = navigator.gpu.getPreferredCanvasFormat();

  const depthTexture = device.resources.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const hiZBuffer = new HiZBuffer(device, canvas.width, canvas.height);

  const cameraBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBindGroupLayout.gpuBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });

  // --- Wall (occluder), drawn with the normal forward Material ---
  const wallGeometry = new Geometry(device, {
    attributes: {
      position: {
        format: 'float32x3',
        data: new Float32Array([
          -3, -3, 0, 3, -3, 0, 3, 3, 0,
          -3, -3, 0, 3, 3, 0, -3, 3, 0,
        ]),
      },
    },
  });

  const wallMaterial = new Material(device, cameraBindGroupLayout, {
    shader: { code: wallShaderSource },
    vertexBufferLayouts: wallGeometry.vertexBufferLayouts,
    fragmentTargets: [{ format }],
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    bindings: {
      tint: { binding: 0, visibility: GPUShaderStage.FRAGMENT, size: 16 },
      transform: { binding: 1, visibility: GPUShaderStage.VERTEX, size: 64 },
    },
  });

  const wallInstance = wallMaterial.createInstance({ tint: new Float32Array([0.3, 0.3, 0.35, 1.0]) });
  wallInstance.set('transform', new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 4, 1, // wall sits between camera and the cube grid
  ]));

  // --- Cube grid (culled objects, drawn via indirect draw) ---
  const cubeGeometry = new Geometry(device, {
    attributes: { position: { format: 'float32x3', data: createCubeData() } },
  });
  const cubeBounds = cubeGeometry.computeBounds();

  const root = new SceneNode('root');
  const gridSize = 5;
  const spacing = 2.0;
  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      const node = new SceneNode(`cube-${x}-${y}`);
      node.setPosition((x - (gridSize - 1) / 2) * spacing, (y - (gridSize - 1) / 2) * spacing, -6);
      root.add(node);
    }
  }

  const flattened = flattenSceneGraph(root);
  const propagation = new TransformPropagation(device, flattened);

  // flattened.nodes[0] is the root itself (identity, not drawn) — cube
  // world matrices are nodes[1..], so the culling/draw object count
  // excludes the root.
  const cubeCount = flattened.nodes.length - 1;
  const cubeBoundsArray = Array.from({ length: cubeCount }, () => cubeBounds);
  const cubeWorldBufferOffset = 16 * 4; // skip the root's mat4x4f (64 bytes)

  const cubeWorldMatrices = device.resources.createBuffer({
    size: cubeCount * 64,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const cullingPass = new CullingPass(device, camera, cubeWorldMatrices, cubeBoundsArray, hiZBuffer);
  const indirectDrawSystem = new IndirectDrawSystem(device, cullingPass, cubeGeometry.vertexCount);

  const cubePipelineLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  const cubeShaderModule = device.device.createShaderModule({ code: cubeShaderSource });
  const cubePipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [cubePipelineLayout] }),
    vertex: { module: cubeShaderModule, entryPoint: 'vertexMain', buffers: cubeGeometry.vertexBufferLayouts },
    fragment: { module: cubeShaderModule, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const cubeBindGroup = device.device.createBindGroup({
    layout: cubePipelineLayout,
    entries: [
      { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
      { binding: 1, resource: { buffer: cubeWorldMatrices.gpuBuffer } },
      { binding: 2, resource: { buffer: indirectDrawSystem.visibleIndicesBuffer.gpuBuffer } },
    ],
  });

  let angle = 0;

  function frame() {
    angle += 0.005;
    camera.setViewMatrix(lookAt([Math.sin(angle) * 10, 2, 14], [0, 0, -6], [0, 1, 0]));
    camera.update();

    propagation.updateLocalMatrices(flattened.localMatrices);

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'propagate-cull-and-build-draws',
      writes: [
        propagation.worldBuffer,
        cubeWorldMatrices,
        hiZBuffer.texture,
        cullingPass.visibilityBuffer,
        indirectDrawSystem.indirectBuffer,
        indirectDrawSystem.visibleIndicesBuffer,
      ],
      execute: (encoder) => {
        propagation.propagate(encoder);
        encoder.copyBufferToBuffer(propagation.worldBuffer.gpuBuffer, cubeWorldBufferOffset, cubeWorldMatrices.gpuBuffer, 0, cubeCount * 64);
        hiZBuffer.build(encoder, depthTexture);
        cullingPass.cull(encoder);
        indirectDrawSystem.build(encoder);
      },
    });

    graph.addPass({
      name: 'forward',
      colorAttachments: [{
        target: CANVAS,
        clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
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
      reads: [camera.buffer, cubeWorldMatrices, indirectDrawSystem.visibleIndicesBuffer, indirectDrawSystem.indirectBuffer],
      execute: (renderPass) => {
        renderPass.setBindGroup(0, cameraBindGroup);

        renderPass.setPipeline(wallMaterial.pipeline);
        renderPass.setBindGroup(1, wallInstance.bindGroup);
        renderPass.setVertexBuffer(0, wallGeometry.attributes.position.buffer.gpuBuffer);
        renderPass.draw(wallGeometry.vertexCount);

        renderPass.setPipeline(cubePipeline);
        renderPass.setBindGroup(0, cubeBindGroup);
        renderPass.setVertexBuffer(0, cubeGeometry.attributes.position.buffer.gpuBuffer);
        renderPass.drawIndirect(indirectDrawSystem.indirectBuffer.gpuBuffer, 0);
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
