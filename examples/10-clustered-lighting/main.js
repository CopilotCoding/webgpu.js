import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { ClusterGrid } from '../../src/lighting/ClusterGrid.js';
import { LightCulling, MAX_LIGHTS_PER_CLUSTER } from '../../src/lighting/LightCulling.js';
import { clusterLightingStructs, clusterIndexFunction, accumulateClusterLightingFunction } from '../../src/lighting/clusterLighting.wgsl.js';
import { perspective, lookAt, multiply, fromTranslationRotationScale } from '../../src/math/mat4.js';
import { createCubeData } from './cube.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

function showError(message) {
  errorEl.hidden = false;
  errorEl.textContent = message;
}

const NEAR = 0.1;
const FAR = 60;
const LIGHT_COUNT = 64;

const cameraStruct = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
`;

// group(0): camera + clustered-lighting data (shared by all draws).
// group(1): per-instance transform + base color.
const litShaderSource = /* wgsl */ `
${cameraStruct}
${clusterLightingStructs}

struct Transform {
  modelMatrix: mat4x4f,
  baseColor: vec3f,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> clusterGridInfo: ClusterGridInfo;
@group(0) @binding(2) var<storage, read> clusterRanges: array<ClusterLightRange>;
@group(0) @binding(3) var<storage, read> lightIndices: array<u32>;
@group(0) @binding(4) var<storage, read> lights: array<PointLight>;

@group(1) @binding(0) var<uniform> transform: Transform;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewZ: f32,
};

@vertex
fn vertexMain(@location(0) position: vec3f, @location(1) normal: vec3f) -> VertexOutput {
  let world = transform.modelMatrix * vec4f(position, 1.0);
  let viewPos = camera.viewMatrix * world;

  var out: VertexOutput;
  out.position = camera.projectionMatrix * viewPos;
  out.worldPos = world.xyz;
  out.normal = (transform.modelMatrix * vec4f(normal, 0.0)).xyz;
  out.viewZ = viewPos.z;
  return out;
}

${clusterIndexFunction}
${accumulateClusterLightingFunction}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let normal = normalize(in.normal);
  let ambient = vec3f(0.03, 0.03, 0.04);

  let lighting = accumulateClusterLighting(in.worldPos, normal, in.position, clusterGridInfo, in.viewZ);

  let color = transform.baseColor * (ambient + lighting);
  return vec4f(color, 1.0);
}
`;

// Tiny unlit cubes mark each light's current position.
const lightMarkerShaderSource = /* wgsl */ `
${cameraStruct}

struct Transform {
  modelMatrix: mat4x4f,
  baseColor: vec3f,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> transform: Transform;

@vertex
fn vertexMain(@location(0) position: vec3f, @location(1) normal: vec3f) -> @builtin(position) vec4f {
  return camera.projectionMatrix * camera.viewMatrix * transform.modelMatrix * vec4f(position, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return vec4f(transform.baseColor, 1.0);
}
`;

function hsv(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
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

  const camera = new Camera(device);
  const projectionMatrix = perspective(Math.PI / 4, canvas.width / canvas.height, NEAR, FAR);
  camera.setProjectionMatrix(projectionMatrix);
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const clusterGrid = new ClusterGrid(device, { clusterCountX: 16, clusterCountY: 9, clusterCountZ: 24 });
  clusterGrid.setProjection(projectionMatrix, canvas.width, canvas.height, NEAR, FAR);

  const lightCulling = new LightCulling(device, clusterGrid, LIGHT_COUNT);

  // group(0) bind group layout shared by the lit pipeline and the marker pipeline's camera binding.
  const sceneBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });

  const cameraOnlyBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });

  // group(1): per-instance transform (modelMatrix + baseColor), 80 bytes.
  const transformBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });

  const sceneBindGroup = device.device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
      { binding: 1, resource: { buffer: clusterGrid.gridInfoBuffer.gpuBuffer } },
      { binding: 2, resource: { buffer: lightCulling.clusterRangesBuffer.gpuBuffer } },
      { binding: 3, resource: { buffer: lightCulling.lightIndicesBuffer.gpuBuffer } },
      { binding: 4, resource: { buffer: lightCulling.lightsBuffer.gpuBuffer } },
    ],
  });

  const cameraOnlyBindGroup = device.device.createBindGroup({
    layout: cameraOnlyBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
    ],
  });

  // --- Geometry ---
  const cubeData = createCubeData();
  const cubeGeometry = new Geometry(device, {
    attributes: {
      position: { format: 'float32x3', data: cubeData.positions },
      normal: { format: 'float32x3', data: cubeData.normals },
    },
  });

  // --- Pipelines ---
  const litShaderModule = device.device.createShaderModule({ code: litShaderSource });
  const litPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [sceneBindGroupLayout, transformBindGroupLayout] }),
    vertex: { module: litShaderModule, entryPoint: 'vertexMain', buffers: cubeGeometry.vertexBufferLayouts },
    fragment: { module: litShaderModule, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const markerShaderModule = device.device.createShaderModule({ code: lightMarkerShaderSource });
  const markerPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [cameraOnlyBindGroupLayout, transformBindGroupLayout] }),
    vertex: { module: markerShaderModule, entryPoint: 'vertexMain', buffers: cubeGeometry.vertexBufferLayouts },
    fragment: { module: markerShaderModule, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  // --- Scene objects: a grid of cubes on the floor + a transform buffer per instance ---
  const TRANSFORM_SIZE = 80; // mat4x4f (64) + vec3f baseColor + pad (16)

  function createInstance(baseColor) {
    const buffer = device.resources.createBuffer({
      size: TRANSFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.device.createBindGroup({
      layout: transformBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: buffer.gpuBuffer } }],
    });
    const data = new Float32Array(20);
    data.set(baseColor, 16);
    return { buffer, bindGroup, data };
  }

  function writeInstance(instance, modelMatrix) {
    instance.data.set(modelMatrix, 0);
    device.queue.writeBuffer(instance.buffer.gpuBuffer, 0, instance.data);
  }

  const cubeInstances = [];
  const gridSize = 9;
  const spacing = 3.0;
  for (let x = 0; x < gridSize; x++) {
    for (let z = 0; z < gridSize; z++) {
      const px = (x - (gridSize - 1) / 2) * spacing;
      const pz = (z - (gridSize - 1) / 2) * spacing - 10;
      const height = 1.0 + ((x * 7 + z * 13) % 5) * 0.6;
      const instance = createInstance([0.7, 0.7, 0.75]);
      const modelMatrix = fromTranslationRotationScale(
        [px, height / 2 - 1, pz],
        [0, 0, 0, 1],
        [1, height, 1],
      );
      writeInstance(instance, modelMatrix);
      cubeInstances.push(instance);
    }
  }

  // Floor: one large flat cube.
  const floorInstance = createInstance([0.4, 0.4, 0.45]);
  writeInstance(floorInstance, fromTranslationRotationScale([0, -1.5, -10], [0, 0, 0, 1], [30, 0.5, 30]));

  // --- Light markers + animated light state ---
  const lightMarkerInstances = [];
  const lightColors = [];
  const lightRadii = [];
  const lightPhase = [];
  for (let i = 0; i < LIGHT_COUNT; i++) {
    const color = hsv(i / LIGHT_COUNT, 0.8, 1.0);
    lightColors.push(color);
    lightRadii.push(4 + Math.random() * 3);
    lightPhase.push(Math.random() * Math.PI * 2);
    const instance = createInstance(color);
    writeInstance(instance, fromTranslationRotationScale([0, 0, 0], [0, 0, 0, 1], [0.15, 0.15, 0.15]));
    lightMarkerInstances.push(instance);
  }

  const lightData = new Float32Array(LIGHT_COUNT * 8); // PointLight: position(3)+radius(1)+color(3)+intensity(1)

  let angle = 0;

  function frame() {
    angle += 0.004;

    const eye = [Math.sin(angle) * 22, 10, Math.cos(angle) * 22 - 10];
    camera.setViewMatrix(lookAt(eye, [0, 0, -10], [0, 1, 0]));
    camera.update();

    const time = performance.now() * 0.001;
    for (let i = 0; i < LIGHT_COUNT; i++) {
      const phase = lightPhase[i];
      const ring = 4 + (i % 5) * 3;
      const px = Math.sin(time * 0.5 + phase) * ring;
      const pz = Math.cos(time * 0.4 + phase * 1.3) * ring - 10;
      const py = 1 + Math.sin(time * 0.7 + phase * 2.0) * 2.5 + 2;

      lightData.set([px, py, pz, lightRadii[i]], i * 8);
      lightData.set([...lightColors[i], 6.0], i * 8 + 4);

      writeInstance(lightMarkerInstances[i], fromTranslationRotationScale([px, py, pz], [0, 0, 0, 1], [0.15, 0.15, 0.15]));
    }
    lightCulling.setLights(lightData);
    lightCulling.setView(camera.viewMatrix, LIGHT_COUNT);

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'cluster-and-cull-lights',
      writes: [clusterGrid.clusterBoundsBuffer, lightCulling.clusterRangesBuffer],
      execute: (encoder) => {
        clusterGrid.build(encoder);
        lightCulling.cull(encoder);
      },
    });

    graph.addPass({
      name: 'forward',
      colorAttachments: [{
        target: CANVAS,
        clearValue: { r: 0.01, g: 0.01, b: 0.02, a: 1.0 },
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
      reads: [camera.buffer, clusterGrid.gridInfoBuffer, lightCulling.clusterRangesBuffer, lightCulling.lightIndicesBuffer, lightCulling.lightsBuffer],
      execute: (renderPass) => {
        renderPass.setPipeline(litPipeline);
        renderPass.setBindGroup(0, sceneBindGroup);
        renderPass.setVertexBuffer(0, cubeGeometry.attributes.position.buffer.gpuBuffer);
        renderPass.setVertexBuffer(1, cubeGeometry.attributes.normal.buffer.gpuBuffer);

        renderPass.setBindGroup(1, floorInstance.bindGroup);
        renderPass.draw(cubeGeometry.vertexCount);

        for (const instance of cubeInstances) {
          renderPass.setBindGroup(1, instance.bindGroup);
          renderPass.draw(cubeGeometry.vertexCount);
        }

        renderPass.setPipeline(markerPipeline);
        renderPass.setBindGroup(0, cameraOnlyBindGroup);
        for (const instance of lightMarkerInstances) {
          renderPass.setBindGroup(1, instance.bindGroup);
          renderPass.draw(cubeGeometry.vertexCount);
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
