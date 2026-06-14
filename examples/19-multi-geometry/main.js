import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Camera } from '../../src/camera/Camera.js';
import { GeometryArena, interleaveStandard } from '../../src/geometry/GeometryArena.js';
import { MultiDrawSystem } from '../../src/culling/MultiDrawSystem.js';
import {
  boxData, cylinderData, coneData, sphereData, octahedronData, dodecahedronData,
} from '../../src/geometry/primitives.js';
import { perspective, lookAt, fromTranslationRotationScale } from '../../src/math/mat4.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
const statsEl = document.getElementById('stats');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const showError = (m) => { errorEl.hidden = false; errorEl.textContent = m; };

// Lit shader. The object id comes from the MultiDrawSystem's draw-slot group:
// slotToObject[slotIndex], where slotIndex is a per-draw dynamic uniform. (We
// avoid firstInstance, which is a no-op without indirect-first-instance.)
const shaderSource = /* wgsl */ `
struct Camera { viewMatrix: mat4x4f, projectionMatrix: mat4x4f, frustumPlanes: array<vec4f,6>, viewport: vec4f, };
struct Material { color: vec4f, };

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> worldMatrices: array<mat4x4f>;
@group(1) @binding(1) var<storage, read> materials: array<Material>;
@group(2) @binding(0) var<storage, read> slotToObject: array<u32>;
@group(2) @binding(1) var<uniform> slotIndex: u32;

struct VOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
) -> VOut {
  let objectIndex = slotToObject[slotIndex];
  let m = worldMatrices[objectIndex];
  var o: VOut;
  let world = m * vec4f(position, 1.0);
  o.position = camera.projectionMatrix * camera.viewMatrix * world;
  o.normal = normalize((m * vec4f(normal, 0.0)).xyz);
  o.color = materials[objectIndex].color.rgb;
  return o;
}

@fragment
fn fragmentMain(i: VOut) -> @location(0) vec4f {
  let l = normalize(vec3f(0.4, 0.8, 0.5));
  let ndotl = max(dot(normalize(i.normal), l), 0.0);
  return vec4f(i.color * (0.2 + 0.8 * ndotl), 1.0);
}
`;

try {
  const device = await createDevice();
  const context = device.getCanvasContext(canvas);
  const format = navigator.gpu.getPreferredCanvasFormat();

  const depthTexture = device.resources.createTexture({
    size: [canvas.width, canvas.height], format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 200));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  // --- Arena: pack several distinct meshes ---
  const arena = new GeometryArena(device);
  const meshDatas = [
    boxData([1, 1, 1]),
    cylinderData(0.5, 0.5, 1.2, 20),
    coneData(0.6, 1.2, 20),
    sphereData(0.6, 20, 14),
    octahedronData(0.7),
    dodecahedronData(0.6),
  ];
  const meshes = meshDatas.map((d) => {
    const { vertexData, indexData } = interleaveStandard(d);
    const alloc = arena.allocate(vertexData, indexData);
    // Local AABB from positions.
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < d.positions.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        min[a] = Math.min(min[a], d.positions[i + a]);
        max[a] = Math.max(max[a], d.positions[i + a]);
      }
    }
    return { alloc, bounds: { min, max } };
  });

  // --- Scene of N objects, each a random mesh on a grid ---
  const GRID = 16; // 16x16 = 256 objects
  const objectCount = GRID * GRID;
  const spacing = 2.4;

  const capacity = 1024;
  const worldMatricesBuffer = device.resources.createBuffer({
    size: capacity * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const materialsBuffer = device.resources.createBuffer({
    size: capacity * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const boundsBuffer = device.resources.createBuffer({
    size: capacity * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const multiDraw = new MultiDrawSystem(device, camera, worldMatricesBuffer, boundsBuffer, capacity);

  const palette = [
    [0.9, 0.45, 0.4], [0.45, 0.75, 0.55], [0.5, 0.65, 0.95],
    [0.9, 0.8, 0.4], [0.75, 0.5, 0.9], [0.4, 0.85, 0.85],
  ];

  let idx = 0;
  for (let x = 0; x < GRID; x++) {
    for (let z = 0; z < GRID; z++) {
      const meshIdx = (x * 7 + z * 13) % meshes.length;
      const mesh = meshes[meshIdx];
      const px = (x - (GRID - 1) / 2) * spacing;
      const pz = (z - (GRID - 1) / 2) * spacing;

      device.queue.writeBuffer(worldMatricesBuffer.gpuBuffer, idx * 64,
        fromTranslationRotationScale([px, 0, pz], [0, 0, 0, 1], [1, 1, 1]));
      device.queue.writeBuffer(materialsBuffer.gpuBuffer, idx * 16,
        new Float32Array([...palette[meshIdx], 1]));
      device.queue.writeBuffer(boundsBuffer.gpuBuffer, idx * 32,
        new Float32Array([...mesh.bounds.min, 0, ...mesh.bounds.max, 0]));

      multiDraw.setRecord(idx, {
        firstIndex: mesh.alloc.firstIndex,
        indexCount: mesh.alloc.indexCount,
        baseVertex: mesh.alloc.baseVertex,
        transformIndex: idx,
        layerMask: 0x1,
        flags: 0,
      });
      idx++;
    }
  }
  multiDraw.setObjectCount(objectCount);

  // --- Bind groups + pipeline ---
  const cameraBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });
  const objectBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBindGroupLayout.gpuBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });
  const objectBindGroup = device.device.createBindGroup({
    layout: objectBindGroupLayout.gpuBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: worldMatricesBuffer.gpuBuffer } },
      { binding: 1, resource: { buffer: materialsBuffer.gpuBuffer } },
    ],
  });

  const module = device.device.createShaderModule({ code: shaderSource });
  const pipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({
      bindGroupLayouts: [cameraBindGroupLayout.gpuBindGroupLayout, objectBindGroupLayout.gpuBindGroupLayout, multiDraw.drawSlotBindGroupLayout],
    }),
    vertex: { module, entryPoint: 'vertexMain', buffers: arena.vertexBufferLayouts },
    fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  let angle = 0;
  function frame() {
    angle += 0.003;
    const r = 26;
    camera.setViewMatrix(lookAt([Math.sin(angle) * r, 14, Math.cos(angle) * r], [0, 0, 0], [0, 1, 0]));
    camera.update();

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'cull-and-compact',
      writes: [multiDraw.drawArgsBuffer, multiDraw.drawCountBuffer, multiDraw.slotToObjectBuffer],
      reads: [camera.buffer, worldMatricesBuffer, boundsBuffer, multiDraw.recordBuffer],
      execute: (encoder) => multiDraw.build(encoder),
    });

    graph.addPass({
      name: 'forward',
      colorAttachments: [{ target: CANVAS, clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { target: depthTexture, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [depthTexture],
      reads: [camera.buffer, worldMatricesBuffer, materialsBuffer, multiDraw.drawArgsBuffer, multiDraw.slotToObjectBuffer],
      execute: (rp) => {
        rp.setPipeline(pipeline);
        rp.setBindGroup(0, cameraBindGroup);
        rp.setBindGroup(1, objectBindGroup);
        arena.bind(rp);
        multiDraw.drawAll(rp, 2);
      },
    });

    graph.execute();
    statsEl.textContent = `${objectCount} objects, ${meshes.length} distinct meshes, 1 arena, ${objectCount} indexed-indirect calls (args from GPU)`;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
