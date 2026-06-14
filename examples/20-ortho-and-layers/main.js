import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { PerspectiveCamera } from '../../src/camera/PerspectiveCamera.js';
import { OrthographicCamera } from '../../src/camera/OrthographicCamera.js';
import { GeometryArena, interleaveStandard } from '../../src/geometry/GeometryArena.js';
import { MultiDrawSystem } from '../../src/culling/MultiDrawSystem.js';
import { boxData, cylinderData, sphereData, octahedronData } from '../../src/geometry/primitives.js';
import { fromTranslationRotationScale } from '../../src/math/mat4.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const showError = (m) => { errorEl.hidden = false; errorEl.textContent = m; };

const LAYER_WORLD = 0x1;   // ordinary scene objects (main view)
const LAYER_MINIMAP = 0x2; // markers shown only in the ortho inset

const shaderSource = /* wgsl */ `
struct Camera { viewMatrix: mat4x4f, projectionMatrix: mat4x4f, frustumPlanes: array<vec4f,6>, viewport: vec4f, };
struct Material { color: vec4f, };
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> worldMatrices: array<mat4x4f>;
@group(1) @binding(1) var<storage, read> materials: array<Material>;
@group(2) @binding(0) var<storage, read> slotToObject: array<u32>;
@group(2) @binding(1) var<uniform> slotIndex: u32;
struct VOut { @builtin(position) position: vec4f, @location(0) normal: vec3f, @location(1) color: vec3f, };
@vertex fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f) -> VOut {
  let oi = slotToObject[slotIndex];
  let m = worldMatrices[oi];
  var o: VOut;
  o.position = camera.projectionMatrix * camera.viewMatrix * m * vec4f(p, 1.0);
  o.normal = normalize((m * vec4f(n, 0.0)).xyz);
  o.color = materials[oi].color.rgb;
  return o;
}
@fragment fn fragmentMain(i: VOut) -> @location(0) vec4f {
  let l = normalize(vec3f(0.4, 0.8, 0.5));
  return vec4f(i.color * (0.25 + 0.75 * max(dot(normalize(i.normal), l), 0.0)), 1.0);
}
`;

try {
  const device = await createDevice();
  const context = device.getCanvasContext(canvas);
  const format = navigator.gpu.getPreferredCanvasFormat();

  const W = canvas.width, H = canvas.height;
  // Two depth textures: full-screen main + the small inset region.
  const depthMain = device.resources.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });

  const mainCam = new PerspectiveCamera(device, { fov: Math.PI / 4, aspect: W / H, near: 0.1, far: 200 });
  mainCam.setViewport(0, 0, W, H);

  const FIELD = 18;
  const insetCam = new OrthographicCamera(device, { near: 0.1, far: 80 });
  insetCam.setExtent(FIELD);
  insetCam.position = [0, 40, 0];
  insetCam.target = [0, 0, 0];
  insetCam.up = [0, 0, -1];

  // --- Arena ---
  const arena = new GeometryArena(device);
  const make = (d) => {
    const { vertexData, indexData } = interleaveStandard(d);
    const alloc = arena.allocate(vertexData, indexData);
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < d.positions.length; i += 3) for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], d.positions[i + a]); max[a] = Math.max(max[a], d.positions[i + a]);
    }
    return { alloc, bounds: { min, max } };
  };
  const box = make(boxData([1, 1, 1]));
  const cyl = make(cylinderData(0.4, 0.4, 1, 16));
  const sph = make(sphereData(0.5, 16, 12));
  const marker = make(octahedronData(0.6));

  const capacity = 512;
  const worldBuf = device.resources.createBuffer({ size: capacity * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const matBuf = device.resources.createBuffer({ size: capacity * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const boundsBuf = device.resources.createBuffer({ size: capacity * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // Main system owns the records; inset shares them + the slot layout and
  // culls the same scene with a different camera/layer mask.
  const mainDraw = new MultiDrawSystem(device, mainCam, worldBuf, boundsBuf, capacity);
  const insetDraw = new MultiDrawSystem(device, insetCam, worldBuf, boundsBuf, capacity, {
    recordBuffer: mainDraw.recordBuffer,
    drawSlotBindGroupLayout: mainDraw.drawSlotBindGroupLayout,
  });
  mainDraw.setCameraLayerMask(LAYER_WORLD);
  insetDraw.setCameraLayerMask(LAYER_MINIMAP);

  const worldMeshes = [box, cyl, sph];
  const worldColors = [[0.8, 0.5, 0.4], [0.45, 0.75, 0.55], [0.5, 0.65, 0.95]];

  // Grid of ordinary objects (LAYER_WORLD only).
  const GRID = 10;
  let idx = 0;
  for (let x = 0; x < GRID; x++) {
    for (let z = 0; z < GRID; z++) {
      const mi = (x + z) % worldMeshes.length;
      const mesh = worldMeshes[mi];
      const px = (x - (GRID - 1) / 2) * 3, pz = (z - (GRID - 1) / 2) * 3;
      device.queue.writeBuffer(worldBuf.gpuBuffer, idx * 64, fromTranslationRotationScale([px, 0, pz], [0, 0, 0, 1], [1, 1, 1]));
      device.queue.writeBuffer(matBuf.gpuBuffer, idx * 16, new Float32Array([...worldColors[mi], 1]));
      device.queue.writeBuffer(boundsBuf.gpuBuffer, idx * 32, new Float32Array([...mesh.bounds.min, 0, ...mesh.bounds.max, 0]));
      mainDraw.setRecord(idx, { firstIndex: mesh.alloc.firstIndex, indexCount: mesh.alloc.indexCount, baseVertex: mesh.alloc.baseVertex, transformIndex: idx, layerMask: LAYER_WORLD });
      idx++;
    }
  }
  // A few tall markers, present in BOTH layers so they show in main and inset.
  const markerSpots = [[-12, -12], [12, 12], [0, 0], [12, -12], [-12, 12]];
  for (const [px, pz] of markerSpots) {
    device.queue.writeBuffer(worldBuf.gpuBuffer, idx * 64, fromTranslationRotationScale([px, 1.5, pz], [0, 0, 0, 1], [1.5, 3, 1.5]));
    device.queue.writeBuffer(matBuf.gpuBuffer, idx * 16, new Float32Array([1.0, 0.85, 0.2, 1]));
    device.queue.writeBuffer(boundsBuf.gpuBuffer, idx * 32, new Float32Array([...marker.bounds.min, 0, ...marker.bounds.max, 0]));
    mainDraw.setRecord(idx, { firstIndex: marker.alloc.firstIndex, indexCount: marker.alloc.indexCount, baseVertex: marker.alloc.baseVertex, transformIndex: idx, layerMask: LAYER_WORLD | LAYER_MINIMAP });
    idx++;
  }
  const objectCount = idx;
  mainDraw.setObjectCount(objectCount);
  insetDraw.setObjectCount(objectCount);

  // --- Pipeline + bind groups (shared by both views) ---
  const cameraBGL = device.resources.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }] });
  const objectBGL = device.resources.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const mainCamBG = device.device.createBindGroup({ layout: cameraBGL.gpuBindGroupLayout, entries: [{ binding: 0, resource: { buffer: mainCam.buffer.gpuBuffer } }] });
  const insetCamBG = device.device.createBindGroup({ layout: cameraBGL.gpuBindGroupLayout, entries: [{ binding: 0, resource: { buffer: insetCam.buffer.gpuBuffer } }] });
  const objectBG = device.device.createBindGroup({
    layout: objectBGL.gpuBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: worldBuf.gpuBuffer } },
      { binding: 1, resource: { buffer: matBuf.gpuBuffer } },
    ],
  });

  const module = device.device.createShaderModule({ code: shaderSource });
  const pipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [cameraBGL.gpuBindGroupLayout, objectBGL.gpuBindGroupLayout, mainDraw.drawSlotBindGroupLayout] }),
    vertex: { module, entryPoint: 'vertexMain', buffers: arena.vertexBufferLayouts },
    fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const insetSize = Math.floor(Math.min(W, H) * 0.28);
  const insetMargin = 16;

  let angle = 0;
  function frame() {
    angle += 0.004;
    const r = 30;
    mainCam.position = [Math.sin(angle) * r, 16, Math.cos(angle) * r];
    mainCam.target = [0, 0, 0];
    mainCam.update();
    insetCam.update();

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'cull-main',
      writes: [mainDraw.drawArgsBuffer, mainDraw.drawCountBuffer, mainDraw.slotToObjectBuffer],
      reads: [mainCam.buffer, worldBuf, boundsBuf, mainDraw.recordBuffer],
      execute: (encoder) => mainDraw.build(encoder),
    });
    graph.addPass({
      name: 'cull-inset',
      writes: [insetDraw.drawArgsBuffer, insetDraw.drawCountBuffer, insetDraw.slotToObjectBuffer],
      reads: [insetCam.buffer, worldBuf, boundsBuf, insetDraw.recordBuffer],
      execute: (encoder) => insetDraw.build(encoder),
    });

    // Main view: clears the canvas + depth, draws LAYER_WORLD.
    graph.addPass({
      name: 'main-view',
      colorAttachments: [{ target: CANVAS, clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { target: depthMain, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [depthMain],
      reads: [mainCam.buffer, worldBuf, matBuf, mainDraw.drawArgsBuffer, mainDraw.slotToObjectBuffer],
      execute: (rp) => {
        rp.setViewport(0, 0, W, H, 0, 1);
        rp.setPipeline(pipeline);
        rp.setBindGroup(0, mainCamBG);
        rp.setBindGroup(1, objectBG);
        arena.bind(rp);
        mainDraw.drawAll(rp, 2);
      },
    });

    // Inset: loads (doesn't clear) the canvas, scissors a corner, clears only
    // that region's depth, draws LAYER_MINIMAP through the ortho camera.
    graph.addPass({
      name: 'inset-view',
      colorAttachments: [{ target: CANVAS, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { target: depthMain, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [depthMain],
      reads: [insetCam.buffer, worldBuf, matBuf, insetDraw.drawArgsBuffer, insetDraw.slotToObjectBuffer],
      execute: (rp) => {
        const x = insetMargin, y = insetMargin;
        rp.setViewport(x, y, insetSize, insetSize, 0, 1);
        rp.setScissorRect(x, y, insetSize, insetSize);
        rp.setPipeline(pipeline);
        rp.setBindGroup(0, insetCamBG);
        rp.setBindGroup(1, objectBG);
        arena.bind(rp);
        insetDraw.drawAll(rp, 2);
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
