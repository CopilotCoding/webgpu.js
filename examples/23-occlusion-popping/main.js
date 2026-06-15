// Example 23 — Occlusion culling in a CITY scene.
//
// A grid of buildings (boxes of varying height) drawn through the real
// GPU-driven draw path: IndirectDrawSystem compacts a drawIndirect arg list and
// draws ONLY the objects whose VISIBLE_OCCLUSION bit is set. The camera flies low
// through the streets, so near buildings naturally occlude the rows behind them —
// a realistic occlusion-culling workload, not a contrived wall.
//
// Occlusion is tested against the PREVIOUS frame's Hi-Z. Press [Space] to toggle
// it on the draw path and compare:
//   - ON:  far buildings hidden behind nearer ones are not drawn ("cubes drawn"
//          drops sharply). Under fast camera motion, buildings right at an
//          occluder edge can pop for a frame (the 1-frame Hi-Z lag) — which is
//          why occlusion ships OFF by default for the draw path.
//   - OFF: every frustum-visible building draws (frustum-only). Solid, higher
//          draw count.
import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { SceneNode } from '../../src/scene/SceneNode.js';
import { flattenSceneGraph } from '../../src/scene/flattenSceneGraph.js';
import { TransformPropagation } from '../../src/scene/TransformPropagation.js';
import { HiZBuffer } from '../../src/culling/HiZBuffer.js';
import { CullingPass } from '../../src/culling/CullingPass.js';
import { IndirectDrawSystem } from '../../src/culling/IndirectDrawSystem.js';
import { boxData } from '../../src/geometry/primitives.js';
import { OrbitControls } from '../../src/controls/OrbitControls.js';
import { perspective } from '../../src/math/mat4.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
const panelEl = document.getElementById('panel');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const showError = (m) => { errorEl.hidden = false; errorEl.textContent = m; };

const cameraStruct = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
`;

// Ground plane — a plain Material-less inline pipeline (one tinted quad).
const groundShaderSource = /* wgsl */ `
${cameraStruct}
@group(0) @binding(0) var<uniform> camera: Camera;
@vertex
fn vertexMain(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return camera.projectionMatrix * camera.viewMatrix * vec4f(position, 1.0);
}
@fragment
fn fragmentMain() -> @location(0) vec4f { return vec4f(0.06, 0.07, 0.09, 1.0); }
`;

// Buildings drawn via one drawIndirect: instance_index ranges over
// [0, visibleCount); visibleIndices maps it back to the original object index.
// Simple top-down-ish shading from the box normal so the city reads with depth.
const buildingShaderSource = /* wgsl */ `
${cameraStruct}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> visibleIndices: array<u32>;

struct VOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) @interpolate(flat) objectIndex: u32,
};

@vertex
fn vertexMain(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f, @builtin(instance_index) instanceIndex: u32) -> VOut {
  let objectIndex = visibleIndices[instanceIndex];
  let m = worldMatrices[objectIndex];
  var out: VOut;
  out.position = camera.projectionMatrix * camera.viewMatrix * m * vec4f(position, 1.0);
  out.normal = normalize((m * vec4f(normal, 0.0)).xyz);
  out.objectIndex = objectIndex;
  return out;
}

@fragment
fn fragmentMain(in: VOut) -> @location(0) vec4f {
  // Per-building hue so individual buildings are distinguishable.
  let h = f32(in.objectIndex) * 0.0739;
  let base = vec3f(0.5 + 0.5 * sin(h * 6.28), 0.5 + 0.5 * sin(h * 6.28 + 2.09), 0.5 + 0.5 * sin(h * 6.28 + 4.18));
  let ndl = max(dot(normalize(in.normal), normalize(vec3f(0.4, 0.9, 0.3))), 0.0);
  return vec4f(base * (0.25 + 0.75 * ndl), 1.0);
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

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 3, canvas.width / canvas.height, 0.1, 200));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const cameraBGL = device.device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });
  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBGL, entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });

  // --- City layout ---
  const GRID = 12;          // 12x12 = 144 buildings
  const SPACING = 6;        // street spacing
  const CITY = (GRID - 1) * SPACING;

  // Ground plane covering the city.
  const g = CITY * 0.6 + SPACING;
  const groundGeometry = new Geometry(device, {
    attributes: { position: { format: 'float32x3', data: new Float32Array([
      -g, 0, -g,  g, 0, -g,  g, 0, g,
      -g, 0, -g,  g, 0, g,  -g, 0, g,
    ]) } },
  });
  const groundModule = device.device.createShaderModule({ code: groundShaderSource });
  const groundPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [cameraBGL] }),
    vertex: { module: groundModule, entryPoint: 'vertexMain', buffers: groundGeometry.vertexBufferLayouts },
    fragment: { module: groundModule, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  // --- Buildings: unit box scaled per-building (height varies) ---
  const boxGeometry = new Geometry(device, {
    attributes: (() => {
      const d = boxData([1, 1, 1]);
      return {
        position: { format: 'float32x3', data: d.positions },
        normal: { format: 'float32x3', data: d.normals },
        uv: { format: 'float32x2', data: d.uvs },
      };
    })(),
  });
  const boxBounds = boxGeometry.computeBounds(); // unit box; world matrix carries the scale

  const root = new SceneNode('root');
  for (let x = 0; x < GRID; x++) {
    for (let z = 0; z < GRID; z++) {
      const h = 4 + ((x * 7 + z * 13) % 7) * 3; // 4..22 tall
      const w = 2.2 + ((x * 3 + z * 5) % 3) * 0.6;
      const node = new SceneNode(`b-${x}-${z}`);
      node.setPosition((x - (GRID - 1) / 2) * SPACING, h / 2, (z - (GRID - 1) / 2) * SPACING);
      node.setScale(w, h, w);
      root.add(node);
    }
  }

  const flattened = flattenSceneGraph(root);
  const propagation = new TransformPropagation(device, flattened);
  const count = flattened.nodes.length - 1;       // exclude the identity root
  const boundsArray = Array.from({ length: count }, () => boxBounds);
  const worldBufferOffset = 16 * 4;               // skip the root's mat4x4f

  const worldMatrices = device.resources.createBuffer({
    size: count * 64,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const cullingPass = new CullingPass(device, camera, worldMatrices, boundsArray, hiZBuffer);
  const indirectDraw = new IndirectDrawSystem(device, cullingPass, boxGeometry.vertexCount);

  let occlusionOn = true;
  cullingPass.setOcclusionEnabled(occlusionOn);
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      occlusionOn = !occlusionOn;
      cullingPass.setOcclusionEnabled(occlusionOn);
    }
  });

  // Orbit camera: drag to rotate, wheel to zoom, right-drag to pan. Orbits the
  // city center at a low polar angle so the view skims across the rooftops and
  // near buildings occlude the rows behind them.
  const controls = new OrbitControls(canvas, {
    target: [0, 6, 0],
    distance: CITY * 0.7,
    minDistance: 8,
    maxDistance: CITY * 1.5,
    polar: 1.25,        // near-horizontal: look across the city, not down on it
  });

  // Drawn-count readback from CullingPass.visibilityBuffer (has COPY_SRC).
  const VISIBLE_OCCLUSION = 2;
  const visStaging = device.resources.createBuffer({
    size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  let readbackBusy = false;
  let drawnCount = count;

  const buildingBGL = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const buildingModule = device.device.createShaderModule({ code: buildingShaderSource });
  const buildingPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [buildingBGL] }),
    vertex: { module: buildingModule, entryPoint: 'vertexMain', buffers: boxGeometry.vertexBufferLayouts },
    fragment: { module: buildingModule, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  const buildingBindGroup = device.device.createBindGroup({
    layout: buildingBGL,
    entries: [
      { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
      { binding: 1, resource: { buffer: worldMatrices.gpuBuffer } },
      { binding: 2, resource: { buffer: indirectDraw.visibleIndicesBuffer.gpuBuffer } },
    ],
  });

  function frame() {
    camera.setViewMatrix(controls.update());
    camera.update();
    propagation.updateLocalMatrices(flattened.localMatrices);

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'propagate-cull-build',
      writes: [propagation.worldBuffer, worldMatrices, hiZBuffer.texture, cullingPass.visibilityBuffer, indirectDraw.indirectBuffer, indirectDraw.visibleIndicesBuffer],
      execute: (encoder) => {
        propagation.propagate(encoder);
        encoder.copyBufferToBuffer(propagation.worldBuffer.gpuBuffer, worldBufferOffset, worldMatrices.gpuBuffer, 0, count * 64);
        hiZBuffer.build(encoder, depthTexture);
        cullingPass.cull(encoder);
        indirectDraw.build(encoder);
        if (!readbackBusy) encoder.copyBufferToBuffer(cullingPass.visibilityBuffer.gpuBuffer, 0, visStaging.gpuBuffer, 0, count * 4);
      },
    });

    graph.addPass({
      name: 'forward',
      colorAttachments: [{ target: CANVAS, clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { target: depthTexture, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [depthTexture],
      reads: [camera.buffer, worldMatrices, indirectDraw.visibleIndicesBuffer, indirectDraw.indirectBuffer],
      execute: (rp) => {
        rp.setBindGroup(0, cameraBindGroup);
        rp.setPipeline(groundPipeline);
        rp.setVertexBuffer(0, groundGeometry.attributes.position.buffer.gpuBuffer);
        rp.draw(groundGeometry.vertexCount);

        rp.setPipeline(buildingPipeline);
        rp.setBindGroup(0, buildingBindGroup);
        rp.setVertexBuffer(0, boxGeometry.attributes.position.buffer.gpuBuffer);
        rp.setVertexBuffer(1, boxGeometry.attributes.normal.buffer.gpuBuffer);
        rp.setVertexBuffer(2, boxGeometry.attributes.uv.buffer.gpuBuffer);
        rp.drawIndirect(indirectDraw.indirectBuffer.gpuBuffer, 0);
      },
    });

    graph.execute();

    if (!readbackBusy) {
      readbackBusy = true;
      visStaging.gpuBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const bits = new Uint32Array(visStaging.gpuBuffer.getMappedRange());
        let n = 0;
        for (let i = 0; i < count; i++) if (bits[i] & VISIBLE_OCCLUSION) n++;
        drawnCount = n;
        visStaging.gpuBuffer.unmap();
        readbackBusy = false;
      }).catch(() => { readbackBusy = false; });
    }

    panelEl.textContent =
      `occlusion culling   ${occlusionOn ? 'ON  (drops hidden buildings)' : 'OFF (frustum only)'}\n` +
      `buildings total     ${count}\n` +
      `buildings drawn     ${drawnCount}\n` +
      (occlusionOn ? 'rows behind nearer buildings are skipped (Hi-Z, 1-frame lag)' : 'every frustum-visible building draws');

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
