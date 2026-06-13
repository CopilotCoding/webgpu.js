import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { ClusterGrid } from '../../src/lighting/ClusterGrid.js';
import { LightCulling } from '../../src/lighting/LightCulling.js';
import { clusterLightingStructs, clusterIndexFunction, accumulateClusterLightingFunction } from '../../src/lighting/clusterLighting.wgsl.js';
import { ShadowMap } from '../../src/lighting/ShadowMap.js';
import { shadowMapStruct, sampleShadowFunction } from '../../src/lighting/shadowMap.wgsl.js';
import { MipGenerator } from '../../src/textures/MipGenerator.js';
import { FullscreenPass } from '../../src/post/FullscreenPass.js';
import {
  brightPassFragmentSource, brightPassLayoutEntries,
  blurFragmentSource, blurLayoutEntries,
  compositeFragmentSource, compositeLayoutEntries,
} from '../../src/post/postEffects.wgsl.js';
import { perspective, lookAt, fromTranslationRotationScale } from '../../src/math/mat4.js';
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
const SHADOW_MAP_SIZE = 2048;
const HDR_FORMAT = 'rgba16float';
const TEXTURE_SIZE = 256;
const TEXTURE_LAYERS = 4;
const TEXTURE_MIP_COUNT = Math.log2(TEXTURE_SIZE) + 1; // 9

// Directional light: points down and slightly to the side.
const LIGHT_DIRECTION = [0.4, -0.7, 0.35];

const cameraStruct = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
`;

// group(0): camera + clustered-lighting data + shadow map + albedo texture
// array (shared by all lit draws).
// group(1): per-instance transform + base color + texture layer + uv scale.
const litShaderSource = /* wgsl */ `
${cameraStruct}
${clusterLightingStructs}
${shadowMapStruct}

struct Transform {
  modelMatrix: mat4x4f,
  baseColor: vec3f,
  textureLayer: f32,
  uvScale: vec2f,
  _pad0: vec2f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> clusterGridInfo: ClusterGridInfo;
@group(0) @binding(2) var<storage, read> clusterRanges: array<ClusterLightRange>;
@group(0) @binding(3) var<storage, read> lightIndices: array<u32>;
@group(0) @binding(4) var<storage, read> lights: array<PointLight>;
@group(0) @binding(5) var<uniform> shadowMap: ShadowMap;
@group(0) @binding(6) var shadowDepth: texture_depth_2d;
@group(0) @binding(7) var shadowSampler: sampler_comparison;
@group(0) @binding(8) var albedoTexture: texture_2d_array<f32>;
@group(0) @binding(9) var albedoSampler: sampler;

@group(1) @binding(0) var<uniform> transform: Transform;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewZ: f32,
  @location(3) uv: vec2f,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
) -> VertexOutput {
  let world = transform.modelMatrix * vec4f(position, 1.0);
  let viewPos = camera.viewMatrix * world;

  var out: VertexOutput;
  out.position = camera.projectionMatrix * viewPos;
  out.worldPos = world.xyz;
  out.normal = (transform.modelMatrix * vec4f(normal, 0.0)).xyz;
  out.viewZ = viewPos.z;
  out.uv = uv * transform.uvScale;
  return out;
}

${clusterIndexFunction}
${accumulateClusterLightingFunction}
${sampleShadowFunction}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let normal = normalize(in.normal);
  let ambient = vec3f(0.03, 0.03, 0.04);

  let albedo = textureSample(albedoTexture, albedoSampler, in.uv, i32(transform.textureLayer)).rgb;

  let shadow = sampleShadow(in.worldPos, normal);
  let sunDir = normalize(-shadowMap.lightDirection);
  let sunNdotl = max(dot(normal, sunDir), 0.0);
  let sun = vec3f(1.0, 0.96, 0.88) * sunNdotl * shadow * 1.4;

  let lighting = accumulateClusterLighting(in.worldPos, normal, in.position, clusterGridInfo, in.viewZ);

  let color = albedo * transform.baseColor * (ambient + sun + lighting);
  return vec4f(color, 1.0);
}
`;

// Light markers render well above 1.0 so the bright pass picks them up.
const lightMarkerShaderSource = /* wgsl */ `
${cameraStruct}

struct Transform {
  modelMatrix: mat4x4f,
  baseColor: vec3f,
  textureLayer: f32,
  uvScale: vec2f,
  _pad0: vec2f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> transform: Transform;

@vertex
fn vertexMain(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> @builtin(position) vec4f {
  return camera.projectionMatrix * camera.viewMatrix * transform.modelMatrix * vec4f(position, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return vec4f(transform.baseColor * 8.0, 1.0);
}
`;

// Depth-only pass: transforms vertices by the shadow map's view-projection
// matrix.
const shadowDepthShaderSource = /* wgsl */ `
${shadowMapStruct}

struct Transform {
  modelMatrix: mat4x4f,
  baseColor: vec3f,
  textureLayer: f32,
  uvScale: vec2f,
  _pad0: vec2f,
};

@group(0) @binding(0) var<uniform> shadowMap: ShadowMap;
@group(1) @binding(0) var<uniform> transform: Transform;

@vertex
fn vertexMain(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> @builtin(position) vec4f {
  let world = transform.modelMatrix * vec4f(position, 1.0);
  return shadowMap.viewProjectionMatrix * world;
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

// --- Procedural texture layers, drawn with 2D canvas ---
// layer 0: brick, layer 1: window grid, layer 2: concrete panels, layer 3: pavement.
function drawTextureLayer(ctx, layer, size) {
  ctx.clearRect(0, 0, size, size);

  if (layer === 0) {
    // Brick.
    ctx.fillStyle = '#9a4a3a';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#c0c0b8';
    const rows = 8, brickH = size / rows;
    for (let r = 0; r < rows; r++) {
      ctx.fillRect(0, r * brickH - 2, size, 4);
      const offset = (r % 2) * (size / 8);
      for (let c = 0; c < 4; c++) {
        ctx.fillRect(((c * size) / 4 + offset) % size - 2, r * brickH, 4, brickH);
      }
    }
  } else if (layer === 1) {
    // Window grid: dark facade, warm lit windows.
    ctx.fillStyle = '#23262e';
    ctx.fillRect(0, 0, size, size);
    const cells = 8, cell = size / cells;
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        const lit = (x * 31 + y * 17) % 7 < 3;
        ctx.fillStyle = lit ? '#ffd98a' : '#11131a';
        ctx.fillRect(x * cell + cell * 0.2, y * cell + cell * 0.2, cell * 0.6, cell * 0.6);
      }
    }
  } else if (layer === 2) {
    // Concrete panels.
    ctx.fillStyle = '#8d8d92';
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = '#6b6b70';
    ctx.lineWidth = 3;
    const panels = 4, panel = size / panels;
    for (let i = 0; i <= panels; i++) {
      ctx.beginPath(); ctx.moveTo(i * panel, 0); ctx.lineTo(i * panel, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * panel); ctx.lineTo(size, i * panel); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(0,0,0,0.12)';
    for (let i = 0; i < 60; i++) {
      ctx.fillRect((i * 37) % size, (i * 91) % size, 6, 6);
    }
  } else {
    // Pavement checker.
    const cells = 8, cell = size / cells;
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        ctx.fillStyle = (x + y) % 2 === 0 ? '#7c7f86' : '#5e6168';
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
    ctx.strokeStyle = '#46484e';
    ctx.lineWidth = 2;
    for (let i = 0; i <= cells; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke();
    }
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

  // --- Albedo texture array: mip 0 uploaded from 2D canvas, the rest
  // generated in compute ---
  const albedoTexture = device.resources.createTexture({
    size: [TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_LAYERS],
    format: 'rgba8unorm',
    mipLevelCount: TEXTURE_MIP_COUNT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const scratch = document.createElement('canvas');
  scratch.width = TEXTURE_SIZE;
  scratch.height = TEXTURE_SIZE;
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  for (let layer = 0; layer < TEXTURE_LAYERS; layer++) {
    drawTextureLayer(ctx, layer, TEXTURE_SIZE);
    const pixels = ctx.getImageData(0, 0, TEXTURE_SIZE, TEXTURE_SIZE).data;
    device.queue.writeTexture(
      { texture: albedoTexture.gpuTexture, origin: [0, 0, layer] },
      pixels,
      { bytesPerRow: TEXTURE_SIZE * 4, rowsPerImage: TEXTURE_SIZE },
      [TEXTURE_SIZE, TEXTURE_SIZE, 1],
    );
  }

  // One-time compute pass filling mips 1..N for every layer.
  const mipGenerator = new MipGenerator(device);
  {
    const encoder = device.device.createCommandEncoder();
    mipGenerator.generate(encoder, albedoTexture);
    device.queue.submit([encoder.finish()]);
  }

  const albedoSampler = device.resources.createSampler({
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 8,
  });

  // --- Post-process targets ---
  const hdrTexture = device.resources.createTexture({
    size: [canvas.width, canvas.height],
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const halfWidth = Math.max(1, canvas.width >> 1);
  const halfHeight = Math.max(1, canvas.height >> 1);
  const brightTexture = device.resources.createTexture({
    size: [halfWidth, halfHeight],
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const blurTexture = device.resources.createTexture({
    size: [halfWidth, halfHeight],
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const bloomTexture = device.resources.createTexture({
    size: [halfWidth, halfHeight],
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  const linearSampler = device.resources.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const camera = new Camera(device);
  const projectionMatrix = perspective(Math.PI / 4, canvas.width / canvas.height, NEAR, FAR);
  camera.setProjectionMatrix(projectionMatrix);
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const clusterGrid = new ClusterGrid(device, { clusterCountX: 16, clusterCountY: 9, clusterCountZ: 24 });
  clusterGrid.setProjection(projectionMatrix, canvas.width, canvas.height, NEAR, FAR);

  const lightCulling = new LightCulling(device, clusterGrid, LIGHT_COUNT);

  const shadowMap = new ShadowMap(device, { mapSize: SHADOW_MAP_SIZE });

  // group(0) bind group layout shared by the lit pipeline.
  const sceneBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });

  const cameraOnlyBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });

  // group(1): per-instance transform (modelMatrix + baseColor + textureLayer + uvScale), 96 bytes.
  const transformBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });

  // group(0) for the shadow depth-only pass: shadow map view-projection.
  const shadowBindGroupLayout = device.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
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
      { binding: 5, resource: { buffer: shadowMap.buffer.gpuBuffer } },
      { binding: 6, resource: shadowMap.getView() },
      { binding: 7, resource: shadowMap.depthSampler.gpuSampler },
      { binding: 8, resource: albedoTexture.gpuTexture.createView({ dimension: '2d-array' }) },
      { binding: 9, resource: albedoSampler.gpuSampler },
    ],
  });

  const cameraOnlyBindGroup = device.device.createBindGroup({
    layout: cameraOnlyBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: camera.buffer.gpuBuffer } },
    ],
  });

  const shadowBindGroup = device.device.createBindGroup({
    layout: shadowBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: shadowMap.buffer.gpuBuffer } },
    ],
  });

  // --- Post-process passes ---
  const brightPass = new FullscreenPass(device, {
    fragmentSource: brightPassFragmentSource,
    bindGroupLayoutEntries: brightPassLayoutEntries,
    targetFormat: HDR_FORMAT,
  });
  const blurPass = new FullscreenPass(device, {
    fragmentSource: blurFragmentSource,
    bindGroupLayoutEntries: blurLayoutEntries,
    targetFormat: HDR_FORMAT,
  });
  const compositePass = new FullscreenPass(device, {
    fragmentSource: compositeFragmentSource,
    bindGroupLayoutEntries: compositeLayoutEntries,
    targetFormat: format,
  });

  const brightParamsBuffer = device.resources.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(brightParamsBuffer.gpuBuffer, 0, new Float32Array([1.0, 1.0, 0, 0]));

  const blurHParamsBuffer = device.resources.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(blurHParamsBuffer.gpuBuffer, 0, new Float32Array([1, 0, 0, 0]));

  const blurVParamsBuffer = device.resources.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(blurVParamsBuffer.gpuBuffer, 0, new Float32Array([0, 1, 0, 0]));

  const brightBindGroup = brightPass.createBindGroup([
    { binding: 0, resource: hdrTexture.gpuTexture.createView() },
    { binding: 1, resource: linearSampler.gpuSampler },
    { binding: 2, resource: { buffer: brightParamsBuffer.gpuBuffer } },
  ]);
  const blurHBindGroup = blurPass.createBindGroup([
    { binding: 0, resource: brightTexture.gpuTexture.createView() },
    { binding: 1, resource: linearSampler.gpuSampler },
    { binding: 2, resource: { buffer: blurHParamsBuffer.gpuBuffer } },
  ]);
  const blurVBindGroup = blurPass.createBindGroup([
    { binding: 0, resource: blurTexture.gpuTexture.createView() },
    { binding: 1, resource: linearSampler.gpuSampler },
    { binding: 2, resource: { buffer: blurVParamsBuffer.gpuBuffer } },
  ]);
  const compositeBindGroup = compositePass.createBindGroup([
    { binding: 0, resource: hdrTexture.gpuTexture.createView() },
    { binding: 1, resource: bloomTexture.gpuTexture.createView() },
    { binding: 2, resource: linearSampler.gpuSampler },
  ]);

  // --- Geometry ---
  const cubeData = createCubeData();
  const cubeGeometry = new Geometry(device, {
    attributes: {
      position: { format: 'float32x3', data: cubeData.positions },
      normal: { format: 'float32x3', data: cubeData.normals },
      uv: { format: 'float32x2', data: cubeData.uvs },
    },
  });

  function setCubeVertexBuffers(renderPass) {
    renderPass.setVertexBuffer(0, cubeGeometry.attributes.position.buffer.gpuBuffer);
    renderPass.setVertexBuffer(1, cubeGeometry.attributes.normal.buffer.gpuBuffer);
    renderPass.setVertexBuffer(2, cubeGeometry.attributes.uv.buffer.gpuBuffer);
  }

  // --- Pipelines ---
  const litShaderModule = device.device.createShaderModule({ code: litShaderSource });
  const litPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [sceneBindGroupLayout, transformBindGroupLayout] }),
    vertex: { module: litShaderModule, entryPoint: 'vertexMain', buffers: cubeGeometry.vertexBufferLayouts },
    fragment: { module: litShaderModule, entryPoint: 'fragmentMain', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const markerShaderModule = device.device.createShaderModule({ code: lightMarkerShaderSource });
  const markerPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [cameraOnlyBindGroupLayout, transformBindGroupLayout] }),
    vertex: { module: markerShaderModule, entryPoint: 'vertexMain', buffers: cubeGeometry.vertexBufferLayouts },
    fragment: { module: markerShaderModule, entryPoint: 'fragmentMain', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const shadowDepthShaderModule = device.device.createShaderModule({ code: shadowDepthShaderSource });
  const shadowPipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout, transformBindGroupLayout] }),
    vertex: { module: shadowDepthShaderModule, entryPoint: 'vertexMain', buffers: cubeGeometry.vertexBufferLayouts },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  });

  // --- Scene objects ---
  // Transform: mat4x4f (64) + baseColor vec3f + textureLayer f32 (16) + uvScale vec2f + pad (16) = 96 bytes.
  const TRANSFORM_SIZE = 96;

  function createInstance(baseColor, textureLayer = 0, uvScale = [1, 1]) {
    const buffer = device.resources.createBuffer({
      size: TRANSFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.device.createBindGroup({
      layout: transformBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: buffer.gpuBuffer } }],
    });
    const data = new Float32Array(TRANSFORM_SIZE / 4);
    data.set(baseColor, 16);
    data[19] = textureLayer;
    data.set(uvScale, 20);
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
      const layer = (x + z) % 3; // brick / windows / concrete
      const instance = createInstance([1, 1, 1], layer, [1, Math.max(1, Math.round(height))]);
      const modelMatrix = fromTranslationRotationScale(
        [px, height / 2 - 1, pz],
        [0, 0, 0, 1],
        [1, height, 1],
      );
      writeInstance(instance, modelMatrix);
      cubeInstances.push(instance);
    }
  }

  // Floor: one large flat cube, pavement texture tiled 30x.
  const floorInstance = createInstance([1, 1, 1], 3, [30, 30]);
  writeInstance(floorInstance, fromTranslationRotationScale([0, -1.5, -10], [0, 0, 0, 1], [30, 0.5, 30]));

  // All shadow-casting instances (floor + buildings).
  const shadowCasters = [floorInstance, ...cubeInstances];

  // World-space AABB enclosing the floor and building grid, used to fit the
  // shadow map's orthographic projection.
  const sceneBounds = {
    min: [-15, -2, -25],
    max: [15, 4, 5],
  };

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

    shadowMap.update(LIGHT_DIRECTION, sceneBounds);

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
      name: 'shadow-map',
      depthStencilAttachment: {
        target: shadowMap.depthTexture,
        view: shadowMap.getView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      writes: [shadowMap.depthTexture],
      execute: (renderPass) => {
        renderPass.setPipeline(shadowPipeline);
        renderPass.setBindGroup(0, shadowBindGroup);
        setCubeVertexBuffers(renderPass);

        for (const instance of shadowCasters) {
          renderPass.setBindGroup(1, instance.bindGroup);
          renderPass.draw(cubeGeometry.vertexCount);
        }
      },
    });

    graph.addPass({
      name: 'forward-hdr',
      colorAttachments: [{
        target: hdrTexture,
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
      writes: [hdrTexture, depthTexture],
      reads: [
        camera.buffer, clusterGrid.gridInfoBuffer,
        lightCulling.clusterRangesBuffer, lightCulling.lightIndicesBuffer, lightCulling.lightsBuffer,
        shadowMap.buffer, shadowMap.depthTexture, albedoTexture,
      ],
      execute: (renderPass) => {
        renderPass.setPipeline(litPipeline);
        renderPass.setBindGroup(0, sceneBindGroup);
        setCubeVertexBuffers(renderPass);

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

    graph.addPass({
      name: 'bright-pass',
      colorAttachments: [{ target: brightTexture, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      writes: [brightTexture],
      reads: [hdrTexture],
      execute: (renderPass) => brightPass.draw(renderPass, brightBindGroup),
    });

    graph.addPass({
      name: 'bloom-blur-h',
      colorAttachments: [{ target: blurTexture, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      writes: [blurTexture],
      reads: [brightTexture],
      execute: (renderPass) => blurPass.draw(renderPass, blurHBindGroup),
    });

    graph.addPass({
      name: 'bloom-blur-v',
      colorAttachments: [{ target: bloomTexture, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      writes: [bloomTexture],
      reads: [blurTexture],
      execute: (renderPass) => blurPass.draw(renderPass, blurVBindGroup),
    });

    graph.addPass({
      name: 'composite',
      colorAttachments: [{ target: CANVAS, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      reads: [hdrTexture, bloomTexture],
      execute: (renderPass) => compositePass.draw(renderPass, compositeBindGroup),
    });

    graph.execute();

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
