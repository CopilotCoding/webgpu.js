import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { Camera } from '../../src/camera/Camera.js';
import { boxData, sphereData, geometryFromData } from '../../src/geometry/primitives.js';
import { BasicMaterial } from '../../src/materials/BasicMaterial.js';
import { LambertMaterial } from '../../src/materials/LambertMaterial.js';
import { PointsMaterial } from '../../src/materials/PointsMaterial.js';
import { Material } from '../../src/materials/Material.js';
import { perspective, lookAt, fromTranslationRotationScale, translation } from '../../src/math/mat4.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const showError = (m) => { errorEl.hidden = false; errorEl.textContent = m; };

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
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const cameraBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
  });
  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBindGroupLayout.gpuBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });

  // --- Geometries ---
  const boxGeo = geometryFromData(device, boxData([1, 1, 1]));
  const sphereGeo = geometryFromData(device, sphereData(0.6, 24, 18));

  // --- Lambert box (opaque, lit) ---
  const lambert = new LambertMaterial(device, { cameraBindGroupLayout, vertexBufferLayouts: boxGeo.vertexBufferLayouts, format });
  const lightsData = LambertMaterial.packLights([0.08, 0.08, 0.10], [
    { direction: [0.4, 0.8, 0.5], color: [1, 0.96, 0.9], intensity: 1.0 },
    { position: [3, 2, 3], color: [0.4, 0.6, 1.0], intensity: 1.0 },
  ]);
  const lambertInstance = lambert.createInstance({ color: [0.8, 0.5, 0.4], lights: lightsData });
  lambertInstance.set('transform', fromTranslationRotationScale([-2.2, 0, 0], [0, 0, 0, 1], [1, 1, 1]));

  // --- Additive glow sphere (order-independent) ---
  const additive = new BasicMaterial(device, {
    cameraBindGroupLayout, vertexBufferLayouts: sphereGeo.vertexBufferLayouts, format,
    blend: 'additive', depthWrite: false,
  });
  const glowA = additive.createInstance({ color: [0.5, 0.7, 1.0], opacity: 0.5 });
  const glowB = additive.createInstance({ color: [1.0, 0.5, 0.3], opacity: 0.5 });

  // --- Alpha-blended panel (Basic transparent box) ---
  const alpha = new BasicMaterial(device, {
    cameraBindGroupLayout, vertexBufferLayouts: boxGeo.vertexBufferLayouts, format,
    transparent: true, depthWrite: false, side: 'double',
  });
  const panel = alpha.createInstance({ color: [0.3, 1.0, 0.6], opacity: 0.4 });
  panel.set('transform', fromTranslationRotationScale([0, 0, 1.5], [0, 0, 0, 1], [2.5, 2.5, 0.05]));

  // --- Vertex-color custom-shader mesh (a tri with per-vertex colors) via Material ---
  const triGeo = new Geometry(device, {
    attributes: {
      position: { format: 'float32x3', data: new Float32Array([0, 0.6, 0, -0.6, -0.5, 0, 0.6, -0.5, 0]) },
      color: { format: 'float32x3', data: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]) },
    },
  });
  const customShader = /* wgsl */ `
struct Camera { viewMatrix: mat4x4f, projectionMatrix: mat4x4f, frustumPlanes: array<vec4f,6>, viewport: vec4f, };
struct Transform { modelMatrix: mat4x4f, };
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> transform: Transform;
struct VOut { @builtin(position) position: vec4f, @location(0) color: vec3f, };
@vertex fn vertexMain(@location(0) position: vec3f, @location(1) color: vec3f) -> VOut {
  var o: VOut;
  o.position = camera.projectionMatrix * camera.viewMatrix * transform.modelMatrix * vec4f(position, 1.0);
  o.color = color;
  return o;
}
@fragment fn fragmentMain(i: VOut) -> @location(0) vec4f { return vec4f(i.color, 1.0); }
`;
  const customMat = new Material(device, cameraBindGroupLayout, {
    shader: { code: customShader },
    vertexBufferLayouts: triGeo.vertexBufferLayouts,
    fragmentTargets: [{ format }],
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    bindings: { transform: { binding: 0, visibility: GPUShaderStage.VERTEX, size: 64 } },
  });
  const customInstance = customMat.createInstance();
  customInstance.set('transform', translation(2.2, 0, 0));

  // --- Points star cloud ---
  const starPositions = new Float32Array(2000 * 3);
  for (let i = 0; i < 2000; i++) {
    const r = 12 + Math.random() * 6;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPositions[i * 3 + 2] = r * Math.cos(phi);
  }
  const points = new PointsMaterial(device, { cameraBindGroupLayout, format, positions: starPositions, color: [1, 1, 1], size: 2 });

  // Binds vertex buffers in the geometry's attribute order (shaderLocation is
  // assigned sequentially by Geometry: e.g. position/normal/uv or
  // position/color). One draw per instance.
  function drawMeshInstance(rp, pipeline, instance, geo) {
    rp.setPipeline(pipeline);
    rp.setBindGroup(1, instance.bindGroup);
    for (const attr of Object.values(geo.attributes)) {
      rp.setVertexBuffer(attr.shaderLocation, attr.buffer.gpuBuffer);
    }
    rp.draw(geo.vertexCount);
  }

  let angle = 0;
  function frame() {
    angle += 0.008;
    camera.setViewMatrix(lookAt([Math.sin(angle) * 6, 2.5, 6], [0, 0, 0], [0, 1, 0]));
    camera.update();
    points.updateViewport(canvas.width, canvas.height);

    const t = performance.now() * 0.001;
    glowA.set('transform', fromTranslationRotationScale([Math.sin(t) * 1.2, 0.5, 0], [0, 0, 0, 1], [1, 1, 1]));
    glowB.set('transform', fromTranslationRotationScale([Math.sin(t + 1) * 1.2, -0.5, 0.3], [0, 0, 0, 1], [1, 1, 1]));

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);
    graph.addPass({
      name: 'forward',
      colorAttachments: [{ target: CANVAS, clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { target: depthTexture, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      writes: [depthTexture],
      reads: [camera.buffer],
      execute: (rp) => {
        rp.setBindGroup(0, cameraBindGroup);
        // Opaque first.
        drawMeshInstance(rp, lambert.pipeline, lambertInstance, boxGeo);
        drawMeshInstance(rp, customMat.pipeline, customInstance, triGeo);
        points.draw(rp);
        // Transparent after opaque.
        drawMeshInstance(rp, alpha.pipeline, panel, boxGeo);
        drawMeshInstance(rp, additive.pipeline, glowA, sphereGeo);
        drawMeshInstance(rp, additive.pipeline, glowB, sphereGeo);
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
