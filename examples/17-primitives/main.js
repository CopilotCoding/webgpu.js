import { createDevice } from '../../src/device/Device.js';
import { RenderGraph, CANVAS } from '../../src/render-graph/RenderGraph.js';
import {
  boxData, cylinderData, coneData, sphereData,
  octahedronData, dodecahedronData, tubeData,
  geometryFromData,
} from '../../src/geometry/primitives.js';
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

// One simple Lambert-ish shader: a single directional light + ambient so the
// generated normals are visible on every primitive.
const shaderSource = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};

struct Object {
  modelMatrix: mat4x4f,
  color: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> obj: Object;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) color: vec3f,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
) -> VertexOut {
  var out: VertexOut;
  out.position = camera.projectionMatrix * camera.viewMatrix * obj.modelMatrix * vec4f(position, 1.0);
  // Uniform scale in this demo, so the model matrix's upper 3x3 is fine for normals.
  out.normal = normalize((obj.modelMatrix * vec4f(normal, 0.0)).xyz);
  out.color = obj.color.rgb;
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let lightDir = normalize(vec3f(0.4, 0.8, 0.5));
  let ndotl = max(dot(normalize(in.normal), lightDir), 0.0);
  let shade = 0.2 + 0.8 * ndotl;
  return vec4f(in.color * shade, 1.0);
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

  // Build one geometry per primitive generator.
  // A gentle S-curve: corners stay wider than the tube radius so the sweep
  // never folds through itself.
  const tubePoints = [
    [-0.7, -0.5, 0], [-0.35, 0.0, 0.25], [0.0, 0.35, 0], [0.35, 0.0, -0.25], [0.7, -0.5, 0],
  ];
  const specs = [
    { data: boxData([0.8, 0.8, 0.8]), color: [0.90, 0.45, 0.40] },
    { data: cylinderData(0.4, 0.4, 0.9, 24), color: [0.45, 0.75, 0.55] },
    { data: coneData(0.45, 0.9, 24), color: [0.50, 0.65, 0.95] },
    { data: sphereData(0.5, 24, 18), color: [0.90, 0.80, 0.40] },
    { data: octahedronData(0.55), color: [0.75, 0.50, 0.90] },
    { data: dodecahedronData(0.50), color: [0.40, 0.85, 0.85] },
    { data: tubeData(tubePoints, 0.12, 12, 80, false), color: [0.95, 0.60, 0.75] },
  ];

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const cameraBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });
  const cameraBindGroup = device.device.createBindGroup({
    layout: cameraBindGroupLayout.gpuBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: camera.buffer.gpuBuffer } }],
  });

  const objectBindGroupLayout = device.resources.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
  });

  const shaderModule = device.device.createShaderModule({ code: shaderSource });

  // Layout is the same for every primitive (position/normal/uv), so one pipeline.
  const sampleGeometry = geometryFromData(device, specs[0].data);
  const pipeline = device.device.createRenderPipeline({
    layout: device.device.createPipelineLayout({
      bindGroupLayouts: [cameraBindGroupLayout.gpuBindGroupLayout, objectBindGroupLayout.gpuBindGroupLayout],
    }),
    vertex: { module: shaderModule, entryPoint: 'vertexMain', buffers: sampleGeometry.vertexBufferLayouts },
    fragment: { module: shaderModule, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const spacing = 1.6;
  const objects = specs.map((spec, i) => {
    const geometry = i === 0 ? sampleGeometry : geometryFromData(device, spec.data);
    const x = (i - (specs.length - 1) / 2) * spacing;

    const uniformBuffer = device.resources.createBuffer({
      size: 80, // mat4x4f (64) + vec4f color (16)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.device.createBindGroup({
      layout: objectBindGroupLayout.gpuBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: uniformBuffer.gpuBuffer } }],
    });
    return { geometry, x, color: spec.color, uniformBuffer, bindGroup };
  });

  let angle = 0;
  const uniformData = new Float32Array(20); // 16 matrix + 4 color

  function frame() {
    angle += 0.01;
    const eye = [Math.sin(angle * 0.3) * 3, 2.0, 8];
    camera.setViewMatrix(lookAt(eye, [0, 0, 0], [0, 1, 0]));
    camera.update();

    for (const obj of objects) {
      // Spin each primitive about Y so all faces (and their normals) are seen.
      const c = Math.cos(angle), s = Math.sin(angle);
      const m = translation(obj.x, 0, 0);
      // Compose a Y rotation into the translation matrix (column-major).
      const rot = new Float32Array([
        c, 0, -s, 0,
        0, 1, 0, 0,
        s, 0, c, 0,
        obj.x, 0, 0, 1,
      ]);
      uniformData.set(rot, 0);
      uniformData.set([...obj.color, 1], 16);
      device.queue.writeBuffer(obj.uniformBuffer.gpuBuffer, 0, uniformData);
    }

    const graph = new RenderGraph(device);
    graph.setCanvasTarget(context);

    graph.addPass({
      name: 'forward',
      colorAttachments: [{
        target: CANVAS,
        clearValue: { r: 0.06, g: 0.07, b: 0.09, a: 1.0 },
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
        for (const obj of objects) {
          renderPass.setBindGroup(1, obj.bindGroup);
          renderPass.setVertexBuffer(0, obj.geometry.attributes.position.buffer.gpuBuffer);
          renderPass.setVertexBuffer(1, obj.geometry.attributes.normal.buffer.gpuBuffer);
          renderPass.setVertexBuffer(2, obj.geometry.attributes.uv.buffer.gpuBuffer);
          renderPass.draw(obj.geometry.vertexCount);
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
