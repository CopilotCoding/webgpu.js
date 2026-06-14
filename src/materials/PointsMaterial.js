import { cameraStruct } from './materialCommon.js';

// Points renderer. WebGPU has no gl_PointSize, so each point is expanded in
// the vertex shader into a camera-facing quad (2 triangles = 6 vertices) sized
// in clip space, giving constant screen-space point size like
// THREE.PointsMaterial (sizeAttenuation off). Point positions live in a
// storage buffer; one instanced draw covers the whole cloud — no per-point CPU
// work.
//
// This material builds its own pipeline (it needs a storage binding, which the
// uniform-only Material/MaterialInstance path doesn't express) but follows the
// same group(0)=camera convention.

export class PointsMaterial {
  /**
   * @param {object} opts
   * @param {object} opts.cameraBindGroupLayout shared camera layout (group 0)
   * @param {GPUTextureFormat} opts.format render target format
   * @param {Float32Array} opts.positions xyz per point
   * @param {number[]} [opts.color=[1,1,1]]
   * @param {number} [opts.size=2] point size in pixels
   * @param {boolean} [opts.depthWrite=false]
   */
  constructor(device, { cameraBindGroupLayout, format, positions, color = [1, 1, 1], size = 2, depthWrite = false }) {
    this.device = device;
    this.pointCount = positions.length / 3;

    this.positionsBuffer = device.resources.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.positionsBuffer.gpuBuffer, 0, positions);

    // params: color.rgb, sizePx ; viewport.xy (filled per-frame), _, _
    this.paramsBuffer = device.resources.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._color = color;
    this._size = size;

    const shader = /* wgsl */ `
${cameraStruct}
struct Params { color: vec4f, viewport: vec4f, };

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<uniform> params: Params;

struct VertexOut { @builtin(position) position: vec4f, };

// 6 corners of a quad (two triangles) in [-1,1].
const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

@vertex
fn vertexMain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOut {
  let base = ii * 3u;
  let world = vec3f(positions[base], positions[base + 1u], positions[base + 2u]);
  var clip = camera.projectionMatrix * camera.viewMatrix * vec4f(world, 1.0);
  // Offset in clip space by the point size converted from pixels. The NDC
  // half-extent (size/viewport) is precomputed CPU-side into viewport.zw.
  let corner = CORNERS[vi];
  let halfSize = params.viewport.zw;
  clip.x = clip.x + corner.x * halfSize.x * clip.w;
  clip.y = clip.y + corner.y * halfSize.y * clip.w;
  var out: VertexOut;
  out.position = clip;
  return out;
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return vec4f(params.color.rgb, 1.0);
}
`;

    const module = device.device.createShaderModule({ code: shader });

    this.bindGroupLayout = device.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({
        bindGroupLayouts: [cameraBindGroupLayout.gpuBindGroupLayout, this.bindGroupLayout],
      }),
      vertex: { module, entryPoint: 'vertexMain' },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: depthWrite, depthCompare: 'less' },
    });

    this.bindGroup = device.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.positionsBuffer.gpuBuffer } },
        { binding: 1, resource: { buffer: this.paramsBuffer.gpuBuffer } },
      ],
    });
  }

  /** Call once per frame (or on resize) with the canvas pixel size. */
  updateViewport(widthPx, heightPx) {
    // Convert the desired pixel size into an NDC half-extent: size/viewport.
    const ndcX = this._size / widthPx;
    const ndcY = this._size / heightPx;
    this.device.queue.writeBuffer(this.paramsBuffer.gpuBuffer, 0, new Float32Array([
      this._color[0], this._color[1], this._color[2], 0,
      0, 0, ndcX, ndcY,
    ]));
  }

  /** Records the points draw into an active render pass (camera at group 0). */
  draw(renderPass) {
    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(1, this.bindGroup);
    renderPass.draw(6, this.pointCount);
  }
}
