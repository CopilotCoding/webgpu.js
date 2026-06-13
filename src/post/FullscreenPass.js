// A fullscreen post-process pass: one pipeline that rasterizes a single
// triangle covering the screen and runs a caller-supplied fragment shader.
//
// The fragment shader declares its own @group(0) bindings (input textures,
// samplers, uniforms); the caller supplies the matching bind group layout
// entries and per-use bind groups. The pass itself owns nothing but the
// pipeline — inputs and render targets are wired up through the render
// graph, like any other pass.

const fullscreenVertexSource = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // One triangle covering the whole screen: clip-space corners at
  // (-1,-1), (3,-1), (-1,3). The parts outside [-1,1] are clipped away.
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );

  let pos = positions[vertexIndex];

  var out: VertexOutput;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = vec2f(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
  return out;
}
`;

export class FullscreenPass {
  /**
   * @param {import('../device/Device.js').Device} device
   * @param {object} options
   * @param {string} options.fragmentSource WGSL source containing the @group(0) binding declarations and a fragment entry point
   * @param {string} [options.entryPoint='fragmentMain'] fragment entry point name
   * @param {GPUBindGroupLayoutEntry[]} options.bindGroupLayoutEntries layout entries matching the fragment shader's @group(0) bindings
   * @param {GPUTextureFormat} options.targetFormat color target format this pass renders to
   */
  constructor(device, { fragmentSource, entryPoint = 'fragmentMain', bindGroupLayoutEntries, targetFormat }) {
    this.device = device;

    this.bindGroupLayout = device.device.createBindGroupLayout({
      entries: bindGroupLayoutEntries,
    });

    const module = device.device.createShaderModule({
      code: fullscreenVertexSource + fragmentSource,
    });

    this.pipeline = device.device.createRenderPipeline({
      layout: device.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: { module, entryPoint: 'vertexMain' },
      fragment: { module, entryPoint, targets: [{ format: targetFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** Creates a bind group matching this pass's @group(0) layout. */
  createBindGroup(entries) {
    return this.device.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries,
    });
  }

  /** Records the fullscreen draw into an active render pass. */
  draw(renderPass, bindGroup) {
    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
  }
}
