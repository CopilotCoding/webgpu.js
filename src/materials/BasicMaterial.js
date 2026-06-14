import { Material } from './Material.js';
import { cameraStruct, fogStruct, fragmentTarget, cullModeFor, packFog } from './materialCommon.js';

// Unlit material — the workhorse for solid colors and additive/alpha glows
// (sun core/corona, lasers, atmosphere, ghosts, markers). Mirrors the useful
// surface of THREE.MeshBasicMaterial: solid or vertex color, opacity, blend
// mode, depth write/test, single/double/back side, wireframe, fog.
//
// Vertex layout expected: position(loc 0), normal(loc 1, ignored), uv(loc 2,
// ignored) — i.e. the standard primitive layout, so a BasicMaterial can draw
// any primitive geometry. With `vertexColors`, an extra color attribute is
// read at loc 3 instead of the uniform color.
//
// group(0): camera (binding 0). group(1): transform (0), material (1), fog (2).

export class BasicMaterial {
  /**
   * @param {object} opts
   * @param {object} opts.cameraBindGroupLayout shared camera layout (group 0)
   * @param {GPUVertexBufferLayout[]} opts.vertexBufferLayouts from the geometry
   * @param {GPUTextureFormat} opts.format render target format
   * @param {boolean} [opts.vertexColors=false] read a per-vertex color (loc 3, float32x3)
   * @param {'normal'|'additive'|null} [opts.blend=null] null = opaque
   * @param {boolean} [opts.transparent=false] convenience: defaults blend to 'normal'
   * @param {boolean} [opts.depthWrite=true]
   * @param {boolean} [opts.depthTest=true]
   * @param {'front'|'back'|'double'} [opts.side='front']
   * @param {boolean} [opts.wireframe=false] line-list topology
   * @param {boolean} [opts.fog=true] participate in scene fog
   */
  constructor(device, opts) {
    const {
      cameraBindGroupLayout, vertexBufferLayouts, format,
      vertexColors = false, transparent = false, depthWrite = true,
      depthTest = true, side = 'front', wireframe = false, fog = true,
    } = opts;
    const blend = opts.blend ?? (transparent ? 'normal' : null);

    this.device = device;
    this.fogEnabled = fog;

    const colorInput = vertexColors
      ? '@location(3) color: vec3f,'
      : '';
    const colorExpr = vertexColors ? 'in.color * mat.color.rgb' : 'mat.color.rgb';
    const colorVarying = vertexColors ? '@location(1) color: vec3f,' : '';
    const colorAssign = vertexColors ? 'out.color = color;' : '';

    const shader = /* wgsl */ `
${cameraStruct}
${fogStruct}
struct Transform { modelMatrix: mat4x4f, };
struct MaterialData { color: vec4f, };

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> transform: Transform;
@group(1) @binding(1) var<uniform> mat: MaterialData;
@group(1) @binding(2) var<uniform> fog: Fog;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) viewDepth: f32,
  ${colorVarying}
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  ${colorInput}
) -> VertexOut {
  var out: VertexOut;
  let world = transform.modelMatrix * vec4f(position, 1.0);
  let view = camera.viewMatrix * world;
  out.position = camera.projectionMatrix * view;
  out.viewDepth = -view.z;
  ${colorAssign}
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let rgb = applyFog(${colorExpr}, fog, in.viewDepth);
  return vec4f(rgb, mat.color.a);
}
`;

    this.material = new Material(device, cameraBindGroupLayout, {
      shader: { code: shader },
      vertexBufferLayouts,
      fragmentTargets: [fragmentTarget(format, blend)],
      primitive: { topology: wireframe ? 'line-list' : 'triangle-list', cullMode: cullModeFor(side) },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: depthWrite, depthCompare: depthTest ? 'less' : 'always' },
      bindings: {
        transform: { binding: 0, visibility: GPUShaderStage.VERTEX, size: 64 },
        material: { binding: 1, visibility: GPUShaderStage.FRAGMENT, size: 16 },
        fog: { binding: 2, visibility: GPUShaderStage.FRAGMENT, size: 32 },
      },
    });
  }

  get pipeline() { return this.material.pipeline; }

  /**
   * Creates a per-object instance. color is [r,g,b]; opacity 0..1.
   * Returns the MaterialInstance plus setters for transform/color.
   */
  createInstance({ color = [1, 1, 1], opacity = 1, fog } = {}) {
    const instance = this.material.createInstance();
    instance.set('material', new Float32Array([color[0], color[1], color[2], opacity]));
    instance.set('fog', packFog(fog ?? { enabled: false }));
    return instance;
  }
}
