import { Material } from './Material.js';
import { cameraStruct, fogStruct, fragmentTarget, cullModeFor, packFog } from './materialCommon.js';

// Diffuse (Lambert) material: N·L over an ambient term plus a small fixed set
// of lights, mirroring THREE.MeshLambertMaterial's role for opaque
// surfaces (buildings, the player, belts). Lighting is a per-instance uniform
// block here (ambient + up to MAX_LIGHTS point/directional lights); an app
// that wants one shared lights buffer can write the same data into every
// instance, or drop to a custom Material with the lights at group 0.
//
// Lights layout per light (32 bytes): position.xyz, kind (0 = directional,
// 1 = point); color.rgb, intensity. Directional uses position as the
// direction TO the light.

const MAX_LIGHTS = 4;
const LIGHTS_SIZE = 16 + MAX_LIGHTS * 32; // ambient vec4 + count, then lights

export class LambertMaterial {
  constructor(device, opts) {
    const {
      cameraBindGroupLayout, vertexBufferLayouts, format,
      side = 'front', fog = true,
    } = opts;

    this.device = device;

    const shader = /* wgsl */ `
${cameraStruct}
${fogStruct}
struct Transform { modelMatrix: mat4x4f, };
struct MaterialData { color: vec4f, };
struct Light { posDir: vec4f, color: vec4f, };
struct Lights {
  ambient: vec4f,    // rgb + count
  lights: array<Light, ${MAX_LIGHTS}>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> transform: Transform;
@group(1) @binding(1) var<uniform> mat: MaterialData;
@group(1) @binding(2) var<uniform> fog: Fog;
@group(1) @binding(3) var<uniform> lighting: Lights;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewDepth: f32,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
) -> VertexOut {
  var out: VertexOut;
  let world = transform.modelMatrix * vec4f(position, 1.0);
  let view = camera.viewMatrix * world;
  out.position = camera.projectionMatrix * view;
  out.worldPos = world.xyz;
  out.normal = normalize((transform.modelMatrix * vec4f(normal, 0.0)).xyz);
  out.viewDepth = -view.z;
  return out;
}

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  var diffuse = lighting.ambient.rgb;
  let count = u32(lighting.ambient.w);
  for (var i = 0u; i < count; i = i + 1u) {
    let L = lighting.lights[i];
    var dir: vec3f;
    var atten: f32 = 1.0;
    if (L.posDir.w < 0.5) {
      dir = normalize(L.posDir.xyz);             // directional
    } else {
      let d = L.posDir.xyz - in.worldPos;        // point
      let dist = length(d);
      dir = d / max(dist, 0.0001);
      atten = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
    }
    let ndotl = max(dot(n, dir), 0.0);
    diffuse = diffuse + L.color.rgb * (L.color.w * ndotl * atten);
  }
  let rgb = applyFog(mat.color.rgb * diffuse, fog, in.viewDepth);
  return vec4f(rgb, mat.color.a);
}
`;

    this.material = new Material(device, cameraBindGroupLayout, {
      shader: { code: shader },
      vertexBufferLayouts,
      fragmentTargets: [fragmentTarget(format, null)],
      primitive: { topology: 'triangle-list', cullMode: cullModeFor(side) },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      bindings: {
        transform: { binding: 0, visibility: GPUShaderStage.VERTEX, size: 64 },
        material: { binding: 1, visibility: GPUShaderStage.FRAGMENT, size: 16 },
        fog: { binding: 2, visibility: GPUShaderStage.FRAGMENT, size: 32 },
        lighting: { binding: 3, visibility: GPUShaderStage.FRAGMENT, size: LIGHTS_SIZE },
      },
    });
  }

  get pipeline() { return this.material.pipeline; }

  /**
   * Packs a lights uniform.
   * @param {number[]} ambient rgb
   * @param {Array<{direction?:number[], position?:number[], color:number[], intensity:number}>} lights
   */
  static packLights(ambient, lights) {
    const data = new Float32Array(LIGHTS_SIZE / 4);
    data.set(ambient, 0);
    data[3] = Math.min(lights.length, MAX_LIGHTS);
    for (let i = 0; i < Math.min(lights.length, MAX_LIGHTS); i++) {
      const L = lights[i];
      const off = 4 + i * 8;
      if (L.position) { data.set(L.position, off); data[off + 3] = 1; }
      else { data.set(L.direction ?? [0, 1, 0], off); data[off + 3] = 0; }
      data.set(L.color ?? [1, 1, 1], off + 4);
      data[off + 7] = L.intensity ?? 1;
    }
    return data;
  }

  createInstance({ color = [1, 1, 1], opacity = 1, lights, fog } = {}) {
    const instance = this.material.createInstance();
    instance.set('material', new Float32Array([color[0], color[1], color[2], opacity]));
    instance.set('fog', packFog(fog ?? { enabled: false }));
    if (lights) instance.set('lighting', lights);
    return instance;
  }
}
