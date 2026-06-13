// Shaders for the Engine's GPU-driven forward renderer. The whole scene is
// drawn from storage buffers: per-object world matrices and material params,
// indexed via the indirect draw's visibleIndices. These mirror the WGSL that
// examples 14/15 inlined, factored out so the Engine owns one copy.

import { clusterLightingStructs, clusterIndexFunction, accumulateClusterLightingFunction } from '../lighting/clusterLighting.wgsl.js';
import { shadowMapStruct, sampleShadowFunction } from '../lighting/shadowMap.wgsl.js';

const cameraStruct = /* wgsl */ `
struct Camera {
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  frustumPlanes: array<vec4f, 6>,
  viewport: vec4f,
};
`;

// Per-object material parameters (48 bytes). emissive is added on top of the
// shaded color and feeds the bloom bright pass — it's also how the picking
// highlight is rendered (the Engine writes a highlight emissive into the
// hovered object's slot).
const materialParamsStruct = /* wgsl */ `
struct MaterialParams {
  baseColor: vec3f,
  textureLayer: f32,
  uvScale: vec2f,
  _pad0: vec2f,
  emissive: vec3f,
  _pad1: f32,
};
`;

export const MATERIAL_PARAMS_SIZE = 48;

export const litShaderSource = /* wgsl */ `
${cameraStruct}
${clusterLightingStructs}
${shadowMapStruct}
${materialParamsStruct}

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

@group(1) @binding(0) var<storage, read> worldMatrices: array<mat4x4f>;
@group(1) @binding(1) var<storage, read> materialParams: array<MaterialParams>;
@group(1) @binding(2) var<storage, read> visibleIndices: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewZ: f32,
  @location(3) uv: vec2f,
  @location(4) @interpolate(flat) objectIndex: u32,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let objectIndex = visibleIndices[instanceIndex];
  let worldMatrix = worldMatrices[objectIndex];
  let params = materialParams[objectIndex];

  let world = worldMatrix * vec4f(position, 1.0);
  let viewPos = camera.viewMatrix * world;

  var out: VertexOutput;
  out.position = camera.projectionMatrix * viewPos;
  out.worldPos = world.xyz;
  out.normal = (worldMatrix * vec4f(normal, 0.0)).xyz;
  out.viewZ = viewPos.z;
  out.uv = uv * params.uvScale;
  out.objectIndex = objectIndex;
  return out;
}

${clusterIndexFunction}
${accumulateClusterLightingFunction}
${sampleShadowFunction}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let params = materialParams[in.objectIndex];
  let normal = normalize(in.normal);
  let ambient = vec3f(0.03, 0.03, 0.04);

  let albedo = textureSample(albedoTexture, albedoSampler, in.uv, i32(params.textureLayer)).rgb;

  let shadow = sampleShadow(in.worldPos, normal);
  let sunDir = normalize(-shadowMap.lightDirection);
  let sunNdotl = max(dot(normal, sunDir), 0.0);
  let sun = vec3f(1.0, 0.96, 0.88) * sunNdotl * shadow * 1.4;

  let lighting = accumulateClusterLighting(in.worldPos, normal, in.position, clusterGridInfo, in.viewZ);

  let color = albedo * params.baseColor * (ambient + sun + lighting) + params.emissive;
  return vec4f(color, 1.0);
}
`;

// Light markers: one instanced draw, reading positions/colors straight from
// the cluster-lighting lights buffer.
export const markerShaderSource = /* wgsl */ `
${cameraStruct}
${clusterLightingStructs}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> lights: array<PointLight>;

struct MarkerOutput {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) lightIndex: u32,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @builtin(instance_index) instanceIndex: u32,
) -> MarkerOutput {
  let light = lights[instanceIndex];
  let world = light.position + position * 0.3;

  var out: MarkerOutput;
  out.position = camera.projectionMatrix * camera.viewMatrix * vec4f(world, 1.0);
  out.lightIndex = instanceIndex;
  return out;
}

@fragment
fn fragmentMain(in: MarkerOutput) -> @location(0) vec4f {
  return vec4f(lights[in.lightIndex].color * 8.0, 1.0);
}
`;

// Shadow depth pass: one instanced draw over every object (no culling — an
// object off-screen can still cast a shadow into the view).
export const shadowDepthShaderSource = /* wgsl */ `
${shadowMapStruct}

@group(0) @binding(0) var<uniform> shadowMap: ShadowMap;
@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @builtin(instance_index) instanceIndex: u32,
) -> @builtin(position) vec4f {
  let world = worldMatrices[instanceIndex] * vec4f(position, 1.0);
  return shadowMap.viewProjectionMatrix * world;
}
`;
