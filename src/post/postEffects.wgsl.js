// Reusable fragment shaders for common post-process passes. Each source
// declares its own @group(0) bindings; the matching bind group layout
// entries are exported alongside so a FullscreenPass can be built without
// re-stating them.

// --- Bright pass: keep only the HDR color above a luminance threshold ---
// Uniform: { threshold: f32, intensity: f32 } (padded to 16 bytes).
export const brightPassFragmentSource = /* wgsl */ `
struct BrightPassParams {
  threshold: f32,
  intensity: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var<uniform> params: BrightPassParams;

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(inputTexture, inputSampler, in.uv).rgb;
  let luminance = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let contribution = max(luminance - params.threshold, 0.0) / max(luminance, 0.0001);
  return vec4f(color * contribution * params.intensity, 1.0);
}
`;

export const brightPassLayoutEntries = [
  { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
  { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
];

// --- Separable Gaussian blur: 9 taps along a direction ---
// Uniform: { direction: vec2f } — (1,0) for horizontal, (0,1) for vertical,
// in texel units (the shader scales by 1/textureDimensions).
export const blurFragmentSource = /* wgsl */ `
struct BlurParams {
  direction: vec2f,
  _pad0: vec2f,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let texelSize = 1.0 / vec2f(textureDimensions(inputTexture, 0));
  let step = params.direction * texelSize;

  // 9-tap Gaussian, sigma ~2.
  var weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

  var result = textureSample(inputTexture, inputSampler, in.uv).rgb * weights[0];
  for (var i = 1; i < 5; i++) {
    let offset = step * f32(i);
    result += textureSample(inputTexture, inputSampler, in.uv + offset).rgb * weights[i];
    result += textureSample(inputTexture, inputSampler, in.uv - offset).rgb * weights[i];
  }

  return vec4f(result, 1.0);
}
`;

export const blurLayoutEntries = [
  { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
  { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
];

// --- Composite: HDR scene + bloom, ACES tonemap, gamma encode ---
export const compositeFragmentSource = /* wgsl */ `
@group(0) @binding(0) var sceneTexture: texture_2d<f32>;
@group(0) @binding(1) var bloomTexture: texture_2d<f32>;
@group(0) @binding(2) var inputSampler: sampler;

// Narkowicz ACES approximation.
fn acesTonemap(color: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let scene = textureSample(sceneTexture, inputSampler, in.uv).rgb;
  let bloom = textureSample(bloomTexture, inputSampler, in.uv).rgb;

  let hdr = scene + bloom;
  let tonemapped = acesTonemap(hdr);
  let gammaEncoded = pow(tonemapped, vec3f(1.0 / 2.2));

  return vec4f(gammaEncoded, 1.0);
}
`;

export const compositeLayoutEntries = [
  { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
  { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
  { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
];
