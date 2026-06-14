import { createDevice } from '../../src/device/Device.js';
import { Camera } from '../../src/camera/Camera.js';
import { Geometry } from '../../src/geometry/Geometry.js';
import { boxData, sphereData, geometryFromData, computeVertexNormals } from '../../src/geometry/primitives.js';
import { SceneRenderer } from '../../src/scene/SceneRenderer.js';
import { Scene, Fog } from '../../src/scene/Scene.js';
import { Mesh, Group } from '../../src/scene/Mesh.js';
import { LambertMaterial, BasicMaterial, PointsMaterial, ShaderMaterial } from '../../src/scene/materials.js';
import { AmbientLight, PointLight } from '../../src/scene/lights.js';
import { Vec3 } from '../../src/math/vec3.js';
import { perspective, lookAt } from '../../src/math/mat4.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const showError = (m) => { errorEl.hidden = false; errorEl.textContent = m; };

try {
  const device = await createDevice();
  const renderer = new SceneRenderer(device, canvas, { antialias: true });

  const camera = new Camera(device);
  camera.setProjectionMatrix(perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 200));
  camera.setViewport(0, 0, canvas.width, canvas.height);

  const scene = new Scene();
  scene.background.setHex(0x05060a);
  scene.fog = new Fog(0x05060a, 30, 90);

  scene.add(new AmbientLight(0x404050, 1.0));
  // A bright, nearby point light so the Lambert buildings read clearly (the
  // shader uses quadratic distance attenuation, so a far light barely lights).
  const sun = new PointLight(0xfff0e0, 40, 0, 0);
  sun.position.set(6, 10, -6);
  scene.add(sun);

  // --- Lambert "buildings" as Groups (body + accent), like the game ---
  const boxGeo = geometryFromData(device, boxData([1, 1, 1]));
  const accentGeo = geometryFromData(device, sphereData(0.18, 8, 6));
  for (let x = -3; x <= 3; x++) {
    for (let z = -3; z <= 3; z++) {
      const h = 1 + ((x * 5 + z * 3 + 100) % 5) * 0.5;
      const group = new Group();
      group.position.set(x * 2.2, 0, z * 2.2 - 10);
      const body = new Mesh(geometryFromData(device, boxData([1, h, 1])), new LambertMaterial({ color: 0x88aacc }));
      const accent = new Mesh(accentGeo, new BasicMaterial({ color: 0x00ff44 }));
      accent.position.set(0, h / 2 + 0.3, 0);
      group.add(body); group.add(accent);
      scene.add(group);
    }
  }

  // --- Additive glow spheres ---
  const glowGeo = geometryFromData(device, sphereData(0.7, 16, 12));
  const glows = [];
  for (let i = 0; i < 3; i++) {
    const g = new Mesh(glowGeo, new BasicMaterial({ color: [0x66aaff, 0xff7744, 0x88ff66][i], opacity: 0.5, blending: 'additive', depthWrite: false }));
    scene.add(g); glows.push(g);
  }

  // --- Custom-shader floor quad (per-vertex color) ---
  // Centered under the building grid (which sits around z = -10), sized to
  // cover it, and just below the building bases.
  const N = 20, size = 26, cz = -10, y = -1.5;
  const pos = [], col = [];
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) {
    const x0 = (x / N - 0.5) * size, x1 = ((x + 1) / N - 0.5) * size;
    const z0 = (z / N - 0.5) * size + cz, z1 = ((z + 1) / N - 0.5) * size + cz;
    // CCW from above so the +Y top face is the front face (cullMode 'back').
    const quad = [[x0, y, z0], [x1, y, z1], [x1, y, z0], [x0, y, z0], [x0, y, z1], [x1, y, z1]];
    for (const p of quad) { pos.push(...p); const c = 0.3 + 0.4 * ((x + z) % 2); col.push(c, c * 0.8, c * 0.5); }
  }
  const positions = new Float32Array(pos);
  const terrainGeo = new Geometry(device, {
    attributes: {
      position: { format: 'float32x3', data: positions },
      normal: { format: 'float32x3', data: computeVertexNormals(positions) },
      color: { format: 'float32x3', data: new Float32Array(col) },
    },
  });
  // A generic custom-shader material: the app brings its own WGSL + a uniform
  // buffer it fills each frame. Here, a simple animated "pulse" light over a
  // moving point, using the geometry's per-vertex color. (This demonstrates the
  // engine's ShaderMaterial path — the engine knows nothing about what it does.)
  const customWGSL = /* wgsl */ `
struct Camera { viewMatrix: mat4x4f, projectionMatrix: mat4x4f, frustumPlanes: array<vec4f,6>, viewport: vec4f, };
struct U { lightPos: vec4f, fogColor: vec4f, fogRange: vec4f, };
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> u: U;
struct VOut { @builtin(position) position: vec4f, @location(0) color: vec3f, @location(1) world: vec3f, @location(2) viewDepth: f32, };
@vertex fn vertexMain(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) color: vec3f) -> VOut {
  var o: VOut; o.color = color; o.world = p;
  let view = camera.viewMatrix * vec4f(p, 1.0);
  o.position = camera.projectionMatrix * view; o.viewDepth = -view.z;
  return o;
}
@fragment fn fragmentMain(i: VOut) -> @location(0) vec4f {
  let d = length(u.lightPos.xyz - i.world);
  let glow = u.lightPos.w / (1.0 + d * d * 0.2);
  var rgb = i.color * (0.15 + glow);
  let f = clamp((i.viewDepth - u.fogRange.x) / max(u.fogRange.y - u.fogRange.x, 0.0001), 0.0, 1.0);
  rgb = mix(rgb, u.fogColor.rgb, f);
  return vec4f(rgb, 1.0);
}`;
  const lightPos = new Vec3(0, 0, -10);
  const terrainMat = new ShaderMaterial({
    wgsl: customWGSL,
    attributes: ['position', 'normal', 'color'],
    uniformSize: 48, // 3 x vec4
    updateUniforms: (v) => {
      v[0] = lightPos.x; v[1] = lightPos.y; v[2] = lightPos.z; v[3] = 2.5; // light + intensity
      v[4] = scene.fog.color.r; v[5] = scene.fog.color.g; v[6] = scene.fog.color.b; v[7] = 0;
      v[8] = scene.fog.near; v[9] = scene.fog.far;
    },
  });
  scene.add(new Mesh(terrainGeo, terrainMat));

  // --- Points star field ---
  const starN = 1500;
  const starPos = new Float32Array(starN * 3);
  for (let i = 0; i < starN; i++) {
    const r = 60 + Math.random() * 30, th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    starPos[i * 3] = r * Math.sin(ph) * Math.cos(th);
    starPos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
    starPos[i * 3 + 2] = r * Math.cos(ph);
  }
  const starGeo = new Geometry(device, { attributes: { position: { format: 'float32x3', data: starPos } } });
  scene.add(new Mesh(starGeo, new PointsMaterial({ color: 0xffffff, size: 2 })));

  let angle = 0;
  function frame() {
    angle += 0.005;
    camera.setViewMatrix(lookAt([Math.sin(angle) * 22, 8, Math.cos(angle) * 22 - 10], [0, 0, -10], [0, 1, 0]));
    camera.update();

    const t = performance.now() * 0.001;
    glows.forEach((g, i) => g.position.set(Math.sin(t + i * 2) * 5, 1 + i, Math.cos(t + i * 2) * 5 - 10));
    // Move the custom shader's light (read in updateUniforms each frame).
    lightPos.set(Math.sin(t) * 6, 0, Math.cos(t) * 6 - 10);

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
