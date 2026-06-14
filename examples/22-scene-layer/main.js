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

  // --- Custom-shader "terrain" quad (per-vertex color + skyAccess) ---
  // Centered under the building grid (which sits around z = -10), sized to
  // cover it, and just below the building bases.
  const N = 20, size = 26, cz = -10, y = -1.5;
  const pos = [], col = [], sky = [];
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) {
    const x0 = (x / N - 0.5) * size, x1 = ((x + 1) / N - 0.5) * size;
    const z0 = (z / N - 0.5) * size + cz, z1 = ((z + 1) / N - 0.5) * size + cz;
    // CCW from above so the +Y top face is the front face (cullMode 'back').
    const quad = [[x0, y, z0], [x1, y, z1], [x1, y, z0], [x0, y, z0], [x0, y, z1], [x1, y, z1]];
    for (const p of quad) { pos.push(...p); const c = 0.3 + 0.4 * ((x + z) % 2); col.push(c, c * 0.8, c * 0.5); sky.push(1); }
  }
  const positions = new Float32Array(pos);
  const terrainGeo = new Geometry(device, {
    attributes: {
      position: { format: 'float32x3', data: positions },
      normal: { format: 'float32x3', data: computeVertexNormals(positions) },
      color: { format: 'float32x3', data: new Float32Array(col) },
      skyAccess: { format: 'float32', data: new Float32Array(sky) },
    },
  });
  const terrainMat = new ShaderMaterial({
    vertexAttributes: ['color', 'skyAccess'],
    uniforms: {
      sunPosition: { value: new Vec3(20, 30, 20) },
      sunIntensity: { value: 1.2 },
      lanternPosition: { value: new Vec3(0, 0, -10) },
      lanternIntensity: { value: 2.0 },
      lanternRange: { value: 1.0 },
      ambientIntensity: { value: 0.05 },
      fog: { value: { color: scene.fog.color, near: scene.fog.near, far: scene.fog.far } },
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
    // animate the lantern in the terrain shader
    terrainMat.uniforms.lanternPosition.value.set(Math.sin(t) * 6, 0, Math.cos(t) * 6 - 10);

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
} catch (err) {
  showError(err.message);
  throw err;
}
