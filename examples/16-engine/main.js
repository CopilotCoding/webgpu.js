import { Engine } from '../../src/engine/Engine.js';
import { OrbitControls } from '../../src/controls/OrbitControls.js';
import { makeTextureLayers } from './textures.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
const hitEl = document.getElementById('hit');

try {
  // 1. Create the engine — wires device, render graph, shadows, clustered
  //    lights, GPU culling + indirect draws, bloom/tonemap, and picking.
  const engine = await Engine.create({
    canvas,
    far: 60,
    lightDirection: [0.4, -0.7, 0.35],
    textureLayers: makeTextureLayers(256), // brick, windows, concrete, pavement
  });

  const box = engine.boxGeometry();

  // 2. Build the scene. A floor + a 9x9 grid of "buildings".
  engine.scene.addMesh({
    geometry: box, position: [0, -1.5, -10], scale: [60, 1, 60],
    textureLayer: 3, uvScale: [30, 30],
  });

  for (let x = 0; x < 9; x++) {
    for (let z = 0; z < 9; z++) {
      const height = 1 + ((x * 7 + z * 13) % 5) * 0.6;
      engine.scene.addMesh({
        geometry: box,
        position: [(x - 4) * 3, height / 2 - 1, (z - 4) * 3 - 10],
        scale: [1, height, 1],
        textureLayer: (x + z) % 3,
        uvScale: [1, Math.max(1, Math.round(height))],
      });
    }
  }

  // 3. Add animated point lights.
  const lights = [];
  for (let i = 0; i < 64; i++) {
    const hue = i / 64;
    lights.push({
      handle: engine.scene.addLight({ color: hsv(hue, 0.8, 1), radius: 4 + Math.random() * 3, intensity: 6 }),
      ring: 4 + (i % 5) * 3,
      phase: Math.random() * Math.PI * 2,
    });
  }

  // 4. Camera controls + per-frame update.
  const controls = new OrbitControls(canvas, { target: [0, 0, -10], distance: 28, polar: 1.0 });

  engine.onUpdate = (dt) => {
    controls.update();
    engine.camera.setViewMatrix(controls.viewMatrix);

    const t = performance.now() * 0.001;
    for (const l of lights) {
      l.handle.setPosition(
        Math.sin(t * 0.5 + l.phase) * l.ring,
        1 + Math.sin(t * 0.7 + l.phase * 2) * 2.5 + 2,
        Math.cos(t * 0.4 + l.phase * 1.3) * l.ring - 10,
      );
    }
    const h = engine.hovered;
    hitEl.textContent = h ? `hit: object #${h.objectIndex} @ ${h.distance.toFixed(1)}` : '';
  };

  engine.start();
} catch (err) {
  errorEl.hidden = false;
  errorEl.textContent = err.stack || err.message;
  throw err;
}

function hsv(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
}
