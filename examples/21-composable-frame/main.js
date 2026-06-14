import { Engine } from '../../src/engine/Engine.js';
import { OrbitControls } from '../../src/controls/OrbitControls.js';

const canvas = document.getElementById('canvas');
const errorEl = document.getElementById('error');
const stateEl = document.getElementById('state');
const showError = (m) => { errorEl.hidden = false; errorEl.textContent = m; };

// Builds the same little scene on a fresh Engine. Called on load and whenever
// the frame composition changes (bloom on/off) or after dispose — proving the
// Engine tears down and rebuilds cleanly.
async function buildEngine(bloom) {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const engine = await Engine.create({ canvas, far: 80, bloom });

  const box = engine.boxGeometry();
  engine.scene.addMesh({ geometry: box, position: [0, -1.5, -10], scale: [40, 1, 40], baseColor: [0.4, 0.42, 0.5] });
  for (let x = 0; x < 7; x++) {
    for (let z = 0; z < 7; z++) {
      const h = 1 + ((x * 5 + z * 11) % 5) * 0.7;
      engine.scene.addMesh({
        geometry: box,
        position: [(x - 3) * 2.4, h / 2 - 1, (z - 3) * 2.4 - 10],
        scale: [1, h, 1],
        baseColor: [0.7, 0.7, 0.8],
        // A few emissive blocks so the bloom on/off difference is obvious.
        emissive: (x === z) ? [1.2, 0.6, 0.2] : [0, 0, 0],
      });
    }
  }
  // Orbiting point lights.
  const lights = [];
  for (let i = 0; i < 6; i++) {
    lights.push(engine.scene.addLight({ position: [0, 3, -10], radius: 8, color: [Math.random(), Math.random(), 1], intensity: 5 }));
  }

  const controls = new OrbitControls(canvas, { target: [0, 0, -10], distance: 22 });
  engine.onUpdate = () => {
    controls.update();
    engine.camera.setViewMatrix(controls.viewMatrix);
    const t = performance.now() * 0.001;
    lights.forEach((light, i) => {
      const a = t * 0.6 + i * (Math.PI * 2 / 6);
      light.setPosition(Math.cos(a) * 7, 3, Math.sin(a) * 7 - 10);
    });
  };
  engine.start();
  return { engine, controls };
}

try {
  let bloom = true;
  let { engine } = await buildEngine(bloom);
  const setState = () => { stateEl.textContent = `bloom: ${bloom ? 'ON' : 'OFF'} | ${canvas.width}x${canvas.height}`; };
  setState();

  document.getElementById('toggle-bloom').addEventListener('click', async () => {
    bloom = !bloom;
    engine.dispose();
    ({ engine } = await buildEngine(bloom));
    setState();
  });

  document.getElementById('resize').addEventListener('click', () => {
    // Toggle between full size and a smaller box to exercise setSize().
    const small = canvas.width === window.innerWidth;
    const w = small ? Math.floor(window.innerWidth * 0.6) : window.innerWidth;
    const h = small ? Math.floor(window.innerHeight * 0.6) : window.innerHeight;
    engine.setSize(w, h);
    setState();
  });

  document.getElementById('dispose').addEventListener('click', async () => {
    engine.dispose();
    ({ engine } = await buildEngine(bloom));
    setState();
  });

  window.addEventListener('resize', () => {
    engine.setSize(window.innerWidth, window.innerHeight);
    setState();
  });
} catch (err) {
  showError(err.message);
  throw err;
}
