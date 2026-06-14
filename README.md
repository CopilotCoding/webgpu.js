# webgpu.js

A GPU-native 3D engine for the web, built on WebGPU from the ground up.

webgpu.js is not a Three.js successor and not a WebGL wrapper. WebGL is a stateful, CPU-driven API from 2011; Three.js is excellent engineering on top of that limited contract. WebGPU changes the contract — explicit pipelines, compute shaders, indirect drawing, bind groups, GPU-driven rendering — and webgpu.js is built _for_ that contract rather than against it.

The design goal is twofold: an experienced graphics programmer reading the source sees exactly what the GPU is doing, and a developer new to graphics can learn GPU programming by reading the engine rather than being shielded from it.

---

## What makes it different

**GPU-driven by default.** The GPU decides what to draw, not the CPU. Transform propagation, frustum + occlusion culling, draw-call generation, and light assignment all happen in compute passes. The CPU submits work; it does not iterate visible objects. In the engine's demo scene the entire city — floor plus 81 buildings — renders from a **single `drawIndirect` call** whose arguments are written by the GPU. CPU cost is constant whether the scene has 80 objects or 80,000.

**Explicit over implicit.** Every pipeline, bind group, render pass, and resource barrier is declared. The render graph resolves pass ordering from declared resource reads/writes and inserts barriers; there is no manual synchronization elsewhere and no hidden global state.

**WGSL only.** No GLSL, no transpilation. Shaders are WGSL, compiled natively by the browser.

**No legacy surface area.** No WebGL fallback, no compatibility shims. When WebGPU isn't available, the engine throws an error referencing the WebGPU concept that failed — it does not degrade.

---

## Requirements

- A browser with WebGPU enabled (Chrome/Edge 113+, or any current Chromium). WebGPU must be available as `navigator.gpu`.
- No build step. The engine is plain ES modules, imported directly. A bundler is the consumer's choice, not the engine's.
- ES modules require HTTP (not `file://`), so the demos must be served by a static web server.

## Running the examples

Serve the repository root with any static file server, then open an example's `index.html`:

```bash
# pick whichever you have
npx serve .
# or
python -m http.server 3000
```

Then visit, e.g., `http://localhost:3000/examples/16-engine/index.html`.

The examples are numbered to mirror how the engine was built, each one self-contained and demonstrating one system on top of the previous:

| # | Example | Demonstrates |
|---|---------|--------------|
| 01 | device-init | Device acquisition, resource manager, a cleared canvas |
| 02 | render-graph | Passes with declared dependencies, automatic ordering |
| 03 | geometry | Immutable vertex/index buffers and layouts |
| 04 | forward-pass | A basic forward render pass |
| 05 | materials | Pipeline descriptors, hashing/caching, bind groups |
| 06 | scene-graph | Hierarchy with CPU transform propagation |
| 07 | compute-transforms | Transform propagation moved to a compute pass |
| 08 | culling | GPU frustum + Hi-Z occlusion culling |
| 09 | indirect-draw | GPU-generated indirect draw arguments |
| 10 | clustered-lighting | Clustered point lights via compute light assignment |
| 11 | shadow-maps | A directional shadow map with PCF |
| 12 | post-processing | HDR, bloom, ACES tonemap as render-graph passes |
| 13 | textures | Compute-generated mipmaps, texture arrays, anisotropy |
| 14 | gpu-driven | The whole scene consolidated into one indirect draw |
| 15 | picking | GPU raycasting against object bounds |
| 16 | **engine** | The high-level Engine layer — the whole scene in ~55 lines |
| 17 | primitives | Cylinder/cone/sphere/octahedron/dodecahedron/tube generators + computed normals |
| 18 | materials-blend | Built-in Basic/Lambert/Points materials, additive + alpha blending, a custom-shader vertex-color mesh |
| 19 | multi-geometry | GPU-driven rendering of *heterogeneous* geometry: a GeometryArena + compute-built indexed multi-draw |
| 20 | ortho-and-layers | Orthographic camera + a layer mask evaluated in the GPU cull pass (a minimap inset) |
| 21 | composable-frame | The Engine's frame as composable passes: bloom opt-in, plus `setSize()` / `dispose()` |

Start with **16-engine** to see the assembly layer, then read **01** through **15** to see what it's made of. Examples **17–21** add the capabilities needed to render a heterogeneous, game-like scene (many distinct meshes, custom shaders, transparency, an ortho minimap) while staying GPU-driven.

### How 17–21 extend the engine

Earlier examples assume one shared geometry, one fixed shader, and a single indirect draw. **17–21** lift those limits so the GPU-driven path can render a real game's mix of meshes and materials:

- **17–18** add the missing content tools: more primitives + `computeVertexNormals`, and Three-style `BasicMaterial`/`LambertMaterial`/`PointsMaterial` plus blend-state plumbing on `Material`.
- **19** is the core generalization: a `GeometryArena` packs many *different* meshes into shared buffers, and `MultiDrawSystem` runs a compute pass that frustum-culls per object and compacts a per-object `DrawIndexedIndirect` arg array — so heterogeneous geometry draws GPU-driven, at constant CPU submit cost.
- **20** adds an orthographic camera wrapper and a per-object layer mask the cull pass honors (the minimap pattern), so a second view culls on the GPU too.
- **21** makes the Engine's frame composable (bloom is opt-in) and adds `setSize()`/`dispose()` for real app lifecycles.

---

## Quick start (the Engine layer)

The `Engine` class composes every system — device, render graph, shadows, clustered lights, GPU culling + indirect draws, bloom/tonemap, and picking — and runs the per-frame render graph for you. It does **not** hide the systems: each is exposed (`engine.camera`, `engine.shadowMap`, `engine.lightCulling`, `engine.cullingPass`, `engine.picker`, …) so you can reach past the Engine whenever you need to.

```js
import { Engine } from './src/engine/Engine.js';
import { OrbitControls } from './src/controls/OrbitControls.js';

const canvas = document.getElementById('canvas');

// Wires the whole pipeline. textureLayers is an array of canvas/ImageBitmap
// sources uploaded into an albedo texture array (mips generated in compute).
const engine = await Engine.create({
  canvas,
  far: 60,
  lightDirection: [0.4, -0.7, 0.35],
  textureLayers: [brickCanvas, windowsCanvas, concreteCanvas, pavementCanvas],
});

const box = engine.boxGeometry();

// A floor + a grid of buildings. Every mesh in a Scene shares one geometry,
// which is what keeps the whole scene a single indirect draw.
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
    });
  }
}

// Point lights.
const light = engine.scene.addLight({
  position: [0, 4, -10], radius: 6, color: [1, 0.8, 0.4], intensity: 6,
});

// Orbit camera.
const controls = new OrbitControls(canvas, { target: [0, 0, -10], distance: 28 });

// Per-frame hook: drive the camera, transforms, and lights here.
engine.onUpdate = (dt) => {
  controls.update();
  engine.camera.setViewMatrix(controls.viewMatrix);

  const t = performance.now() * 0.001;
  light.setPosition(Math.sin(t) * 8, 4, Math.cos(t) * 8 - 10);

  // Picking is built in; engine.hovered is the current hit (or null).
  if (engine.hovered) console.log('hovering object', engine.hovered.objectIndex);
};

engine.start();
```

That is the complete setup. The full version of this scene — textures, shadows, 64 animated clustered lights, bloom, GPU-driven indirect rendering, and hover picking — is [`examples/16-engine/main.js`](examples/16-engine/main.js), about 55 lines.

### Engine API surface

- `Engine.create(options)` → `Promise<Engine>`. Options: `canvas` (required), `near`, `far`, `fov`, `maxObjects` (≤ 4096), `maxLights`, `shadowMapSize`, `shadowBounds` (`{ min, max }` world AABB the shadow map fits), `lightDirection`, `textureLayers`.
- `engine.boxGeometry(size?)` → a shared box `Geometry` registered as the batch geometry.
- `engine.scene.addMesh({ geometry, position, rotation, scale, baseColor, textureLayer, uvScale, emissive, bounds? })` → a mesh handle with `setTransform`, `setMatrix`, `setMaterial`.
- `engine.scene.addLight({ position, radius, color, intensity })` → a light handle with `set`, `setPosition`.
- `engine.onUpdate = (dt) => { … }` — called once per frame before rendering.
- `engine.start()` / `engine.stop()`.
- `engine.hovered` — the latest pick result `{ objectIndex, distance, point }`, or `null`.

### Camera controls

```js
import { OrbitControls } from './src/controls/OrbitControls.js';
import { FlyControls }   from './src/controls/FlyControls.js';
```

- **OrbitControls** — drag to rotate, wheel to zoom, right-drag to pan. Call `update()` each frame, then `camera.setViewMatrix(controls.viewMatrix)`.
- **FlyControls** — WASD to move, Q/E down/up, drag to look, Shift to boost. Call `update(dt)` each frame.

---

## Architecture

The engine is a set of composable systems with narrow interfaces. Nothing is load-bearing by accident; the renderer does not depend on the scene graph, the scene graph does not depend on materials, and each system can be used on its own.

- **Device layer** (`src/device/`) — owns the WebGPU adapter, device, and queue. All GPU object creation flows through the `ResourceManager` (`src/resources/`); nothing else touches the raw WebGPU API for resource creation. Resources are immutable once created and carry an explicit `destroy()`.
- **Render graph** (`src/render-graph/`) — passes declare their resource reads, writes, and attachments. The graph topologically orders them and resolves attachment views; pass order comes from dependency, not call order.
- **Geometry** (`src/geometry/`) — immutable vertex/index buffers described by a layout that maps directly to WebGPU vertex buffer layouts. `primitives.js` provides `boxGeometry`/`boxData`.
- **Materials** (`src/materials/`) — a material is a pipeline descriptor plus typed bindings; pipelines are compiled once and cached by descriptor hash.
- **Scene graph** (`src/scene/`) — dirty-flagged hierarchy; a compute pass propagates world transforms one depth level at a time.
- **Culling** (`src/culling/`) — frustum culling against the camera planes and occlusion culling against a hierarchical depth (Hi-Z) buffer from the previous frame, both in compute, writing a visibility buffer that feeds indirect draw generation.
- **Lighting** (`src/lighting/`) — clustered lighting: the view frustum is subdivided into a 3D grid of clusters and a compute pass assigns lights to clusters each frame. Plus a directional shadow map with PCF.
- **Textures** (`src/textures/`) — mipmaps generated in compute; native texture arrays and cubemaps.
- **Post** (`src/post/`) — a fullscreen-pass helper and reusable bright/blur/composite effects (bloom + ACES tonemap).
- **Picking** (`src/picking/`) — GPU raycasting against object bounds, sharing the same world-matrix and bounds buffers the culling pass uses, so picking always agrees with what's rendered.
- **Engine** (`src/engine/`) — the assembly layer that wires the above into one GPU-driven forward renderer.

### The frame

Each frame, the Engine's render graph runs:

1. Cluster build + light assignment (compute)
2. Hi-Z build + frustum/occlusion cull + indirect draw-argument generation (compute)
3. Shadow depth pass (one instanced draw over all casters)
4. Forward HDR pass — the whole scene as one indirect draw, plus light markers
5. Bloom: bright pass → separable Gaussian blur (H, then V)
6. Composite: scene + bloom, ACES tonemap, gamma encode, to the canvas

---

## Using the systems directly

You don't have to use the Engine. Every system is independently constructable — see examples 01–15, each of which wires a subset by hand. The Engine is just the convenient default; when a project outgrows it, drop down a layer rather than fighting an abstraction.

---

## Status and non-goals

The engine implements its full intended pipeline: device layer, render graph, geometry, materials, scene graph with compute transforms, GPU culling, indirect rendering, clustered lighting, shadows, post-processing, textures, GPU-driven consolidation, picking, and the Engine assembly layer.

Explicitly out of scope for the core engine: physics, audio, input handling, a WebGL fallback, a Three.js compatibility layer, and a visual editor. Some may arrive as external modules. A glTF mesh loader and PBR materials are natural next additions on top of the current foundation.

## License

MIT