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

- A browser with WebGPU enabled (Chrome/Edge 113+, or Firefox 141+). WebGPU must be available as `navigator.gpu`. The engine runs cross-browser: it uses the standard `indirect-first-instance` feature (Firefox + Chromium) for GPU-driven indirect draws, and only opts into Chromium's experimental multi-draw-indirect when present — falling back to a bounded indirect-draw loop (still GPU-decided visibility) elsewhere. The active path is logged at startup, e.g. `indirect-first-instance: on | multi-draw: off`.
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

Then visit any example below. The links assume you serve the repository **root**
(the folder containing both `webgpu.js/` and `PlanetVoxel/`) on port 3000 — e.g.
`http://localhost:3000/webgpu.js/examples/16-engine/index.html`. If you serve the
`webgpu.js/` folder directly instead, drop the `/webgpu.js` segment from the URLs.

The examples are numbered to mirror how the engine was built, each one self-contained and demonstrating one system on top of the previous:

| # | Example | Demonstrates |
|---|---------|--------------|
| 01 | [device-init](http://localhost:3000/webgpu.js/examples/01-device-init/index.html) | Device acquisition, resource manager, a cleared canvas |
| 02 | [render-graph](http://localhost:3000/webgpu.js/examples/02-render-graph/index.html) | Passes with declared dependencies, automatic ordering |
| 03 | [geometry](http://localhost:3000/webgpu.js/examples/03-geometry/index.html) | Immutable vertex/index buffers and layouts |
| 04 | [forward-pass](http://localhost:3000/webgpu.js/examples/04-forward-pass/index.html) | A basic forward render pass |
| 05 | [materials](http://localhost:3000/webgpu.js/examples/05-materials/index.html) | Pipeline descriptors, hashing/caching, bind groups |
| 06 | [scene-graph](http://localhost:3000/webgpu.js/examples/06-scene-graph/index.html) | Hierarchy with CPU transform propagation |
| 07 | [compute-transforms](http://localhost:3000/webgpu.js/examples/07-compute-transforms/index.html) | Transform propagation moved to a compute pass |
| 08 | [culling](http://localhost:3000/webgpu.js/examples/08-culling/index.html) | GPU frustum + Hi-Z occlusion culling, with a live per-object visibility counter (visible / occluded / frustum-culled) |
| 09 | [indirect-draw](http://localhost:3000/webgpu.js/examples/09-indirect-draw/index.html) | GPU-generated indirect draw arguments |
| 10 | [clustered-lighting](http://localhost:3000/webgpu.js/examples/10-clustered-lighting/index.html) | Clustered point lights via compute light assignment |
| 11 | [shadow-maps](http://localhost:3000/webgpu.js/examples/11-shadow-maps/index.html) | A directional shadow map with PCF |
| 12 | [post-processing](http://localhost:3000/webgpu.js/examples/12-post-processing/index.html) | HDR, bloom, ACES tonemap as render-graph passes |
| 13 | [textures](http://localhost:3000/webgpu.js/examples/13-textures/index.html) | Compute-generated mipmaps, texture arrays, anisotropy |
| 14 | [gpu-driven](http://localhost:3000/webgpu.js/examples/14-gpu-driven/index.html) | The whole scene consolidated into one indirect draw |
| 15 | [picking](http://localhost:3000/webgpu.js/examples/15-picking/index.html) | GPU raycasting against object bounds |
| 16 | [**engine**](http://localhost:3000/webgpu.js/examples/16-engine/index.html) | The high-level Engine layer — the whole scene in ~55 lines |
| 17 | [primitives](http://localhost:3000/webgpu.js/examples/17-primitives/index.html) | Cylinder/cone/sphere/octahedron/dodecahedron/tube generators + computed normals |
| 18 | [materials-blend](http://localhost:3000/webgpu.js/examples/18-materials-blend/index.html) | Built-in Basic/Lambert/Points materials, additive + alpha blending, a custom-shader vertex-color mesh |
| 19 | [multi-geometry](http://localhost:3000/webgpu.js/examples/19-multi-geometry/index.html) | GPU-driven rendering of *heterogeneous* geometry: a GeometryArena + compute-built indexed multi-draw |
| 20 | [ortho-and-layers](http://localhost:3000/webgpu.js/examples/20-ortho-and-layers/index.html) | Orthographic camera + a layer mask evaluated in the GPU cull pass (a minimap inset) |
| 21 | [composable-frame](http://localhost:3000/webgpu.js/examples/21-composable-frame/index.html) | The Engine's frame as composable passes: bloom opt-in, plus `setSize()` / `dispose()` |
| 22 | [scene-layer](http://localhost:3000/webgpu.js/examples/22-scene-layer/index.html) | Retained-mode scene graph (Mesh/Group/lights/materials) drawn GPU-driven by SceneRenderer: Lambert + additive + custom-shader + points |
| 23 | [occlusion-city](http://localhost:3000/webgpu.js/examples/23-occlusion-popping/index.html) | Hi-Z occlusion culling in a city scene: buildings hidden behind nearer ones are dropped from the indirect-draw path as the camera flies the streets ("buildings drawn" falls); [Space] toggles occlusion (off = frustum-only) |

Start with **16-engine** to see the assembly layer, then read **01** through **15** to see what it's made of. Examples **17–22** add the capabilities needed to render a heterogeneous, game-like scene (many distinct meshes, custom shaders, transparency, an ortho minimap, a retained scene graph) while staying GPU-driven.

### How 17–22 extend the engine

Earlier examples assume one shared geometry, one fixed shader, and a single indirect draw. **17–22** lift those limits so the GPU-driven path can render a real game's mix of meshes and materials:

- **17–18** add the missing content tools: more primitives + `computeVertexNormals`, and Three-style `BasicMaterial`/`LambertMaterial`/`PointsMaterial` plus blend-state plumbing on `Material`.
- **19** is the core generalization: a `GeometryArena` packs many *different* meshes into shared buffers, and `MultiDrawSystem` runs a compute pass that frustum-culls per object and compacts a per-object `DrawIndexedIndirect` arg array — so heterogeneous geometry draws GPU-driven, at constant CPU submit cost.
- **20** adds an orthographic camera wrapper and a per-object layer mask the cull pass honors (the minimap pattern), so a second view culls on the GPU too.
- **21** makes the Engine's frame composable (bloom is opt-in) and adds `setSize()`/`dispose()` for real app lifecycles.
- **22** is the **retained-mode scene layer** (`src/scene/`): imperative `Mesh`/`Group`/`Scene` nodes a game mutates in place (`mesh.position.copy(...)`, `material.color.setHex(...)`, `group.add(...)`), plus `Vec3`/`Quat`/`Color` classes and lights/fog. A `SceneRenderer` walks the graph and draws it through the GPU-driven path — batching meshes by pipeline (each batch its own `GeometryArena` + `MultiDrawSystem` with GPU cull), with dedicated pipelines for a custom-shader material and points. This is the layer a Three.js-style app ports onto.

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
- **Math** (`src/math/`) — column-major `mat4` (perspective/orthographic/lookAt/TRS/invert), plus chainable `Vec3`/`Quat` classes and a `Color` for CPU-side scene and game math. The mat4/array helpers stay allocation-light for the hot paths; the classes are the ergonomic surface for app code.
- **Render graph** (`src/render-graph/`) — passes declare their resource reads, writes, and attachments. The graph topologically orders them, resolves attachment views (including optional MSAA `resolveTarget`s), and inserts barriers; pass order comes from dependency, not call order.
- **Geometry** (`src/geometry/`) — immutable vertex/index buffers described by a layout that maps directly to WebGPU vertex buffer layouts. `primitives.js` provides box/cylinder/cone/sphere/octahedron/dodecahedron/tube generators plus `computeVertexNormals`. `GeometryArena` packs many distinct meshes (one vertex layout) into shared buffers so heterogeneous geometry can be drawn together.
- **Materials** (`src/materials/`) — a `Material` is a pipeline descriptor plus typed bindings; pipelines are compiled once and cached by descriptor hash. Built-in factories (`BasicMaterial`/`LambertMaterial`/`PointsMaterial`) cover the common unlit/diffuse/points cases with blend, depth, side, and fog options.
- **Camera** (`src/camera/`) — the GPU camera uniform (view/projection/frustum planes/viewport), plus `PerspectiveCamera`/`OrthographicCamera` wrappers (position/target/up/layers) that mirror a conventional camera API.
- **Scene graph (GPU-driven)** (`src/scene/`) — `SceneNode` hierarchy whose world transforms are propagated by a compute pass (`TransformPropagation`), one depth level at a time.
- **Scene layer (retained-mode)** (`src/scene/`) — `Node`/`Mesh`/`Group`/`Scene` an app mutates imperatively, with `Color`, lights, fog, and material descriptors. `SceneRenderer` draws the graph through the GPU-driven path: meshes are batched by pipeline (each batch its own `GeometryArena` + `MultiDrawSystem` + GPU cull), with dedicated pipelines for custom-shader materials and points.
- **Culling** (`src/culling/`) — frustum culling against the camera planes and occlusion culling against a hierarchical depth (Hi-Z) buffer from the previous frame, both in compute. `IndirectDrawSystem` compacts a single instanced draw; `MultiDrawSystem` compacts a per-object `DrawIndexedIndirect` arg array for heterogeneous geometry, honoring a per-object layer mask and frustum-cull-disable flag.
- **Lighting** (`src/lighting/`) — clustered lighting: the view frustum is subdivided into a 3D grid of clusters and a compute pass assigns lights to clusters each frame. Plus a directional shadow map with PCF. (Both optional — the retained scene layer uses a small forward lights block instead.)
- **Textures** (`src/textures/`) — mipmaps generated in compute; native texture arrays and cubemaps.
- **Post** (`src/post/`) — a fullscreen-pass helper and reusable bright/blur/composite effects (bloom + ACES tonemap).
- **Picking** (`src/picking/`) — GPU raycasting against object bounds, sharing the same world-matrix and bounds buffers the culling pass uses, so picking always agrees with what's rendered.
- **Engine** (`src/engine/`) — the assembly layer that wires the above into one GPU-driven forward renderer. Clustered lighting, shadows, and bloom are opt-in; it exposes `setSize()`/`dispose()`.

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

You don't have to use the Engine. Every system is independently constructable — see examples 01–15, each of which wires a subset by hand. The Engine is just the convenient default; when a project outgrows it, drop down a layer rather than fighting an abstraction. For a heterogeneous, game-like scene that doesn't fit the Engine's single-geometry batch, use the **retained scene layer** (`SceneRenderer` + `Mesh`/`Group`, examples 19–22) instead.

---

## Status and non-goals

The engine implements its full intended pipeline: device layer, render graph, math, geometry (primitives + arena), materials (with built-in factories), cameras (perspective + orthographic), scene graph with compute transforms, GPU culling (single + heterogeneous multi-draw), indirect rendering, clustered lighting, shadows, post-processing, textures, GPU-driven consolidation, picking, the Engine assembly layer, and a retained-mode scene layer (`SceneRenderer`) that draws an imperative `Mesh`/`Group` graph through the GPU-driven path.

Explicitly out of scope for the core engine: physics, audio, input handling, a WebGL fallback, a Three.js compatibility layer, and a visual editor. Some may arrive as external modules. A glTF mesh loader and PBR materials are natural next additions on top of the current foundation.

## License

MIT