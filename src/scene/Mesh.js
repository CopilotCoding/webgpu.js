import { Node } from './Node.js';

// A drawable scene node: a transform (from Node) plus a geometry and a
// material. The geometry is an engine Geometry (its own vertex/index buffers);
// the SceneRenderer decides how to draw it based on the material kind (batched
// through the GPU-driven arena for lambert/basic, or via a dedicated pipeline
// for the terrain shader / points).
//
// `frustumCulled = false` opts a mesh out of culling (sun, corona, atmosphere —
// objects so large/always-relevant that culling them is wrong).

export class Mesh extends Node {
  constructor(geometry, material) {
    super();
    this.geometry = geometry;
    this.material = material;
    this.frustumCulled = true;
    // Renderer-owned bookkeeping (arena handle, draw-record slot, cached
    // per-mesh GPU resources). The renderer sets/reads these; the game doesn't.
    this._render = null;
  }

  dispose() {
    if (this.geometry && this.geometry.destroy) this.geometry.destroy();
  }
}

// A transform-only container (no geometry). Pure grouping for composed
// transforms (e.g. a building body + accent, or the sun + its coronas).
export class Group extends Node {}
