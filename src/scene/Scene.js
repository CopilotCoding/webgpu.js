import { Node } from './Node.js';
import { Color } from './Color.js';

// Linear fog descriptor (matches a typical { color, near, far } fog).
export class Fog {
  constructor(color = 0x000000, near = 1, far = 1000) {
    this.color = color instanceof Color ? color : new Color(color);
    this.near = near;
    this.far = far;
  }
}

// Root of the retained scene graph: a Node that also carries a background color
// and optional fog. Meshes and lights are add()ed into it (or into child
// Groups). The SceneRenderer traverses it each frame.
export class Scene extends Node {
  constructor() {
    super();
    this.background = new Color(0x000000);
    this.fog = null;
  }
}
