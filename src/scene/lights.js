import { Node } from './Node.js';
import { Color } from './Color.js';

// Scene lights. The SceneRenderer packs the ambient term + each point/
// directional light into a small lights uniform block consumed by lit
// materials (Lambert). A game drives them imperatively: light.position.copy
// (...), light.intensity = ..., light.color.setHex(...).

export class AmbientLight extends Node {
  constructor(color = 0x111111, intensity = 1) {
    super();
    this.isAmbient = true;
    this.color = color instanceof Color ? color : new Color(color);
    this.intensity = intensity;
  }
}

export class PointLight extends Node {
  constructor(color = 0xffffff, intensity = 1, distance = 0, decay = 1) {
    super();
    this.isPointLight = true;
    this.color = color instanceof Color ? color : new Color(color);
    this.intensity = intensity;
    this.distance = distance; // 0 = no falloff cutoff
    this.decay = decay;
  }
}

export class DirectionalLight extends Node {
  constructor(color = 0xffffff, intensity = 1) {
    super();
    this.isDirectional = true;
    this.color = color instanceof Color ? color : new Color(color);
    this.intensity = intensity;
    this.direction = null; // set by caller, or derived from position
  }
}
