import { Vec3 } from '../math/vec3.js';
import { Quat } from '../math/quat.js';
import { fromTranslationRotationScale, multiply, identity } from '../math/mat4.js';

// Retained-mode scene node: a transform (position / quaternion / scale) with
// optional parent and children, a visibility flag, a render-order hint, and a
// layer bitmask. This is the imperative scene-graph surface a game mutates
// (node.position.copy(...), node.quaternion.setFromUnitVectors(...), node.scale
// .set(...), node.visible = false, group.add(child)); a SceneRenderer walks the
// graph and feeds world matrices + draw records to the GPU-driven path.
//
// World matrices are lazily recomputed: any transform mutation marks the node
// (and its subtree) dirty via a per-node version counter the renderer compares
// against. Because position/quaternion/scale are mutable objects the game edits
// in place, the node can't catch every write — so updateWorldMatrix() always
// recomposes from the current local components when asked. The dirty flag is an
// optimization hint, not a correctness gate.

let _nodeId = 0;

export class Node {
  constructor() {
    this.id = ++_nodeId;
    this.position = new Vec3(0, 0, 0);
    this.quaternion = new Quat(0, 0, 0, 1);
    this.scale = new Vec3(1, 1, 1);

    this.visible = true;
    this.renderOrder = 0;
    this.layers = 0x1; // bitmask, ANDed against a camera's mask in culling

    this.parent = null;
    this.children = [];

    this.worldMatrix = identity();
    this._localMatrix = identity();
  }

  add(child) {
    if (child.parent) child.parent.remove(child);
    child.parent = this;
    this.children.push(child);
    return this;
  }

  remove(child) {
    const i = this.children.indexOf(child);
    if (i !== -1) { this.children.splice(i, 1); child.parent = null; }
    return this;
  }

  /** Recomposes localMatrix from position/quaternion/scale. */
  updateLocalMatrix() {
    fromTranslationRotationScale(
      [this.position.x, this.position.y, this.position.z],
      [this.quaternion.x, this.quaternion.y, this.quaternion.z, this.quaternion.w],
      [this.scale.x, this.scale.y, this.scale.z],
      this._localMatrix,
    );
    return this._localMatrix;
  }

  /**
   * Recomputes this node's world matrix (and its subtree's) from current local
   * transforms. `parentWorld` is the parent's world matrix (or null for a root).
   */
  updateWorldMatrix(parentWorld = null) {
    this.updateLocalMatrix();
    if (parentWorld) {
      multiply(parentWorld, this._localMatrix, this.worldMatrix);
    } else {
      this.worldMatrix.set(this._localMatrix);
    }
    for (const child of this.children) child.updateWorldMatrix(this.worldMatrix);
    return this.worldMatrix;
  }

  /** Depth-first traversal including this node. */
  traverse(fn) {
    fn(this);
    for (const child of this.children) child.traverse(fn);
  }

  /** Three-compat alias used by some game code: orient from a quaternion. */
  setRotationFromQuaternion(q) {
    this.quaternion.copy(q);
    return this;
  }
}
