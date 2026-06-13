import { identity, multiply, fromTranslationRotationScale } from '../math/mat4.js';

/**
 * A node in the scene graph: local transform (position/rotation/scale),
 * hierarchy (parent/children), and visibility. Writing to the transform
 * marks the node dirty; propagateTransforms() recomputes world matrices
 * for dirty subtrees only.
 */
export class SceneNode {
  constructor(name = '') {
    this.name = name;
    this.parent = null;
    this.children = [];

    this.position = new Float32Array([0, 0, 0]);
    this.rotation = new Float32Array([0, 0, 0, 1]); // quaternion [x, y, z, w]
    this.scale = new Float32Array([1, 1, 1]);

    this.localMatrix = identity();
    this.worldMatrix = identity();

    this.visible = true;
    this.dirty = true;
  }

  setPosition(x, y, z) {
    this.position.set([x, y, z]);
    this.dirty = true;
  }

  setRotation(x, y, z, w) {
    this.rotation.set([x, y, z, w]);
    this.dirty = true;
  }

  setScale(x, y, z) {
    this.scale.set([x, y, z]);
    this.dirty = true;
  }

  add(child) {
    if (child.parent) child.parent.remove(child);
    child.parent = this;
    child.dirty = true;
    this.children.push(child);
  }

  remove(child) {
    const index = this.children.indexOf(child);
    if (index !== -1) this.children.splice(index, 1);
    child.parent = null;
    child.dirty = true;
  }
}

/**
 * Recomputes local/world matrices for this node and its descendants.
 * A node is recomputed if it is dirty or an ancestor was recomputed —
 * dirty status propagates down the subtree without touching clean
 * branches elsewhere in the graph.
 */
export function propagateTransforms(node, parentWorldMatrix = null, parentDirty = false) {
  const dirty = node.dirty || parentDirty;

  if (dirty) {
    fromTranslationRotationScale(node.position, node.rotation, node.scale, node.localMatrix);

    if (parentWorldMatrix) {
      multiply(parentWorldMatrix, node.localMatrix, node.worldMatrix);
    } else {
      node.worldMatrix.set(node.localMatrix);
    }

    node.dirty = false;
  }

  for (const child of node.children) {
    propagateTransforms(child, node.worldMatrix, dirty);
  }
}
