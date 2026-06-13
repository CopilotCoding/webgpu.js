import { fromTranslationRotationScale } from '../math/mat4.js';

/**
 * Flattens a scene graph rooted at `root` into per-node arrays ordered by
 * depth level (root first, then its children, then grandchildren, ...).
 * Within this ordering, every node's parent is guaranteed to appear in an
 * earlier level, which is what allows the GPU transform-propagation
 * compute shader to process one level per dispatch: level N only reads
 * world matrices written by level N-1.
 *
 * Returns:
 *   nodes: SceneNode[] in level order — nodes[i] corresponds to row i in localMatrices/parentIndices
 *   localMatrices: Float32Array, 16 floats per node
 *   parentIndices: Int32Array, -1 for the root
 *   levels: { offset, count }[] — contiguous ranges of `nodes` for each depth level
 */
export function flattenSceneGraph(root) {
  const nodes = [];
  const parentIndices = [];
  const levels = [];

  let currentLevel = [root];
  let parentIndexMap = new Map([[root, -1]]);

  while (currentLevel.length > 0) {
    const offset = nodes.length;

    for (const node of currentLevel) {
      parentIndices.push(parentIndexMap.get(node));
      nodes.push(node);
    }

    levels.push({ offset, count: currentLevel.length });

    const nextLevel = [];
    const nextParentIndexMap = new Map();
    for (const node of currentLevel) {
      const parentIndex = nodes.indexOf(node);
      for (const child of node.children) {
        nextParentIndexMap.set(child, parentIndex);
        nextLevel.push(child);
      }
    }

    currentLevel = nextLevel;
    parentIndexMap = nextParentIndexMap;
  }

  const localMatrices = new Float32Array(nodes.length * 16);
  for (let i = 0; i < nodes.length; i++) {
    fromTranslationRotationScale(nodes[i].position, nodes[i].rotation, nodes[i].scale, localMatrices.subarray(i * 16, i * 16 + 16));
  }

  return {
    nodes,
    localMatrices,
    parentIndices: new Int32Array(parentIndices),
    levels,
  };
}
