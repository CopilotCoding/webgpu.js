// Unit cube (positions only), centered at the origin, side length 1.
export function createCubeData() {
  const positions = [];

  const faces = [
    // +X, -X, +Y, -Y, +Z, -Z — each as two triangles
    [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, 1, 1], [1, -1, 1]],
    [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1]],
    [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [-1, 1, -1], [1, 1, 1], [1, 1, -1]],
    [[-1, -1, 1], [-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, -1], [1, -1, 1]],
    [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1]],
  ];

  for (const face of faces) {
    for (const vertex of face) {
      positions.push(vertex[0] * 0.5, vertex[1] * 0.5, vertex[2] * 0.5);
    }
  }

  return new Float32Array(positions);
}
