// Unit cube (position + normal), centered at the origin, side length 1.
export function createCubeData() {
  const positions = [];
  const normals = [];

  const faces = [
    [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, 1, 1], [1, -1, 1], [1, 0, 0]],
    [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1], [-1, 0, 0]],
    [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [-1, 1, -1], [1, 1, 1], [1, 1, -1], [0, 1, 0]],
    [[-1, -1, 1], [-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, -1], [1, -1, 1], [0, -1, 0]],
    [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 1]],
    [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [0, 0, -1]],
  ];

  for (const face of faces) {
    const normal = face[6];
    for (let i = 0; i < 6; i++) {
      const vertex = face[i];
      positions.push(vertex[0] * 0.5, vertex[1] * 0.5, vertex[2] * 0.5);
      normals.push(normal[0], normal[1], normal[2]);
    }
  }

  return { positions: new Float32Array(positions), normals: new Float32Array(normals) };
}
