// Unit cube (position + normal + uv), centered at the origin, side length 1.
export function createCubeData() {
  const positions = [];
  const normals = [];
  const uvs = [];

  const faces = [
    [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, 1, 1], [1, -1, 1], [1, 0, 0]],
    [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, 1, -1], [-1, -1, -1], [-1, 0, 0]],
    [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [-1, 1, -1], [1, 1, 1], [1, 1, -1], [0, 1, 0]],
    [[-1, -1, 1], [-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, -1], [1, -1, 1], [0, -1, 0]],
    [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 1]],
    [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [0, 0, -1]],
  ];

  // Per-face UVs for the two triangles (quad corners 00, 10, 11 / 00, 11, 01
  // matching the winding above).
  const faceUVs = [[0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]];

  for (const face of faces) {
    const normal = face[6];
    for (let i = 0; i < 6; i++) {
      const vertex = face[i];
      positions.push(vertex[0] * 0.5, vertex[1] * 0.5, vertex[2] * 0.5);
      normals.push(normal[0], normal[1], normal[2]);
      uvs.push(faceUVs[i][0], faceUVs[i][1]);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}
