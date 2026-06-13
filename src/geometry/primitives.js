import { Geometry } from './Geometry.js';

// Procedural primitive geometry. Each returns raw typed-array attribute data
// (position, normal, uv); pass it to `geometryFromPrimitive` or build a
// Geometry directly. Positions are centered at the origin.

/**
 * Axis-aligned box. `size` is the full extent on each axis (default unit
 * cube). 36 vertices (two triangles per face), flat per-face normals, and a
 * 0..1 UV per face.
 */
export function boxData(size = [1, 1, 1]) {
  const hx = size[0] / 2, hy = size[1] / 2, hz = size[2] / 2;

  // Per face: 4 corner positions (CCW from outside), the face normal, and
  // the two triangles' corner ordering (0,1,2, 0,2,3).
  const faces = [
    { n: [1, 0, 0], c: [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]] },   // +X
    { n: [-1, 0, 0], c: [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]] }, // -X
    { n: [0, 1, 0], c: [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]] },    // +Y
    { n: [0, -1, 0], c: [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]] }, // -Y
    { n: [0, 0, 1], c: [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]] },    // +Z
    { n: [0, 0, -1], c: [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]] }, // -Z
  ];

  const cornerUV = [[0, 1], [1, 1], [1, 0], [0, 0]];
  const tris = [0, 1, 2, 0, 2, 3];

  const positions = [];
  const normals = [];
  const uvs = [];

  for (const face of faces) {
    for (const i of tris) {
      positions.push(...face.c[i]);
      normals.push(...face.n);
      uvs.push(...cornerUV[i]);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}

/** Builds an immutable Geometry (position + normal + uv) from primitive data. */
export function geometryFromData(device, data) {
  return new Geometry(device, {
    attributes: {
      position: { format: 'float32x3', data: data.positions },
      normal: { format: 'float32x3', data: data.normals },
      uv: { format: 'float32x2', data: data.uvs },
    },
  });
}

/** Convenience: a box Geometry with the given full-extent size. */
export function boxGeometry(device, size = [1, 1, 1]) {
  return geometryFromData(device, boxData(size));
}
