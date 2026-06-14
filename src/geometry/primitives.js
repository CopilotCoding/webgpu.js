import { Geometry } from './Geometry.js';

// Procedural primitive geometry. Each returns raw typed-array attribute data
// (position, normal, uv); pass it to `geometryFromData` or build a
// Geometry directly. Positions are centered at the origin.
//
// All generators here produce NON-INDEXED triangle soup (three vertices per
// triangle), matching `boxData` — the engine's geometry path and the future
// terrain mesher both work with non-indexed buffers, and it keeps every
// generator a flat list of triangles with no shared-vertex bookkeeping.

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

/**
 * Cylinder along +Y, centered at the origin. Tapered (radiusTop !=
 * radiusBottom) covers cones-with-a-flat-top; `coneData` is the degenerate
 * radiusTop=0 case. Side normals are smooth (per-vertex), caps are flat.
 */
export function cylinderData(radiusTop = 0.5, radiusBottom = 0.5, height = 1, radialSegments = 16) {
  const positions = [];
  const normals = [];
  const uvs = [];
  const hy = height / 2;

  // Slope of the side surface, used to tilt side normals so a cone shades
  // correctly (the normal is not purely radial when the radii differ).
  const slope = (radiusBottom - radiusTop) / height;

  const ring = (radius, y, v) => {
    const out = [];
    for (let i = 0; i <= radialSegments; i++) {
      const theta = (i / radialSegments) * Math.PI * 2;
      const cos = Math.cos(theta), sin = Math.sin(theta);
      out.push({
        p: [radius * cos, y, radius * sin],
        // Radial direction tilted by the slope toward +Y, then normalized.
        n: normalize3([cos, slope, sin]),
        uv: [i / radialSegments, v],
      });
    }
    return out;
  };

  const top = ring(radiusTop, hy, 1);
  const bottom = ring(radiusBottom, -hy, 0);

  // Side quads (two triangles each), wound CCW as seen from outside.
  for (let i = 0; i < radialSegments; i++) {
    const a = bottom[i], b = bottom[i + 1], c = top[i + 1], d = top[i];
    for (const v of [a, d, c, a, c, b]) {
      positions.push(...v.p); normals.push(...v.n); uvs.push(...v.uv);
    }
  }

  // Caps (fan from the center), only if the radius is non-zero.
  const cap = (radius, y, ny, flip) => {
    if (radius <= 0) return;
    for (let i = 0; i < radialSegments; i++) {
      const t0 = (i / radialSegments) * Math.PI * 2;
      const t1 = ((i + 1) / radialSegments) * Math.PI * 2;
      const e0 = [radius * Math.cos(t0), y, radius * Math.sin(t0)];
      const e1 = [radius * Math.cos(t1), y, radius * Math.sin(t1)];
      const center = [0, y, 0];
      const verts = flip ? [center, e1, e0] : [center, e0, e1];
      for (const p of verts) {
        positions.push(...p); normals.push(0, ny, 0); uvs.push(0.5, 0.5);
      }
    }
  };
  cap(radiusTop, hy, 1, true);
  cap(radiusBottom, -hy, -1, false);

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}

/** Cone along +Y (apex at +height/2), centered at the origin. */
export function coneData(radius = 0.5, height = 1, radialSegments = 16) {
  return cylinderData(0, radius, height, radialSegments);
}

/** UV sphere centered at the origin. Smooth per-vertex normals. */
export function sphereData(radius = 0.5, widthSegments = 16, heightSegments = 12) {
  const positions = [];
  const normals = [];
  const uvs = [];

  const vertex = (iy, ix) => {
    const u = ix / widthSegments;
    const v = iy / heightSegments;
    const theta = u * Math.PI * 2;
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const n = [
      Math.cos(theta) * sinPhi,
      Math.cos(phi),
      Math.sin(theta) * sinPhi,
    ];
    return { p: [n[0] * radius, n[1] * radius, n[2] * radius], n, uv: [u, 1 - v] };
  };

  for (let iy = 0; iy < heightSegments; iy++) {
    for (let ix = 0; ix < widthSegments; ix++) {
      const a = vertex(iy, ix);
      const b = vertex(iy, ix + 1);
      const c = vertex(iy + 1, ix + 1);
      const d = vertex(iy + 1, ix);
      // Skip degenerate triangles at the poles.
      const push = (...vs) => {
        for (const vtx of vs) { positions.push(...vtx.p); normals.push(...vtx.n); uvs.push(...vtx.uv); }
      };
      if (iy !== 0) push(a, b, d);
      if (iy !== heightSegments - 1) push(b, c, d);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}

// --- Platonic-ish solids built from a face triangulation of unit vertices ---

/**
 * Builds non-indexed triangle-soup data from a list of unit-sphere-projected
 * vertices and triangle indices, scaled to `radius`. Flat per-face normals,
 * a trivial planar UV. Shared by octahedron/dodecahedron.
 */
function polyhedronData(vertices, indices, radius) {
  const positions = [];
  const normals = [];
  const uvs = [];

  for (let i = 0; i < indices.length; i += 3) {
    const tri = [indices[i], indices[i + 1], indices[i + 2]].map((idx) => {
      const v = vertices[idx];
      return normalize3([v[0], v[1], v[2]]);
    });
    const n = faceNormal(tri[0], tri[1], tri[2]);
    for (const v of tri) {
      positions.push(v[0] * radius, v[1] * radius, v[2] * radius);
      normals.push(...n);
      // Spherical UV from the unit direction.
      uvs.push(Math.atan2(v[2], v[0]) / (Math.PI * 2) + 0.5, Math.asin(clamp(v[1], -1, 1)) / Math.PI + 0.5);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}

export function octahedronData(radius = 0.5) {
  const vertices = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  ];
  const indices = [
    0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2,
    1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2,
  ];
  return polyhedronData(vertices, indices, radius);
}

export function dodecahedronData(radius = 0.5) {
  // 20 vertices of a regular dodecahedron, on the unit sphere. Faces are
  // derived by tracing the edge graph (each vertex has exactly 3 neighbours at
  // the minimum edge length) into 12 pentagons — guaranteed watertight and
  // consistently wound, with no hand-listed connectivity to get wrong.
  const t = (1 + Math.sqrt(5)) / 2; // φ
  const r = 1 / t;
  const verts = [
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    [0, -r, -t], [0, -r, t], [0, r, -t], [0, r, t],
    [-r, -t, 0], [-r, t, 0], [r, -t, 0], [r, t, 0],
    [-t, 0, -r], [t, 0, -r], [-t, 0, r], [t, 0, r],
  ].map((v) => normalize3(v));

  const faces = tracePolyhedronFaces(verts, 3);

  const positions = [], normals = [], uvs = [];
  for (const face of faces) {
    const dir = normalize3(face.reduce((s, i) => add3(s, verts[i]), [0, 0, 0]));
    for (let k = 1; k < face.length - 1; k++) {
      const tri = [verts[face[0]], verts[face[k]], verts[face[k + 1]]];
      const fn = cross3(sub3(tri[1], tri[0]), sub3(tri[2], tri[0]));
      if (dot3(fn, dir) < 0) tri.reverse();
      for (const v of tri) {
        positions.push(v[0] * radius, v[1] * radius, v[2] * radius);
        normals.push(...dir);
        uvs.push(0.5, 0.5);
      }
    }
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}

/**
 * Traces the faces of a convex polyhedron from its vertices, given the valence
 * (neighbours per vertex). Builds the edge graph from nearest neighbours, then
 * walks each directed edge turning to the most clockwise next edge in the
 * face's plane — yielding each face as an ordered vertex-index loop. Each
 * undirected edge is shared by exactly two faces.
 */
function tracePolyhedronFaces(verts, valence) {
  const n = verts.length;
  // Min edge length -> adjacency.
  let minD = Infinity;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const d = dist3(verts[i], verts[j]);
    if (d < minD) minD = d;
  }
  const eps = minD * 1e-3;
  const adj = verts.map((_, i) => {
    const near = [];
    for (let j = 0; j < n; j++) if (j !== i && Math.abs(dist3(verts[i], verts[j]) - minD) < eps) near.push(j);
    return near.slice(0, valence);
  });

  const faces = [];
  const used = new Set(); // directed edges "a->b" already consumed
  for (let a = 0; a < n; a++) {
    for (const b of adj[a]) {
      if (used.has(a + '->' + b)) continue;
      const face = [a];
      let prev = a, cur = b;
      while (cur !== a) {
        face.push(cur);
        used.add(prev + '->' + cur);
        // Turn left: among cur's neighbours (excluding prev) pick the one that
        // continues the same face — the most clockwise around the outward normal.
        const center = normalize3(verts[cur]);
        const inDir = normalize3(sub3(verts[cur], verts[prev]));
        let best = -1, bestAngle = Infinity;
        for (const nb of adj[cur]) {
          if (nb === prev) continue;
          const outDir = normalize3(sub3(verts[nb], verts[cur]));
          // Signed turn angle about the outward normal at cur.
          const crossV = cross3(inDir, outDir);
          const sign = dot3(crossV, center);
          let ang = Math.atan2(sign, dot3(inDir, outDir));
          if (ang < bestAngle) { bestAngle = ang; best = nb; }
        }
        used.add(prev + '->' + cur);
        prev = cur; cur = best;
      }
      used.add(prev + '->' + cur);
      faces.push(face);
    }
  }
  return faces;
}

/**
 * A tube swept along a 3D polyline. `points` is an array of [x,y,z]; the path
 * is resampled with a Catmull-Rom spline (`pathSegments` samples), then a ring
 * of `radialSegments` is extruded along it. Smooth radial normals. Used for
 * conveyor belts. Open (not closed) by default.
 */
export function tubeData(points, radius = 0.1, radialSegments = 8, pathSegments = 64, closed = false) {
  const path = sampleCatmullRom(points, pathSegments, closed);
  const frames = computeFrames(path, closed);

  const positions = [];
  const normals = [];
  const uvs = [];

  const ring = (i) => {
    const c = path[i];
    const { normal, binormal } = frames[i];
    const out = [];
    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * Math.PI * 2;
      const sin = Math.sin(v), cos = -Math.cos(v);
      const nx = cos * normal[0] + sin * binormal[0];
      const ny = cos * normal[1] + sin * binormal[1];
      const nz = cos * normal[2] + sin * binormal[2];
      out.push({
        p: [c[0] + radius * nx, c[1] + radius * ny, c[2] + radius * nz],
        n: [nx, ny, nz],
        uv: [i / (path.length - 1), j / radialSegments],
      });
    }
    return out;
  };

  for (let i = 0; i < path.length - 1; i++) {
    const r0 = ring(i), r1 = ring(i + 1);
    for (let j = 0; j < radialSegments; j++) {
      const a = r0[j], b = r0[j + 1], c = r1[j + 1], d = r1[j];
      for (const v of [a, d, c, a, c, b]) {
        positions.push(...v.p); normals.push(...v.n); uvs.push(...v.uv);
      }
    }
  }

  // End caps (open tubes only): a fan over each end ring, facing along the
  // path tangent, so the tube reads as a solid object rather than a hollow
  // pipe you can see through.
  if (!closed && path.length > 1) {
    const capFan = (ringVerts, center, capNormal, flip) => {
      for (let j = 0; j < radialSegments; j++) {
        const e0 = ringVerts[j].p, e1 = ringVerts[j + 1].p;
        const tri = flip ? [center, e1, e0] : [center, e0, e1];
        for (const pt of tri) { positions.push(...pt); normals.push(...capNormal); uvs.push(0.5, 0.5); }
      }
    };
    const startRing = ring(0);
    const endRing = ring(path.length - 1);
    const startTan = frames[0].tangent;
    const endTan = frames[path.length - 1].tangent;
    // Start cap faces backward (-tangent); end cap faces forward (+tangent).
    capFan(startRing, path[0], [-startTan[0], -startTan[1], -startTan[2]], false);
    capFan(endRing, path[path.length - 1], endTan, true);
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
  };
}

// --- Vertex normals ---

/**
 * Computes per-vertex normals for a position buffer (Float32Array, xyz per
 * vertex). For non-indexed geometry every triangle's three vertices get that
 * triangle's flat face normal — matching THREE.BufferGeometry.computeVertexNormals
 * on a non-indexed buffer. If `indices` is given, normals are accumulated per
 * shared vertex and normalized (smooth shading).
 *
 * Returns a Float32Array the same length as `positions`.
 */
export function computeVertexNormals(positions, indices = null) {
  const normals = new Float32Array(positions.length);

  const addFace = (ia, ib, ic) => {
    const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
    const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
    const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    // Cross product, left un-normalized so larger triangles weight more
    // (matches Three's area-weighted accumulation for indexed geometry).
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  };

  if (indices) {
    for (let i = 0; i < indices.length; i += 3) {
      addFace(indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3);
    }
  } else {
    for (let i = 0; i < positions.length; i += 9) {
      addFace(i, i + 3, i + 6);
    }
  }

  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= len; normals[i + 1] /= len; normals[i + 2] /= len;
  }
  return normals;
}

// --- Geometry construction ---

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

// --- internal helpers ---

function normalize3(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Rotates vector v around unit axis k by `angle` radians (Rodrigues). */
function rotateAroundAxis(v, k, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const kv = cross3(k, v);
  const kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
  return [
    v[0] * c + kv[0] * s + k[0] * kd * (1 - c),
    v[1] * c + kv[1] * s + k[1] * kd * (1 - c),
    v[2] * c + kv[2] * s + k[2] * kd * (1 - c),
  ];
}

function faceNormal(a, b, c) {
  const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  return normalize3([
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  ]);
}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

/**
 * Samples a CENTRIPETAL Catmull-Rom spline (alpha = 0.5) through `points`,
 * producing `segments` total samples. Centripetal parameterization is the key
 * choice here: unlike the uniform variant it provably never forms cusps or
 * self-intersecting loops between control points, so a swept tube doesn't fold
 * through itself at sharp corners. Endpoints are duplicated for the open case
 * so the curve passes through the first and last control points.
 */
function sampleCatmullRom(points, segments, closed) {
  const n = points.length;
  if (n < 2) return points.map((p) => [...p]);

  const get = (i) => {
    if (closed) return points[((i % n) + n) % n];
    return points[clamp(i, 0, n - 1)];
  };

  const out = [];
  const spans = closed ? n : n - 1;
  const perSpan = Math.max(2, Math.floor(segments / spans));
  for (let i = 0; i < spans; i++) {
    const p0 = get(i - 1), p1 = get(i), p2 = get(i + 1), p3 = get(i + 2);
    for (let s = 0; s < perSpan; s++) {
      out.push(catmullRomCentripetal(p0, p1, p2, p3, s / perSpan));
    }
  }
  if (!closed) out.push([...points[n - 1]]);
  return out;
}

// Centripetal Catmull-Rom (Barry-Goldman), alpha = 0.5. `t` in [0,1] maps
// across the p1->p2 span.
function catmullRomCentripetal(p0, p1, p2, p3, t) {
  // Knot sequence with alpha = 0.5 (centripetal): t_{i+1} = t_i + |p_{i+1}-p_i|^0.5
  const t0 = 0;
  const t1 = t0 + Math.pow(Math.max(dist3(p0, p1), 1e-6), 0.5);
  const t2 = t1 + Math.pow(Math.max(dist3(p1, p2), 1e-6), 0.5);
  const t3v = t2 + Math.pow(Math.max(dist3(p2, p3), 1e-6), 0.5);
  const tt = t1 + (t2 - t1) * t;

  const lerpKnot = (a, b, ta, tb) => {
    const f = (tb - tt) / (tb - ta);
    const g = (tt - ta) / (tb - ta);
    return [a[0] * f + b[0] * g, a[1] * f + b[1] * g, a[2] * f + b[2] * g];
  };
  const A1 = lerpKnot(p0, p1, t0, t1);
  const A2 = lerpKnot(p1, p2, t1, t2);
  const A3 = lerpKnot(p2, p3, t2, t3v);
  const B1 = lerpKnot(A1, A2, t0, t2);
  const B2 = lerpKnot(A2, A3, t1, t3v);
  return lerpKnot(B1, B2, t1, t2);
}

/**
 * Computes a per-sample tangent/normal/binormal frame along a path using the
 * parallel-transport method (a stable frame with minimal twist), so the swept
 * tube doesn't spin around its axis.
 */
function computeFrames(path, closed) {
  const n = path.length;
  const tangents = [];
  for (let i = 0; i < n; i++) {
    const prev = path[Math.max(0, i - 1)];
    const next = path[Math.min(n - 1, i + 1)];
    tangents.push(normalize3([next[0] - prev[0], next[1] - prev[1], next[2] - prev[2]]));
  }

  // Seed an initial normal perpendicular to the first tangent.
  const t0 = tangents[0];
  let normal = Math.abs(t0[1]) < 0.99
    ? normalize3(cross([0, 1, 0], t0))
    : normalize3(cross([1, 0, 0], t0));

  const frames = [];
  let prevT = tangents[0];
  for (let i = 0; i < n; i++) {
    const t = tangents[i];
    // Parallel-transport the carried normal by rotating it with the same
    // rotation that takes the previous tangent to the current one (Rodrigues).
    // This preserves the frame's handedness and length even at sharp bends —
    // a plain re-projection can collapse and flip the normal there, which
    // inverted whole sections of the swept tube (and made them vanish under
    // back-face culling).
    const axis = cross(prevT, t);
    const axisLen = Math.hypot(axis[0], axis[1], axis[2]);
    if (axisLen > 1e-6) {
      const a = [axis[0] / axisLen, axis[1] / axisLen, axis[2] / axisLen];
      const cosA = clamp(prevT[0] * t[0] + prevT[1] * t[1] + prevT[2] * t[2], -1, 1);
      const angle = Math.acos(cosA);
      normal = rotateAroundAxis(normal, a, angle);
    }
    // Guard against drift: re-orthogonalize against the current tangent.
    const d = normal[0] * t[0] + normal[1] * t[1] + normal[2] * t[2];
    normal = normalize3([normal[0] - t[0] * d, normal[1] - t[1] * d, normal[2] - t[2] * d]);
    const binormal = normalize3(cross(t, normal));
    frames.push({ tangent: t, normal, binormal });
    prevT = t;
  }
  return frames;
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
