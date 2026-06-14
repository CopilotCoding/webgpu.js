// Quaternions as [x, y, z, w], matching the rotation argument of
// mat4.fromTranslationRotationScale. These cover the orientation math a game
// needs when placing/moving objects (aligning a building to a surface normal,
// pointing a belt/laser along a direction) — computed once on change, not per
// frame.

export function identity() { return [0, 0, 0, 1]; }

/** Quaternion from an axis (need not be unit) and an angle in radians. */
export function fromAxisAngle(axis, angle) {
  const len = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(angle / 2) / len;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

/**
 * Shortest-arc rotation taking unit vector `from` to unit vector `to`
 * (THREE.Quaternion.setFromUnitVectors). Handles the antiparallel case.
 */
export function fromUnitVectors(from, to) {
  let r = from[0] * to[0] + from[1] * to[1] + from[2] * to[2] + 1;
  let x, y, z;
  if (r < 1e-6) {
    // from and to are opposite — pick any perpendicular axis.
    r = 0;
    if (Math.abs(from[0]) > Math.abs(from[2])) { x = -from[1]; y = from[0]; z = 0; }
    else { x = 0; y = -from[2]; z = from[1]; }
  } else {
    x = from[1] * to[2] - from[2] * to[1];
    y = from[2] * to[0] - from[0] * to[2];
    z = from[0] * to[1] - from[1] * to[0];
  }
  return normalize([x, y, z, r]);
}

/** Quaternion from yaw/pitch/roll (radians), applied Z*Y*X order. */
export function fromEuler(x, y, z) {
  const cx = Math.cos(x / 2), sx = Math.sin(x / 2);
  const cy = Math.cos(y / 2), sy = Math.sin(y / 2);
  const cz = Math.cos(z / 2), sz = Math.sin(z / 2);
  return [
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  ];
}

/** Hamilton product a * b. */
export function multiply(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/**
 * Quaternion from an orthonormal basis given as three column axes
 * (THREE.Quaternion.setFromRotationMatrix). Useful to build an orientation
 * from a surface normal plus a tangent.
 */
export function fromBasis(xAxis, yAxis, zAxis) {
  const m00 = xAxis[0], m10 = xAxis[1], m20 = xAxis[2];
  const m01 = yAxis[0], m11 = yAxis[1], m21 = yAxis[2];
  const m02 = zAxis[0], m12 = zAxis[1], m22 = zAxis[2];
  const trace = m00 + m11 + m22;
  let x, y, z, w;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    w = 0.25 / s; x = (m21 - m12) * s; y = (m02 - m20) * s; z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  return normalize([x, y, z, w]);
}

function normalize(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}
