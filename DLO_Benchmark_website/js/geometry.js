/**
 * geometry.js
 * ---------------------------------------------------------------------
 * Low level vector math and path primitives. Mirrors the numpy helpers
 * in the reference Python script (compute_arc_length, sample_line,
 * sample_arc, the circle/segment intersection used for arc-length
 * accurate resampling).
 *
 * Points are represented throughout the app as plain [x, y] arrays so
 * they can be passed to canvas / JSON without conversion.
 * ---------------------------------------------------------------------
 */
const Geometry = (() => {

  function sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
  function scale(a, s) { return [a[0] * s, a[1] * s]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
  function norm(a) { return Math.hypot(a[0], a[1]); }
  function lerp(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; }

  /** Cumulative arc length along a polyline. Returns Float64Array of length N. */
  function computeArcLength(points) {
    const n = points.length;
    const cum = new Float64Array(n);
    if (n <= 1) return cum;
    for (let i = 1; i < n; i++) {
      cum[i] = cum[i - 1] + norm(sub(points[i], points[i - 1]));
    }
    return cum;
  }

  function totalLength(points) {
    const cum = computeArcLength(points);
    return cum[cum.length - 1] || 0;
  }

  /** Evenly spaced points (inclusive) between p1 and p2. */
  function sampleLine(p1, p2, numPoints = 200) {
    const out = [];
    for (let i = 0; i < numPoints; i++) {
      const t = numPoints === 1 ? 0 : i / (numPoints - 1);
      out.push(lerp(p1, p2, t));
    }
    return out;
  }

  /** Evenly spaced points (inclusive) along a circular arc. */
  function sampleArc(center, radius, startAngle, endAngle, numPoints = 200) {
    const out = [];
    for (let i = 0; i < numPoints; i++) {
      const t = numPoints === 1 ? 0 : i / (numPoints - 1);
      const angle = startAngle + (endAngle - startAngle) * t;
      out.push([center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)]);
    }
    return out;
  }

  /** Dense samples of a sine curve y = amplitude*sin(x * freq * PI) over x in [x0, x1]. */
  function sampleSine(x0, x1, amplitude, freqMultiple, numPoints = 200, invert = false) {
    const out = [];
    for (let i = 0; i < numPoints; i++) {
      const t = i / (numPoints - 1);
      const x = x0 + (x1 - x0) * t;
      let y = Math.sin(x * freqMultiple * Math.PI) * amplitude;
      if (invert) y = -y;
      out.push([x, y]);
    }
    return out;
  }

  /**
   * Real roots (parametric t along segment p1->p2) where the segment
   * crosses a circle of given center/radius. Mirrors
   * find_circle_segment_intersection from the Python script.
   */
  function circleSegmentIntersection(p1, p2, center, radius) {
    const d = sub(p2, p1);
    const f = sub(p1, center);
    const a = dot(d, d);
    if (a < 1e-8) return [];
    const b = 2 * dot(f, d);
    const c = dot(f, f) - radius * radius;
    const disc = b * b - 4 * a * c;
    if (disc < 0) return [];
    const sq = Math.sqrt(disc);
    const t1 = (-b - sq) / (2 * a);
    const t2 = (-b + sq) / (2 * a);
    const valid = [];
    for (const t of [t1, t2]) {
      if (t >= -1e-6 && t <= 1 + 1e-6) valid.push(Math.min(1, Math.max(0, t)));
    }
    return valid;
  }

  /** Axis aligned bounding box of a point set. */
  function boundingBox(points) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
  }

  /**
   * Discrete curvature estimate at interior node i using the
   * standard turning-angle formula:
   *   k = 2*sin(theta/2) / chord(p_{i-1}, p_{i+1})
   * Returns array aligned with input nodes (0 at the endpoints).
   */
  function curvatureEstimate(points) {
    const n = points.length;
    const k = new Float64Array(n);
    for (let i = 1; i < n - 1; i++) {
      const v1 = sub(points[i], points[i - 1]);
      const v2 = sub(points[i + 1], points[i]);
      const l1 = norm(v1), l2 = norm(v2);
      if (l1 < 1e-9 || l2 < 1e-9) continue;
      const cross = v1[0] * v2[1] - v1[1] * v2[0];
      const dp = dot(v1, v2);
      const theta = Math.atan2(cross, dp); // signed turning angle
      const chord = norm(sub(points[i + 1], points[i - 1]));
      if (chord < 1e-9) continue;
      k[i] = Math.abs(2 * Math.sin(theta / 2) / chord);
    }
    return k;
  }

  return {
    sub, add, scale, dot, norm, lerp,
    computeArcLength, totalLength,
    sampleLine, sampleArc, sampleSine,
    circleSegmentIntersection, boundingBox, curvatureEstimate
  };
})();
