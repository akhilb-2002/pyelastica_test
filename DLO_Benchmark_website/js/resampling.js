/**
 * resampling.js
 * ---------------------------------------------------------------------
 * Ports resample_curve() / trace_polyline() from the Python script.
 *
 * Rather than resampling by fractional arc-length (which can drift
 * under floating point error), the original approach walks the dense
 * polyline with a compass of fixed radius ("segment_length"), finding
 * the exact next intersection point, then binary-searches on that
 * radius until exactly N nodes are produced and the final node lands
 * as close as possible to the end of the dense curve. This guarantees
 * genuinely uniform spacing rather than approximately uniform spacing.
 * ---------------------------------------------------------------------
 */
const Resampling = (() => {
  const G = Geometry;

  function dedupeConsecutive(points) {
    const out = [points[0]];
    for (let i = 1; i < points.length; i++) {
      const d = G.norm(G.sub(points[i], points[i - 1]));
      if (d > 1e-6) out.push(points[i]);
    }
    return out;
  }

  /**
   * Walk the dense polyline with compass radius `segmentLength`,
   * producing exactly numNodes points, or null if the polyline runs
   * out before reaching numNodes.
   */
  function tracePolyline(points, segmentLength, numNodes) {
    const resampled = [points[0]];
    let currentIdx = 0;
    let currentPos = points[0];

    for (let step = 1; step < numNodes; step++) {
      let found = false;

      for (let i = currentIdx; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i + 1];
        const ts = G.circleSegmentIntersection(p1, p2, currentPos, segmentLength);

        const validForwardTs = [];
        for (const t of ts) {
          if (i === currentIdx) {
            const d = G.sub(p2, p1);
            const denom = G.dot(d, d);
            if (denom > 1e-8) {
              const currT = G.dot(G.sub(currentPos, p1), d) / denom;
              if (t > currT + 1e-5) validForwardTs.push(t);
            }
          } else {
            validForwardTs.push(t);
          }
        }

        if (validForwardTs.length > 0) {
          const t = Math.min(...validForwardTs);
          const newPos = G.lerp(p1, p2, t);
          resampled.push(newPos);
          currentIdx = i;
          currentPos = newPos;
          found = true;
          break;
        }
      }

      if (!found) return null;
    }

    return resampled;
  }

  /**
   * Binary-search the compass radius so that walking the dense
   * polyline produces exactly numNodes uniformly spaced points whose
   * last node lands as close as possible to the true curve end.
   */
  function resampleCurve(rawPoints, numNodes) {
    const points = dedupeConsecutive(rawPoints);
    const cumDists = G.computeArcLength(points);
    const totalLength = cumDists[cumDists.length - 1];

    let low = 0.0;
    let high = (totalLength / (numNodes - 1)) * 1.5;
    let best = null;
    let bestError = Infinity;

    for (let iter = 0; iter < 60; iter++) {
      const mid = (low + high) / 2;
      const res = tracePolyline(points, mid, numNodes);

      if (res !== null) {
        const distToEnd = G.norm(G.sub(res[res.length - 1], points[points.length - 1]));
        if (distToEnd < bestError) {
          bestError = distToEnd;
          best = res;
        }
        low = mid; // try to reach further toward the end
      } else {
        high = mid; // radius too large, ran out of polyline
      }
    }

    if (best === null) {
      throw new Error('Failed to resample curve: could not reach the requested node count.');
    }
    return best;
  }

  function scaleToRopeLength(points, targetLength) {
    const currentLength = G.totalLength(points);
    if (currentLength === 0) throw new Error('Cannot scale a zero-length rope.');
    const factor = targetLength / currentLength;
    return points.map(p => [p[0] * factor, p[1] * factor]);
  }

  /** Translate so the LAST node (Node N-1) lands exactly at origin. */
  function translateToOrigin(points, origin) {
    const last = points[points.length - 1];
    const shift = [origin[0] - last[0], origin[1] - last[1]];
    return points.map(p => [p[0] + shift[0], p[1] + shift[1]]);
  }

  return { resampleCurve, scaleToRopeLength, translateToOrigin, dedupeConsecutive };
})();
