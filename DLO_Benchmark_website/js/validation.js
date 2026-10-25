/**
 * validation.js
 * ---------------------------------------------------------------------
 * Ports validate_rope() from the Python script. Every generated rope
 * must pass all five checks before it is drawn or exported.
 * ---------------------------------------------------------------------
 */
const Validation = (() => {
  const G = Geometry;

  /**
   * @returns {{ok: boolean, checks: {label:string, pass:boolean, detail:string}[]}}
   */
  function validateRope(points, targetLength, targetNodes) {
    const checks = [];

    // 1. exact node count
    const nodeCountOk = points.length === targetNodes;
    checks.push({
      label: 'Exactly N nodes',
      pass: nodeCountOk,
      detail: `expected ${targetNodes}, got ${points.length}`
    });

    // 2. no NaNs
    const hasNaN = points.some(([x, y]) => Number.isNaN(x) || Number.isNaN(y));
    checks.push({ label: 'No NaN values', pass: !hasNaN, detail: hasNaN ? 'NaN found in node coordinates' : 'clean' });

    // 3. total length matches
    const cum = G.computeArcLength(points);
    const actualLength = cum[cum.length - 1] || 0;
    const lengthOk = Math.abs(actualLength - targetLength) < 1e-3 + targetLength * 1e-6;
    checks.push({
      label: 'Total length equals rope length',
      pass: lengthOk,
      detail: `expected ${targetLength.toFixed(3)} mm, got ${actualLength.toFixed(3)} mm`
    });

    // 4. no duplicate consecutive nodes / continuous polyline
    const segLens = [];
    for (let i = 1; i < cum.length; i++) segLens.push(cum[i] - cum[i - 1]);
    const noDup = segLens.every(d => d > 1e-6);
    checks.push({
      label: 'Continuous polyline, no duplicate nodes',
      pass: noDup,
      detail: noDup ? 'all segment lengths positive' : 'zero-length segment detected'
    });

    // 5. uniform spacing within tolerance
    const expectedSpacing = targetLength / (targetNodes - 1);
    const tol = targetLength * 0.01;
    const uniform = segLens.every(d => Math.abs(d - expectedSpacing) <= tol);
    const minSpacing = segLens.length ? Math.min(...segLens) : 0;
    const maxSpacing = segLens.length ? Math.max(...segLens) : 0;
    checks.push({
      label: 'Uniform node spacing',
      pass: uniform,
      detail: `expected ${expectedSpacing.toFixed(3)} mm, range [${minSpacing.toFixed(3)}, ${maxSpacing.toFixed(3)}] mm`
    });

    const ok = checks.every(c => c.pass);
    return { ok, checks };
  }

  return { validateRope };
})();
