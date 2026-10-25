/**
 * letters.js
 * ---------------------------------------------------------------------
 * Procedural centerline construction for every supported shape.
 * Each shape is built ONLY from the three geometric primitives
 * (lines, arcs, sine curves) on a shared unit grid, exactly mirroring
 * create_centerline() in the reference Python script. No fonts, no
 * SVG text, no image tracing.
 *
 * createCenterline(shape, opts) -> dense [x,y][] polyline
 *   opts.waveAmplitude  multiplier applied to the base amplitude of
 *                       every sine-based shape (S, Z, N, M, W)
 * ---------------------------------------------------------------------
 */
const Letters = (() => {

  const SHAPES = ['I', 'J', 'L', 'C', 'D', 'O', 'P', 'S', 'U', 'V', 'Z', 'N', 'M', 'W', 'T'];

  const DESCRIPTIONS = {
    I: 'Single vertical line',
    J: 'Circular hook + vertical line',
    L: 'Horizontal line, rounded quarter-circle, vertical line',
    C: 'Single circular arc',
    D: 'Vertical line, connector, semicircle, connector',
    O: 'Full circle',
    P: 'Vertical stem + semicircular loop',
    S: 'Smooth sine wave',
    U: 'Horizontal line, semicircle, horizontal line',
    V: 'Two straight lines',
    Z: 'Inverted smooth sine wave',
    N: 'Smooth zig-zag (single sine period)',
    M: 'Smooth double wave (two sine periods)',
    W: 'Smooth triple wave (three sine periods)',
    T: 'Vertical line with a bar on one side (inverted L)'
  };

  function build(shape, waveAmplitude) {
    const G = Geometry;
    const parts = [];
    const amp = waveAmplitude; // multiplier, 1.0 = base python amplitude

    switch (shape) {
      case 'I':
        parts.push(G.sampleLine([0.5, 0], [0.5, 2]));
        break;

      case 'J':
        parts.push(G.sampleArc([0.5, 0.5], 0.5, Math.PI, 2 * Math.PI));
        parts.push(G.sampleLine([1, 0.5], [1, 2]));
        break;

      case 'L':
        parts.push(G.sampleLine([2, 1.5], [0.5, 1.5]));
        parts.push(G.sampleArc([0.5, 1], 0.5, Math.PI / 2, Math.PI));
        parts.push(G.sampleLine([0, 1], [0, 0]));
        break;

      case 'C':
        parts.push(G.sampleArc([1, 1], 1, Math.PI / 4, 7 * Math.PI / 4));
        break;

      case 'D':
        parts.push(G.sampleLine([0, 0], [0, 1.75]));
        parts.push(G.sampleArc([0.25, 1.75], 0.25, Math.PI, Math.PI / 2));
        parts.push(G.sampleLine([0.25, 2], [0.5, 2]));
        parts.push(G.sampleArc([0.5, 1], 1, Math.PI / 2, -Math.PI / 2));
        parts.push(G.sampleLine([0.5, 0], [0, 0]));
        break;

      case 'O':
        parts.push(G.sampleArc([1, 1], 1, 0, 2 * Math.PI));
        break;

      case 'P':
        parts.push(G.sampleLine([1, 0], [1, 0.5]));
        parts.push(G.sampleArc([0.5, 0.5], 0.5, 0, Math.PI));
        parts.push(G.sampleLine([0, 0.5], [0, 0]));
        parts.push(G.sampleLine([0, 0], [2.088772, 0]));
        break;

      case 'S':
        parts.push(G.sampleSine(0, 2, 0.5 * amp, 1, 200, false));
        break;

      case 'U':
        parts.push(G.sampleLine([2, 1.5], [0.5, 1.5]));
        parts.push(G.sampleArc([0.5, 1], 0.5, Math.PI / 2, 1.5 * Math.PI));
        parts.push(G.sampleLine([0.5, 0.5], [2, 0.5]));
        break;

      case 'V':
        const theta1 = Math.atan2(0.1875, -0.0703125);
        const theta2 = 2 * Math.PI - theta1;
        parts.push(G.sampleLine([2, 1.5], [0.5, 0.9375]));
        parts.push(G.sampleArc([0.5703125, 0.75], 0.200244, theta1, theta2));
        parts.push(G.sampleLine([0.5, 0.5625], [2, 0]));
        break;

      case 'Z':
        parts.push(G.sampleSine(0, 2, 0.25 * amp, 1, 200, true));
        break;

      case 'N':
        parts.push(G.sampleSine(0, 2, 0.25 * amp, 1, 200, false));
        break;

      case 'M':
        parts.push(G.sampleSine(0, 3, 0.25 * amp, 1, 200, false));
        break;

      case 'W':
        parts.push(G.sampleSine(0, 3, 0.25 * amp, 1, 200, true));
        break;

      case 'T':
        parts.push(G.sampleLine([0.5, 0], [0.5, 2]));
        parts.push(G.sampleLine([0.5, 2], [1, 2]));
        break;

      default:
        throw new Error(`Shape "${shape}" is not implemented.`);
    }

    // Flatten parts into one dense continuous polyline (concatenation
    // preserves continuity because every part starts where the
    // previous part ended, exactly as in the Python script).
    const dense = [];
    for (const part of parts) {
      for (const p of part) dense.push(p);
    }
    return dense;
  }

  return {
    SHAPES,
    DESCRIPTIONS,
    createCenterline: build
  };
})();
