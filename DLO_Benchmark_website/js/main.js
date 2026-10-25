/**
 * main.js
 * ---------------------------------------------------------------------
 * Wires the UI to the geometry pipeline:
 *   1. createCenterline(shape)          dense ideal curve
 *   2. resampleCurve(dense, N)          exactly N uniformly spaced nodes
 *   3. scaleToRopeLength(nodes, L)      total arc length == L
 *   4. translateToOrigin(nodes, O)      last node lands at origin
 *   5. validateRope(nodes, L, N)        five-point validation
 * The centerline is always regenerated first; resampling, scaling and
 * translation always happen after, so changing rope length or node
 * count never touches the underlying shape geometry.
 * ---------------------------------------------------------------------
 */
(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const state = {
    shape: 'C',
    ropeLength: 300,
    numNodes: 30,
    originX: 300,
    originY: 0,
    waveAmplitude: 1.0,
    previewScale: 1,
    nodeSize: 4,
    showGrid: true,
    darkMode: false,
    showNumbers: true,
    showMarkers: true,
    result: null // { nodes, record, validation, measurements }
  };

  const canvasEl = $('#ropeCanvas');
  const ropeCanvas = new RopeCanvas(canvasEl);

  // ---------------------------------------------------------------
  // Shape list (left sidebar)
  // ---------------------------------------------------------------
  function buildShapeList() {
    const list = $('#shapeList');
    list.innerHTML = '';
    Letters.SHAPES.forEach((shape, i) => {
      const li = document.createElement('li');
      li.className = 'shape-item';
      li.dataset.shape = shape;
      const t = i / (Letters.SHAPES.length - 1);
      li.innerHTML = `
        <span class="shape-dot" style="--dot:${spectrumCss(t)}"></span>
        <span class="shape-glyph">${shape}</span>
      `;
      li.addEventListener('click', () => {
        state.shape = shape;
        $('#shapeSelect').value = shape;
        setActiveShape(shape);
        regenerate();
      });
      list.appendChild(li);
    });
    setActiveShape(state.shape);
  }

  function setActiveShape(shape) {
    $$('.shape-item').forEach(el => el.classList.toggle('active', el.dataset.shape === shape));
  }

  function spectrumCss(t) {
    const stops = ['#2E3192', '#6A2C9B', '#962C90', '#BE2F71', '#D2542E', '#C77B15', '#9C7C18'];
    const n = stops.length - 1;
    const idx = Math.min(Math.round(t * n), n);
    return stops[idx];
  }

  // ---------------------------------------------------------------
  // Populate the <select> fallback (kept in sync with sidebar)
  // ---------------------------------------------------------------
  function buildShapeSelect() {
    const sel = $('#shapeSelect');
    sel.innerHTML = Letters.SHAPES.map(s => `<option value="${s}">${s}</option>`).join('');
    sel.value = state.shape;
    sel.addEventListener('change', () => {
      state.shape = sel.value;
      setActiveShape(state.shape);
      regenerate();
    });
  }

  // ---------------------------------------------------------------
  // Pipeline
  // ---------------------------------------------------------------
  function generateRope() {
    if (state.numNodes < 2) throw new Error('Node count must be at least 2.');
    if (state.ropeLength <= 0) throw new Error('Rope length must be positive.');

    const dense = Letters.createCenterline(state.shape, state.waveAmplitude);
    const resampled = Resampling.resampleCurve(dense, state.numNodes);
    const scaled = Resampling.scaleToRopeLength(resampled, state.ropeLength);
    const origin = [state.originX, state.originY];
    const final = Resampling.translateToOrigin(scaled, origin);
    const validation = Validation.validateRope(final, state.ropeLength, state.numNodes);
    const measurements = computeMeasurements(final, state.ropeLength, state.numNodes);
    const record = Exporter.buildRecord(state.shape, state.ropeLength, state.numNodes, origin, final);

    return { nodes: final, origin, validation, measurements, record };
  }

  function computeMeasurements(points, ropeLength, numNodes) {
    const bb = Geometry.boundingBox(points);
    const spacing = ropeLength / (numNodes - 1);
    const curv = Geometry.curvatureEstimate(points);
    const maxCurv = curv.length ? Math.max(...curv) : 0;
    return {
      totalLength: ropeLength,
      numNodes,
      segmentLength: spacing,
      bbox: bb,
      maxCurvature: maxCurv
    };
  }

  let lastFitShape = null;

  function regenerate({ refit = false } = {}) {
    const errorBox = $('#errorBox');
    try {
      const result = generateRope();
      state.result = result;
      errorBox.hidden = true;

      ropeCanvas.setData(result.nodes, result.origin);
      ropeCanvas.setOptions({
        showGrid: state.showGrid,
        darkMode: state.darkMode,
        showNumbers: state.showNumbers,
        showMarkers: state.showMarkers,
        nodeSize: state.nodeSize
      });

      if (refit || lastFitShape !== state.shape) {
        ropeCanvas.fitToView();
        lastFitShape = state.shape;
      } else {
        ropeCanvas.render();
      }

      renderMeasurements(result.measurements);
      renderValidation(result.validation);
      document.body.classList.toggle('dark', state.darkMode);
    } catch (err) {
      errorBox.hidden = false;
      errorBox.textContent = `⚠ Generation failed — ${err.message}`;
      console.error(err);
    }
  }

  function renderMeasurements(m) {
    $('#mTotalLength').textContent = `${m.totalLength.toFixed(2)} mm`;
    $('#mNumNodes').textContent = `${m.numNodes}`;
    $('#mSegment').textContent = `${m.segmentLength.toFixed(3)} mm`;
    $('#mBbox').textContent = `${m.bbox.width.toFixed(2)} × ${m.bbox.height.toFixed(2)} mm`;
    $('#mCurvature').textContent = `${m.maxCurvature.toExponential(3)} mm⁻¹`;
  }

  function renderValidation(v) {
    const box = $('#validationList');
    box.innerHTML = '';
    v.checks.forEach(c => {
      const row = document.createElement('div');
      row.className = 'check-row' + (c.pass ? '' : ' fail');
      row.innerHTML = `
        <span class="check-mark">${c.pass ? '✓' : '✕'}</span>
        <span class="check-label">${c.label}</span>
        <span class="check-detail">${c.detail}</span>
      `;
      box.appendChild(row);
    });
    $('#validationStatus').textContent = v.ok ? 'All checks passed' : 'Validation failed';
    $('#validationStatus').className = 'validation-status ' + (v.ok ? 'ok' : 'bad');
  }

  // ---------------------------------------------------------------
  // Controls wiring
  // ---------------------------------------------------------------
  function bindNumberInput(id, key, { min, refit = false } = {}) {
    const el = $(id);
    el.addEventListener('input', () => {
      let v = parseFloat(el.value);
      if (Number.isNaN(v)) return;
      if (min !== undefined) v = Math.max(min, v);
      state[key] = v;
      regenerate({ refit });
    });
  }

  function bindToggle(id, key) {
    const el = $(id);
    el.addEventListener('change', () => {
      state[key] = el.checked;
      regenerate();
    });
  }

  function initControls() {
    bindNumberInput('#ropeLength', 'ropeLength', { min: 0.001 });
    bindNumberInput('#numNodes', 'numNodes', { min: 2, refit: true });
    bindNumberInput('#originX', 'originX');
    bindNumberInput('#originY', 'originY');
    bindNumberInput('#waveAmplitude', 'waveAmplitude', { min: 0, refit: true });
    bindNumberInput('#nodeSize', 'nodeSize');

    bindToggle('#toggleGrid', 'showGrid');
    bindToggle('#toggleDark', 'darkMode');
    bindToggle('#toggleNumbers', 'showNumbers');
    bindToggle('#toggleMarkers', 'showMarkers');

    $('#btnFit').addEventListener('click', () => ropeCanvas.fitToView());
    $('#btnZoomIn').addEventListener('click', () => ropeCanvas.zoomAt(1.25, [ropeCanvas.cssW / 2, ropeCanvas.cssH / 2]));
    $('#btnZoomOut').addEventListener('click', () => ropeCanvas.zoomAt(1 / 1.25, [ropeCanvas.cssW / 2, ropeCanvas.cssH / 2]));

    $('#exportJSON').addEventListener('click', () => state.result && Exporter.exportJSON(state.result.record));
    $('#exportCSV').addEventListener('click', () => state.result && Exporter.exportCSV(state.result.record));
    $('#exportTXT').addEventListener('click', () => state.result && Exporter.exportTXT(state.result.record));

    $('#exportAllJSON').addEventListener('click', () => {
      const allRecords = Letters.SHAPES.map(shape => {
        const dense = Letters.createCenterline(shape, state.waveAmplitude);
        const resampled = Resampling.resampleCurve(dense, state.numNodes);
        const scaled = Resampling.scaleToRopeLength(resampled, state.ropeLength);
        const origin = [state.originX, state.originY];
        const final = Resampling.translateToOrigin(scaled, origin);
        return Exporter.buildRecord(shape, state.ropeLength, state.numNodes, origin, final);
      });
      Exporter.exportAllJSON(allRecords);
    });
  }

  // ---------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------
  function boot() {
    buildShapeList();
    buildShapeSelect();
    initControls();
    regenerate({ refit: true });
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
