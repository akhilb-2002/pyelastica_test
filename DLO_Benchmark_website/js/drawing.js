/**
 * drawing.js
 * ---------------------------------------------------------------------
 * Canvas 2D renderer for the rope visualization: continuous polyline,
 * nodes, node index labels, origin marker, background grid, pan/zoom/
 * fit-to-view, and a light/dark canvas theme independent of the page
 * theme (both follow the "dark mode" toggle).
 * ---------------------------------------------------------------------
 */
class RopeCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.points = [];
    this.origin = [0, 0];
    this.opts = {
      showGrid: true,
      darkMode: false,
      showNumbers: true,
      showMarkers: true,
      nodeSize: 4,
      previewScale: 1
    };

    // world -> screen transform: screen = world * scale + offset (y flipped)
    this.scale = 1;
    this.offset = [0, 0];

    this._bindInteraction();
    this._resizeObserver = new ResizeObserver(() => this._handleResize());
    this._resizeObserver.observe(canvas.parentElement);
    this._handleResize();
  }

  _handleResize() {
    const parent = this.canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cssW = w;
    this.cssH = h;
    this.render();
  }

  setData(points, origin) {
    this.points = points;
    this.origin = origin;
  }

  setOptions(opts) {
    Object.assign(this.opts, opts);
  }

  worldToScreen([x, y]) {
    return [
      x * this.scale + this.offset[0],
      this.cssH - (y * this.scale + this.offset[1])
    ];
  }

  screenToWorld([sx, sy]) {
    return [
      (sx - this.offset[0]) / this.scale,
      (this.cssH - sy - this.offset[1]) / this.scale
    ];
  }

  fitToView(padding = 56) {
    if (!this.points.length) return;
    const bb = Geometry.boundingBox([...this.points, this.origin]);
    const w = Math.max(bb.width, 1e-6);
    const h = Math.max(bb.height, 1e-6);
    const availW = Math.max(this.cssW - padding * 2, 10);
    const availH = Math.max(this.cssH - padding * 2, 10);
    this.scale = Math.min(availW / w, availH / h);
    const cx = (bb.minX + bb.maxX) / 2;
    const cy = (bb.minY + bb.maxY) / 2;
    this.offset = [
      this.cssW / 2 - cx * this.scale,
      this.cssH / 2 - cy * this.scale
    ];
    this.render();
  }

  zoomAt(factor, screenPoint) {
    const before = this.screenToWorld(screenPoint);
    this.scale *= factor;
    this.scale = Math.min(Math.max(this.scale, 0.02), 400);
    const after = this.worldToScreen(before);
    this.offset[0] += screenPoint[0] - after[0];
    this.offset[1] -= (screenPoint[1] - after[1]); // y flipped
    this.render();
  }

  panBy(dx, dy) {
    this.offset[0] += dx;
    this.offset[1] -= dy;
    this.render();
  }

  _bindInteraction() {
    let dragging = false;
    let last = [0, 0];

    this.canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      last = [e.offsetX, e.offsetY];
      this.canvas.setPointerCapture(e.pointerId);
      this.canvas.style.cursor = 'grabbing';
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.offsetX - last[0];
      const dy = e.offsetY - last[1];
      last = [e.offsetX, e.offsetY];
      this.panBy(dx, dy);
    });
    const stop = (e) => {
      dragging = false;
      this.canvas.style.cursor = 'grab';
    };
    this.canvas.addEventListener('pointerup', stop);
    this.canvas.addEventListener('pointerleave', stop);
    this.canvas.style.cursor = 'grab';

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      this.zoomAt(factor, [e.offsetX, e.offsetY]);
    }, { passive: false });
  }

  _spectrumColor(t) {
    // interpolate through the editorial spectrum tokens s0..s6
    const stops = ['#2E3192', '#6A2C9B', '#962C90', '#BE2F71', '#D2542E', '#C77B15', '#9C7C18'];
    const n = stops.length - 1;
    const scaled = Math.min(Math.max(t, 0), 1) * n;
    const i = Math.min(Math.floor(scaled), n - 1);
    const frac = scaled - i;
    return this._lerpColor(stops[i], stops[i + 1], frac);
  }

  _lerpColor(a, b, t) {
    const pa = this._hexToRgb(a), pb = this._hexToRgb(b);
    const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
    const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
    const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
    return `rgb(${r},${g},${bl})`;
  }

  _hexToRgb(hex) {
    const v = hex.replace('#', '');
    return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
  }

  render() {
    const ctx = this.ctx;
    const { cssW, cssH } = this;
    const dark = this.opts.darkMode;

    const bg = dark ? '#12181F' : '#FFFFFF';
    const ruleColor = dark ? '#26313B' : '#E3E8EC';
    const axisColor = dark ? '#3A4652' : '#C7CFD6';
    const textColor = dark ? '#B9C4CC' : '#5E6B77';
    const originColor = dark ? '#F4F6F7' : '#0F151B';

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, cssW, cssH);

    if (this.opts.showGrid) this._drawGrid(ruleColor, axisColor);

    if (!this.points.length) return;

    // rope stroke: segment-by-segment gradient along arc-length position
    const n = this.points.length;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (let i = 0; i < n - 1; i++) {
      const p1 = this.worldToScreen(this.points[i]);
      const p2 = this.worldToScreen(this.points[i + 1]);
      const t = i / Math.max(n - 2, 1);
      ctx.strokeStyle = this._spectrumColor(t);
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
    }

    // nodes
    if (this.opts.showMarkers) {
      for (let i = 0; i < n; i++) {
        const [sx, sy] = this.worldToScreen(this.points[i]);
        const t = i / Math.max(n - 1, 1);
        ctx.beginPath();
        ctx.arc(sx, sy, this.opts.nodeSize, 0, Math.PI * 2);
        ctx.fillStyle = this._spectrumColor(t);
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = bg;
        ctx.stroke();
      }
    }

    // node numbers
    if (this.opts.showNumbers) {
      ctx.font = '10px "IBM Plex Mono", ui-monospace, monospace';
      ctx.fillStyle = textColor;
      ctx.textBaseline = 'bottom';
      for (let i = 0; i < n; i++) {
        const [sx, sy] = this.worldToScreen(this.points[i]);
        ctx.fillText(String(i), sx + 6, sy - 4);
      }
    }

    // origin marker
    const [ox, oy] = this.worldToScreen(this.origin);
    ctx.strokeStyle = originColor;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(ox - 9, oy); ctx.lineTo(ox + 9, oy);
    ctx.moveTo(ox, oy - 9); ctx.lineTo(ox, oy + 9);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(ox, oy, 3, 0, Math.PI * 2);
    ctx.fillStyle = originColor;
    ctx.fill();
    ctx.font = '10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillStyle = textColor;
    ctx.fillText('origin', ox + 12, oy + 14);
  }

  _drawGrid(ruleColor, axisColor) {
    const ctx = this.ctx;
    const { cssW, cssH } = this;
    // choose a "nice" grid step in world units based on scale
    const targetPx = 64;
    const rawStep = targetPx / this.scale;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const candidates = [1, 2, 5, 10].map(m => m * mag);
    const step = candidates.reduce((a, b) => Math.abs(b - rawStep) < Math.abs(a - rawStep) ? b : a);

    const topLeft = this.screenToWorld([0, 0]);
    const bottomRight = this.screenToWorld([cssW, cssH]);
    const xStart = Math.floor(topLeft[0] / step) * step;
    const xEnd = Math.ceil(bottomRight[0] / step) * step;
    const yStart = Math.floor(bottomRight[1] / step) * step;
    const yEnd = Math.ceil(topLeft[1] / step) * step;

    ctx.lineWidth = 1;
    ctx.strokeStyle = ruleColor;
    ctx.beginPath();
    for (let x = xStart; x <= xEnd; x += step) {
      const [sx] = this.worldToScreen([x, 0]);
      ctx.moveTo(sx, 0); ctx.lineTo(sx, cssH);
    }
    for (let y = yStart; y <= yEnd; y += step) {
      const [, sy] = this.worldToScreen([0, y]);
      ctx.moveTo(0, sy); ctx.lineTo(cssW, sy);
    }
    ctx.stroke();

    // axes (x=0 / y=0) slightly stronger
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    const [zx] = this.worldToScreen([0, 0]);
    const [, zy] = this.worldToScreen([0, 0]);
    ctx.moveTo(zx, 0); ctx.lineTo(zx, cssH);
    ctx.moveTo(0, zy); ctx.lineTo(cssW, zy);
    ctx.stroke();
  }
}
