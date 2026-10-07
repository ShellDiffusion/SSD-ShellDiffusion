/*
 * Interactive 3D (SVG) illustrations for the SSD project page.
 * Everything here is a toy simulation for intuition; quantitative results are in the paper.
 */
(() => {
  "use strict";

  const TAU = Math.PI * 2;
  const NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- small math helpers ----------
  function mulberry32(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(r) {
    let u = 0; while (!u) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r());
  }
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const lerp = (a, b, t) => a + (b - a) * t;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- a small drug-like toy molecule (Å) ----------
  const ATOMS = [];
  const BONDS = [];
  (() => {
    const ring = [];
    for (let k = 0; k < 6; k++) {
      const a = (k * TAU) / 6;
      ring.push(["C", [1.39 * Math.cos(a), 1.39 * Math.sin(a), 0]]);
    }
    const extra = [
      ["C", [2.88, 0.02, 0.05], 0],
      ["O", [3.52, 1.06, 0.32], 6],
      ["N", [3.55, -1.16, -0.22], 6],
      ["C", [5.0, -1.24, -0.1], 8],
      ["O", [-2.74, 0.0, 0.12], 3],
      ["C", [-3.5, 1.2, -0.25], 10],
      ["F", [1.24, 2.15, 0.0], 1],
      ["N", [-1.26, -2.2, 0.1], 4],
    ];
    ring.forEach(([e, p]) => ATOMS.push({ e, p }));
    for (let k = 0; k < 6; k++) BONDS.push([k, (k + 1) % 6]);
    extra.forEach(([e, p, to]) => { BONDS.push([ATOMS.length, to]); ATOMS.push({ e, p }); });
    // zero centre of mass (all states live in the CoM-free subspace)
    const c = ATOMS.reduce((m, a) => add(m, scl(a.p, 1 / ATOMS.length)), [0, 0, 0]);
    ATOMS.forEach((a) => (a.p = sub(a.p, c)));
  })();
  const N = ATOMS.length;
  const X0 = ATOMS.map((a) => a.p);
  const R_CHEM = X0.reduce((s, p) => s + len(p), 0) / N; // mean atom radius of the "data"
  const R_SSD = R_CHEM; // r_SSD is calibrated to the mean radius of centred conformations
  const R_EXTENT = Math.max(...X0.map(len)); // farthest atom of the target molecule
  const R_GAUSS = 2.6 * R_CHEM; // illustrative: Gaussian prior far larger than the molecule
  const ELEM = {
    C: { r: 0.36, fill: "var(--el-c)" },
    O: { r: 0.36, fill: "#e5484d" },
    N: { r: 0.36, fill: "#3e63dd" },
    F: { r: 0.34, fill: "#30a46c" },
  };

  // Fibonacci points on the unit sphere = candidate shell sites
  function fibSphere(n) {
    const pts = [];
    const g = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const y = 1 - (2 * (i + 0.5)) / n;
      const r = Math.sqrt(1 - y * y);
      pts.push([Math.cos(g * i) * r, y, Math.sin(g * i) * r]);
    }
    return pts;
  }

  // ---------- trajectory simulation ----------
  const K = 180; // frames from t = T (k = 0) to t = 0 (k = K)

  function bridge(r, amp) {
    // 3D Brownian bridge pinned to 0 at both ends
    const W = [[0, 0, 0]];
    const sd = 1 / Math.sqrt(K);
    for (let k = 1; k <= K; k++) W.push(add(W[k - 1], [normal(r) * sd, normal(r) * sd, normal(r) * sd]));
    const end = W[K];
    return W.map((w, k) => scl(sub(w, scl(end, k / K)), amp));
  }

  function centre(frame) {
    const c = frame.reduce((m, p) => add(m, scl(p, 1 / frame.length)), [0, 0, 0]);
    return frame.map((p) => sub(p, c));
  }

  function simulate(seed) {
    const r = mulberry32(seed);
    // SSD: atoms on the chemically scaled shell, random permutation of shell sites
    const sites = fibSphere(N);
    const perm = sites.map((_, i) => i);
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const ssdT = centre(perm.map((j) => scl(sites[j], R_SSD)));
    // Gaussian: isotropic prior whose radius is set by sigma_T, not chemistry
    const sigma = R_GAUSS / 1.596;
    const gT = centre(X0.map(() => [normal(r) * sigma, normal(r) * sigma, normal(r) * sigma]));

    const ssdNoise = X0.map(() => bridge(r, 0.35));
    const gNoise = X0.map(() => bridge(r, 5.2));

    const ssd = [], gauss = [];
    for (let k = 0; k <= K; k++) {
      const s = 1 - k / K; // s = t / T
      // radial attraction contracts at uniform speed -> linear path
      ssd.push(centre(X0.map((x0, i) => add(add(x0, scl(sub(ssdT[i], x0), s)), ssdNoise[i][k]))));
      // VE/VP-style schedule: most of the travel happens late, with large diffusion
      const g = Math.pow(s, 0.55);
      gauss.push(centre(X0.map((x0, i) => add(add(x0, scl(sub(gT[i], x0), g)), gNoise[i][k]))));
    }
    return { ssd, gauss, ssdT, gT };
  }

  function pathStats(frames) {
    let excess = 0, dev = 0;
    for (let i = 0; i < N; i++) {
      let L = 0;
      for (let k = 1; k <= K; k++) L += len(sub(frames[k][i], frames[k - 1][i]));
      excess += L / Math.max(1e-6, len(sub(frames[0][i], frames[K][i])));
    }
    for (let k = 0; k <= K; k++) {
      dev += Math.abs(frames[k].reduce((s, p) => s + len(p), 0) / N - R_CHEM);
    }
    return { excess: excess / N, dev: dev / (K + 1) };
  }

  // ---------- projection ----------
  function makeCamera(size, worldRadius) {
    return { size, scale: (size * 0.46) / worldRadius, dist: worldRadius * 4.2 };
  }
  function rotate(p, view) {
    const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
    const x1 = cy * p[0] + sy * p[2];
    const z1 = -sy * p[0] + cy * p[2];
    const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    return [x1, cp * p[1] - sp * z1, sp * p[1] + cp * z1];
  }
  function project(p, view, cam) {
    const q = rotate(p, view);
    const f = cam.dist / (cam.dist - q[2]);
    return { x: cam.size / 2 + q[0] * f * cam.scale, y: cam.size / 2 - q[1] * f * cam.scale, z: q[2], f };
  }

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function sphereWire(R, view, cam) {
    // latitude + longitude circles, split into back (z<0) and front (z>=0) halves
    const circles = [];
    for (const lat of [-60, -30, 0, 30, 60]) {
      const phi = (lat * Math.PI) / 180, c = [];
      for (let k = 0; k <= 72; k++) {
        const a = (k * TAU) / 72;
        c.push([R * Math.cos(phi) * Math.cos(a), R * Math.sin(phi), R * Math.cos(phi) * Math.sin(a)]);
      }
      circles.push(c);
    }
    for (let m = 0; m < 6; m++) {
      const th = (m * Math.PI) / 6, c = [];
      for (let k = 0; k <= 72; k++) {
        const a = (k * TAU) / 72;
        c.push([R * Math.cos(a) * Math.cos(th), R * Math.sin(a), R * Math.cos(a) * Math.sin(th)]);
      }
      circles.push(c);
    }
    let back = "", front = "";
    for (const c of circles) {
      let prev = null;
      for (const p of c) {
        const q = project(p, view, cam);
        const isFront = q.z >= 0;
        const cmd = `${q.x.toFixed(1)},${q.y.toFixed(1)}`;
        if (prev && prev.front === isFront) {
          if (isFront) front += `L${cmd}`; else back += `L${cmd}`;
        } else {
          if (isFront) front += `M${cmd}`; else back += `M${cmd}`;
        }
        prev = { front: isFront };
      }
    }
    // silhouette
    const outline = project([0, 0, 0], view, cam);
    const rr = R * cam.scale * (cam.dist / Math.sqrt(cam.dist * cam.dist - R * R));
    return { back, front, cx: outline.x, cy: outline.y, r: rr };
  }

  // ---------- drift decomposition (SSD reverse process) ----------
  const ALPHA = 0.95; // display length of the uniform-speed radial drift (Å)
  const D_MIN = 1.25;
  function drifts(frame, k, sim) {
    const out = [];
    const meanTravel = X0.reduce((s, x0, i) => s + len(sub(x0, sim.ssdT[i])), 0) / N;
    for (let i = 0; i < N; i++) {
      const x = frame[i];
      const n = len(x) || 1;
      const vRad = scl(x, -ALPHA / n);
      let vRep = [0, 0, 0];
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const d = sub(x, frame[j]);
        const dl = len(d);
        if (dl < D_MIN && dl > 1e-6) vRep = add(vRep, scl(d, ((D_MIN - dl) / D_MIN) * 2.2 / dl));
      }
      // deterministic transport toward the data, minus what radial + repulsion already supply
      const vTot = scl(sub(X0[i], sim.ssdT[i]), ALPHA / meanTravel);
      let vScore = sub(sub(vTot, vRad), vRep);
      // the correction matters most late in sampling, when local geometry is resolved
      vScore = scl(vScore, 0.45 + 0.55 * (k / K));
      out.push({ vRad, vRep, vScore, repActive: len(vRep) > 1e-3 });
    }
    return out;
  }

  // ---------- viewer ----------
  class Viewer {
    constructor(svg, opts) {
      this.svg = svg;
      this.opts = opts;
      this.size = 400;
      svg.setAttribute("viewBox", `0 0 ${this.size} ${this.size}`);
      this.cam = makeCamera(this.size, opts.worldRadius);
      const defs = el("defs", {}, svg);
      for (const [e, spec] of Object.entries(ELEM)) {
        const g = el("radialGradient", { id: `${opts.id}-g-${e}`, cx: "35%", cy: "30%", r: "70%" }, defs);
        el("stop", { offset: "0%", "stop-color": "#fff", "stop-opacity": "0.9" }, g);
        el("stop", { offset: "38%", "stop-color": spec.fill }, g);
        el("stop", { offset: "100%", "stop-color": spec.fill, "stop-opacity": "1" }, g);
      }
      for (const name of ["rad", "rep", "score"]) {
        const m = el("marker", { id: `${opts.id}-m-${name}`, viewBox: "0 0 10 10", refX: "8", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" }, defs);
        el("path", { d: "M0,0 L10,5 L0,10 z", fill: `var(--${name})` }, m);
      }
      this.gShellHalo = el("circle", { fill: opts.haloFill, stroke: "none" }, svg);
      this.gBack = el("path", { fill: "none", stroke: "var(--wire-back)", "stroke-width": "1" }, svg);
      this.gTrails = el("g", { fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" }, svg);
      this.gStart = el("g", {}, svg);
      this.gBonds = el("g", { stroke: "var(--ink-2)", "stroke-linecap": "round" }, svg);
      this.gAtoms = el("g", {}, svg);
      this.gArrows = el("g", { "stroke-linecap": "round" }, svg);
      this.gFront = el("path", { fill: "none", stroke: "var(--wire)", "stroke-width": "1" }, svg);
      this.gOutline = el("circle", { fill: "none", stroke: opts.accent, "stroke-width": "1.5", "stroke-dasharray": opts.dashed ? "4 5" : "none" }, svg);
      this.gRef = el("circle", { fill: "none", stroke: "var(--ink-3)", "stroke-width": "1", "stroke-dasharray": "2 4" }, svg);
    }

    render(st) {
      const { view } = st;
      const cam = this.cam;
      const frames = this.opts.frames();
      const k = Math.round(st.k);
      const frame = frames[k];
      const s = 1 - k / K;

      // shell wireframe at the prior radius
      const w = sphereWire(this.opts.shellRadius, view, cam);
      this.gBack.setAttribute("d", w.back);
      this.gFront.setAttribute("d", w.front);
      this.gOutline.setAttribute("cx", w.cx); this.gOutline.setAttribute("cy", w.cy); this.gOutline.setAttribute("r", w.r);
      this.gShellHalo.setAttribute("cx", w.cx); this.gShellHalo.setAttribute("cy", w.cy); this.gShellHalo.setAttribute("r", w.r);
      // reference: chemical radius r_chem
      if (this.opts.showRef) {
        const rr = R_CHEM * cam.scale;
        this.gRef.setAttribute("cx", w.cx); this.gRef.setAttribute("cy", w.cy); this.gRef.setAttribute("r", rr);
        this.gRef.style.display = "";
      } else this.gRef.style.display = "none";

      const P = frame.map((p) => project(p, view, cam));

      // trails
      let trails = "";
      if (st.layers.trails) {
        const step = 2;
        for (let i = 0; i < N; i++) {
          let d = "";
          for (let j = 0; j <= k; j += step) {
            const q = project(frames[j][i], view, cam);
            d += `${j ? "L" : "M"}${q.x.toFixed(1)},${q.y.toFixed(1)}`;
          }
          d += `L${P[i].x.toFixed(1)},${P[i].y.toFixed(1)}`;
          trails += `<path d="${d}" stroke="${this.opts.accent}" stroke-opacity="${this.opts.trailOpacity || 0.38}" stroke-width="1.3"/>`;
        }
      }
      this.gTrails.innerHTML = trails;

      // starting positions x_T
      let starts = "";
      if (st.layers.trails && k > 0) {
        for (let i = 0; i < N; i++) {
          const q = project(frames[0][i], view, cam);
          starts += `<circle cx="${q.x.toFixed(1)}" cy="${q.y.toFixed(1)}" r="${(2.4 * q.f).toFixed(2)}" fill="${this.opts.accent}" fill-opacity="0.55"/>`;
        }
      }
      this.gStart.innerHTML = starts;

      // bonds fade in as the structure resolves
      const bondAlpha = Math.max(0, 1 - s / 0.3);
      let bonds = "";
      if (bondAlpha > 0) {
        for (const [a, b] of BONDS) {
          if (len(sub(frame[a], frame[b])) > 2.0) continue; // only draw bonds that have actually formed
          bonds += `<line x1="${P[a].x.toFixed(1)}" y1="${P[a].y.toFixed(1)}" x2="${P[b].x.toFixed(1)}" y2="${P[b].y.toFixed(1)}" stroke-width="${(3.2 * (P[a].f + P[b].f) / 2).toFixed(2)}" stroke-opacity="${(bondAlpha * 0.75).toFixed(2)}"/>`;
        }
      }
      this.gBonds.innerHTML = bonds;

      // atoms, painter's order
      const zMax = this.opts.worldRadius;
      const order = P.map((q, i) => i).sort((a, b) => P[a].z - P[b].z);
      let atoms = "";
      for (const i of order) {
        const q = P[i];
        const spec = ELEM[ATOMS[i].e];
        const rad = spec.r * cam.scale * q.f;
        const depth = 0.6 + 0.4 * Math.min(1, Math.max(0, (q.z + zMax) / (2 * zMax)));
        const sel = this.opts.drifts && st.selected === i;
        atoms += `<circle data-i="${i}" cx="${q.x.toFixed(1)}" cy="${q.y.toFixed(1)}" r="${Math.max(3, rad).toFixed(2)}" fill="url(#${this.opts.id}-g-${ATOMS[i].e})" opacity="${depth.toFixed(2)}"${sel ? ` stroke="${this.opts.accent}" stroke-width="2.5"` : ""} style="cursor:pointer"/>`;
        if (this.opts.drifts && st.layers.rep && st.focus === "rep") {
          atoms += `<circle cx="${q.x.toFixed(1)}" cy="${q.y.toFixed(1)}" r="${((D_MIN / 2) * cam.scale * q.f).toFixed(1)}" fill="none" stroke="var(--rep)" stroke-opacity="0.35" stroke-dasharray="2 3"/>`;
        }
      }
      this.gAtoms.innerHTML = atoms;

      // drift arrows
      let arrows = "";
      if (this.opts.drifts && k < K) {
        const D = drifts(frame, k, this.opts.sim());
        const kinds = [["rad", "vRad"], ["rep", "vRep"], ["score", "vScore"]];
        for (let i = 0; i < N; i++) {
          const emph = st.selected === i;
          for (const [name, key] of kinds) {
            if (!st.layers[name]) continue;
            let v = D[i][key];
            const vl = len(v);
            if (vl < 0.05) continue;
            if (vl > 1.0) v = scl(v, 1.0 / vl); // keep arrows readable
            const a = P[i];
            const b = project(add(frame[i], scl(v, 1.25)), view, cam);
            const dim = st.selected != null && !emph ? 0.45 : 0.95;
            arrows += `<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="var(--${name})" stroke-width="${emph ? 2.8 : 1.7}" stroke-opacity="${dim}" marker-end="url(#${this.opts.id}-m-${name})"/>`;
          }
        }
      }
      this.gArrows.innerHTML = arrows;
    }
  }

  // ---------- method explorer ----------
  function initExplorer() {
    const root = document.getElementById("explorer");
    if (!root) return;
    let seed = 7;
    let sim = simulate(seed);
    let statsS = pathStats(sim.ssd), statsG = pathStats(sim.gauss);

    const st = {
      k: 0,
      view: { yaw: -0.6, pitch: 0.38 },
      layers: { trails: false, rad: false, rep: false, score: false },
      selected: 2,
      focus: "init",
      playing: false,
    };

    const vS = new Viewer(document.getElementById("view-ssd"), {
      id: "vs", accent: "var(--ssd)", haloFill: "var(--ssd-soft)", shellRadius: R_SSD, worldRadius: R_EXTENT * 1.15,
      frames: () => sim.ssd, sim: () => sim, drifts: true, showRef: false,
    });
    const vG = new Viewer(document.getElementById("view-gauss"), {
      id: "vg", accent: "var(--gauss)", haloFill: "var(--gauss-soft)", shellRadius: R_GAUSS, worldRadius: R_GAUSS * 1.12,
      frames: () => sim.gauss, sim: () => sim, drifts: false, showRef: true, dashed: true, trailOpacity: 0.22,
    });

    const slider = document.getElementById("t-slider");
    const tOut = document.getElementById("t-out");
    const playBtn = document.getElementById("play");
    const metaS = document.getElementById("meta-ssd");
    const metaG = document.getElementById("meta-gauss");
    const chart = document.getElementById("radius-chart");
    slider.max = K;
    const ZOOM = ((R_GAUSS * 1.12) / (R_EXTENT * 1.15)).toFixed(1);

    // radius-over-time chart
    function drawChart() {
      const W = 720, H = 170, pl = 44, pr = 12, pt = 12, pb = 28;
      const meanR = (frames) => frames.map((f) => f.reduce((s, p) => s + len(p), 0) / N);
      const a = meanR(sim.ssd), b = meanR(sim.gauss);
      const yMax = Math.max(...b, ...a) * 1.08;
      const x = (k) => pl + ((W - pl - pr) * k) / K;
      const y = (v) => pt + (H - pt - pb) * (1 - v / yMax);
      const line = (arr) => arr.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join("");
      let ticks = "";
      const step = yMax > 12 ? 4 : 2;
      for (let v = 0; v <= yMax; v += step) {
        ticks += `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)"/>` +
          `<text x="${pl - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--ink-3)" font-family="var(--mono)">${v}</text>`;
      }
      chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
      chart.innerHTML = ticks +
        `<text x="${pl}" y="${H - 6}" font-size="11" fill="var(--ink-3)" font-family="var(--mono)">t = T (prior)</text>` +
        `<text x="${W - pr}" y="${H - 6}" text-anchor="end" font-size="11" fill="var(--ink-3)" font-family="var(--mono)">t = 0 (molecule)</text>` +
        `<line x1="${pl}" x2="${W - pr}" y1="${y(R_CHEM)}" y2="${y(R_CHEM)}" stroke="var(--ink-2)" stroke-dasharray="4 4"/>` +
        `<path d="${line(b)}" fill="none" stroke="var(--gauss)" stroke-width="2"/>` +
        `<path d="${line(a)}" fill="none" stroke="var(--ssd)" stroke-width="2.4"/>` +
        `<line id="chart-cursor" y1="${pt}" y2="${H - pb}" stroke="var(--ink)" stroke-width="1"/>` +
        `<circle id="cur-a" r="4" fill="var(--ssd)"/><circle id="cur-b" r="4" fill="var(--gauss)"/>`;
      chart._map = { x, y, a, b };
    }

    function render() {
      vS.render(st);
      vG.render(st);
      const k = Math.round(st.k);
      slider.value = k;
      tOut.textContent = `t = ${(1 - k / K).toFixed(2)}·T`;
      const rS = sim.ssd[k].reduce((s, p) => s + len(p), 0) / N;
      const rG = sim.gauss[k].reduce((s, p) => s + len(p), 0) / N;
      metaS.innerHTML = `<span>mean ‖x‖ = ${rS.toFixed(2)} Å · zoom ×${ZOOM}</span><span>path excess ${statsS.excess.toFixed(2)}×</span>`;
      metaG.innerHTML = `<span>mean ‖x‖ = ${rG.toFixed(2)} Å</span><span>path excess ${statsG.excess.toFixed(2)}×</span>`;
      const m = chart._map;
      if (m) {
        const cx = m.x(k);
        const c = chart.querySelector("#chart-cursor");
        c.setAttribute("x1", cx); c.setAttribute("x2", cx);
        chart.querySelector("#cur-a").setAttribute("cx", cx); chart.querySelector("#cur-a").setAttribute("cy", m.y(m.a[k]));
        chart.querySelector("#cur-b").setAttribute("cx", cx); chart.querySelector("#cur-b").setAttribute("cy", m.y(m.b[k]));
      }
    }

    let raf = null;
    function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = null; render(); }); }

    // playback
    let last = 0;
    function tick(ts) {
      if (!st.playing) return;
      const dt = last ? Math.min(64, ts - last) : 16;
      last = ts;
      st.k = Math.min(K, st.k + dt * 0.045);
      if (!dragging) st.view.yaw += dt * 0.00012;
      render();
      if (st.k >= K) { setPlaying(false); return; }
      requestAnimationFrame(tick);
    }
    function setPlaying(p) {
      st.playing = p;
      playBtn.innerHTML = p
        ? `<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="3.5" height="12"/><rect x="9.5" y="2" width="3.5" height="12"/></svg>Pause`
        : `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2l10 6-10 6z"/></svg>${st.k >= K ? "Replay" : "Play"}`;
      if (p) { if (st.k >= K) st.k = 0; last = 0; requestAnimationFrame(tick); }
    }
    playBtn.addEventListener("click", () => setPlaying(!st.playing));
    slider.addEventListener("input", () => { setPlaying(false); st.k = +slider.value; schedule(); });
    document.getElementById("resample").addEventListener("click", () => {
      seed += 1; sim = simulate(seed); statsS = pathStats(sim.ssd); statsG = pathStats(sim.gauss);
      drawChart(); schedule();
    });

    // layer chips
    root.querySelectorAll("[data-layer]").forEach((inp) => {
      inp.checked = !!st.layers[inp.dataset.layer];
      inp.addEventListener("change", () => { st.layers[inp.dataset.layer] = inp.checked; schedule(); });
    });
    function syncChips() { root.querySelectorAll("[data-layer]").forEach((inp) => (inp.checked = !!st.layers[inp.dataset.layer])); }

    // method steps
    const STEPS = {
      init: { k: 0, layers: { trails: false, rad: false, rep: false, score: false } },
      rad: { k: 40, layers: { trails: true, rad: true, rep: false, score: false } },
      rep: { k: 95, layers: { trails: true, rad: false, rep: true, score: false } },
      score: { k: 150, layers: { trails: true, rad: false, rep: false, score: true } },
      all: { k: 0, layers: { trails: true, rad: true, rep: true, score: true }, play: true },
    };
    const stepBtns = root.querySelectorAll(".step");
    stepBtns.forEach((b) => b.addEventListener("click", () => {
      const s = STEPS[b.dataset.step];
      stepBtns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      st.focus = b.dataset.step;
      st.layers = { ...s.layers };
      syncChips();
      setPlaying(false);
      st.k = s.k;
      if (s.play && !reduceMotion) setPlaying(true); else schedule();
    }));

    // rotate by dragging either viewer; click an atom to focus its drifts
    let dragging = false, px = 0, py = 0, moved = 0;
    for (const svg of [vS.svg, vG.svg]) {
      svg.addEventListener("pointerdown", (e) => { dragging = true; moved = 0; px = e.clientX; py = e.clientY; svg.setPointerCapture(e.pointerId); });
      svg.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - px, dy = e.clientY - py; px = e.clientX; py = e.clientY;
        moved += Math.abs(dx) + Math.abs(dy);
        st.view.yaw += dx * 0.01;
        st.view.pitch = Math.max(-1.4, Math.min(1.4, st.view.pitch + dy * 0.01));
        schedule();
      });
      svg.addEventListener("pointerup", (e) => {
        dragging = false;
        if (moved < 4 && svg === vS.svg) {
          const hit = document.elementFromPoint(e.clientX, e.clientY);
          const i = hit && hit.dataset ? hit.dataset.i : undefined;
          st.selected = i !== undefined ? (st.selected === +i ? null : +i) : null;
          schedule();
        }
      });
    }

    drawChart();
    render();
  }

  // ---------- hero: a looping SSD sample ----------
  function initHero() {
    const svg = document.getElementById("hero-svg");
    if (!svg) return;
    const sim = simulate(3);
    const st = { k: 0, view: { yaw: -0.5, pitch: 0.32 }, layers: { trails: true, rad: false, rep: false, score: false }, selected: null };
    const v = new Viewer(svg, {
      id: "vh", accent: "var(--ssd)", haloFill: "var(--ssd-soft)", shellRadius: R_SSD, worldRadius: R_EXTENT * 1.12,
      frames: () => sim.ssd, sim: () => sim, drifts: false, showRef: false,
    });
    if (reduceMotion) { st.k = K; v.render(st); return; }
    let t0 = null, visible = true;
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) requestAnimationFrame(loop); }).observe(svg);
    function loop(ts) {
      if (!visible) return;
      if (t0 === null) t0 = ts;
      const cyc = ((ts - t0) / 7000) % 1; // 0..1: sample, hold, reset
      const u = Math.min(1, cyc / 0.7);
      st.k = K * (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
      st.view.yaw = -0.5 + (ts - t0) * 0.00015;
      v.render(st);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  // ---------- Gaussian annulus demo ----------
  function initAnnulus() {
    const svg = document.getElementById("annulus-svg");
    if (!svg) return;
    const slider = document.getElementById("n-slider");
    const out = document.getElementById("n-out");
    const r = mulberry32(11);
    const M = 700;
    const W = 460, H = 250, pl = 20, pr = 16, pt = 46, pb = 34;
    const XMAX = 16;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    function draw() {
      const n = +slider.value;
      out.textContent = n;
      const d = 3 * n;
      const norms = [];
      for (let m = 0; m < M; m++) {
        let s = 0;
        for (let j = 0; j < d; j++) { const z = normal(r); s += z * z; }
        norms.push(Math.sqrt(s));
      }
      const B = 64, bw = XMAX / B, hist = new Array(B).fill(0);
      norms.forEach((v) => { const b = Math.min(B - 1, Math.floor(v / bw)); hist[b]++; });
      const hMax = Math.max(...hist);
      const x = (v) => pl + ((W - pl - pr) * v) / XMAX;
      const y = (c) => pt + (H - pt - pb) * (1 - c / (hMax * 1.1));
      let bars = "";
      hist.forEach((c, b) => {
        if (!c) return;
        bars += `<rect x="${x(b * bw).toFixed(1)}" y="${y(c).toFixed(1)}" width="${Math.max(1, x(bw) - x(0) - 1).toFixed(1)}" height="${(H - pb - y(c)).toFixed(1)}" fill="var(--gauss)" fill-opacity="0.75" rx="1"/>`;
      });
      const rTheory = Math.sqrt(d);
      let axis = "";
      for (let v = 0; v <= XMAX; v += 4) {
        axis += `<line x1="${x(v)}" x2="${x(v)}" y1="${H - pb}" y2="${H - pb + 4}" stroke="var(--ink-3)"/>` +
          `<text x="${x(v)}" y="${H - pb + 18}" text-anchor="middle" font-size="11" fill="var(--ink-3)" font-family="var(--mono)">${v}σ</text>`;
      }
      const mean = norms.reduce((a, b) => a + b, 0) / M;
      const sd = Math.sqrt(norms.reduce((a, b) => a + (b - mean) ** 2, 0) / M);
      svg.innerHTML =
        `<line x1="${pl}" x2="${W - pr}" y1="${H - pb}" y2="${H - pb}" stroke="var(--ink-3)"/>` + axis + bars +
        `<line x1="${x(rTheory)}" x2="${x(rTheory)}" y1="${pt}" y2="${H - pb}" stroke="var(--ink)" stroke-dasharray="3 3"/>` +
        `<text x="${Math.min(x(rTheory) + 6, W - 150)}" y="${pt + 8}" font-size="13" fill="var(--ink)" font-family="var(--mono)">σ√(3n) = ${rTheory.toFixed(2)}σ</text>` +
        `<text x="${pl}" y="16" font-size="13" fill="var(--ink-3)">‖x_T‖ over ${M} prior draws (σ = 1)</text>`;
      document.getElementById("ann-r").textContent = `${mean.toFixed(2)}σ`;
      document.getElementById("ann-w").textContent = `${sd.toFixed(2)}σ`;
      document.getElementById("ann-rel").textContent = `${((sd / mean) * 100).toFixed(1)}%`;
    }
    slider.addEventListener("input", draw);
    draw();
  }

  document.documentElement.style.setProperty("--el-c", "#5b6270");
  initHero();
  initAnnulus();
  initExplorer();
})();
