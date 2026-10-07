/*
 * SSD project page: 3D (SVG) animations of the three stages of Shell-guided Spherical Diffusion
 * (initialization, forward process, reverse process), plus the hero loop and small charts.
 * The molecule and trajectories are a toy simulation for intuition, not model outputs.
 */
(() => {
  "use strict";

  const TAU = Math.PI * 2;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- math ----------
  function mulberry32(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r()); }
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const nrm = (a) => scl(a, 1 / (len(a) || 1));
  const mix = (a, b, t) => add(a, scl(sub(b, a), t));
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const easeOut = (x) => 1 - Math.pow(1 - x, 3);
  const f1 = (v) => v.toFixed(1);

  // ---------- toy molecule (Å), zero centre of mass ----------
  const ATOMS = [], BONDS = [];
  (() => {
    for (let k = 0; k < 6; k++) {
      const a = (k * TAU) / 6;
      ATOMS.push({ e: "C", p: [1.39 * Math.cos(a), 1.39 * Math.sin(a), 0] });
    }
    for (let k = 0; k < 6; k++) BONDS.push([k, (k + 1) % 6]);
    [
      ["C", [2.88, 0.02, 0.05], 0], ["O", [3.52, 1.06, 0.32], 6], ["N", [3.55, -1.16, -0.22], 6],
      ["C", [5.0, -1.24, -0.1], 8], ["O", [-2.74, 0.0, 0.12], 3], ["C", [-3.5, 1.2, -0.25], 10],
      ["F", [1.24, 2.15, 0.0], 1], ["N", [-1.26, -2.2, 0.1], 4],
    ].forEach(([e, p, to]) => { BONDS.push([ATOMS.length, to]); ATOMS.push({ e, p }); });
    const c = ATOMS.reduce((m, a) => add(m, scl(a.p, 1 / ATOMS.length)), [0, 0, 0]);
    ATOMS.forEach((a) => (a.p = sub(a.p, c)));
  })();
  const N = ATOMS.length;
  const X0 = ATOMS.map((a) => a.p);
  const REST = BONDS.map(([a, b]) => len(sub(X0[a], X0[b])));
  const R_CHEM = X0.reduce((s, p) => s + len(p), 0) / N;
  const R_SSD = R_CHEM; // calibrated to the mean radius of centred conformations
  const R_EXT = Math.max(...X0.map(len));
  const R_GAUSS = 2.2 * R_CHEM; // illustrative: Gaussian shell set by sigma_T * sqrt(3n), not chemistry
  const D_MIN = 1.25;
  const ELEM = { C: "#a7b0bf", O: "#ff5a5f", N: "#5b8cff", F: "#3ddc84" };
  const COL = { ssd: "#ff7a2f", gauss: "#5aa2ff", rad: "#ffc14d", rep: "#b48cff", score: "#3ee6c1", bond: "#cdd3dd" };

  // ---------- simulation ----------
  const K = 200;
  function bridge(r) {
    const W = [[0, 0, 0]], sd = 1 / Math.sqrt(K);
    for (let k = 1; k <= K; k++) W.push(add(W[k - 1], [normal(r) * sd, normal(r) * sd, normal(r) * sd]));
    const end = W[K];
    return W.map((w, k) => sub(w, scl(end, k / K)));
  }
  function at(b, u) {
    const x = clamp(u) * K, k = Math.min(K - 1, Math.floor(x));
    return mix(b[k], b[k + 1], x - k);
  }
  function makeSim(seed) {
    const r = mulberry32(seed);
    const shuffle = () => {
      const p = [...Array(N).keys()];
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
      return p;
    };
    const perms = [shuffle(), shuffle(), shuffle()];
    // x_T^(i) = r v / ||v||, v ~ N(0, I), then re-centre (paper, Sec. 3)
    const raw = X0.map(() => [normal(r), normal(r), normal(r)]);
    const onShell = raw.map((v) => scl(nrm(v), R_SSD));
    const shift = onShell.reduce((m, p) => add(m, scl(p, 1 / N)), [0, 0, 0]);
    const sites = onShell.map((p) => sub(p, shift));
    const S = X0.map((_, i) => sites[perms[2][i]]);
    const d = X0.map((x, i) => len(sub(S[i], x)));
    const sigma = R_GAUSS / 1.596;
    let gT = X0.map(() => [normal(r) * sigma, normal(r) * sigma, normal(r) * sigma]);
    const c = gT.reduce((m, p) => add(m, scl(p, 1 / N)), [0, 0, 0]);
    gT = gT.map((p) => sub(p, c));
    const mk = () => X0.map(() => bridge(r));
    return { perms, raw, onShell, shift, sites, S, d, dMax: Math.max(...d), gT, bF: mk(), bR: mk(), gF: mk(), gR: mk() };
  }

  // ---------- renderer ----------
  let uid = 0;
  class Scene3D {
    constructor(svg, w, h) {
      this.svg = svg; this.w = w; this.h = h; this.id = `s${uid++}`;
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      const id = this.id;
      let defs = "";
      for (const [e, c] of Object.entries(ELEM)) {
        defs += `<radialGradient id="${id}-a-${e}" cx="35%" cy="30%" r="75%"><stop offset="0" stop-color="#fff"/><stop offset=".35" stop-color="${c}"/><stop offset="1" stop-color="${c}" stop-opacity=".85"/></radialGradient>`;
        defs += `<radialGradient id="${id}-g-${e}"><stop offset="0" stop-color="${c}" stop-opacity=".55"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></radialGradient>`;
      }
      for (const k of ["ssd", "gauss"]) {
        defs += `<radialGradient id="${id}-cl-${k}"><stop offset="0" stop-color="${COL[k]}" stop-opacity=".22"/><stop offset=".6" stop-color="${COL[k]}" stop-opacity=".08"/><stop offset="1" stop-color="${COL[k]}" stop-opacity="0"/></radialGradient>`;
        defs += `<radialGradient id="${id}-sh-${k}" cx="40%" cy="35%" r="70%"><stop offset="0" stop-color="${COL[k]}" stop-opacity=".02"/><stop offset=".8" stop-color="${COL[k]}" stop-opacity=".10"/><stop offset="1" stop-color="${COL[k]}" stop-opacity=".28"/></radialGradient>`;
      }
      for (const k of ["rad", "rep", "score", "ssd", "gauss"]) {
        defs += `<marker id="${id}-m-${k}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${COL[k]}"/></marker>`;
      }
      defs += `<filter id="${id}-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
      defs += `<filter id="${id}-blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5"/></filter>`;
      svg.innerHTML = `<defs>${defs}</defs>`;
      const names = ["orbitsBack", "shellFill", "back", "sites", "links", "trails", "bonds", "atoms", "halos", "arrows", "front", "orbitsFront", "pulses", "labels"];
      this.L = {};
      for (const n of names) {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        svg.appendChild(g); this.L[n] = g;
      }
    }

    setCamera(worldR, view) {
      this.view = view;
      this.scale = (Math.min(this.w, this.h) * 0.45) / worldR;
      this.dist = worldR * 4.5;
    }
    P(p) {
      const v = this.view;
      const cy = Math.cos(v.yaw), sy = Math.sin(v.yaw);
      const x1 = cy * p[0] + sy * p[2], z1 = -sy * p[0] + cy * p[2];
      const cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
      const y2 = cp * p[1] - sp * z1, z2 = sp * p[1] + cp * z1;
      const f = this.dist / (this.dist - z2);
      return { x: this.w / 2 + x1 * f * this.scale, y: this.h / 2 - y2 * f * this.scale, z: z2, f };
    }

    circlePath(pts, draw) {
      // split a 3D polyline into back / front SVG paths, drawing only the first `draw` fraction
      let back = "", front = "", prev = null;
      const lim = Math.floor(draw * (pts.length - 1));
      for (let k = 0; k <= lim; k++) {
        const q = this.P(pts[k]);
        const fr = q.z >= 0, c = `${f1(q.x)},${f1(q.y)}`;
        const cmd = prev === fr ? "L" : "M";
        if (prev !== null && prev !== fr) {
          // bridge the seam so lines stay continuous
          if (fr) back += `L${c}`; else front += `L${c}`;
        }
        if (fr) front += cmd + c; else back += cmd + c;
        prev = fr;
      }
      return { back, front };
    }

    shellRings(R) {
      if (this._ringsR === R) return this._rings;
      const rings = [], M = 72;
      for (const lat of [-60, -30, 0, 30, 60]) {
        const ph = (lat * Math.PI) / 180, c = [];
        for (let k = 0; k <= M; k++) { const a = (k * TAU) / M; c.push([R * Math.cos(ph) * Math.cos(a), R * Math.sin(ph), R * Math.cos(ph) * Math.sin(a)]); }
        rings.push(c);
      }
      for (let m = 0; m < 6; m++) {
        const th = (m * Math.PI) / 6, c = [];
        for (let k = 0; k <= M; k++) { const a = (k * TAU) / M; c.push([R * Math.cos(a) * Math.cos(th), R * Math.sin(a), R * Math.cos(a) * Math.sin(th)]); }
        rings.push(c);
      }
      this._ringsR = R; this._rings = rings;
      return rings;
    }

    render(sc, time) {
      const id = this.id, L = this.L;
      let shellFill = "", back = "", front = "", labels = "", orbB = "", orbF = "";

      // shells
      for (const sh of sc.shells || []) {
        if (sh.alpha <= 0.001) continue;
        const col = COL[sh.kind];
        const o = this.P([0, 0, 0]);
        const rr = sh.R * this.scale * (this.dist / Math.sqrt(this.dist * this.dist - sh.R * sh.R));
        if (sh.cloud) {
          // an unstructured Gaussian cloud: no shell, just a soft blob
          shellFill += `<circle cx="${f1(o.x)}" cy="${f1(o.y)}" r="${f1(rr)}" fill="url(#${id}-cl-${sh.kind})" opacity="${sh.alpha.toFixed(3)}"/>`;
          continue;
        }
        shellFill += `<circle cx="${f1(o.x)}" cy="${f1(o.y)}" r="${f1(rr)}" fill="url(#${id}-sh-${sh.kind})" opacity="${(sh.alpha * clamp(sh.draw * 1.4)).toFixed(3)}"/>`;
        const circ = TAU * rr;
        shellFill += `<circle cx="${f1(o.x)}" cy="${f1(o.y)}" r="${f1(rr)}" fill="none" stroke="${col}" stroke-width="1.6" opacity="${sh.alpha.toFixed(3)}" stroke-dasharray="${sh.dashed ? "5 6" : `${f1(circ * sh.draw)} ${f1(circ)}`}" transform="rotate(-90 ${f1(o.x)} ${f1(o.y)})"/>`;
        let b = "", f = "";
        for (const ring of this.shellRings(sh.R)) { const p = this.circlePath(ring, sh.draw); b += p.back; f += p.front; }
        back += `<path d="${b}" fill="none" stroke="rgba(236,235,230,${(0.07 * sh.alpha).toFixed(3)})" stroke-width="1"/>`;
        front += `<path d="${f}" fill="none" stroke="rgba(236,235,230,${(0.22 * sh.alpha).toFixed(3)})" stroke-width="1"/>`;
        if (sh.scan) {
          // a glowing latitude ring sweeping over the shell
          const ph = Math.sin(time * 0.9) * 1.25, c = [];
          for (let k = 0; k <= 72; k++) { const a = (k * TAU) / 72; c.push([sh.R * Math.cos(ph) * Math.cos(a), sh.R * Math.sin(ph), sh.R * Math.cos(ph) * Math.sin(a)]); }
          const p = this.circlePath(c, 1);
          back += `<path d="${p.back}" fill="none" stroke="${col}" stroke-opacity="${(0.25 * sh.alpha * sh.scan).toFixed(3)}" stroke-width="1.5"/>`;
          front += `<path d="${p.front}" fill="none" stroke="${col}" stroke-opacity="${(0.95 * sh.alpha * sh.scan).toFixed(3)}" stroke-width="2.2" filter="url(#${id}-glow)"/>`;
        }
      }
      L.shellFill.innerHTML = shellFill;
      L.back.innerHTML = back;
      L.front.innerHTML = front;

      // orbits (hero decoration)
      for (const ob of sc.orbits || []) {
        const c = [];
        for (let k = 0; k <= 96; k++) {
          const a = (k * TAU) / 96;
          let p = [ob.R * Math.cos(a), 0, ob.R * Math.sin(a)];
          p = [p[0], p[1] * Math.cos(ob.tilt) - p[2] * Math.sin(ob.tilt), p[1] * Math.sin(ob.tilt) + p[2] * Math.cos(ob.tilt)];
          c.push(p);
        }
        const p = this.circlePath(c, 1);
        orbB += `<path d="${p.back}" fill="none" stroke="${ob.color}" stroke-opacity=".10" stroke-width="1"/>`;
        orbF += `<path d="${p.front}" fill="none" stroke="${ob.color}" stroke-opacity=".28" stroke-width="1"/>`;
        for (let j = 0; j < ob.n; j++) {
          const k = Math.floor((((time * ob.speed + j / ob.n) % 1) + 1) % 1 * 96);
          const q = this.P(c[k]);
          const tgt = q.z >= 0 ? "F" : "B";
          const dot = `<circle cx="${f1(q.x)}" cy="${f1(q.y)}" r="${f1(2.6 * q.f)}" fill="${ob.color}" opacity="${q.z >= 0 ? 0.95 : 0.3}" filter="url(#${id}-glow)"/>`;
          if (tgt === "F") orbF += dot; else orbB += dot;
        }
      }
      L.orbitsBack.innerHTML = orbB;
      L.orbitsFront.innerHTML = orbF;

      // sites
      let sites = "";
      for (const s of sc.sites || []) {
        if (s.alpha <= 0.01) continue;
        const q = this.P(s.p);
        sites += `<circle cx="${f1(q.x)}" cy="${f1(q.y)}" r="${f1((s.size || 4.6) * q.f * (0.6 + 0.4 * s.alpha))}" fill="${s.color || COL.ssd}" filter="url(#${this.id}-glow)" opacity="${(s.alpha * (q.z >= 0 ? 1 : 0.45)).toFixed(2)}"/>`;
      }
      L.sites.innerHTML = sites;

      // links (atom -> assigned shell site)
      let links = "";
      for (const l of sc.links || []) {
        if (l.alpha <= 0.01) continue;
        const a = this.P(l.a), b = this.P(l.b);
        links += `<line x1="${f1(a.x)}" y1="${f1(a.y)}" x2="${f1(b.x)}" y2="${f1(b.y)}" stroke="${l.color || COL.ssd}" stroke-width="1.7" stroke-dasharray="4 4" opacity="${l.alpha.toFixed(2)}"/>`;
      }
      L.links.innerHTML = links;

      // trails: drawn in chunks with increasing opacity for a comet-tail look
      let trails = "";
      for (const tr of sc.trails || []) {
        const pts = tr.pts.map((p) => this.P(p));
        const chunks = 4, n = pts.length;
        if (n < 2) continue;
        for (let c = 0; c < chunks; c++) {
          const i0 = Math.floor((c * (n - 1)) / chunks), i1 = Math.floor(((c + 1) * (n - 1)) / chunks);
          if (i1 <= i0) continue;
          let d = "";
          for (let i = i0; i <= i1; i++) d += `${i === i0 ? "M" : "L"}${f1(pts[i].x)},${f1(pts[i].y)}`;
          trails += `<path d="${d}" fill="none" stroke="${tr.color}" stroke-width="${tr.width || 1.4}" stroke-linecap="round" stroke-linejoin="round" opacity="${(tr.alpha * (0.15 + (0.85 * (c + 1)) / chunks)).toFixed(3)}"/>`;
        }
      }
      L.trails.innerHTML = trails;

      // atoms (projected once, reused by bonds/halos/arrows)
      const P = sc.atoms.map((a) => this.P(a.p));
      let bonds = "";
      for (const b of sc.bonds || []) {
        if (b.alpha <= 0.01) continue;
        const p = P[b.a], q = P[b.b];
        const w = 3.4 * (p.f + q.f) / 2;
        if (b.glow > 0.01) bonds += `<line x1="${f1(p.x)}" y1="${f1(p.y)}" x2="${f1(q.x)}" y2="${f1(q.y)}" stroke="${COL.ssd}" stroke-width="${f1(w * 3)}" stroke-linecap="round" opacity="${(b.glow * 0.8).toFixed(2)}" filter="url(#${id}-blur)"/>`;
        bonds += `<line x1="${f1(p.x)}" y1="${f1(p.y)}" x2="${f1(q.x)}" y2="${f1(q.y)}" stroke="${COL.bond}" stroke-width="${f1(w)}" stroke-linecap="round" opacity="${(b.alpha * 0.8).toFixed(2)}"/>`;
      }
      L.bonds.innerHTML = bonds;

      const order = P.map((_, i) => i).sort((a, b) => P[a].z - P[b].z);
      let atoms = "";
      for (const i of order) {
        const a = sc.atoms[i], q = P[i];
        if (a.alpha <= 0.01) continue;
        const r = 0.36 * this.scale * q.f;
        const g = a.glow == null ? 0.5 : a.glow;
        if (g > 0.01) atoms += `<circle cx="${f1(q.x)}" cy="${f1(q.y)}" r="${f1(r * 2.6)}" fill="url(#${id}-g-${ATOMS[i].e})" opacity="${(g * a.alpha).toFixed(2)}"/>`;
        atoms += `<circle cx="${f1(q.x)}" cy="${f1(q.y)}" r="${f1(Math.max(2.5, r))}" fill="url(#${id}-a-${ATOMS[i].e})" opacity="${(a.alpha * (0.7 + 0.3 * clamp((q.z + 6) / 12))).toFixed(2)}"/>`;
      }
      L.atoms.innerHTML = atoms;

      let halos = "";
      for (const h of sc.halos || []) {
        if (h.alpha <= 0.01) continue;
        const q = P[h.i];
        halos += `<circle cx="${f1(q.x)}" cy="${f1(q.y)}" r="${f1(h.r * this.scale * q.f)}" fill="${COL.rep}" fill-opacity="${(h.alpha * 0.12).toFixed(3)}" stroke="${COL.rep}" stroke-width="1.2" stroke-dasharray="3 3" opacity="${h.alpha.toFixed(2)}"/>`;
      }
      L.halos.innerHTML = halos;

      let arrows = "";
      for (const ar of sc.arrows || []) {
        if (ar.alpha <= 0.01) continue;
        const a = P[ar.i], b = this.P(add(sc.atoms[ar.i].p, ar.v));
        arrows += `<line x1="${f1(a.x)}" y1="${f1(a.y)}" x2="${f1(b.x)}" y2="${f1(b.y)}" stroke="${COL[ar.kind]}" stroke-width="2.2" stroke-linecap="round" opacity="${ar.alpha.toFixed(2)}" marker-end="url(#${id}-m-${ar.kind})"/>`;
      }
      L.arrows.innerHTML = arrows;

      let pulses = "";
      for (const pu of sc.pulses || []) {
        if (pu.k <= 0 || pu.k >= 1) continue;
        const q = this.P(pu.p);
        const r = (0.38 + pu.k * 0.6) * this.scale * q.f;
        pulses += `<circle cx="${f1(q.x)}" cy="${f1(q.y)}" r="${f1(r)}" fill="none" stroke="${pu.color || COL.ssd}" stroke-width="${f1(2 * (1 - pu.k) + 0.4)}" opacity="${(1 - pu.k).toFixed(2)}" filter="url(#${id}-glow)"/>`;
      }
      L.pulses.innerHTML = pulses;

      // 2D labels: centre-of-mass cross and radius measure
      if (sc.com && sc.com > 0.01) {
        const o = this.P([0, 0, 0]);
        labels += `<g opacity="${sc.com.toFixed(2)}" stroke="#ecebe6" stroke-width="1.4"><line x1="${f1(o.x - 9)}" y1="${f1(o.y)}" x2="${f1(o.x + 9)}" y2="${f1(o.y)}"/><line x1="${f1(o.x)}" y1="${f1(o.y - 9)}" x2="${f1(o.x)}" y2="${f1(o.y + 9)}"/></g>` +
          `<circle cx="${f1(o.x)}" cy="${f1(o.y)}" r="${f1(5 + 6 * ((time * 1.2) % 1))}" fill="none" stroke="#ecebe6" opacity="${(sc.com * (1 - ((time * 1.2) % 1))).toFixed(2)}"/>`;
      }
      if (sc.measure && sc.measure.alpha > 0.01) {
        const m = sc.measure, o = this.P([0, 0, 0]);
        const rr = m.R * this.scale * (this.dist / Math.sqrt(this.dist * this.dist - m.R * m.R)) * m.grow;
        const ang = -0.55, ex = o.x + Math.cos(ang) * rr, ey = o.y + Math.sin(ang) * rr;
        labels += `<g opacity="${m.alpha.toFixed(2)}"><line x1="${f1(o.x)}" y1="${f1(o.y)}" x2="${f1(ex)}" y2="${f1(ey)}" stroke="${COL[m.kind]}" stroke-width="2" marker-end="url(#${id}-m-${m.kind})"/>` +
          `<text x="${f1(ex + 10)}" y="${f1(ey - 6)}" fill="${COL[m.kind]}" font-family="IBM Plex Mono, monospace" font-size="${m.size || 15}">${m.text}</text></g>`;
      }
      L.labels.innerHTML = labels;
    }
  }

  // ---------- scene builders ----------
  function bondsFor(pos, glowFn) {
    return BONDS.map(([a, b], j) => {
      const l = len(sub(pos[a], pos[b]));
      return { a, b, alpha: clamp(1 - (l - REST[j]) / 0.9), glow: glowFn ? glowFn(j) : 0 };
    });
  }
  function trailPts(fn, u, steps = 26) {
    const pts = [];
    for (let s = 0; s <= steps; s++) pts.push(fn((u * s) / steps));
    return pts;
  }

  // 01 · Initialization: draw the shell and sample x_T on it
  const INIT = {
    dur: 7.5,
    captions: [
      [0, "Draw a shell of radius <span class='k'>r<sub>SSD</sub></span>, the mean radius of centred training molecules"],
      [1.4, "Sample n random Gaussian vectors v<sup>(i)</sup> ~ N(0, I)"],
      [2.6, "Normalise each onto the shell: <span class='k'>x<sub>T</sub><sup>(i)</sup> = r · v<sup>(i)</sup> / ‖v<sup>(i)</sup>‖</span>"],
      [3.9, "Re-centre to zero centre of mass"],
      [4.9, "<span class='k'>x<sub>T</sub></span>: one shell point per atom, resampled for every trajectory"],
    ],
    gaussCaptions: [
      [0, "Gaussian prior"],
      [1.4, "x<sub>T</sub> ~ N(0, σ<sub>T</sub><sup>2</sup> I): scattered points"],
      [4.9, "Its scale is set by the noise level, not by chemistry"],
    ],
    drifts: () => ({}),
    ssd(t, sim) {
      const draw = ease(clamp(t / 1.2));
      const proj = ease(clamp((t - 2.6) / 1.1));
      const cen = ease(clamp((t - 3.9) / 0.7));
      const sites = [], links = [], pulses = [];
      sim.raw.forEach((v, j) => {
        const inner = scl(v, R_SSD * 0.3);
        const p = sub(mix(inner, sim.onShell[j], proj), scl(sim.shift, cen));
        sites.push({ p, alpha: clamp((t - 1.4 - j * 0.06) / 0.3), size: 5.5 });
        links.push({ a: [0, 0, 0], b: p, alpha: 0.55 * clamp((t - 2.5) / 0.3) * (1 - clamp((t - 3.8) / 0.4)) });
        pulses.push({ p, k: (t - 3.7) / 0.7 });
      });
      return {
        shells: [{ kind: "ssd", R: R_SSD, draw, alpha: 1, scan: clamp((t - 4.9) / 0.5) }],
        sites, links, pulses, atoms: [], bonds: [],
        com: clamp((t - 3.9) / 0.3) * (1 - clamp((t - 5.0) / 0.4)),
        measure: { kind: "ssd", R: R_SSD, grow: ease(clamp((t - 0.3) / 0.9)), alpha: clamp((t - 0.3) / 0.3) * (1 - clamp((t - 2.4) / 0.4)), text: "r<tspan baseline-shift='sub' font-size='11'>SSD</tspan>" },
      };
    },
    gauss(t, sim) {
      return {
        shells: [{ kind: "gauss", R: R_GAUSS * 1.1, cloud: true, alpha: clamp(t / 1.2) }],
        sites: sim.gT.map((p, j) => ({ p, alpha: clamp((t - 1.4 - j * 0.06) / 0.3), color: COL.gauss, size: 5.5 })),
        pulses: sim.gT.map((p, j) => ({ p, k: (t - 1.4 - j * 0.06) / 0.6, color: COL.gauss })),
        atoms: [], bonds: [],
      };
    },
  };

  // 02 · Forward process (data -> shell), used to corrupt training data
  const FWD = {
    dur: 8.5,
    captions: [
      [0, "Forward (training): start from a centred molecule x<sub>0</sub>"],
      [0.8, "Pair atoms with shell points by a <span class='k'>random permutation π</span>"],
      [1.9, "Each atom moves straight to its point at the <span class='k' style='color:var(--rad)'>same speed α<sub>t</sub></span>"],
      [4.6, "Nearer atoms arrive first and stop: linear, not exponential, contraction"],
      [7.0, "x<sub>T</sub>: every atom on the shell"],
    ],
    gaussCaptions: [
      [0, "Gaussian forward: the same molecule"],
      [1.9, "Noise diffuses atoms toward N(0, σ<sub>T</sub><sup>2</sup> I), far beyond the molecule's size"],
    ],
    drifts: (t) => ({ rad: t > 1.9 && t < 7.0 }),
    ssd(t, sim) {
      const t0 = 1.9, v = sim.dMax / 4.8;
      const u = (i) => clamp((v * (t - t0)) / sim.d[i]);
      const posAt = (i, uu) => add(mix(X0[i], sim.S[i], uu), scl(at(sim.bF[i], uu), 0.3));
      const permIdx = Math.min(2, Math.max(0, Math.floor((t - 0.8) / 0.35)));
      const flash = t >= 0.8 && t < 1.9 ? 1 - ((t - 0.8) % 0.35) / 0.35 : 0;
      const pos = [], atoms = [], arrows = [], trails = [], pulses = [], links = [];
      for (let i = 0; i < N; i++) {
        const ui = u(i), p = posAt(i, ui);
        pos.push(p); atoms.push({ p, alpha: 1, glow: 0.45 + 0.4 * ui });
        if (ui > 0 && ui < 1) arrows.push({ i, kind: "rad", v: scl(nrm(sub(sim.S[i], p)), 0.85), alpha: clamp(t - t0) });
        trails.push({ pts: trailPts((s) => posAt(i, s), ui), color: COL.ssd, alpha: 0.75 });
        if (t >= 0.8) links.push({ a: p, b: sim.sites[sim.perms[permIdx][i]], alpha: (0.4 + 0.5 * flash) * clamp((t - 0.8) / 0.2) * (1 - ui) });
        pulses.push({ p: sim.S[i], k: (t - (t0 + sim.d[i] / v)) / 0.7 });
      }
      return {
        shells: [{ kind: "ssd", R: R_SSD, draw: 1, alpha: 1, scan: 1 }],
        sites: sim.sites.map((p) => ({ p, alpha: 0.75 })),
        atoms, arrows, trails, pulses, links, bonds: bondsFor(pos),
        com: clamp(t / 0.4) * (1 - clamp((t - 1.5) / 0.4)),
      };
    },
    gauss(t, sim) {
      const t0 = 1.9;
      const posAt = (i, g) => add(mix(X0[i], sim.gT[i], Math.sqrt(g)), scl(at(sim.gF[i], g), 4.2));
      const g = clamp((t - t0) / 5.4);
      const pos = [], atoms = [], trails = [];
      for (let i = 0; i < N; i++) {
        const p = posAt(i, g);
        pos.push(p); atoms.push({ p, alpha: 1, glow: 0.4 });
        trails.push({ pts: trailPts((s) => posAt(i, s), g, 60), color: COL.gauss, alpha: 0.55, width: 1.1 });
      }
      return { shells: [{ kind: "gauss", R: R_GAUSS * 1.1, cloud: true, alpha: clamp((t - t0) / 2) }], atoms, trails, bonds: bondsFor(pos), com: clamp(t / 0.4) * (1 - clamp((t - 1.5) / 0.4)) };
    },
  };

  // 03 · Reverse process (shell -> molecule)
  const REV = {
    dur: 10,
    captions: [
      [0, "Reverse (generation): start from a shell sample x<sub>T</sub>"],
      [0.7, "<span class='k' style='color:var(--rad)'>v<sub>rad</sub></span>: pulls every atom inward at the same speed"],
      [2.8, "<span class='k' style='color:var(--rep)'>v<sub>rep</sub></span>: keeps atoms at least d<sub>min</sub> apart"],
      [4.8, "<span class='k' style='color:var(--score)'>v<sub>score</sub></span>: the backbone's SE(3)-equivariant network refines local geometry"],
      [7.4, "Short, direct path to a valid molecule"],
    ],
    gaussCaptions: [
      [0, "Reverse from the Gaussian prior"],
      [2.8, "Long, wandering trajectories with high spatial entropy"],
      [7.4, "Most of the path is spent undoing the scale mismatch"],
    ],
    drifts: (t) => ({ rad: t > 0.7 && t < 6.6, rep: t > 2.8 && t < 6.8, score: t > 4.8 && t < 8.6 }),
    ssd(t, sim) {
      const t0 = 0.7, v = sim.dMax / 6.2;
      const land = (i) => t0 + sim.d[i] / v;
      const u = (i) => clamp((v * (t - t0)) / sim.d[i]);
      const posAt = (i, uu) => add(mix(sim.S[i], X0[i], uu), scl(at(sim.bR[i], uu), 0.3));
      const pos = [], atoms = [], arrows = [], trails = [], halos = [], pulses = [];
      for (let i = 0; i < N; i++) {
        const ui = u(i), p = posAt(i, ui);
        pos.push(p);
        atoms.push({ p, alpha: 1, glow: 0.45 + 0.35 * (1 - ui) });
        trails.push({ pts: trailPts((s) => posAt(i, s), ui), color: COL.ssd, alpha: 0.75 });
        pulses.push({ p: X0[i], k: (t - land(i)) / 0.7, color: COL.score });
      }
      for (let i = 0; i < N; i++) {
        const ui = u(i), p = pos[i];
        // radial: uniform-length arrows toward the centre
        if (t < 6.6 && ui < 0.95) arrows.push({ i, kind: "rad", v: scl(nrm(p), -0.85), alpha: clamp((t - t0) / 0.3) * (1 - clamp((t - 6.2) / 0.4)) * (1 - clamp((ui - 0.8) / 0.15)) });
        // repulsion: halo brightens as a neighbour enters d_min
        if (t > 2.8 && t < 6.8) {
          let dmin = Infinity, away = [0, 0, 0];
          for (let j = 0; j < N; j++) if (j !== i) {
            const dv = sub(p, pos[j]), dl = len(dv);
            if (dl < dmin) { dmin = dl; away = dv; }
          }
          const vis = clamp((t - 2.8) / 0.4) * (1 - clamp((t - 6.4) / 0.4));
          const near = clamp((D_MIN * 1.35 - dmin) / (D_MIN * 0.5));
          halos.push({ i, r: D_MIN / 2, alpha: vis * (0.18 + 0.82 * near) });
          if (dmin < D_MIN * 1.15) arrows.push({ i, kind: "rep", v: scl(nrm(away), 0.7), alpha: vis * near });
        }
        // score: the remaining, molecule-specific correction
        if (t > 4.8) {
          const dv = sub(X0[i], p), dl = len(dv);
          if (dl > 0.15) arrows.push({ i, kind: "score", v: scl(nrm(dv), Math.min(0.95, 0.35 + dl)), alpha: clamp((t - 4.8) / 0.4) * clamp(dl / 0.4) });
        }
      }
      const bonds = bondsFor(pos, (j) => {
        const [a, b] = BONDS[j], tf = Math.max(land(a), land(b));
        return clamp(1 - Math.abs(t - tf - 0.2) / 0.6);
      });
      return {
        shells: [{ kind: "ssd", R: R_SSD, draw: 1, alpha: 1 - 0.55 * clamp((t - 7.0) / 1.0), scan: 1 - clamp((t - 6.5) / 0.6) }],
        atoms, arrows, trails, halos, pulses, bonds,
      };
    },
    gauss(t, sim) {
      const t0 = 0.7;
      const posAt = (i, s) => add(mix(sim.gT[i], X0[i], Math.pow(s, 1.6)), scl(at(sim.gR[i], s), 4.6));
      const g = clamp((t - t0) / 7.0);
      const pos = [], atoms = [], trails = [];
      for (let i = 0; i < N; i++) {
        const p = posAt(i, g);
        pos.push(p); atoms.push({ p, alpha: 1, glow: 0.4 });
        trails.push({ pts: trailPts((s) => posAt(i, s), g, 70), color: COL.gauss, alpha: 0.55, width: 1.1 });
      }
      return { shells: [{ kind: "gauss", R: R_GAUSS * 1.1, cloud: true, alpha: 1 - 0.7 * g }], atoms, trails, bonds: bondsFor(pos) };
    },
  };

  const STORIES = [INIT, FWD, REV];

  // ---------- stage ----------
  function initStage() {
    const root = document.getElementById("stage");
    if (!root) return;
    const svgS = document.getElementById("pane-ssd"), svgG = document.getElementById("pane-gauss");
    // both panes share one scale so the size mismatch is visible
    const sceneS = new Scene3D(svgS, 600, 520);
    const sceneG = new Scene3D(svgG, 600, 520);
    const paneEls = [...root.querySelectorAll(".pane")];
    const capEl = document.getElementById("stage-caption");
    const capG = document.getElementById("stage-caption-g");
    const tabs = [...root.querySelectorAll(".stage-tab")];
    const fill = root.querySelector(".scrub .fill"), knob = root.querySelector(".scrub .knob");
    const playBtn = root.querySelector(".play");
    const pills = [...root.querySelectorAll(".pill")];

    let sim = makeSim(7), story = 0, t = 0, playing = !reduceMotion, visible = false;
    const view = { yaw: -0.5, pitch: 0.32 }, drag = { yaw: 0, pitch: 0 };
    let clock = 0;

    // phones show one pane at a time with a switch; desktops show both side by side
    const switches = [...root.querySelectorAll(".pane-switch button")];
    switches.forEach((b, i) => b.addEventListener("click", () => {
      switches.forEach((x, j) => x.setAttribute("aria-selected", String(i === j)));
      paneEls.forEach((el, j) => el.classList.toggle("active", i === j));
      draw();
    }));
    const shown = (el) => el.offsetParent !== null;

    function caption(list, el) {
      let txt = list[0][1];
      for (const [ts, s] of list) if (t >= ts) txt = s;
      if (el._txt !== txt) { el._txt = txt; el.innerHTML = txt; }
    }

    function draw() {
      const st = STORIES[story];
      const v = { yaw: view.yaw + drag.yaw + clock * 0.12, pitch: clamp(view.pitch + drag.pitch, -1.3, 1.3) };
      const world = R_GAUSS * 1.02;
      if (shown(paneEls[0])) {
        sceneS.setCamera(world, v);
        sceneS.render(st.ssd(t, sim), clock);
        caption(st.captions, capEl);
      }
      if (shown(paneEls[1])) {
        sceneG.setCamera(world, v);
        sceneG.render(st.gauss(t, sim), clock);
        caption(st.gaussCaptions, capG);
      }
      const fr = clamp(t / st.dur);
      fill.style.width = `${fr * 100}%`;
      knob.style.left = `${fr * 100}%`;
      tabs.forEach((b, i) => {
        b.setAttribute("aria-selected", String(i === story));
        b.querySelector(".bar").style.width = i === story ? `${fr * 100}%` : i < story ? "0" : "0";
      });
      const d = st.drifts(t);
      pills.forEach((p) => p.classList.toggle("on", !!d[p.dataset.k]));
    }

    function setPlaying(p) {
      playing = p;
      playBtn.innerHTML = p
        ? `<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="3.5" height="12" rx="1"/><rect x="9.5" y="2" width="3.5" height="12" rx="1"/></svg>`
        : `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2l10 6-10 6z"/></svg>`;
      playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
    }
    playBtn.addEventListener("click", () => setPlaying(!playing));

    function go(i) {
      story = i; t = 0;
      if (i === 0) sim = makeSim(7 + Math.floor(Math.random() * 1000));
      draw();
    }
    tabs.forEach((b, i) => b.addEventListener("click", () => { go(i); setPlaying(true); }));

    // scrubbing
    const scrub = root.querySelector(".scrub");
    let scrubbing = false;
    const scrubTo = (e) => {
      const r = scrub.getBoundingClientRect();
      t = clamp((e.clientX - r.left) / r.width) * STORIES[story].dur;
      draw();
    };
    scrub.addEventListener("pointerdown", (e) => { scrubbing = true; setPlaying(false); scrub.setPointerCapture(e.pointerId); scrubTo(e); });
    scrub.addEventListener("pointermove", (e) => { if (scrubbing) scrubTo(e); });
    scrub.addEventListener("pointerup", () => { scrubbing = false; });

    // drag to rotate
    for (const svg of [svgS, svgG]) {
      let on = false, px = 0, py = 0;
      svg.addEventListener("pointerdown", (e) => { on = true; px = e.clientX; py = e.clientY; svg.setPointerCapture(e.pointerId); });
      svg.addEventListener("pointermove", (e) => {
        if (!on) return;
        drag.yaw += (e.clientX - px) * 0.008; drag.pitch += (e.clientY - py) * 0.008;
        px = e.clientX; py = e.clientY;
        if (!playing) draw();
      });
      svg.addEventListener("pointerup", () => { on = false; });
    }

    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0.15 }).observe(root);

    let last = 0;
    function loop(ts) {
      const dt = last ? Math.min(0.05, (ts - last) / 1000) : 0;
      last = ts;
      if (visible) {
        if (playing) {
          clock += dt;
          t += dt;
          if (t > STORIES[story].dur) go((story + 1) % STORIES.length);
        }
        draw();
      }
      requestAnimationFrame(loop);
    }
    // seek hook (also handy for screenshots): SSD.seek(storyIndex, seconds)
    window.SSD = Object.assign(window.SSD || {}, { seek(i, tt) { story = i; t = tt; clock = tt; setPlaying(false); draw(); } });
    setPlaying(playing);
    if (reduceMotion) { t = STORIES[0].dur; }
    draw();
    requestAnimationFrame(loop);
  }

  // ---------- hero loop: shell -> molecule -> shell ----------
  function initHero() {
    const svg = document.getElementById("hero-svg");
    if (!svg) return;
    const sc = new Scene3D(svg, 600, 600);
    const sim = makeSim(3);
    const cap = document.getElementById("hero-cap");
    const orbits = [
      { R: R_EXT * 1.25, tilt: 1.1, speed: 0.07, n: 2, color: COL.ssd },
      { R: R_EXT * 1.42, tilt: -0.7, speed: -0.05, n: 3, color: COL.gauss },
      { R: R_EXT * 1.1, tilt: 0.25, speed: 0.1, n: 1, color: COL.score },
    ];
    const CYCLE = 13;
    let clock = reduceMotion ? 5.5 : 0, visible = true, last = 0;
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(svg);
    function frame() {
      const c = clock % CYCLE;
      let scene, label;
      if (c < 7) {
        scene = REV.ssd(Math.min(c * 1.15, 9.4), sim);
        scene.arrows = (scene.arrows || []).filter((a) => a.kind === "rad").map((a) => ({ ...a, alpha: a.alpha * 0.8 }));
        scene.halos = [];
        label = c < 4.5 ? "<b>reverse</b> · shell → molecule" : "a valid molecule";
      } else {
        scene = FWD.ssd(Math.min(1.9 + (c - 7) * 1.25, 8.5), sim);
        scene.links = []; scene.sites = []; scene.com = 0;
        label = "<b>forward</b> · molecule → shell";
      }
      scene.orbits = orbits;
      sc.setCamera(R_EXT * 1.55, { yaw: -0.4 + clock * 0.18, pitch: 0.35 + 0.08 * Math.sin(clock * 0.4) });
      sc.render(scene, clock);
      if (cap && cap._t !== label) { cap._t = label; cap.innerHTML = label; }
    }
    function loop(ts) {
      const dt = last ? Math.min(0.05, (ts - last) / 1000) : 0;
      last = ts;
      if (visible && !reduceMotion) { clock += dt; frame(); }
      requestAnimationFrame(loop);
    }
    window.SSD = Object.assign(window.SSD || {}, { hero(c) { clock = c; frame(); } });
    frame();
    requestAnimationFrame(loop);
  }

  // ---------- Gaussian annulus: auto-sweeping histogram ----------
  function initAnnulus() {
    const svg = document.getElementById("annulus-svg");
    if (!svg) return;
    const slider = document.getElementById("n-slider"), out = document.getElementById("n-out");
    const r = mulberry32(11), M = 600;
    const W = 520, H = 250, pl = 16, pr = 16, pt = 40, pb = 30, XMAX = 16;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const x = (v) => pl + ((W - pl - pr) * v) / XMAX;
    function draw(n) {
      out.textContent = `n = ${n}`;
      const d = 3 * n, norms = [];
      for (let m = 0; m < M; m++) { let s = 0; for (let j = 0; j < d; j++) { const z = normal(r); s += z * z; } norms.push(Math.sqrt(s)); }
      const B = 64, bw = XMAX / B, hist = new Array(B).fill(0);
      norms.forEach((v) => hist[Math.min(B - 1, Math.floor(v / bw))]++);
      const y = (c) => pt + (H - pt - pb) * (1 - c / 140);
      let s = `<line x1="${pl}" x2="${W - pr}" y1="${H - pb}" y2="${H - pb}" stroke="#2a3340"/>`;
      for (let v = 0; v <= XMAX; v += 4) s += `<text x="${x(v)}" y="${H - pb + 18}" text-anchor="middle" font-size="12" fill="#6c7380" font-family="IBM Plex Mono, monospace">${v}σ</text>`;
      hist.forEach((c, b) => {
        if (!c) return;
        const yy = Math.max(pt, y(c));
        s += `<rect x="${f1(x(b * bw))}" y="${f1(yy)}" width="${f1(x(bw) - x(0) - 1.2)}" height="${f1(H - pb - yy)}" rx="1.5" fill="#5aa2ff" opacity=".85"/>`;
      });
      const rt = Math.sqrt(d);
      s += `<line x1="${x(rt)}" x2="${x(rt)}" y1="${pt - 6}" y2="${H - pb}" stroke="#ff7a2f" stroke-width="1.5" stroke-dasharray="3 3"/>`;
      s += `<text x="${Math.min(x(rt) + 8, W - 150)}" y="${pt + 6}" font-size="14" fill="#ff7a2f" font-family="IBM Plex Mono, monospace">σ√3n = ${rt.toFixed(1)}σ</text>`;
      s += `<text x="${pl}" y="18" font-size="13" fill="#a9afba">‖x<tspan baseline-shift="sub" font-size="10">T</tspan>‖ for ${M} prior draws</text>`;
      svg.innerHTML = s;
    }
    let auto = !reduceMotion, n = 3, dir = 1, visible = false, acc = 0, last = 0;
    slider.addEventListener("input", () => { auto = false; n = +slider.value; draw(n); });
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(svg);
    function loop(ts) {
      const dt = last ? ts - last : 0; last = ts;
      if (auto && visible) {
        acc += dt;
        if (acc > 110) {
          acc = 0; n += dir;
          if (n >= 60 || n <= 2) dir *= -1;
          slider.value = n; draw(n);
        }
      }
      requestAnimationFrame(loop);
    }
    draw(+slider.value);
    requestAnimationFrame(loop);
  }

  // ---------- results dumbbell ----------
  const DB = {
    cond: {
      label: "COV-R (%) ↑", min: 60, max: 100, rows: [
        ["GeoDiff · QM9", 80.36, 92.0], ["GeoDiff · Drugs", 64.12, 91.69],
        ["SubGDiff · QM9", 90.91, 93.2], ["SubGDiff · Drugs", 76.16, 92.5],
        ["MCF · QM9", 95.0, 96.15], ["MCF · Drugs", 84.7, 85.37],
      ],
    },
    uncond: {
      label: "Molecule stability (%) ↑", min: 60, max: 100, rows: [
        ["EDM · QM9", 82.0, 99.8], ["SemlaFlow · QM9", 99.7, 99.9], ["SemlaFlow · Drugs", 97.3, 97.6],
      ],
    },
  };
  function initDumbbell() {
    const svg = document.getElementById("db-svg");
    if (!svg) return;
    let key = "cond", anim = 0, started = false;
    const W = 860, rowH = 46, pl = 150, pr = 70, pt = 30;
    function draw(p) {
      const d = DB[key], H = pt + d.rows.length * rowH + 10;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const x = (v) => pl + ((W - pl - pr) * (v - d.min)) / (d.max - d.min);
      let s = "";
      for (let v = d.min; v <= d.max; v += 10) {
        s += `<line x1="${x(v)}" x2="${x(v)}" y1="${pt - 8}" y2="${H - 6}" stroke="#1f2733"/>`;
        s += `<text x="${x(v)}" y="${pt - 14}" text-anchor="middle" font-size="12" fill="#6c7380" font-family="IBM Plex Mono, monospace">${v}</text>`;
      }
      d.rows.forEach(([name, a, b], i) => {
        const y = pt + i * rowH + rowH / 2;
        const q = ease(clamp(p * 1.6 - i * 0.12));
        const bx = x(a + (b - a) * q);
        s += `<text x="${pl - 16}" y="${y + 5}" text-anchor="end" font-size="14.5" fill="#ecebe6" font-family="IBM Plex Sans, sans-serif">${name}</text>`;
        s += `<line x1="${x(a)}" x2="${bx}" y1="${y}" y2="${y}" stroke="#ff7a2f" stroke-width="4" stroke-linecap="round" opacity=".55"/>`;
        s += `<circle cx="${x(a)}" cy="${y}" r="7" fill="#5aa2ff"/>`;
        if (q > 0) s += `<circle cx="${bx}" cy="${y}" r="${7 + 5 * (1 - q) * q * 4}" fill="#ff7a2f" opacity=".25"/><circle cx="${bx}" cy="${y}" r="8" fill="#ff7a2f"/>`;
        s += `<text x="${Math.max(bx, x(a)) + 16}" y="${y + 5}" font-size="14" fill="#ff7a2f" font-family="IBM Plex Mono, monospace" opacity="${q.toFixed(2)}">+${(b - a).toFixed(2)}</text>`;
      });
      svg.innerHTML = s;
      document.getElementById("db-metric").textContent = d.label;
    }
    function run() {
      anim = 0; const t0 = performance.now();
      const step = (ts) => { anim = clamp((ts - t0) / 1600); draw(anim); if (anim < 1) requestAnimationFrame(step); };
      reduceMotion ? draw(1) : requestAnimationFrame(step);
    }
    document.querySelectorAll("[data-db]").forEach((b) => b.addEventListener("click", () => {
      key = b.dataset.db;
      document.querySelectorAll("[data-db]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      run();
    }));
    draw(0);
    new IntersectionObserver((es) => { if (es[0].isIntersecting && !started) { started = true; run(); } }, { threshold: 0.4 }).observe(svg);
  }

  // ---------- count-up + reveal ----------
  function initReveal() {
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      e.target.querySelectorAll("[data-count]").forEach((el) => {
        const to = parseFloat(el.dataset.count), dec = (el.dataset.count.split(".")[1] || "").length, pre = el.dataset.prefix || "";
        if (reduceMotion) { el.textContent = pre + to.toFixed(dec); return; }
        const t0 = performance.now();
        const step = (ts) => { const p = easeOut(clamp((ts - t0) / 1400)); el.textContent = pre + (to * p).toFixed(dec); if (p < 1) requestAnimationFrame(step); };
        requestAnimationFrame(step);
      });
      io.unobserve(e.target);
    }), { threshold: 0.2 });
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  }

  initHero();
  initStage();
  initAnnulus();
  initDumbbell();
  initReveal();
})();
