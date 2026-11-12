// Antara: the building, the hang, walking, and the audio guide.
// Everything is laid out from the programme data embedded in the page.
import * as THREE from 'three';
import { configure, loadHearts, heartButton, heartBadge, mountGuestbook, showStats } from './community.js';

const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
configure(D.api);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

// ---------------------------------------------------------------- dimensions (metres)
const NW = 8, NH = 7.6, T = 0.5, RH = 5.4, DW = 2.8, DH = 3.6, EYE = 1.62, RAD = 0.35;
const Z_START = 11, PZ = 3.2, PW = 4.4, PH = 3.7, PT = 0.36, GAP = 1.5;
const NAVE_WALL = '#E6E3DB', NAVE_FLOOR = '#8D8A83';
// Evening light: ambient is low, each work has its own warm spot. Levels are in sRGB space.
const AMB = 0.34, AMB_D = 1.0;          // wall brightness away from the works (light rooms, dark rooms)
const TXT = '#F8F3E7', TXT2 = '#D6CFC1'; // lettering on the dim walls
const WARM = '#FFC98A';
const ZMAX_DEG = 7;                       // narrowest field of view when zoomed
const UP = new THREE.Vector3(0, 1, 0);

// ---------------------------------------------------------------- renderer
const canvas = $('c');
let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' }); }
catch (err) {
  const b = $('enter'); b.textContent = 'This browser cannot show the rooms in 3D';
  $('gate-note').textContent = 'Every work, its text, recording and code is on the plain page.'; $('gate-note').insertAdjacentHTML('beforeend', ' <a href="works/">Open it</a>');
  throw err;
}
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const MAXA = renderer.capabilities.getMaxAnisotropy();
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0B0B0A');
const camera = new THREE.PerspectiveCamera(62, 1, 0.08, 220);
const world = new THREE.Group();   // everything that reflects in the floor
const floors = new THREE.Group();
scene.add(world, floors);
const solids = [];                 // what a click can land on

// ---------------------------------------------------------------- colour helpers
const col = h => new THREE.Color(h);
const lum = h => { const c = col(h); return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; };
const mix = (a, b, t) => '#' + col(a).lerp(col(b), t).getHexString();
const rgbOf = h => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const hexOf = a => '#' + a.map(v => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0')).join('');
const dim = (h, k) => hexOf(rgbOf(h).map(v => v * k));              // scale in sRGB, as the eye reads it
const mixs = (a, b, t) => { const x = rgbOf(a), y = rgbOf(b); return hexOf(x.map((v, i) => v + (y[i] - v) * t)); };
const rgbs = h => rgbOf(h).join(',');

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = Math.max(2, Math.round(w)); c.height = Math.max(2, Math.round(h));
  draw(c.getContext('2d'), c.width, c.height);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = MAXA;
  return t;
}
function speckle(g, w, h, n, amt) {   // plaster / concrete mottling
  for (let i = 0; i < n; i++) {
    const x = Math.random() * w, y = Math.random() * h, r = (0.04 + Math.random() * 0.16) * Math.min(w, h);
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    const v = Math.random() < 0.5 ? 255 : 0;
    gr.addColorStop(0, `rgba(${v},${v},${v},${amt * Math.random()})`); gr.addColorStop(1, `rgba(${v},${v},${v},0)`);
    g.fillStyle = gr; g.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
}

// ---------------------------------------------------------------- lettering
// Lettering is never painted into the wall's own texture (a wall is tens of metres at ~40 px/m, which smears it).
// Each block gets its own transparent plane just off the wall, with a canvas sized for ~300 to 500 px per metre.
const fontOf = (ln, px) => `${ln.style || ''} ${px}px ${ln.font || 'Marcellus'}`;
const scratch = document.createElement('canvas').getContext('2d');
function layoutText(t) {   // block size in metres, and where its left edge sits along the wall
  const REF = 200, lines = []; let bw = 0, th = 0;
  for (const ln of t.lines) {
    const px = ln.size * REF; scratch.font = fontOf(ln, px); scratch.letterSpacing = ln.track ? `${ln.track * px}px` : '0px';
    const w = (scratch.measureText(ln.text).width - (ln.track ? ln.track * px : 0)) / REF, h = ln.size * (ln.lead || 1.35);
    lines.push({ ln, w, h }); bw = Math.max(bw, w); th += h;
  }
  scratch.letterSpacing = '0px';
  const al = t.align || 'left';
  return { lines, bw, th, al, x0: al === 'center' ? t.a - bw / 2 : al === 'right' ? t.a - bw : t.a };
}
function letterPlane(wall, t) {
  const L = layoutText(t), pad = 0.14, Wm = L.bw + 2 * pad, Hm = L.th + 2 * pad;
  const ppm = Math.min(t.ppm || 320, 2048 / Wm, 2048 / Hm);
  const c = document.createElement('canvas'); c.width = Math.ceil(Wm * ppm); c.height = Math.ceil(Hm * ppm);
  const g = c.getContext('2d'); g.textBaseline = 'top';
  let y = pad * ppm;
  for (const { ln, w, h } of L.lines) {
    const px = ln.size * ppm; g.font = fontOf(ln, px); g.letterSpacing = ln.track ? `${ln.track * px}px` : '0px';
    g.fillStyle = ln.color || TXT; g.shadowColor = 'rgba(0,0,0,.5)'; g.shadowBlur = px * 0.06; g.shadowOffsetY = px * 0.02;
    g.fillText(ln.text, (pad + (L.al === 'center' ? (L.bw - w) / 2 : L.al === 'right' ? L.bw - w : 0)) * ppm, y);
    y += h * ppm;
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = MAXA; tex.premultiplyAlpha = true;   // premultiplied, so mip levels do not grow dark fringes
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = true;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(Wm, Hm), new THREE.MeshBasicMaterial({ map: tex, transparent: true, premultipliedAlpha: true, depthWrite: false }));
  m.position.copy(wall.at(L.x0 + L.bw / 2, t.y - L.th / 2, 0.006)); m.rotation.y = Math.atan2(wall.nrm.x, wall.nrm.z);
  m.renderOrder = 2; m.userData.px = [c.width, c.height, +ppm.toFixed(0)];
  return m;
}

// ---------------------------------------------------------------- walls
// A wall is a vertical plane seen from the side its normal points to.
// `a` runs along it left to right as seen from that side, `y` is height.
function makeWall({ c, n, len, h, color, dark = false, amb = dark ? AMB_D : AMB, openings = [], pools = [], texts = [], ppm = 44, into = world, solid = true }) {
  const nrm = new THREE.Vector3(n[0], 0, n[1]);
  const r = new THREE.Vector3().crossVectors(UP, nrm);
  const o = new THREE.Vector3(c[0], 0, c[1]).addScaledVector(r, -len / 2);
  const at = (a, y, off = 0) => o.clone().addScaledVector(r, a).addScaledVector(nrm, off).setY(y);
  const aOf = (x, z) => (x - o.x) * r.x + (z - o.z) * r.z;
  ppm = Math.min(ppm, 4096 / len);
  const base = dim(color, amb), tint = rgbs(mixs(color, WARM, 0.3));
  const glow = (g, cx, cy, rx, ry, a, rgb = tint, stops = [[0, 1], [0.4, 0.74], [0.7, 0.34], [0.9, 0.08], [1, 0]]) => {   // soft elliptical light, added to what is there
    g.save(); g.globalCompositeOperation = 'lighter'; g.translate(cx, cy); g.scale(rx, ry);
    const gr = g.createRadialGradient(0, 0, 0, 0, 0, 1);
    for (const [t, k] of stops) gr.addColorStop(t, `rgba(${rgb},${a * k})`);
    g.fillStyle = gr; g.fillRect(-1, -1, 2, 2); g.restore();
  };
  const tex = canvasTex(len * ppm, h * ppm, (g, W, H) => {
    g.fillStyle = base; g.fillRect(0, 0, W, H);
    speckle(g, W, H, 60, 0.02);
    let gr = g.createLinearGradient(0, 0, 0, H);   // light dies away toward the ceiling and the floor
    gr.addColorStop(0, 'rgba(0,0,0,.5)'); gr.addColorStop(0.4, 'rgba(0,0,0,.05)'); gr.addColorStop(1, 'rgba(0,0,0,.3)');
    g.fillStyle = gr; g.fillRect(0, 0, W, H);
    for (const [x0, x1] of [[0, 1.4 * ppm], [W, W - 1.4 * ppm]]) {   // corners
      gr = g.createLinearGradient(x0, 0, x1, 0);
      gr.addColorStop(0, `rgba(0,0,0,${dark ? 0.3 : 0.25})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), H);
    }
    for (const p of pools) {   // a warm spot on each work: broad pool, hot scallop under the fixture, soft shadow below the frame
      const cx = p.a * ppm, cy = H - p.y * ppm, top = cy - p.h * ppm / 2;
      glow(g, cx, cy - 0.1 * p.h * ppm, (p.w * 0.5 + 1.2) * ppm, (p.h * 0.5 + 1.05) * ppm, dark ? 0.9 : 0.78);
      glow(g, cx, top - 0.12 * ppm, (p.w * 0.45 + 0.35) * ppm, 0.7 * ppm, dark ? 0.2 : 0.26, rgbs(mixs(color, '#FFDDAA', 0.5)));
      g.save(); g.shadowColor = 'rgba(0,0,0,.8)'; g.shadowBlur = 0.2 * ppm; g.shadowOffsetY = 0.09 * ppm;
      g.fillStyle = base; g.fillRect(cx - p.w * ppm / 2, cy - p.h * ppm / 2, p.w * ppm, p.h * ppm); g.restore();
    }
    g.fillStyle = 'rgba(0,0,0,.6)'; g.fillRect(0, H - 0.03 * ppm, W, 0.03 * ppm);   // shadow gap at the floor
    for (const t of texts) {   // lettering is pale, so the wall behind it is shaded a little darker (never lightened) for contrast
      const L = layoutText(t), cx = (L.x0 + L.bw / 2) * ppm, cy = H - (t.y - L.th / 2) * ppm, rx = (L.bw / 2 + 0.9) * ppm, ry = (L.th / 2 + 0.7) * ppm, k = t.shade ?? 0.42;
      g.save(); g.translate(cx, cy); g.scale(rx, ry);
      const gr = g.createRadialGradient(0, 0, 0, 0, 0, 1);
      for (const [q, m] of [[0, 1], [0.55, 0.8], [0.8, 0.35], [1, 0]]) gr.addColorStop(q, `rgba(0,0,0,${k * m})`);
      g.fillStyle = gr; g.fillRect(-1, -1, 2, 2); g.restore();
    }
  });
  // geometry: full-height strips between openings, plus a lintel over each
  const pos = [], uv = [], idx = [];
  const quad = (a0, y0, a1, y1) => {
    if (a1 - a0 < 1e-4 || y1 - y0 < 1e-4) return;
    const i = pos.length / 3;
    for (const [a, y] of [[a0, y0], [a1, y0], [a1, y1], [a0, y1]]) { const p = at(a, y); pos.push(p.x, p.y, p.z); uv.push(a / len, y / h); }
    idx.push(i, i + 1, i + 2, i, i + 2, i + 3);
  };
  let a = 0;
  for (const op of [...openings].sort((p, q) => p.a0 - q.a0)) { quad(a, 0, op.a0, h); quad(op.a0, op.h, op.a1, h); a = op.a1; }
  quad(a, 0, len, h);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex }));
  into.add(mesh); if (solid) solids.push(mesh);
  const wall = { at, aOf, nrm, r, len, h, mesh };
  for (const t of texts) into.add(letterPlane(wall, t));
  return wall;
}

function flat(x0, x1, z0, z1, y, tex, up, into, opacity = 1) {   // floor (up) or ceiling
  const g = new THREE.PlaneGeometry(Math.abs(x1 - x0), Math.abs(z1 - z0));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, transparent: opacity < 1, opacity }));
  m.rotation.x = up ? -Math.PI / 2 : Math.PI / 2;
  m.position.set((x0 + x1) / 2, y, (z0 + z1) / 2);
  into.add(m);
  return m;
}
function floorTex(wm, dm, color, amb, spills = []) {   // spills: pools of light on the floor, in metres from the top-left
  const ppm = Math.min(28, 2048 / Math.max(wm, dm));
  return canvasTex(wm * ppm, dm * ppm, (g, W, H) => {
    g.fillStyle = dim(color, amb); g.fillRect(0, 0, W, H);
    speckle(g, W, H, 140, 0.03);
    for (const s of spills) {
      g.save(); g.globalCompositeOperation = 'lighter'; g.translate(s.u * ppm, s.v * ppm); g.scale(s.rx * ppm, s.rz * ppm);
      const gr = g.createRadialGradient(0, 0, 0, 0, 0, 1);
      gr.addColorStop(0, `rgba(${s.c},${s.a})`); gr.addColorStop(0.5, `rgba(${s.c},${s.a * 0.45})`); gr.addColorStop(1, `rgba(${s.c},0)`);
      g.fillStyle = gr; g.fillRect(-1, -1, 2, 2); g.restore();
    }
    const e = 1.1 * ppm, k = 0.3;   // darker where floor meets wall
    for (const [x0, y0, x1, y1] of [[0, 0, e, 0], [W, 0, W - e, 0], [0, 0, 0, e], [0, H, 0, H - e]]) {
      const gr = g.createLinearGradient(x0, y0, x1, y1);
      gr.addColorStop(0, `rgba(0,0,0,${k})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(0, 0, W, H);
    }
  });
}
function ceilTex(wm, dm, color, amb, light, lamps = [], bars = []) {   // light = [u0,v0,u1,v1] of the laylight; lamps and bars in metres from the top-left
  const ppm = Math.min(24, 2048 / Math.max(wm, dm));
  return canvasTex(wm * ppm, dm * ppm, (g, W, H) => {
    g.fillStyle = dim(color, amb); g.fillRect(0, 0, W, H);
    if (light) {   // the laylight, at dusk
      const [x0, y0, x1, y1] = [light[0] * W, light[1] * H, light[2] * W, light[3] * H];
      g.save(); g.shadowColor = 'rgba(120,135,165,.12)'; g.shadowBlur = 1.2 * ppm;
      g.fillStyle = '#272C39'; g.fillRect(x0, y0, x1 - x0, y1 - y0); g.restore();
      g.strokeStyle = 'rgba(0,0,0,.4)'; g.lineWidth = Math.max(1, 0.05 * ppm);
      const horiz = x1 - x0 > y1 - y0, n = Math.round((horiz ? x1 - x0 : y1 - y0) / (1.6 * ppm));
      for (let i = 1; i < n; i++) {
        g.beginPath();
        if (horiz) { const x = x0 + (x1 - x0) * i / n; g.moveTo(x, y0); g.lineTo(x, y1); }
        else { const y = y0 + (y1 - y0) * i / n; g.moveTo(x0, y); g.lineTo(x1, y); }
        g.stroke();
      }
    }
    g.strokeStyle = '#0B0B0A'; g.lineWidth = 0.07 * ppm; g.lineCap = 'round';
    for (const b of bars) { g.beginPath(); g.moveTo(b[0] * ppm, b[1] * ppm); g.lineTo(b[2] * ppm, b[3] * ppm); g.stroke(); }
    for (const l of lamps) {   // a small black fixture with a warm lens
      g.fillStyle = '#0B0B0A'; g.beginPath(); g.arc(l[0] * ppm, l[1] * ppm, 0.11 * ppm, 0, 7); g.fill();
      g.save(); g.globalCompositeOperation = 'lighter'; g.translate(l[0] * ppm, l[1] * ppm); g.scale(0.55 * ppm, 0.55 * ppm);
      const gr = g.createRadialGradient(0, 0, 0, 0, 0, 1); gr.addColorStop(0, 'rgba(255,214,150,.32)'); gr.addColorStop(1, 'rgba(255,214,150,0)');
      g.fillStyle = gr; g.fillRect(-1, -1, 2, 2); g.restore();
      g.fillStyle = '#FFE6B8'; g.beginPath(); g.arc(l[0] * ppm, l[1] * ppm, 0.045 * ppm, 0, 7); g.fill();
    }
  });
}

// ---------------------------------------------------------------- the hang
const works = [];   // every hung work, in walking order
const recOf = new Map();   // picture mesh -> its record
const loader = new THREE.TextureLoader();
const loadTex = url => new Promise(res => loader.load(url, t => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = MAXA; res(t); }, undefined, () => res(null)));

const size = (w, e) => { const m = e.mat || 0, f = 0.035; return { ww: w.cm[0] / 100, wh: w.cm[1] / 100, m, f, fw: w.cm[0] / 100 + 2 * m + 2 * f, fh: w.cm[1] / 100 + 2 * m + 2 * f }; };
const centreY = s => (s.fh > 2.25 ? s.fh / 2 + 0.52 : 1.58);

function hang(wall, w, e, a, room, maxDist, dark) {
  const s = size(w, e), y = centreY(s), rotY = Math.atan2(wall.nrm.x, wall.nrm.z);
  const put = (mesh, off, dx = 0, yy = y) => { mesh.position.copy(wall.at(a + dx, yy, off)); mesh.rotation.y = rotY; world.add(mesh); return mesh; };
  put(new THREE.Mesh(new THREE.BoxGeometry(s.fw, s.fh, 0.05), new THREE.MeshBasicMaterial({ color: e.frame })), 0.025);
  if (s.m) put(new THREE.Mesh(new THREE.PlaneGeometry(s.ww + 2 * s.m, s.wh + 2 * s.m), new THREE.MeshBasicMaterial({ color: dark ? '#CFCABF' : '#DDD9CF' })), 0.0515);
  const art = put(new THREE.Mesh(new THREE.PlaneGeometry(s.ww, s.wh), new THREE.MeshBasicMaterial({ color: '#777' })), 0.054);
  // wall label: 0.56 x 0.33 m, drawn at about 1800 px per metre so it stays sharp when you step up to it
  const lw = 0.56, lh = 0.33, K = 2.2;
  const label = canvasTex(1010, 594, (g, W, H) => {
    g.fillStyle = dark ? '#BDB8AE' : '#F8F6F0'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#121210'; g.textBaseline = 'top';
    const X = 30 * K, MW = W - 60 * K;
    g.font = `600 ${30 * K}px "Albert Sans"`; g.fillText(e.artist, X, 24 * K, MW);
    g.font = `italic ${25 * K}px "Albert Sans"`;
    let line = '', yy = 68 * K;
    for (const word of w.title.split(' ')) {
      if (g.measureText(line + word).width > MW && line) { g.fillText(line.trim(), X, yy); yy += 31 * K; line = ''; }
      line += word + ' ';
    }
    g.fillText(line.trim() + ', ' + w.year, X, yy); yy += 44 * K;
    g.font = `${21 * K}px "Albert Sans"`; g.fillStyle = '#2E2D2A';
    g.fillText(w.medium, X, yy, MW); g.fillText(`${w.cm[0]} × ${w.cm[1]} cm`, X, yy + 28 * K);
  });
  put(new THREE.Mesh(new THREE.PlaneGeometry(lw, lh), new THREE.MeshBasicMaterial({ map: label, color: '#DAD5CB' })), 0.006, s.fw / 2 + 0.3 + lw / 2, 1.22);
  const rec = { w, e, room, art, s, y, pos: wall.at(a, y, 0), n: wall.nrm.clone(), maxDist, level: -1, want: 0, busy: false };
  recOf.set(art, rec); solids.push(art); works.push(rec);
  return rec;
}

// ---------------------------------------------------------------- layout
const rooms = [];
{
  let zL = -2.5, zR = -2.5;
  D.exhibitions.forEach((e, i) => {
    const side = i % 2 === 0 ? -1 : 1;
    const list = e.works.filter(w => !w.nave);
    if (!list.length) return;
    let hi = list.findIndex(w => w.hero); if (hi < 0) hi = Math.floor(list.length / 2);
    const A = list.slice(0, hi), B = list.slice(hi + 1), hero = list[hi];
    const need = arr => arr.reduce((t, w) => t + size(w, e).fw, 0) + GAP * (arr.length + 1);
    const depth = Math.max(8.5, need(A), need(B)), width = Math.max(9.5, size(hero, e).fw + 5.5);
    const z0 = side < 0 ? zL : zR, z1 = z0 - width;
    if (side < 0) zL = z1 - 1.4; else zR = z1 - 1.4;
    rooms.push({ e, side, z0, z1, zc: (z0 + z1) / 2, xNear: side * (NW / 2 + T), xFar: side * (NW / 2 + T + depth), depth, width, A, B, hero, dark: lum(e.wall) < 0.12, seen: false });
  });
  var Z_END = Math.min(zL, zR) - 2.5;
}
const NL = Z_START - Z_END;

function buildRoom(R) {
  const { e, side, z0, z1, zc, xNear, xFar, depth, width, dark } = R;
  const xm = (xNear + xFar) / 2;
  // positions first, so the walls know where the light falls
  const spread = (arr, wallZ) => {
    const tot = arr.reduce((t, w) => t + size(w, e).fw, 0), gap = (depth - tot) / (arr.length + 1);
    let d = gap; return arr.map(w => { const s = size(w, e); const p = { w, x: xNear + side * (d + s.fw / 2), z: wallZ, s }; d += s.fw + gap; return p; });
  };
  const leftZ = side < 0 ? z0 : z1, rightZ = side < 0 ? z1 : z0;   // left-hand wall as you walk in
  const pa = spread(R.A, leftZ), pb = spread([...R.B].reverse(), rightZ).reverse();
  const hs = size(R.hero, e);
  const pool = (wall, p) => ({ a: wall.aOf(p.x, p.z), y: centreY(p.s), w: p.s.fw, h: p.s.fh });
  const mk = (c, n, len, pts, extra = {}) => {
    const probe = { aOf: (x, z) => { const r = new THREE.Vector3().crossVectors(UP, new THREE.Vector3(n[0], 0, n[1])); return (x - (c[0] - r.x * len / 2)) * r.x + (z - (c[1] - r.z * len / 2)) * r.z; } };
    return makeWall({ c, n, len, h: RH, color: e.wall, dark, pools: pts.map(p => pool(probe, p)), ...extra });
  };
  const far = mk([xFar, zc], [-side, 0], width, [{ x: xFar, z: zc, s: hs }]);
  const wl = mk([xm, leftZ], [0, side < 0 ? -1 : 1], depth, pa);
  const wr = mk([xm, rightZ], [0, side < 0 ? 1 : -1], depth, pb);
  mk([xNear, zc], [side, 0], width, [], { openings: [{ a0: width / 2 - DW / 2, a1: width / 2 + DW / 2, h: DH }] });
  pa.forEach(p => hang(wl, p.w, e, wl.aOf(p.x, p.z), R, width - 1.0, dark));
  hang(far, R.hero, e, width / 2, R, depth - 1.0, dark);
  pb.forEach(p => hang(wr, p.w, e, wr.aOf(p.x, p.z), R, width - 1.0, dark));
  // floor and ceiling: a pool of light on the floor in front of each work, and a spot fixture above it
  const fx0 = side * NW / 2, fMinX = Math.min(fx0, xFar), cMinX = Math.min(xNear, xFar), spills = [], lamps = [], bars = [];
  const inward = p => (p.z === z0 ? -1 : 1);
  const wc = rgbs(mixs(e.wall, WARM, 0.55));
  for (const grp of [pa, pb]) {
    const ls = [];
    for (const p of grp) {
      const nz = inward(p);
      spills.push({ u: p.x - fMinX, v: p.z + nz * 1.15 - z1, rx: p.s.fw * 0.55 + 0.8, rz: 1.5, a: dark ? 0.3 : 0.26, c: wc });
      ls.push([p.x - cMinX, z0 - (p.z + nz * 1.0)]);
    }
    lamps.push(...ls);
    if (ls.length) { const xs = ls.map(l => l[0]); bars.push([Math.min(...xs) - 0.5, ls[0][1], Math.max(...xs) + 0.5, ls[0][1]]); }
  }
  spills.push({ u: xFar - side * 1.15 - fMinX, v: zc - z1, rx: 1.5, rz: hs.fw * 0.55 + 0.8, a: dark ? 0.3 : 0.26, c: wc });
  lamps.push([xFar - side * 1.0 - cMinX, z0 - zc]); bars.push([xFar - side * 1.0 - cMinX, z0 - zc - hs.fw / 2 - 0.5, xFar - side * 1.0 - cMinX, z0 - zc + hs.fw / 2 + 0.5]);
  const fcol = dark ? mix(e.wall, '#000000', 0.45) : mix(NAVE_FLOOR, e.wall, 0.18);
  solids.push(flat(Math.min(fx0, xFar), Math.max(fx0, xFar), z1, z0, 0, floorTex(Math.abs(xFar - fx0), width, fcol, dark ? 1.0 : 0.38, spills), true, floors, 0.84));
  const cc = dark ? mix(e.wall, '#000000', 0.7) : mix(e.wall, '#ffffff', 0.25);
  flat(xNear, xFar, z1, z0, RH, ceilTex(depth, width, cc, dark ? 1.0 : 0.3, dark ? null : [0.16, 0.16, 0.84, 0.84], lamps, bars), false, world);
  const jm = new THREE.MeshBasicMaterial({ color: mixs(dim(NAVE_WALL, 0.24), WARM, 0.12), side: THREE.DoubleSide });
  for (const zz of [zc - DW / 2, zc + DW / 2]) {
    const j = new THREE.Mesh(new THREE.PlaneGeometry(T, DH), jm); j.position.set(side * (NW / 2 + T / 2), DH / 2, zz); world.add(j);
  }
  const so = new THREE.Mesh(new THREE.PlaneGeometry(T, DW), new THREE.MeshBasicMaterial({ color: dim(NAVE_WALL, 0.2), side: THREE.DoubleSide }));
  so.rotation.x = Math.PI / 2; so.position.set(side * (NW / 2 + T / 2), DH, zc); world.add(so);
}

// The guest book, open on a lectern near the entrance. A click on it opens the book.
const LECT = { x: 1.9, z: 6.9, yaw: -0.7 };
function buildLectern() {
  const g = new THREE.Group(); g.position.set(LECT.x, 0, LECT.z); g.rotation.y = LECT.yaw;
  const wood = new THREE.MeshBasicMaterial({ color: '#4A3A2E' }), woodTop = new THREE.MeshBasicMaterial({ color: '#5C4a3B' });
  const part = (to, geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); to.add(m); return m; };
  part(g, new THREE.BoxGeometry(0.52, 0.06, 0.44), wood, 0, 0.03, 0);            // foot
  part(g, new THREE.BoxGeometry(0.3, 0.92, 0.22), wood, 0, 0.51, 0);             // stem
  const desk = new THREE.Group(); desk.position.set(0, 1.02, 0); desk.rotation.x = 0.42;   // sloped top, facing the visitor (+z)
  g.add(desk);
  part(desk, new THREE.BoxGeometry(0.64, 0.05, 0.46), woodTop, 0, 0, 0);
  part(desk, new THREE.BoxGeometry(0.64, 0.045, 0.05), wood, 0, 0.03, 0.21);     // the lip that keeps the book from sliding
  // the open book: pale leaves on a dark cover
  const c = document.createElement('canvas'); c.width = 512; c.height = 360; const x = c.getContext('2d');
  x.fillStyle = '#EDE6D4'; x.fillRect(0, 0, 512, 360);
  x.fillStyle = 'rgba(60,50,40,.16)'; x.fillRect(250, 0, 12, 360);
  x.fillStyle = '#3A3128'; x.font = '34px Marcellus, serif'; x.textAlign = 'center'; x.fillText('Guest book', 384, 60);
  x.strokeStyle = 'rgba(58,49,40,.28)'; x.lineWidth = 2;
  for (let i = 0; i < 6; i++) { const y = 104 + i * 40; x.beginPath(); x.moveTo(296, y); x.lineTo(470, y); x.stroke(); x.beginPath(); x.moveTo(40, y); x.lineTo(214, y); x.stroke(); }
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = MAXA;
  const cover = new THREE.MeshBasicMaterial({ color: '#5B2F27' });
  part(desk, new THREE.BoxGeometry(0.6, 0.012, 0.4), cover, 0, 0.031, -0.01);
  part(desk, new THREE.BoxGeometry(0.54, 0.018, 0.36), [cover, cover, new THREE.MeshBasicMaterial({ map: tex }), cover, cover, cover], 0, 0.046, -0.01);
  g.traverse(m => { if (m.isMesh) { m.userData.book = true; solids.push(m); } });
  world.add(g);
}
let naveWork = null;
function buildNave() {
  const lettering = (wall, r, alignRight) => ({
    a: wall.aOf(r.side * NW / 2, r.zc + DW / 2 + 0.9), y: 2.55, align: alignRight ? 'left' : 'right',
    lines: [{ text: r.e.artist, size: 0.36, lead: 1.5 }, { text: r.e.title, size: 0.2, font: '"Albert Sans"', style: 'italic 300', lead: 1.7 },
      { text: D.dates, size: 0.105, font: '"Albert Sans"', color: TXT2 }],
  });
  const open = (wall, side) => rooms.filter(r => r.side === side).map(r => { const a = wall.aOf(side * NW / 2, r.zc); return { a0: a - DW / 2, a1: a + DW / 2, h: DH }; });
  for (const side of [-1, 1]) {
    // left wall runs away from the entrance, so its lettering sits before each door and reads toward it
    const probe = makeProbe([side * NW / 2, (Z_START + Z_END) / 2], [-side, 0], NL);
    makeWall({ c: [side * NW / 2, (Z_START + Z_END) / 2], n: [-side, 0], len: NL, h: NH, color: NAVE_WALL, ppm: 40,
      openings: open(probe, side), texts: rooms.filter(r => r.side === side).map(r => ({ ...lettering(probe, r, side > 0), ppm: 340 })) });
  }
  makeWall({ c: [0, Z_START], n: [0, -1], len: NW, h: NH, color: NAVE_WALL });
  const nw = D.exhibitions.flatMap(e => e.works.filter(w => w.nave).map(w => ({ w, e })))[0];
  const s = nw && size(nw.w, nw.e);
  const end = makeWall({ c: [0, Z_END], n: [0, 1], len: NW, h: NH, color: NAVE_WALL,
    pools: nw ? [{ a: NW / 2, y: centreY(s), w: s.fw, h: s.fh }] : [],
    texts: nw ? [] : [{ a: NW / 2, y: 3.3, align: 'center', lines: [
      { text: 'Stand further back than you think.', size: 0.25, lead: 1.55 }, { text: 'Then come much closer than is polite.', size: 0.25, lead: 2.3 },
      { text: D.curator.name, size: 0.12, font: '"Albert Sans"', color: TXT2 }] }] });
  if (nw) naveWork = hang(end, nw.w, nw.e, NW / 2, null, 30, false);
  // the light of each room spills out of its door onto the nave floor; the end work has its own pool and fixture
  const spills = rooms.map(r => ({ u: r.side * 2.9 + NW / 2, v: r.zc - Z_END, rx: 2.4, rz: DW * 0.9, a: 0.3, c: rgbs(mixs(r.e.wall, WARM, 0.55)) }));
  const lamps = [];
  if (nw) {
    spills.push({ u: NW / 2, v: 1.2, rx: s.fw * 0.55 + 0.8, rz: 1.5, a: 0.26, c: rgbs(WARM) });
    lamps.push([NW / 2, NL - 1.0]);
  }
  solids.push(flat(-NW / 2, NW / 2, Z_END, Z_START, 0, floorTex(NW, NL, NAVE_FLOOR, 0.38, spills), true, floors, 0.84));
  flat(-NW / 2, NW / 2, Z_END, Z_START, NH, ceilTex(NW, NL, '#CFCBC2', 0.3, [0.3, 0.012, 0.7, 0.988], lamps), false, world);
  // the title wall in the foyer
  const lines = [{ text: D.name.toUpperCase(), size: 0.6, track: 0.22, lead: 1.75 }, { text: D.line, size: 0.16, font: '"Albert Sans"', style: '400', color: TXT2 }];
  const title = makeWall({ c: [0, PZ + PT / 2], n: [0, 1], len: PW, h: PH, color: '#F1EFE9', texts: [{ a: PW / 2, y: 2.45, align: 'center', lines, ppm: 560, shade: 0.5 }] });
  title.mesh.userData.start = true;   // a click on it begins the visit
  makeWall({ c: [0, PZ - PT / 2], n: [0, -1], len: PW, h: PH, color: '#F1EFE9', texts: [{ a: 0.45, y: 2.6, lines: [
    { text: D.season, size: 0.3, lead: 1.5 }, { text: D.dates, size: 0.13, font: '"Albert Sans"', color: TXT2, lead: 2.6 },
    ...D.exhibitions.map(e => ({ text: `${e.artist}, ${e.title}`, size: 0.115, font: '"Albert Sans"', lead: 1.62 })),
    { text: '', size: 0.1 }, { text: `Hung by ${D.curator.name}`, size: 0.105, font: '"Albert Sans"', style: 'italic', color: TXT2 }] }] });
  buildLectern();
  const edge = new THREE.MeshBasicMaterial({ color: dim('#D5D2CA', 0.3) });
  const box = new THREE.Mesh(new THREE.BoxGeometry(PW - 0.002, PH - 0.002, PT - 0.004), edge); box.position.set(0, PH / 2, PZ); world.add(box);
}
function makeProbe(c, n, len) {
  const r = new THREE.Vector3().crossVectors(UP, new THREE.Vector3(n[0], 0, n[1]));
  return { aOf: (x, z) => (x - (c[0] - r.x * len / 2)) * r.x + (z - (c[1] - r.z * len / 2)) * r.z };
}

// ---------------------------------------------------------------- where you can stand
const inRect = (x, z, r) => x >= r[0] && x <= r[1] && z >= r[2] && z <= r[3];
let walkable = [], blocked = [];
function buildCollision() {
  walkable = [[-NW / 2 + RAD, NW / 2 - RAD, Z_END + RAD, Z_START - RAD]];
  for (const r of rooms) {
    const xs = [r.xNear + r.side * RAD, r.xFar - r.side * RAD].sort((a, b) => a - b);
    walkable.push([xs[0], xs[1], r.z1 + RAD, r.z0 - RAD]);
    const ds = [r.side * (NW / 2 - 0.6), r.side * (NW / 2 + T + 0.6)].sort((a, b) => a - b);
    walkable.push([ds[0], ds[1], r.zc - DW / 2 + RAD, r.zc + DW / 2 - RAD]);
  }
  blocked = [[-PW / 2 - RAD, PW / 2 + RAD, PZ - PT / 2 - RAD, PZ + PT / 2 + RAD], [LECT.x - 0.45 - RAD, LECT.x + 0.45 + RAD, LECT.z - 0.4 - RAD, LECT.z + 0.4 + RAD]];
}
const canStand = (x, z) => walkable.some(r => inRect(x, z, r)) && !blocked.some(r => inRect(x, z, r));
const spaceOf = (x, z) => rooms.findIndex(r => Math.abs(x) > NW / 2 + T / 2 && Math.sign(x) === r.side && z <= r.z0 && z >= r.z1);

function route(a, b) {
  const sa = spaceOf(a.x, a.z), sb = spaceOf(b.x, b.z), pts = [];
  if (sa !== sb) {
    if (sa >= 0) { const r = rooms[sa]; pts.push({ x: r.xNear + r.side, z: r.zc }, { x: r.side * (NW / 2 - 1.3), z: r.zc }); }
    if (sb >= 0) { const r = rooms[sb]; pts.push({ x: r.side * (NW / 2 - 1.3), z: r.zc }, { x: r.xNear + r.side, z: r.zc }); }
  }
  pts.push(b);
  const out = []; let p = a;
  for (const q of pts) {   // walk round the title wall, not through it
    if ((p.z - PZ) * (q.z - PZ) < 0) {
      const x = p.x + (q.x - p.x) * (PZ - p.z) / (q.z - p.z);
      if (Math.abs(x) < PW / 2 + 0.7) {
        const sx = (x >= 0 ? 1 : -1) * (PW / 2 + 0.95), d = Math.sign(q.z - p.z);
        out.push({ x: sx, z: PZ - d * 0.9 }, { x: sx, z: PZ + d * 0.9 });
      }
    }
    out.push(q); p = q;
  }
  return out;
}

// ---------------------------------------------------------------- the visitor
const me = { x: 0, z: 8.6, yaw: 0, pitch: 0 };
let walk = null, focus = null, entered = false, lastInput = 0;
const keys = new Set();
const vel = { f: 0, s: 0, t: 0 };   // smoothed walking, strafing and turning
let baseFov = 62, zoom = 1, zoomT = 1, zoomMax = 9, anchor = null;   // optical zoom: narrows the field of view toward a point on screen
const fovFor = z => THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(baseFov / 2)) / z));
let shiftX = 0, shiftY = 0, shiftTX = 0, shiftTY = 0;   // view offset so the panel does not cover the work

const angDiff = (a, b) => { let d = (b - a) % (2 * Math.PI); if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return d; };
const yawTo = (dx, dz) => Math.atan2(-dx, -dz);
const wide = () => innerWidth >= 820;

function go(to, endYaw, endPitch, onArrive) {
  const pts = route({ x: me.x, z: me.z }, to);
  let len = 0, p = me; for (const q of pts) { len += Math.hypot(q.x - p.x, q.z - p.z); p = q; }
  zoomT = 1; anchor = null;
  walk = { pts, i: 0, endYaw, endPitch: endPitch ?? 0, onArrive, speed: clamp(len / 3.6, 2.4, 8), free: false, settle: false, t: 0 };
}

function viewPose(rec) {
  const vt = Math.tan(THREE.MathUtils.degToRad(baseFov / 2)), ht = vt * camera.aspect;
  const fx = wide() ? (innerWidth - 430) / innerWidth * 0.8 : 0.9, fy = wide() ? 0.74 : 0.37;
  let d = Math.max(rec.s.fh / 2 / (vt * fy), rec.s.fw / 2 / (ht * fx), 1.25);
  d = Math.min(d, rec.maxDist);
  const x = rec.pos.x + rec.n.x * d, z = rec.pos.z + rec.n.z * d;
  return { x, z, yaw: yawTo(-rec.n.x, -rec.n.z), pitch: Math.atan2(rec.y - EYE, d) };
}

// ---------------------------------------------------------------- stops on the guided walk
const stops = [];
function buildStops() {
  for (const r of rooms) {
    stops.push({ kind: 'intro', room: r, pose: () => ({ x: r.xNear + r.side * 1.7, z: r.zc, yaw: yawTo(r.side, 0), pitch: 0.02 }) });
    for (const rec of works.filter(w => w.room === r)) stops.push({ kind: 'work', rec, pose: () => viewPose(rec) });
  }
  if (naveWork) stops.push({ kind: 'work', rec: naveWork, pose: () => viewPose(naveWork) });
}
let cur = -1;
function toStop(i) {
  cur = (i + stops.length) % stops.length;
  const s = stops[cur], p = s.pose();
  if (s.kind === 'work') { s.rec.want = 2; if (s.rec.room) s.rec.room.seen = true; }
  else s.room.seen = true;
  closePanel(true);
  go({ x: p.x, z: p.z }, p.yaw, p.pitch, () => (s.kind === 'work' ? showWork(s.rec) : showIntro(s.room)));
}
const nextStop = d => {
  if (cur < 0) {
    const sp = spaceOf(me.x, me.z);
    cur = sp >= 0 ? stops.findIndex(s => s.kind === 'intro' && s.room === rooms[sp]) - (d > 0 ? 1 : 0) : (d > 0 ? -1 : 0);
  }
  toStop(cur + d);
};

// ---------------------------------------------------------------- sound: the curator's voice (off until asked for) and ambient music
const voice = $('voice');
const store = { get: (k, d) => { try { const v = localStorage.getItem('antara.' + k); return v == null ? d : v === '1'; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem('antara.' + k, v ? '1' : '0'); } catch {} } };
let voiceOn = store.get('voice', false), musicOn = store.get('music', true), playing = null;
const MUSIC_VOL = 0.22;
const music = D.music ? Object.assign(new Audio(), { loop: true, preload: 'auto', volume: 0 }) : null;
let musicLive = false, musicTried = false;
if (music) music.addEventListener('error', () => { music.dataset.bad = '1'; });

function setTrack(a) {
  playing = a; $('player').hidden = !a;
  if (!a) return;
  if (voice.dataset.src !== a.src) { voice.src = a.src; voice.dataset.src = a.src; }
  $('p-bar').style.width = '0%'; $('p-time').textContent = mmss(a.dur); $('p-play').textContent = 'Listen';
}
function play() { voice.play().then(() => { $('p-play').textContent = 'Pause'; }).catch(() => {}); }
$('p-play').addEventListener('click', () => (voice.paused ? play() : voice.pause()));
voice.addEventListener('pause', () => { $('p-play').textContent = voice.ended ? 'Listen again' : 'Listen'; });
voice.addEventListener('play', () => { $('p-play').textContent = 'Pause'; });
voice.addEventListener('timeupdate', () => {
  if (!playing) return;
  const d = voice.duration || playing.dur;
  $('p-bar').style.width = `${100 * voice.currentTime / d}%`; $('p-time').textContent = mmss(Math.max(0, d - voice.currentTime));
});

function startMusic() {   // browsers want a gesture first; called from every early input until it takes
  if (!music || !musicOn || musicLive || music.dataset.bad) return;
  if (!music.src) music.src = 'ambient.mp3';
  music.play().then(() => { musicLive = true; }).catch(() => {});
}
for (const t of ['pointerdown', 'keydown', 'touchend']) addEventListener(t, startMusic, { capture: true, passive: true });
let lastAudio = performance.now();
setInterval(() => {   // fade in over about 3 s, duck to a third while the voice speaks
  const now = performance.now(), dt = Math.min(0.5, (now - lastAudio) / 1000); lastAudio = now;
  if (!music || !musicLive) return;
  const speaking = !voice.paused && !voice.ended;
  const target = musicOn ? (speaking ? MUSIC_VOL / 3 : MUSIC_VOL) : 0;
  const rate = target < music.volume ? MUSIC_VOL / 0.7 : MUSIC_VOL / (speaking ? 1.2 : 3);
  music.volume = clamp(music.volume + clamp(target - music.volume, -rate * dt, rate * dt), 0, 1);
  if (!musicOn && music.volume <= 0.002 && !music.paused) music.pause();
}, 80);
document.addEventListener('visibilitychange', () => { if (!music || !musicLive) return; if (document.hidden) music.pause(); else if (musicOn) music.play().catch(() => {}); });

const SND = { voice: { on: 'Voice is on. Turn off the curator\'s voice', off: 'Voice is off. Turn on the curator\'s voice' },
  music: { on: 'Music is on. Turn off the music', off: 'Music is off. Turn on the music' } };
function syncSound() {
  for (const b of document.querySelectorAll('[data-snd]')) {
    const k = b.dataset.snd, on = k === 'voice' ? voiceOn : musicOn;
    b.setAttribute('aria-pressed', on); b.title = SND[k][on ? 'on' : 'off'];
  }
}
document.addEventListener('click', e => {
  const b = e.target.closest('[data-snd]'); if (!b) return;
  if (b.dataset.snd === 'voice') {
    voiceOn = !voiceOn; store.set('voice', voiceOn);
    if (!voiceOn) voice.pause();
    else if (playing && !panel.hidden && voice.paused && !voice.ended) play();
  } else {
    musicOn = !musicOn; store.set('music', musicOn);
    if (musicOn) { if (musicLive) music.play().catch(() => {}); else startMusic(); }
  }
  syncSound();
});
syncSound();

// ---------------------------------------------------------------- the panel
const panel = $('panel'), body = $('p-body'), acts = $('p-acts');
let panelAt = { x: 0, z: 0 };
function openPanel(audio, auto) {
  panelAt = { x: me.x, z: me.z };
  panel.hidden = false; panel.scrollTop = 0;
  requestAnimationFrame(() => panel.classList.add('on'));
  setTrack(audio);
  if (audio && auto && voiceOn) play();
  const r = panel.getBoundingClientRect();
  if (wide()) { shiftTX = r.width / 2 + 10; shiftTY = 0; } else { shiftTX = 0; shiftTY = Math.min(r.height, innerHeight * 0.5) / 2; }
}
function closePanel(keepAudio) {
  panel.classList.remove('on'); panel.hidden = true; focus = null; shiftTX = shiftTY = 0;
  if (!keepAudio) voice.pause();
}
$('p-close').addEventListener('click', () => closePanel(false));
const btn = (text, fn, cls) => { const b = el('button', cls, text); b.type = 'button'; b.addEventListener('click', fn); return b; };

// ---------------------------------------------------------------- sharing
function shareCopy(url, b) {
  const done = () => { b.textContent = 'Copied'; b.classList.add('ok'); setTimeout(() => { b.textContent = 'Copy link'; b.classList.remove('ok'); }, 2200); };
  const old = () => { const t = document.createElement('textarea'); t.value = url; t.style.cssText = 'position:fixed;opacity:0'; document.body.append(t); t.select(); try { document.execCommand('copy'); done(); } catch { window.prompt('Copy this link', url); } t.remove(); };
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(url).then(done, old); else old();
}
function networks({ title, text, url }) {
  const u = encodeURIComponent(url), t = encodeURIComponent(title);
  return [['X', `https://x.com/intent/post?text=${t}&url=${u}`], ['Facebook', `https://www.facebook.com/sharer/sharer.php?u=${u}`],
    ['WhatsApp', `https://wa.me/?text=${encodeURIComponent(title + ' ' + url)}`], ['LinkedIn', `https://www.linkedin.com/sharing/share-offsite/?url=${u}`],
    ['Email', `mailto:?subject=${t}&body=${encodeURIComponent(text + '\n\n' + url)}`]];
}
// Returns [button, popover]. `get` gives {title, text, url} when asked. The system share sheet if there is one, else a small row of plain links.
function sharer(get) {
  const pop = el('div', 'sharepop'); pop.hidden = true;
  const b = btn('Share', async () => {
    const s = get();
    if (navigator.share) { try { await navigator.share({ title: s.title, text: s.text, url: s.url }); } catch {} return; }
    if (!pop.hidden) { pop.hidden = true; b.setAttribute('aria-expanded', 'false'); return; }
    const copy = btn('Copy link', () => shareCopy(s.url, copy), 'link');
    pop.replaceChildren(copy, ...networks(s).map(([n, h]) => { const a = el('a', null, n); a.href = h; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a; }));
    pop.hidden = false; b.setAttribute('aria-expanded', 'true');
    requestAnimationFrame(() => { if (panel.contains(pop)) panel.scrollTo({ top: panel.scrollHeight, behavior: 'smooth' }); });
  }, 'share');
  b.setAttribute('aria-expanded', 'false');
  return [b, pop];
}
const shareOf = {
  work: rec => () => ({ title: `${rec.w.title}, ${rec.e.artist}: ${D.name}`, text: rec.w.blurb, url: D.url + rec.w.share }),
  room: e => () => ({ title: `${e.artist}, ${e.title}: ${D.name}`, text: e.blurb, url: D.url + e.share }),
  gallery: () => ({ title: `${D.name}: ${D.line}`, text: D.description, url: D.url }),
};
function setAddress(q) { try { history.replaceState(null, '', location.pathname + (q ? '?' + q : '') + location.hash); } catch {} }

function codeLine(rec) {
  const c = rec.w.code, p = el('p', 'codeline', `Computed by one program, ${c.lines.toLocaleString('en-US')} lines of Python. `);
  p.append(btn('Read the code', () => openCode(rec), 'link'));
  return p;
}
function noteLine(rec) {
  const p = el('p', 'codeline');
  p.append(btn('Leave a note about this work', () => openGuest(rec.w.id), 'link'));
  return p;
}
function showWork(rec) {
  focus = rec; cur = stops.findIndex(s => s.rec === rec);
  body.replaceChildren(
    el('p', 'kicker', rec.e.artist), el('h2', null, rec.w.title),
    el('p', 'meta', `${rec.w.year}. ${rec.w.medium}. ${rec.w.cm[0]} × ${rec.w.cm[1]} cm`),
    ...rec.w.story.map(t => el('p', null, t)), codeLine(rec), noteLine(rec));
  acts.replaceChildren(btn('Look closer', () => openCloser(rec)), heartButton(rec.w.id), ...sharer(shareOf.work(rec)));
  setAddress('w=' + encodeURIComponent(rec.w.id));
  openPanel(rec.w.audio, true);
}
function showIntro(R) {
  const e = R.e; R.seen = true; cur = stops.findIndex(s => s.kind === 'intro' && s.room === R);
  const more = el('div', 'more'); more.hidden = true;
  more.append(el('h3', null, 'The artist'), el('p', 'meta', e.origin), ...e.bio.map(t => el('p', null, t)), ...e.said.map(t => el('blockquote', null, t)));
  body.replaceChildren(el('p', 'kicker', 'Exhibition'), el('h2', null, e.artist), el('p', 'sub', e.title), ...e.intro.map(t => el('p', null, t)), more);
  const b = btn('About the artist', () => { more.hidden = !more.hidden; b.textContent = more.hidden ? 'About the artist' : 'Hide'; });
  acts.replaceChildren(b, btn('Go to the first work', () => nextStop(1)), ...sharer(shareOf.room(e)));
  setAddress('x=' + encodeURIComponent(e.slug));
  openPanel(e.audio, true);
}
function showWelcome() {
  body.replaceChildren(el('p', 'kicker', `${D.curator.name}, ${D.curator.role.toLowerCase()}`), el('h2', null, `Welcome to ${D.name}`), ...D.welcome.text.map(t => el('p', null, t)));
  const about = el('a', null, 'About Antara'); about.href = 'about/';
  acts.replaceChildren(btn('Begin the guided walk', () => toStop(0)), about, ...sharer(shareOf.gallery));
  setAddress('');
  openPanel(D.welcome.audio, true);
}

// ---------------------------------------------------------------- looking closer (flat, zoomable)
const dlg = $('closer');
{
  const stage = dlg.querySelector('.stage'), img = dlg.querySelector('img'), cap = dlg.querySelector('.cap');
  let s = 1, x = 0, y = 0, fitS = 1, nw = 1, nh = 1;
  const maxS = () => Math.max(fitS, 1.5);
  const apply = () => {
    const W = stage.clientWidth, H = stage.clientHeight, w = nw * s, h = nh * s;
    x = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, x)); y = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, y));
    img.style.transform = `translate(${x}px,${y}px) scale(${s})`;
  };
  const fit = () => { fitS = Math.min(stage.clientWidth / nw, stage.clientHeight / nh); s = fitS; x = y = 0; apply(); };
  const zoomAt = (cx, cy, f) => { const ns = Math.min(maxS(), Math.max(fitS, s * f)); x = cx - (cx - x) * ns / s; y = cy - (cy - y) * ns / s; s = ns; apply(); };
  window.openCloser = rec => {
    [nw, nh] = rec.w.px; img.style.width = nw + 'px'; img.style.height = nh + 'px';
    img.src = rec.w.w; img.alt = `${rec.w.title}, by ${rec.e.artist}`;
    cap.textContent = `${rec.w.title}. Scroll or pinch to go closer, drag to look around.`;
    dlg.showModal(); fit();
    const big = new Image(); big.onload = () => { if (dlg.open) img.src = big.src; }; big.src = rec.w.z;
  };
  dlg.querySelector('[data-close]').addEventListener('click', () => dlg.close());
  dlg.querySelectorAll('[data-zoom]').forEach(b => b.addEventListener('click', () => zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, +b.dataset.zoom)));
  stage.addEventListener('wheel', e => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0016)); }, { passive: false });
  stage.addEventListener('dblclick', e => zoomAt(e.clientX, e.clientY, s > fitS * 1.05 ? 0 : maxS() / s));
  const pts = new Map(); let lastD = 0;
  stage.addEventListener('pointerdown', e => { stage.setPointerCapture(e.pointerId); pts.set(e.pointerId, e); lastD = 0; });
  stage.addEventListener('pointermove', e => {
    const p = pts.get(e.pointerId); if (!p) return;
    if (pts.size === 1) { x += e.clientX - p.clientX; y += e.clientY - p.clientY; apply(); }
    pts.set(e.pointerId, e);
    if (pts.size === 2) { const [a, b] = [...pts.values()]; const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); if (lastD) zoomAt((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2, d / lastD); lastD = d; }
  });
  const up = e => { pts.delete(e.pointerId); lastD = 0; };
  stage.addEventListener('pointerup', up); stage.addEventListener('pointercancel', up);
  addEventListener('resize', () => { if (dlg.open) fit(); });
}

// ---------------------------------------------------------------- floor plan
const plan = $('plan');
function buildPlan() {
  const svg = $('plan-svg'), NS = 'http://www.w3.org/2000/svg';
  const x0 = Math.min(...rooms.map(r => Math.min(r.xFar, r.xNear)), -NW / 2) - 1, x1 = Math.max(...rooms.map(r => Math.max(r.xFar, r.xNear)), NW / 2) + 1;
  svg.setAttribute('viewBox', `${x0} ${Z_END - 1} ${x1 - x0} ${NL + 2}`);
  const add = (tag, at, text) => { const n = document.createElementNS(NS, tag); for (const k in at) n.setAttribute(k, at[k]); if (text) n.textContent = text; svg.append(n); return n; };
  add('rect', { x: -NW / 2, y: Z_END, width: NW, height: NL, class: 'nave' });
  rooms.forEach((r, i) => {
    const g = add('g', { class: 'rm', tabindex: 0, role: 'button', 'aria-label': `${r.e.artist}, ${r.e.title}` });
    const rc = document.createElementNS(NS, 'rect');
    for (const [k, v] of Object.entries({ x: Math.min(r.xNear, r.xFar), y: r.z1, width: r.depth, height: r.width, fill: r.e.wall })) rc.setAttribute(k, v);
    const dr = document.createElementNS(NS, 'rect');
    for (const [k, v] of Object.entries({ x: r.side < 0 ? -NW / 2 - T : NW / 2, y: r.zc - DW / 2, width: T, height: DW, class: 'door' })) dr.setAttribute(k, v);
    const tx = document.createElementNS(NS, 'text');
    tx.setAttribute('x', (r.xNear + r.xFar) / 2); tx.setAttribute('y', r.zc + 0.4); tx.setAttribute('fill', r.dark ? '#EDEAE2' : '#1B1A18'); tx.textContent = i + 1;
    g.append(rc, dr, tx); svg.append(g);
    const goto = () => { plan.close(); toStop(stops.findIndex(s => s.kind === 'intro' && s.room === r)); };
    g.addEventListener('click', goto); g.addEventListener('keydown', e => { if (e.key === 'Enter') goto(); });
    const li = el('li'); li.append(btn(`${r.e.artist}`, goto), el('span', null, r.e.title)); $('plan-list').append(li);
  });
  add('rect', { x: -PW / 2, y: PZ - PT / 2, width: PW, height: PT, class: 'title' });
  add('circle', { id: 'plan-me', r: 0.7, cx: 0, cy: 0 });
}
$('plan-share').replaceChildren(...sharer(shareOf.gallery));
$('b-plan').addEventListener('click', () => { $('plan-me').setAttribute('cx', me.x); $('plan-me').setAttribute('cy', me.z); plan.showModal(); });
plan.querySelector('[data-close]').addEventListener('click', () => plan.close());

// ---------------------------------------------------------------- the code behind a work
const codeD = $('code'), coPre = $('co-pre'), coCache = new Map();
const PY_KW = 'def|class|import|from|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|lambda|with|as|try|except|finally|raise|pass|break|continue|yield|global|nonlocal|assert|del';
const PY_RE = new RegExp(`("{3}[\\s\\S]*?"{3}|'{3}[\\s\\S]*?'{3}|#[^\\n]*|"(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*')|\\b(${PY_KW})\\b`, 'g');
const escHtml = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function colour(src) {   // light syntax colouring: comments, strings, keywords. No library.
  let out = '', i = 0, m;
  PY_RE.lastIndex = 0;
  while ((m = PY_RE.exec(src))) {
    out += escHtml(src.slice(i, m.index));
    const cls = m[2] ? 'k' : m[1][0] === '#' ? 'c' : 's';
    out += `<span class="${cls}">${escHtml(m[0])}</span>`; i = m.index + m[0].length;
  }
  return out + escHtml(src.slice(i));
}
async function openCode(rec) {
  const c = rec.w.code, seq = (openCode.seq = (openCode.seq || 0) + 1);
  $('co-h').textContent = rec.w.title;
  $('co-meta').textContent = `${rec.e.artist}. ${c.lines.toLocaleString('en-US')} lines of Python, ${c.kb} KB. Plain text: select and copy freely.`;
  const dl = $('co-dl'); dl.href = c.src; dl.setAttribute('download', c.src.split('/').pop());
  coPre.replaceChildren(el('span', 'note', 'Loading the program'));
  $('co-scroll').scrollTo(0, 0);
  if (!codeD.open) codeD.showModal();
  $('co-scroll').focus({ preventScroll: true });
  try {
    let src = coCache.get(c.src);
    if (src == null) { const r = await fetch(c.src); if (!r.ok) throw new Error(r.status); src = await r.text(); coCache.set(c.src, src); }
    if (seq !== openCode.seq || !codeD.open) return;
    const n = src.replace(/\n$/, '').split('\n').length;
    const gut = el('span', 'gut'); gut.setAttribute('aria-hidden', 'true'); gut.textContent = Array.from({ length: n }, (_, i) => i + 1).join('\n');
    const code = el('code'); code.innerHTML = colour(src);
    coPre.replaceChildren(gut, code);
  } catch { coPre.replaceChildren(el('span', 'note', 'The program could not be loaded. Use Download .py, or open the code page.')); }
}
codeD.querySelector('[data-close]').addEventListener('click', () => codeD.close());

// ---------------------------------------------------------------- the artists
const artD = $('artists'), arList = $('ar-list'), arDetail = $('ar-detail');
function buildArtists() {
  const roomOf = e => rooms.find(r => r.e === e);
  const pick = (e, btns, b) => {
    btns.forEach(x => x.setAttribute('aria-current', x === b ? 'true' : 'false'));
    const r = roomOf(e), thumbs = el('div', 'ar-thumbs');
    for (const w of e.works) {
      const rec = works.find(q => q.w === w), t = el('button', 'ar-th'); t.type = 'button';
      t.setAttribute('aria-label', `${w.title}. Go to this work`);
      const im = new Image(); im.src = w.t; im.alt = ''; im.loading = 'lazy'; im.decoding = 'async'; t.append(im, el('span', null, w.title), heartBadge(w.id));
      t.addEventListener('click', () => { artD.close(); if (rec) toStop(stops.findIndex(s => s.rec === rec)); });
      thumbs.append(t);
    }
    const go = btn('Go to this exhibition', () => { artD.close(); if (r) toStop(stops.findIndex(s => s.kind === 'intro' && s.room === r)); }, 'primary');
    arDetail.replaceChildren(el('h3', null, e.artist), el('p', 'ar-origin', e.origin), el('p', 'ar-show', e.title),
      ...e.bio.map(t => el('p', null, t)), el('h4', null, e.works.length === 1 ? 'The work' : 'The works'), thumbs, go);
    arDetail.scrollTop = 0; arDetail.closest('.ar-body').scrollTop = 0;
  };
  const btns = D.exhibitions.map(e => {
    const b = el('button', 'ar-b'); b.type = 'button'; b.append(el('b', null, e.artist), el('span', null, e.title));
    b.addEventListener('click', () => pick(e, btns, b));
    const li = el('li'); li.append(b); arList.append(li); return b;
  });
  arList.addEventListener('keydown', e => {
    const i = btns.indexOf(document.activeElement); if (i < 0) return;
    const j = { ArrowDown: i + 1, ArrowRight: i + 1, ArrowUp: i - 1, ArrowLeft: i - 1, Home: 0, End: btns.length - 1 }[e.key];
    if (j == null || j < 0 || j >= btns.length) return;
    e.preventDefault(); btns[j].focus(); btns[j].click();
  });
  $('b-artists').addEventListener('click', () => {
    artD.showModal();
    const here = rooms[spaceOf(me.x, me.z)], i = Math.max(0, D.exhibitions.findIndex(e => here && e === here.e));
    btns[i].click(); btns[i].focus();
  });
  artD.querySelector('[data-close]').addEventListener('click', () => artD.close());
}

// ---------------------------------------------------------------- the guest book
const gbD = $('guestbook');
const titleOf = Object.fromEntries(D.exhibitions.flatMap(e => e.works.map(w => [w.id, w.title])));
let gb = null;
function openGuest(slug) {
  if (!gb) {
    gb = mountGuestbook($('gb-body'), { titles: titleOf, workLink: (id, a) => {
      a.addEventListener('click', ev => {
        ev.preventDefault(); gbD.close();
        const i = stops.findIndex(s => s.rec && s.rec.w.id === id); if (i >= 0) toStop(i);
      });
    } });
    gbD.querySelector('[data-close]').addEventListener('click', () => gbD.close());
    $('gb-write').addEventListener('click', () => gb.focusForm());
    showStats($('gb-stats'));
  }
  gb.open(slug || null);
  gbD.showModal();
  $('gb-body').scrollTop = 0;
  if (slug) gb.focusForm();
}
$('b-guest').addEventListener('click', () => openGuest(null));

// ---------------------------------------------------------------- zoom
// Optical zoom: the field of view narrows and yaw/pitch shift so the point under the pointer stays under the pointer.
const tanH = f => Math.tan(THREE.MathUtils.degToRad(f / 2));
function setZoom(z, cx, cy, now) {
  zoomT = clamp(z, 1, zoomMax);
  anchor = cx == null ? null : { x: cx, y: cy };
  if (now) applyZoom(zoomT);
  hideHint();
}
function applyZoom(nz) {   // move `zoom` to nz, holding the anchored direction still
  const f0 = fovFor(zoom), f1 = fovFor(nz);
  if (anchor && f0 !== f1) {   // solve for the yaw and pitch that keep the world direction under the pointer where it was
    const W = innerWidth, H = innerHeight, nx = (anchor.x - (W / 2 - shiftX)) / (W / 2), ny = -(anchor.y - (H / 2 - shiftY)) / (H / 2);
    const cam = f => { const v = [nx * tanH(f) * camera.aspect, ny * tanH(f), -1], l = Math.hypot(...v); return v.map(x => x / l); };
    const [c0, c1] = [cam(f0), cam(f1)], cp = Math.cos(me.pitch), sp = Math.sin(me.pitch), cy = Math.cos(me.yaw), sy = Math.sin(me.yaw);
    const ry = c0[1] * cp - c0[2] * sp, rz = c0[1] * sp + c0[2] * cp;          // c0 turned by the current pitch, then yaw
    const d = [c0[0] * cy + rz * sy, ry, -c0[0] * sy + rz * cy];
    const A = d[0], B = -d[2], m = Math.hypot(A, B), base = Math.atan2(B, A), off = Math.acos(clamp(c1[0] / m, -1, 1));
    const cands = [base + off, base - off], yaw = cands.reduce((u, v) => (Math.abs(angDiff(me.yaw, v)) < Math.abs(angDiff(me.yaw, u)) ? v : u));
    const vy = d[1], vz = d[0] * Math.sin(yaw) + d[2] * Math.cos(yaw);
    me.yaw += angDiff(me.yaw, yaw); me.pitch = clamp(Math.atan2(vy, -vz) - Math.atan2(c1[1], -c1[2]), -0.9, 0.9);
  }
  zoom = nz; camera.fov = f1;
}
function stepZoom(dt) {
  if (zoom === zoomT) return;
  const l = Math.log(zoomT / zoom);
  applyZoom(Math.abs(l) < 0.004 ? zoomT : zoom * Math.exp(l * Math.min(1, dt * 9)));
  if (zoom === 1) anchor = null;
}
const zoomStep = (x, y) => setZoom(zoomT >= zoomMax * 0.55 ? 1 : zoomT * 3, x, y);

// ---------------------------------------------------------------- input
const ray = new THREE.Raycaster();
const pick = (cx, cy) => {
  ray.setFromCamera(new THREE.Vector2(cx / innerWidth * 2 - 1, -(cy / innerHeight) * 2 + 1), camera);
  return ray.intersectObjects(solids, false)[0];
};
function clickAt(cx, cy) {   // the single click: walk to the picture or the spot on the floor
  const h = pick(cx, cy); if (!h) return;
  if (h.object.userData.book) { openGuest(null); return; }
  if (h.object.userData.start) { if (cur < 0) nextStop(1); else toStop(0); return; }
  if (recOf.has(h.object)) { toStop(stops.findIndex(s => s.rec === recOf.get(h.object))); return; }
  if (Math.abs(h.point.y) < 0.05) {
    let { x, z } = h.point;
    if (!canStand(x, z)) {   // nudge a click near a wall back onto the floor
      let best = null;
      for (const r of walkable) { const px = clamp(x, r[0], r[1]), pz = clamp(z, r[2], r[3]), dd = Math.hypot(px - x, pz - z); if (!best || dd < best.d) best = { x: px, z: pz, d: dd }; }
      if (!best || best.d > 1.5 || !canStand(best.x, best.z)) return; x = best.x; z = best.z;
    }
    if (focus) closePanel(false);
    go({ x, z }, null, 0, null);
  }
}
const DBL = 260;   // a second click inside this window is a double-click, not two walks
let drag = null, pend = null;
const ptrs = new Map(); let pinchD = 0, pinchCounted = false;
canvas.addEventListener('pointerdown', e => {
  if (!entered) return;
  canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, e);
  if (ptrs.size === 2) { drag = null; const [a, b] = [...ptrs.values()]; pinchD = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); if (pend) { clearTimeout(pend.timer); pend = null; } return; }
  if (ptrs.size === 1) drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, t: performance.now(), moved: false };
});
canvas.addEventListener('pointermove', e => {
  if (!entered) return;
  if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, e);
  if (ptrs.size === 2) {   // two-finger pinch
    const [a, b] = [...ptrs.values()], d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (pinchD && d && !pinchCounted) { pinchCounted = true; usedControl('pinch'); }
    if (pinchD && d) { walk = walk && walk.settle ? null : walk; setZoom(zoomT * d / pinchD, (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2, true); }
    pinchD = d; return;
  }
  if (!drag) { if (e.pointerType === 'mouse') { const h = pick(e.clientX, e.clientY); canvas.style.cursor = h && (recOf.has(h.object) || h.object.userData.start || h.object.userData.book) ? 'pointer' : 'grab'; } return; }
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 6) drag.moved = true;
  if (!drag.moved) return;
  if (!drag.counted) { drag.counted = true; usedControl('drag'); }
  const k = 0.0034 * tanH(camera.fov) / tanH(62);   // finer drag the further in you are
  me.yaw += dx * k; me.pitch = clamp(me.pitch + dy * k, -0.9, 0.9);
  if (walk) { walk.free = true; if (walk.settle) walk = null; }
  lastInput = performance.now(); hideHint();
});
const release = e => {
  ptrs.delete(e.pointerId); if (ptrs.size < 2) { pinchD = 0; pinchCounted = false; }
  if (!drag) return; const d = drag; drag = null;
  if (d.moved || performance.now() - d.t > 500) return;
  hideHint();
  const now = performance.now(), x = e.clientX, y = e.clientY;
  if (pend && Math.hypot(pend.x - x, pend.y - y) < 28) { clearTimeout(pend.timer); pend = null; zoomStep(x, y); return; }   // double-click
  if (pend) { clearTimeout(pend.timer); pend = null; }
  pend = { x, y, timer: setTimeout(() => { pend = null; clickAt(x, y); }, DBL) };
};
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', e => { ptrs.delete(e.pointerId); pinchD = 0; drag = null; });
canvas.addEventListener('wheel', e => {   // wheel, and trackpad pinch (which arrives as wheel with ctrlKey)
  if (!entered) return;
  e.preventDefault();
  const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1);
  setZoom(zoomT * Math.exp(-dy * (e.ctrlKey ? 0.012 : 0.0016)), e.clientX, e.clientY);
}, { passive: false });

// keys: listened for on the window, whatever has focus, so a focused button or the open panel never swallows them
const KEYS = { arrowup: 'f', w: 'f', arrowdown: 'b', s: 'b', arrowleft: 'l', q: 'l', arrowright: 'r', e: 'r', a: 'sl', d: 'sr' };
const typing = t => t && (t.isContentEditable || /^(input|textarea|select)$/i.test(t.tagName));
addEventListener('keydown', e => {
  if (!entered || plan.open || dlg.open || codeD.open || artD.open || gbD.open || e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key.toLowerCase();
  if (KEYS[k]) { keys.add(KEYS[k]); e.preventDefault(); hideHint(); }
  else if (k === 'n' || k === 'pagedown') { e.preventDefault(); nextStop(1); }
  else if (k === 'p' || k === 'pageup') { e.preventDefault(); nextStop(-1); }
  else if (k === 'escape') { if (zoomT > 1) setZoom(1, null); else if (!panel.hidden) closePanel(false); }
  else if (k === '+' || k === '=') { e.preventDefault(); setZoom(zoomT * 1.5, innerWidth / 2 - shiftX, innerHeight / 2 - shiftY); }
  else if (k === '-' || k === '_') { e.preventDefault(); setZoom(zoomT / 1.5, innerWidth / 2 - shiftX, innerHeight / 2 - shiftY); }
}, { capture: true });
addEventListener('keyup', e => {
  const k = KEYS[e.key.toLowerCase()]; if (k) keys.delete(k);
}, { capture: true });
addEventListener('blur', () => keys.clear());
document.addEventListener('visibilitychange', () => keys.clear());
$('t-next').addEventListener('click', () => nextStop(1));
$('t-prev').addEventListener('click', () => nextStop(-1));

const hint = $('hint');
function hideHint() { hint.classList.remove('on'); }

// The standing controls line fades to very faint once the visitor has used the controls a few times (remembered between visits).
const keysEl = $('keys'), coarse = matchMedia('(pointer: coarse)').matches;
const used = { key: 0, wheel: 0, drag: 0, pinch: 0 };
let wheelAt = 0;
if (store.get('knowsControls', false)) keysEl.classList.add('quiet');
function usedControl(k) {
  used[k]++;
  const knows = coarse ? used.drag >= 3 && used.pinch >= 2 : used.key >= 6 && used.wheel >= 3;
  if (knows && !keysEl.classList.contains('quiet')) { keysEl.classList.add('quiet'); store.set('knowsControls', true); }
}
addEventListener('keydown', e => { if (entered && !e.repeat && !typing(e.target) && KEYS[e.key.toLowerCase()] && !e.metaKey && !e.ctrlKey && !e.altKey) usedControl('key'); }, { capture: true });
canvas.addEventListener('wheel', e => { const n = performance.now(); if (n - wheelAt > 500) usedControl('wheel'); wheelAt = n; }, { passive: true });

// ---------------------------------------------------------------- textures, nearest first
// tiers: 0 thumbnail 512, 1 1280, 2 2048, 3 4000 (only for what you are looking at through the zoom)
const TIER = ['t', 'm', 'w', 'z'], TIER_PX = [512, 1280, 2048, 4000];
const hi = [];   // works currently holding a large texture
async function setLevel(rec, lv) {
  if (rec.busy || rec.level === lv) return;
  rec.busy = true;
  const t = await loadTex(rec.w[TIER[lv]]);
  rec.busy = false;
  if (!t) return;
  const old = rec.art.material.map;
  rec.art.material.map = t; rec.art.material.color.set('#fff'); rec.art.material.needsUpdate = true;
  if (old) old.dispose();
  rec.level = lv;
  const i = hi.indexOf(rec); if (i >= 0) hi.splice(i, 1);
  if (lv >= 2) {
    hi.push(rec);
    const big = hi.filter(r => r.level === 3);   // 4000 px textures are heavy: keep one
    if (big.length > 1) setLevel(big[0] === rec ? big[1] : big[0], 2);
    if (hi.length > 5) { const o = hi.find(r => r !== rec && r.level < 3) || hi[0]; if (o !== rec) setLevel(o, 1); }
  }
}
const fwd = new THREE.Vector3(), toRec = new THREE.Vector3();
function streamTextures() {
  let n = works.filter(w => w.busy).length;
  const sp = spaceOf(me.x, me.z), dpr = renderer.getPixelRatio();
  camera.getWorldDirection(fwd);
  const halfDiag = Math.atan(tanH(camera.fov) * Math.hypot(1, camera.aspect)), pxPerUnit = innerHeight * dpr / (2 * tanH(camera.fov));
  const todo = [];
  for (const rec of works) {
    toRec.set(rec.pos.x - me.x, rec.pos.y - EYE, rec.pos.z - me.z);
    const d = toRec.length(), here = rec.room ? rooms.indexOf(rec.room) === sp : sp < 0;
    const seen = d > 0 && fwd.angleTo(toRec) < halfDiag + 0.15 && rec.n.dot(toRec) < 0;
    let want = rec === focus || (here && d < 3.2) ? 2 : (here || d < 13 || rec.want) ? 1 : 0;
    if (seen) {   // enough pixels for how big the work appears on screen
      const px = Math.max(rec.s.ww, rec.s.wh) / d * pxPerUnit * 1.2;
      const need = TIER_PX.findIndex(t => t >= px);
      want = Math.max(want, need < 0 ? 3 : need === 3 ? 3 : need);
      if (zoom >= 4 && fwd.angleTo(toRec) < THREE.MathUtils.degToRad(camera.fov) * 0.7) want = 3;   // zoomed far in on it: the sharpest tier
      if (!here && d > 13) want = Math.min(want, 1);
    }
    if (rec.level === 3 && want < 3) todo.push({ rec, lv: 2, pr: -1 });   // let go of the big one when you pull back
    else if (want > rec.level) todo.push({ rec, lv: want, pr: (seen ? 0 : 1000) + d });
  }
  todo.sort((a, b) => a.pr - b.pr);
  for (const t of todo) { if (n >= 3) break; if (!t.rec.busy) { setLevel(t.rec, t.lv); n++; } }
}

// ---------------------------------------------------------------- frame loop
let last = performance.now(), tick = 0, space = -2, gateT = 0;
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  baseFov = w >= h ? 62 : clamp(THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(27)) / camera.aspect)), 62, 84);
  zoomMax = tanH(baseFov) / tanH(ZMAX_DEG);
  zoomT = Math.min(zoomT, zoomMax); zoom = Math.min(zoom, zoomMax);
  camera.fov = fovFor(zoom); camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

function step(dt) {
  if (!entered) {   // behind the title card: a slow look down the long gallery
    gateT += dt;
    me.x = Math.sin(gateT * 0.07) * 1.2; me.z = 0.5 - ((gateT * 0.35) % 14); me.yaw = Math.sin(gateT * 0.11) * 0.12; me.pitch = 0.03;
    return;
  }
  const mv = (keys.has('f') ? 1 : 0) - (keys.has('b') ? 1 : 0);
  const st = (keys.has('sr') ? 1 : 0) - (keys.has('sl') ? 1 : 0);
  const tn = (keys.has('l') ? 1 : 0) - (keys.has('r') ? 1 : 0);
  const ease = Math.min(1, dt * 9);
  vel.f += (mv * 2.6 - vel.f) * ease; vel.s += (st * 2.6 - vel.s) * ease;
  vel.t += (tn * 1.7 * clamp(tanH(camera.fov) / tanH(62), 0.15, 1.4) - vel.t) * ease;
  if (mv || st || tn) {   // a key always takes over from a walk in progress
    if (walk) walk = null;
    if (mv || st) { zoomT = 1; anchor = null; }
  }
  if (Math.abs(vel.f) + Math.abs(vel.s) + Math.abs(vel.t) > 0.01) {
    me.yaw += vel.t * dt;
    const fx = -Math.sin(me.yaw), fz = -Math.cos(me.yaw);
    const dx = (fx * vel.f - fz * vel.s) * dt, dz = (fz * vel.f + fx * vel.s) * dt;
    if (canStand(me.x + dx, me.z + dz)) { me.x += dx; me.z += dz; }
    else if (canStand(me.x + dx, me.z)) me.x += dx;
    else if (canStand(me.x, me.z + dz)) me.z += dz;
    if (!tn && (mv || st)) me.pitch += (0 - me.pitch) * Math.min(1, dt * 1.5);
  }
  if (walk && !walk.settle) {
    const q = walk.pts[walk.i], dx = q.x - me.x, dz = q.z - me.z, d = Math.hypot(dx, dz);
    let rem = d; for (let i = walk.i; i < walk.pts.length - 1; i++) rem += Math.hypot(walk.pts[i + 1].x - walk.pts[i].x, walk.pts[i + 1].z - walk.pts[i].z);
    walk.t += dt;
    const v = walk.speed * clamp(rem / 2.2, 0.16, 1) * clamp(walk.t / 0.5 + 0.15, 0, 1), stepLen = v * dt;
    if (d <= stepLen) { me.x = q.x; me.z = q.z; walk.i++; if (walk.i >= walk.pts.length) { walk.settle = true; walk.t = 0; if (walk.onArrive) walk.onArrive(); } }
    else { me.x += dx / d * stepLen; me.z += dz / d * stepLen; }
    if (!walk.free) {
      let target = d > 0.05 ? yawTo(dx, dz) : me.yaw;
      if (walk.endYaw != null && rem < 2.6) target = me.yaw + angDiff(me.yaw, target) * (rem / 2.6) + angDiff(me.yaw, walk.endYaw) * (1 - rem / 2.6);
      me.yaw += angDiff(me.yaw, target) * Math.min(1, dt * 3.4);
      me.pitch += ((rem < 2.6 ? walk.endPitch : 0) - me.pitch) * Math.min(1, dt * 2.5);
    }
  } else if (walk && walk.settle) {
    walk.t += dt;
    if (walk.free || walk.endYaw == null || walk.t > 2.5) walk = null;
    else { me.yaw += angDiff(me.yaw, walk.endYaw) * Math.min(1, dt * 4); me.pitch += (walk.endPitch - me.pitch) * Math.min(1, dt * 4); }
  }
  if (!walk && !panel.hidden) {   // wander off and the panel puts itself away; a panel for a room stays a good while longer
    const dist = Math.hypot((focus ? viewPose(focus).x : panelAt.x) - me.x, (focus ? viewPose(focus).z : panelAt.z) - me.z);
    if (dist > (focus ? 1.8 : 5)) closePanel(false);
  }
  // which room are we in
  const sp = spaceOf(me.x, me.z);
  if (sp !== space) {
    space = sp;
    const w = $('where');
    if (sp >= 0) { const r = rooms[sp]; w.replaceChildren(el('b', null, r.e.artist), el('span', null, r.e.title)); }
    else w.replaceChildren(el('b', null, 'The long gallery'), el('span', null, `${D.season}, ${D.dates}`));
    if (sp >= 0 && !rooms[sp].seen && !(walk && !walk.settle && walk.onArrive)) showIntro(rooms[sp]);
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (document.hidden) return;
  if (dlg.open || plan.open || codeD.open || artD.open || gbD.open) { keys.clear(); vel.f = vel.s = vel.t = 0; return; }
  step(dt); stepZoom(dt);
  shiftX += (shiftTX - shiftX) * Math.min(1, dt * 5); shiftY += (shiftTY - shiftY) * Math.min(1, dt * 5);
  if (Math.abs(shiftX) > 0.5 || Math.abs(shiftY) > 0.5) camera.setViewOffset(innerWidth, innerHeight, shiftX, shiftY, innerWidth, innerHeight);
  else { if (camera.view) camera.clearViewOffset(); }
  camera.updateProjectionMatrix();
  camera.position.set(me.x, EYE + (walk && !walk.settle ? Math.sin(now * 0.011) * 0.012 : 0), me.z);
  camera.rotation.set(me.pitch, me.yaw, 0, 'YXZ');
  if ((tick++ & 7) === 0) streamTextures();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- opening
async function init() {
  resize(); loadHearts();
  try { await Promise.all([document.fonts.load('40px Marcellus'), document.fonts.load('300 20px "Albert Sans"'), document.fonts.load('italic 20px "Albert Sans"'), document.fonts.load('600 20px "Albert Sans"')]); } catch {}
  rooms.forEach(buildRoom); buildNave(); buildCollision(); buildStops(); buildPlan(); buildArtists();
  const mirror = world.clone(true); mirror.scale.y = -1; scene.add(mirror);
  requestAnimationFrame(frame);
  await Promise.all(works.map(rec => setLevel(rec, 0)));
  const b = $('enter'); b.disabled = false; b.textContent = 'Enter the gallery';
  b.addEventListener('click', () => {
    startMusic();
    document.body.classList.add('fade');
    setTimeout(() => {
      entered = true; me.x = 0; me.z = 8.6; me.yaw = 0; me.pitch = 0; space = -2;
      $('gate').hidden = true; $('hud').hidden = false; $('where').hidden = false; $('tour').hidden = false;
      document.body.classList.remove('fade'); canvas.style.cursor = 'grab';
      hint.textContent = matchMedia('(pointer: coarse)').matches
        ? 'Drag to look around. Tap the floor or a picture to walk to it. Pinch or double-tap to zoom.'
        : 'Drag to look around. Click the floor or a picture to walk there. Scroll or double-click to zoom. Arrow keys also move you.';
      hint.hidden = false; requestAnimationFrame(() => hint.classList.add('on')); setTimeout(hideHint, 16000);
      keysEl.hidden = false;
      const q = new URLSearchParams(location.search), wid = q.get('w'), xid = q.get('x');
      const at = wid ? stops.findIndex(s => s.kind === 'work' && s.rec.w.id === wid) : xid ? stops.findIndex(s => s.kind === 'intro' && s.room.e.slug === xid) : -1;
      if (at >= 0) toStop(at); else showWelcome();
    }, 650);
  }, { once: true });
}
window.antara = { me, toStop, go, camera, setZoom, state: () => ({ zoom, zoomT, voiceOn, musicOn, musicLive, vol: music ? music.volume : null, walking: !!walk }), works };
init();
