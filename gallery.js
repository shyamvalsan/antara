// Antara: the building, the hang, walking, and the audio guide.
// Everything is laid out from the programme data embedded in the page.
import * as THREE from 'three';

const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

// ---------------------------------------------------------------- dimensions (metres)
const NW = 8, NH = 7.6, T = 0.5, RH = 5.4, DW = 2.8, DH = 3.6, EYE = 1.62, RAD = 0.35;
const Z_START = 11, PZ = 3.2, PW = 4.4, PH = 3.7, PT = 0.36, GAP = 1.5;
const NAVE_WALL = '#E6E3DB', NAVE_FLOOR = '#8D8A83', INK = '#1B1A18';
const UP = new THREE.Vector3(0, 1, 0);

// ---------------------------------------------------------------- renderer
const canvas = $('c');
let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' }); }
catch (err) {
  const b = $('enter'); b.textContent = 'This browser cannot show the rooms in 3D';
  document.querySelector('#gate .fine').textContent = 'Every work, text and recording is on the plain page below.';
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

// ---------------------------------------------------------------- walls
// A wall is a vertical plane seen from the side its normal points to.
// `a` runs along it left to right as seen from that side, `y` is height.
function makeWall({ c, n, len, h, color, dark = false, openings = [], pools = [], texts = [], ppm = 44, into = world, solid = true }) {
  const nrm = new THREE.Vector3(n[0], 0, n[1]);
  const r = new THREE.Vector3().crossVectors(UP, nrm);
  const o = new THREE.Vector3(c[0], 0, c[1]).addScaledVector(r, -len / 2);
  const at = (a, y, off = 0) => o.clone().addScaledVector(r, a).addScaledVector(nrm, off).setY(y);
  const aOf = (x, z) => (x - o.x) * r.x + (z - o.z) * r.z;
  ppm = Math.min(ppm, 4096 / len);
  const tex = canvasTex(len * ppm, h * ppm, (g, W, H) => {
    g.fillStyle = color; g.fillRect(0, 0, W, H);
    speckle(g, W, H, 60, dark ? 0.03 : 0.022);
    let gr = g.createLinearGradient(0, 0, 0, H);
    if (dark) { gr.addColorStop(0, 'rgba(0,0,0,.62)'); gr.addColorStop(0.5, 'rgba(0,0,0,.12)'); gr.addColorStop(1, 'rgba(0,0,0,.38)'); }
    else { gr.addColorStop(0, 'rgba(255,255,255,.16)'); gr.addColorStop(0.45, 'rgba(255,255,255,0)'); gr.addColorStop(1, 'rgba(0,0,0,.13)'); }
    g.fillStyle = gr; g.fillRect(0, 0, W, H);
    for (const [x0, x1] of [[0, 1.4 * ppm], [W, W - 1.4 * ppm]]) {   // corners
      gr = g.createLinearGradient(x0, 0, x1, 0);
      gr.addColorStop(0, `rgba(0,0,0,${dark ? 0.3 : 0.12})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), H);
    }
    for (const p of pools) {   // light on and around each work
      const cx = p.a * ppm, cy = H - p.y * ppm, rx = (p.w * 0.5 + (dark ? 1.0 : 1.3)) * ppm, ry = (p.h * 0.5 + (dark ? 0.9 : 1.2)) * ppm;
      g.save(); g.translate(cx, cy - 0.15 * ry); g.scale(rx, ry);
      gr = g.createRadialGradient(0, 0, 0.2, 0, 0, 1);
      gr.addColorStop(0, dark ? 'rgba(255,228,190,.30)' : 'rgba(255,255,255,.26)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = gr; g.fillRect(-1, -1, 2, 2); g.restore();
      g.save(); g.shadowColor = `rgba(0,0,0,${dark ? 0.7 : 0.4})`; g.shadowBlur = 0.16 * ppm; g.shadowOffsetY = 0.07 * ppm;
      g.fillStyle = color; g.fillRect(cx - p.w * ppm / 2, cy - p.h * ppm / 2, p.w * ppm, p.h * ppm); g.restore();
    }
    g.fillStyle = 'rgba(0,0,0,.55)'; g.fillRect(0, H - 0.03 * ppm, W, 0.03 * ppm);   // shadow gap at the floor
    for (const t of texts) {
      let y = H - t.y * ppm;
      g.textBaseline = 'top'; g.textAlign = t.align || 'left';
      for (const ln of t.lines) {
        const px = ln.size * ppm;
        g.font = `${ln.style || ''} ${px}px ${ln.font || 'Marcellus'}`; g.fillStyle = ln.color || INK;
        if (ln.track) g.letterSpacing = `${ln.track * px}px`;
        g.fillText(ln.text, t.a * ppm, y);
        if (ln.track) g.letterSpacing = '0px';
        y += px * (ln.lead || 1.35);
      }
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
  return { at, aOf, nrm, r, len, h, mesh };
}

function flat(x0, x1, z0, z1, y, tex, up, into, opacity = 1) {   // floor (up) or ceiling
  const g = new THREE.PlaneGeometry(Math.abs(x1 - x0), Math.abs(z1 - z0));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, transparent: opacity < 1, opacity }));
  m.rotation.x = up ? -Math.PI / 2 : Math.PI / 2;
  m.position.set((x0 + x1) / 2, y, (z0 + z1) / 2);
  into.add(m);
  return m;
}
function floorTex(wm, dm, color, dark, spills = []) {
  const ppm = Math.min(28, 2048 / Math.max(wm, dm));
  return canvasTex(wm * ppm, dm * ppm, (g, W, H) => {
    g.fillStyle = color; g.fillRect(0, 0, W, H);
    speckle(g, W, H, 140, dark ? 0.05 : 0.04);
    for (const s of spills) {   // light falling from a work onto the floor
      const gr = g.createRadialGradient(s.u * W, s.v * H, 0, s.u * W, s.v * H, s.r * ppm);
      gr.addColorStop(0, 'rgba(255,225,185,.16)'); gr.addColorStop(1, 'rgba(255,225,185,0)');
      g.fillStyle = gr; g.fillRect(0, 0, W, H);
    }
    const e = 1.1 * ppm, k = dark ? 0.4 : 0.2;   // darker where floor meets wall
    for (const [x0, y0, x1, y1] of [[0, 0, e, 0], [W, 0, W - e, 0], [0, 0, 0, e], [0, H, 0, H - e]]) {
      const gr = g.createLinearGradient(x0, y0, x1, y1);
      gr.addColorStop(0, `rgba(0,0,0,${k})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(0, 0, W, H);
    }
  });
}
function ceilTex(wm, dm, color, dark, light) {   // light = [u0,v0,u1,v1] of the laylight
  const ppm = Math.min(24, 2048 / Math.max(wm, dm));
  return canvasTex(wm * ppm, dm * ppm, (g, W, H) => {
    g.fillStyle = color; g.fillRect(0, 0, W, H);
    if (!light) return;
    const [x0, y0, x1, y1] = [light[0] * W, light[1] * H, light[2] * W, light[3] * H];
    g.save(); g.shadowColor = dark ? 'rgba(255,240,220,.25)' : 'rgba(255,255,255,.9)'; g.shadowBlur = 1.2 * ppm;
    g.fillStyle = dark ? '#3A3632' : '#FDFCF8'; g.fillRect(x0, y0, x1 - x0, y1 - y0); g.restore();
    g.strokeStyle = dark ? 'rgba(0,0,0,.5)' : 'rgba(120,118,110,.35)'; g.lineWidth = Math.max(1, 0.05 * ppm);
    const horiz = x1 - x0 > y1 - y0, n = Math.round((horiz ? x1 - x0 : y1 - y0) / (1.6 * ppm));
    for (let i = 1; i < n; i++) {
      g.beginPath();
      if (horiz) { const x = x0 + (x1 - x0) * i / n; g.moveTo(x, y0); g.lineTo(x, y1); }
      else { const y = y0 + (y1 - y0) * i / n; g.moveTo(x0, y); g.lineTo(x1, y); }
      g.stroke();
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
  if (s.m) put(new THREE.Mesh(new THREE.PlaneGeometry(s.ww + 2 * s.m, s.wh + 2 * s.m), new THREE.MeshBasicMaterial({ color: dark ? '#E4DFD3' : '#F1EEE6' })), 0.0515);
  const art = put(new THREE.Mesh(new THREE.PlaneGeometry(s.ww, s.wh), new THREE.MeshBasicMaterial({ color: '#777' })), 0.054);
  // wall label
  const lw = 0.34, lh = 0.2;
  const label = canvasTex(510, 300, (g, W, H) => {
    g.fillStyle = dark ? '#8E8A82' : '#F6F4EE'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#141412'; g.textBaseline = 'top';
    g.font = '600 30px "Albert Sans"'; g.fillText(e.artist, 30, 28);
    g.font = 'italic 25px "Albert Sans"';
    let line = '', yy = 76;
    for (const word of w.title.split(' ')) {
      if (g.measureText(line + word).width > W - 60 && line) { g.fillText(line.trim(), 30, yy); yy += 32; line = ''; }
      line += word + ' ';
    }
    g.fillText(line.trim() + ', ' + w.year, 30, yy); yy += 46;
    g.font = '21px "Albert Sans"'; g.fillStyle = '#3A3936';
    g.fillText(w.medium, 30, yy, W - 60); g.fillText(`${w.cm[0]} × ${w.cm[1]} cm`, 30, yy + 29);
  });
  put(new THREE.Mesh(new THREE.PlaneGeometry(lw, lh), new THREE.MeshBasicMaterial({ map: label })), 0.006, s.fw / 2 + 0.3 + lw / 2, 1.22);
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
  // floor, ceiling, door reveal
  const fx0 = side * NW / 2, spills = [];
  if (dark) for (const p of [...pa, ...pb, { x: xFar, z: zc }]) {
    const px = p.x === xFar ? xFar - side * 0.5 : p.x, pz = p.z === zc && p.x === xFar ? zc : p.z + (p.z === z0 ? -0.5 : 0.5);
    spills.push({ u: (px - Math.min(fx0, xFar)) / Math.abs(xFar - fx0), v: (pz - z1) / width, r: 2.6 });
  }
  const fcol = dark ? mix(e.wall, '#000000', 0.45) : mix(NAVE_FLOOR, e.wall, 0.18);
  solids.push(flat(Math.min(fx0, xFar), Math.max(fx0, xFar), z1, z0, 0, floorTex(Math.abs(xFar - fx0), width, fcol, dark, spills), true, floors, dark ? 0.87 : 0.86));
  const cc = dark ? mix(e.wall, '#000000', 0.7) : mix(e.wall, '#ffffff', 0.25);
  flat(xNear, xFar, z1, z0, RH, ceilTex(depth, width, cc, dark, dark ? null : [0.16, 0.16, 0.84, 0.84]), false, world);
  const jm = new THREE.MeshBasicMaterial({ color: mix(NAVE_WALL, "#000000", 0.2), side: THREE.DoubleSide });
  for (const zz of [zc - DW / 2, zc + DW / 2]) {
    const j = new THREE.Mesh(new THREE.PlaneGeometry(T, DH), jm); j.position.set(side * (NW / 2 + T / 2), DH / 2, zz); world.add(j);
  }
  const so = new THREE.Mesh(new THREE.PlaneGeometry(T, DW), new THREE.MeshBasicMaterial({ color: mix(NAVE_WALL, '#000000', 0.22), side: THREE.DoubleSide }));
  so.rotation.x = Math.PI / 2; so.position.set(side * (NW / 2 + T / 2), DH, zc); world.add(so);
}

let naveWork = null;
function buildNave() {
  const lettering = (wall, r, alignRight) => ({
    a: wall.aOf(r.side * NW / 2, r.zc + DW / 2 + 0.9), y: 2.55, align: alignRight ? 'left' : 'right',
    lines: [{ text: r.e.artist, size: 0.36, lead: 1.5 }, { text: r.e.title, size: 0.2, font: '"Albert Sans"', style: 'italic 300', lead: 1.7 },
      { text: D.dates, size: 0.105, font: '"Albert Sans"', color: '#55534E' }],
  });
  const open = (wall, side) => rooms.filter(r => r.side === side).map(r => { const a = wall.aOf(side * NW / 2, r.zc); return { a0: a - DW / 2, a1: a + DW / 2, h: DH }; });
  for (const side of [-1, 1]) {
    // left wall runs away from the entrance, so its lettering sits before each door and reads toward it
    const probe = makeProbe([side * NW / 2, (Z_START + Z_END) / 2], [-side, 0], NL);
    makeWall({ c: [side * NW / 2, (Z_START + Z_END) / 2], n: [-side, 0], len: NL, h: NH, color: NAVE_WALL, ppm: 40,
      openings: open(probe, side), texts: rooms.filter(r => r.side === side).map(r => lettering(probe, r, side > 0)) });
  }
  makeWall({ c: [0, Z_START], n: [0, -1], len: NW, h: NH, color: NAVE_WALL });
  const nw = D.exhibitions.flatMap(e => e.works.filter(w => w.nave).map(w => ({ w, e })))[0];
  const s = nw && size(nw.w, nw.e);
  const end = makeWall({ c: [0, Z_END], n: [0, 1], len: NW, h: NH, color: NAVE_WALL,
    pools: nw ? [{ a: NW / 2, y: centreY(s), w: s.fw, h: s.fh }] : [],
    texts: nw ? [] : [{ a: NW / 2, y: 3.3, align: 'center', lines: [
      { text: 'Stand further back than you think.', size: 0.25, lead: 1.55 }, { text: 'Then come much closer than is polite.', size: 0.25, lead: 2.3 },
      { text: D.curator.name, size: 0.12, font: '"Albert Sans"', color: '#55534E' }] }] });
  if (nw) naveWork = hang(end, nw.w, nw.e, NW / 2, null, 30, false);
  solids.push(flat(-NW / 2, NW / 2, Z_END, Z_START, 0, floorTex(NW, NL, NAVE_FLOOR, false), true, floors, 0.86));
  flat(-NW / 2, NW / 2, Z_END, Z_START, NH, ceilTex(NW, NL, '#CFCBC2', false, [0.3, 0.012, 0.7, 0.988]), false, world);
  // the title wall in the foyer
  const lines = [{ text: D.name.toUpperCase(), size: 0.62, track: 0.24, lead: 1.75 }, { text: D.line, size: 0.15, font: '"Albert Sans"', style: '300', color: '#3D3B37' }];
  makeWall({ c: [0, PZ + PT / 2], n: [0, 1], len: PW, h: PH, color: '#F1EFE9', texts: [{ a: PW / 2 + 0.07, y: 2.45, align: 'center', lines }] });
  makeWall({ c: [0, PZ - PT / 2], n: [0, -1], len: PW, h: PH, color: '#F1EFE9', texts: [{ a: 0.45, y: 2.6, lines: [
    { text: D.season, size: 0.3, lead: 1.5 }, { text: D.dates, size: 0.13, font: '"Albert Sans"', color: '#55534E', lead: 2.6 },
    ...D.exhibitions.map(e => ({ text: `${e.artist}, ${e.title}`, size: 0.115, font: '"Albert Sans"', lead: 1.62 })),
    { text: '', size: 0.1 }, { text: `Hung by ${D.curator.name}`, size: 0.105, font: '"Albert Sans"', style: 'italic', color: '#55534E' }] }] });
  const edge = new THREE.MeshBasicMaterial({ color: '#D5D2CA' });
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
  blocked = [[-PW / 2 - RAD, PW / 2 + RAD, PZ - PT / 2 - RAD, PZ + PT / 2 + RAD]];
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
let shiftX = 0, shiftY = 0, shiftTX = 0, shiftTY = 0;   // view offset so the panel does not cover the work

const angDiff = (a, b) => { let d = (b - a) % (2 * Math.PI); if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return d; };
const yawTo = (dx, dz) => Math.atan2(-dx, -dz);
const wide = () => innerWidth >= 820;

function go(to, endYaw, endPitch, onArrive) {
  const pts = route({ x: me.x, z: me.z }, to);
  let len = 0, p = me; for (const q of pts) { len += Math.hypot(q.x - p.x, q.z - p.z); p = q; }
  walk = { pts, i: 0, endYaw, endPitch: endPitch ?? 0, onArrive, speed: clamp(len / 3.6, 2.4, 8), free: false, settle: false, t: 0 };
}

function viewPose(rec) {
  const vt = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)), ht = vt * camera.aspect;
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

// ---------------------------------------------------------------- audio guide
const voice = $('voice');
let guide = true, playing = null;
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
$('b-guide').addEventListener('click', () => {
  guide = !guide; $('b-guide').textContent = guide ? 'Audio guide on' : 'Audio guide off'; $('b-guide').setAttribute('aria-pressed', guide);
  if (!guide) voice.pause();
});

// ---------------------------------------------------------------- the panel
const panel = $('panel'), body = $('p-body'), acts = $('p-acts');
function openPanel(audio, auto) {
  panel.hidden = false; panel.scrollTop = 0;
  requestAnimationFrame(() => panel.classList.add('on'));
  setTrack(audio);
  if (audio && auto && guide) play();
  const r = panel.getBoundingClientRect();
  if (wide()) { shiftTX = r.width / 2 + 10; shiftTY = 0; } else { shiftTX = 0; shiftTY = Math.min(r.height, innerHeight * 0.5) / 2; }
}
function closePanel(keepAudio) {
  panel.classList.remove('on'); panel.hidden = true; focus = null; shiftTX = shiftTY = 0;
  if (!keepAudio) voice.pause();
}
$('p-close').addEventListener('click', () => closePanel(false));
const btn = (text, fn, cls) => { const b = el('button', cls, text); b.type = 'button'; b.addEventListener('click', fn); return b; };

function showWork(rec) {
  focus = rec; cur = stops.findIndex(s => s.rec === rec);
  body.replaceChildren(
    el('p', 'kicker', rec.e.artist), el('h2', null, rec.w.title),
    el('p', 'meta', `${rec.w.year}. ${rec.w.medium}. ${rec.w.cm[0]} × ${rec.w.cm[1]} cm`),
    ...rec.w.story.map(t => el('p', null, t)));
  acts.replaceChildren(btn('Look closer', () => openCloser(rec)));
  openPanel(rec.w.audio, true);
}
function showIntro(R) {
  const e = R.e; R.seen = true; cur = stops.findIndex(s => s.kind === 'intro' && s.room === R);
  const more = el('div', 'more'); more.hidden = true;
  more.append(el('h3', null, 'The artist'), el('p', 'meta', e.origin), ...e.bio.map(t => el('p', null, t)), ...e.said.map(t => el('blockquote', null, t)));
  body.replaceChildren(el('p', 'kicker', 'Exhibition'), el('h2', null, e.artist), el('p', 'sub', e.title), ...e.intro.map(t => el('p', null, t)), more);
  const b = btn('About the artist', () => { more.hidden = !more.hidden; b.textContent = more.hidden ? 'About the artist' : 'Hide'; });
  acts.replaceChildren(b, btn('Go to the first work', () => nextStop(1)));
  openPanel(e.audio, true);
}
function showWelcome() {
  body.replaceChildren(el('p', 'kicker', `${D.curator.name}, ${D.curator.role.toLowerCase()}`), el('h2', null, `Welcome to ${D.name}`), ...D.welcome.text.map(t => el('p', null, t)));
  acts.replaceChildren(btn('Begin the guided walk', () => toStop(0)));
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
$('b-plan').addEventListener('click', () => { $('plan-me').setAttribute('cx', me.x); $('plan-me').setAttribute('cy', me.z); plan.showModal(); });
plan.querySelector('[data-close]').addEventListener('click', () => plan.close());

// ---------------------------------------------------------------- input
const ray = new THREE.Raycaster();
const pick = (cx, cy) => {
  ray.setFromCamera(new THREE.Vector2(cx / innerWidth * 2 - 1, -(cy / innerHeight) * 2 + 1), camera);
  return ray.intersectObjects(solids, false)[0];
};
let drag = null;
canvas.addEventListener('pointerdown', e => { if (!entered) return; canvas.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, t: performance.now(), moved: false }; });
canvas.addEventListener('pointermove', e => {
  if (!entered) return;
  if (!drag) { if (e.pointerType === 'mouse') { const h = pick(e.clientX, e.clientY); canvas.style.cursor = h && recOf.has(h.object) ? 'pointer' : 'grab'; } return; }
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 6) drag.moved = true;
  if (!drag.moved) return;
  const k = 0.0034 * (camera.fov / 62);
  me.yaw += dx * k; me.pitch = clamp(me.pitch + dy * k, -0.75, 0.75);
  if (walk) { walk.free = true; if (walk.settle) walk = null; }
  lastInput = performance.now(); hideHint();
});
const release = e => {
  if (!drag) return; const d = drag; drag = null;
  if (d.moved || performance.now() - d.t > 500) return;
  const h = pick(e.clientX, e.clientY); if (!h) return;
  hideHint();
  if (recOf.has(h.object)) { toStop(stops.findIndex(s => s.rec === recOf.get(h.object))); return; }
  if (Math.abs(h.point.y) < 0.05) {
    let { x, z } = h.point;
    if (!canStand(x, z)) {   // nudge a click near a wall back onto the floor
      let best = null;
      for (const r of walkable) { const px = clamp(x, r[0], r[1]), pz = clamp(z, r[2], r[3]), dd = Math.hypot(px - x, pz - z); if (!best || dd < best.d) best = { x: px, z: pz, d: dd }; }
      if (!best || best.d > 1.5 || !canStand(best.x, best.z)) return; x = best.x; z = best.z;
    }
    if (focus) closePanel(true);
    go({ x, z }, null, 0, null);
  }
};
canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', () => { drag = null; });
addEventListener('keydown', e => {
  if (!entered || plan.open || dlg.open || e.metaKey || e.ctrlKey) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (['w', 'a', 's', 'd', 'q', 'e', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) { keys.add(k); e.preventDefault(); hideHint(); if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); }
  else if (k === 'n' || k === 'PageDown') { e.preventDefault(); nextStop(1); }
  else if (k === 'p' || k === 'PageUp') { e.preventDefault(); nextStop(-1); }
  else if (k === 'Escape' && !panel.hidden) closePanel(false);
});
addEventListener('keyup', e => keys.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key));
addEventListener('blur', () => keys.clear());
$('t-next').addEventListener('click', () => nextStop(1));
$('t-prev').addEventListener('click', () => nextStop(-1));

const hint = $('hint');
function hideHint() { hint.classList.remove('on'); }

// ---------------------------------------------------------------- textures, nearest first
const hi = [];   // works currently holding the large texture
async function setLevel(rec, lv) {
  if (rec.busy || rec.level === lv) return;
  rec.busy = true;
  const t = await loadTex(rec.w[['t', 'm', 'w'][lv]]);
  rec.busy = false;
  if (!t) return;
  const old = rec.art.material.map;
  rec.art.material.map = t; rec.art.material.color.set('#fff'); rec.art.material.needsUpdate = true;
  if (old) old.dispose();
  rec.level = lv;
  if (lv === 2) { hi.push(rec); if (hi.length > 5) { const o = hi.shift(); if (o !== rec) setLevel(o, 1); } }
  else { const i = hi.indexOf(rec); if (i >= 0) hi.splice(i, 1); }
}
function streamTextures() {
  let n = works.filter(w => w.busy).length;
  const sp = spaceOf(me.x, me.z);
  for (const rec of works) {
    if (n >= 3) break;
    const d = Math.hypot(rec.pos.x - me.x, rec.pos.z - me.z), here = rec.room ? rooms.indexOf(rec.room) === sp : sp < 0;
    const want = rec === focus || (here && d < 3.2) ? 2 : (here || d < 13 || rec.want) ? 1 : 0;
    if (want > rec.level && !rec.busy) { setLevel(rec, want); n++; }
  }
}

// ---------------------------------------------------------------- frame loop
let last = performance.now(), tick = 0, space = -2, gateT = 0;
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w >= h ? 62 : clamp(THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(27)) / camera.aspect)), 62, 84);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

function step(dt) {
  if (!entered) {   // behind the title card: a slow look down the long gallery
    gateT += dt;
    me.x = Math.sin(gateT * 0.07) * 1.2; me.z = 0.5 - ((gateT * 0.35) % 14); me.yaw = Math.sin(gateT * 0.11) * 0.12; me.pitch = 0.03;
    return;
  }
  const mv = (keys.has('w') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('s') || keys.has('ArrowDown') ? 1 : 0);
  const st = (keys.has('d') ? 1 : 0) - (keys.has('a') ? 1 : 0);
  const tn = (keys.has('ArrowLeft') || keys.has('q') ? 1 : 0) - (keys.has('ArrowRight') || keys.has('e') ? 1 : 0);
  if (mv || st || tn) {
    if (walk) walk = null;
    me.yaw += tn * 1.7 * dt;
    const sp = 2.6 * dt, fx = -Math.sin(me.yaw), fz = -Math.cos(me.yaw);
    const dx = (fx * mv - fz * st) * sp, dz = (fz * mv + fx * st) * sp;
    if (canStand(me.x + dx, me.z + dz)) { me.x += dx; me.z += dz; }
    else if (canStand(me.x + dx, me.z)) me.x += dx;
    else if (canStand(me.x, me.z + dz)) me.z += dz;
    if (!tn) me.pitch += (0 - me.pitch) * Math.min(1, dt * 1.5);
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
  if (focus) { const p = viewPose(focus); if (Math.hypot(p.x - me.x, p.z - me.z) > 3 && !walk) closePanel(true); }
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
  if (document.hidden || dlg.open || plan.open) return;
  step(dt);
  shiftX += (shiftTX - shiftX) * Math.min(1, dt * 5); shiftY += (shiftTY - shiftY) * Math.min(1, dt * 5);
  if (Math.abs(shiftX) > 0.5 || Math.abs(shiftY) > 0.5) camera.setViewOffset(innerWidth, innerHeight, shiftX, shiftY, innerWidth, innerHeight);
  else camera.clearViewOffset();
  camera.position.set(me.x, EYE + (walk && !walk.settle ? Math.sin(now * 0.011) * 0.012 : 0), me.z);
  camera.rotation.set(me.pitch, me.yaw, 0, 'YXZ');
  if ((tick++ & 15) === 0) streamTextures();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- opening
async function init() {
  resize();
  try { await Promise.all([document.fonts.load('40px Marcellus'), document.fonts.load('300 20px "Albert Sans"'), document.fonts.load('italic 20px "Albert Sans"'), document.fonts.load('600 20px "Albert Sans"')]); } catch {}
  rooms.forEach(buildRoom); buildNave(); buildCollision(); buildStops(); buildPlan();
  const mirror = world.clone(true); mirror.scale.y = -1; scene.add(mirror);
  requestAnimationFrame(frame);
  await Promise.all(works.map(rec => setLevel(rec, 0)));
  const b = $('enter'); b.disabled = false; b.textContent = 'Enter the gallery';
  b.addEventListener('click', () => {
    const quiet = $('quiet').checked;
    document.body.classList.add('fade');
    setTimeout(() => {
      entered = true; me.x = 0; me.z = 8.6; me.yaw = 0; me.pitch = 0; space = -2;
      $('gate').hidden = true; $('hud').hidden = false; $('where').hidden = false; $('tour').hidden = false;
      document.body.classList.remove('fade'); canvas.style.cursor = 'grab';
      hint.textContent = matchMedia('(pointer: coarse)').matches
        ? 'Drag to look around. Tap the floor or a picture to walk to it.'
        : 'Drag to look around. Click the floor or a picture to walk to it. Arrow keys and W A S D also move you.';
      hint.hidden = false; requestAnimationFrame(() => hint.classList.add('on')); setTimeout(hideHint, 14000);
      if (quiet) { guide = false; $('b-guide').textContent = 'Audio guide off'; $('b-guide').setAttribute('aria-pressed', false); }
      showWelcome();
    }, 650);
  }, { once: true });
}
window.antara = { me, toStop, go };
init();
