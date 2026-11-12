#!/usr/bin/env python3
"""
The Nila at Thirunavaya, noon, cloth drying

    python nila_thirunavaya_noon_cloth_drying.py          # the picture, 3200 x 2130
    python nila_thirunavaya_noon_cloth_drying.py 0.5      # a study at half size

Needs numpy, scipy, numba, pillow.  Nothing else goes in.
"""
import sys, math, time
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from PIL import Image

# =====================================================================
#  THE BRUSH
#  A stroke is four decisions: direction, load, pressure, and where it
#  runs dry.  Everything below is those four, written down.
# =====================================================================

@njit(cache=True, fastmath=True, error_model='numpy')
def _stroke(col, hgt, tooth, sx, sy, n, r, c0, sv, bl, bc, bend,
            opac, thick, dry, drysens, pickup, edge, ridge):
    H = hgt.shape[0]; W = hgt.shape[1]
    nb = bl.shape[0]
    seg = np.empty(n); cum = np.empty(n); cum[0] = 0.0
    minx = 1e9; maxx = -1e9; miny = 1e9; maxy = -1e9
    for i in range(n):
        if sx[i] < minx: minx = sx[i]
        if sx[i] > maxx: maxx = sx[i]
        if sy[i] < miny: miny = sy[i]
        if sy[i] > maxy: maxy = sy[i]
        if i < n - 1:
            dx = sx[i + 1] - sx[i]; dy = sy[i + 1] - sy[i]
            seg[i] = math.sqrt(dx * dx + dy * dy) + 1e-6
            cum[i + 1] = cum[i] + seg[i]
    L = cum[n - 1] + 1e-6
    # the brush picks up what is already wet on the canvas as it travels;
    # pigments meet as pigments (log mix), not as light
    cs = np.empty((n, 3))
    cur0 = c0[0]; cur1 = c0[1]; cur2 = c0[2]
    for i in range(n):
        if pickup > 0.0 and i > 0:
            xi = int(sx[i]); yi = int(sy[i])
            if xi >= 0 and xi < W and yi >= 0 and yi < H:
                cur0 = math.exp((1 - pickup) * math.log(max(cur0, 0.004)) + pickup * math.log(max(col[yi, xi, 0], 0.004)))
                cur1 = math.exp((1 - pickup) * math.log(max(cur1, 0.004)) + pickup * math.log(max(col[yi, xi, 1], 0.004)))
                cur2 = math.exp((1 - pickup) * math.log(max(cur2, 0.004)) + pickup * math.log(max(col[yi, xi, 2], 0.004)))
        cs[i, 0] = cur0; cs[i, 1] = cur1; cs[i, 2] = cur2
    x0 = max(0, int(minx - r - 2)); x1 = min(W, int(maxx + r + 3))
    y0 = max(0, int(miny - r - 2)); y1 = min(H, int(maxy + r + 3))
    rr = r * r * 1.1
    for y in range(y0, y1):
        for x in range(x0, x1):
            best = 1e18; bi = 0; bt = 0.0; cr = 0.0
            for i in range(n - 1):
                ax = x - sx[i]; ay = y - sy[i]
                bx = sx[i + 1] - sx[i]; by = sy[i + 1] - sy[i]
                t = (ax * bx + ay * by) / (seg[i] * seg[i])
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                ex = ax - t * bx; ey = ay - t * by
                d2 = ex * ex + ey * ey
                if d2 < best:
                    best = d2; bi = i; bt = t; cr = bx * ay - by * ax
            if best > rr:
                continue
            tau = (cum[bi] + bt * seg[bi]) / L
            wf = 0.93 + 0.07 * math.sin(3.14159 * min(1.0, 0.12 + tau))   # a flat brush: the width barely changes
            dd = math.sqrt(best); over = 0.0
            if bi == 0 and bt <= 0.0:                                     # beyond the heel of the stroke
                over = -((x - sx[0]) * (sx[1] - sx[0]) + (y - sy[0]) * (sy[1] - sy[0])) / seg[0]
                dd = abs(cr) / seg[0]
            elif bi == n - 2 and bt >= 1.0:                               # beyond the toe
                over = ((x - sx[n - 1]) * (sx[n - 1] - sx[n - 2]) + (y - sy[n - 1]) * (sy[n - 1] - sy[n - 2])) / seg[n - 2]
                dd = abs(cr) / seg[n - 2]
            q = dd / (r * wf)
            if q >= 1.0:
                continue
            e = 1.0
            if q > 1.0 - edge:
                e = (1.0 - q) / edge
                e = e * e * (3 - 2 * e)
            if cr < 0: sq = -q
            else: sq = q
            k = int((sq * 0.5 + 0.5) * (nb - 1) + 0.5)
            if k < 0: k = 0
            if k >= nb: k = nb - 1
            if over > r * 0.5 * (bl[k] - 0.5):                            # square end, ragged by the bristles
                continue
            # each bristle lifts off at its own moment
            ce = (1.0 - bend[k] - tau) / 0.05
            if ce <= 0.0: continue
            if ce > 1.0: ce = 1.0
            cb = (tau + 0.02 - 0.3 * bend[nb - 1 - k]) / 0.04
            if cb <= 0.0: continue
            if cb > 1.0: cb = 1.0
            # the load runs out; a dry brush only touches the tops of the weave
            avail = bl[k] * (1.0 - dry * tau ** 1.5)
            cov = (avail - drysens * (1.0 - tooth[y, x]) - 0.08) / 0.2
            if cov <= 0.0: continue
            if cov > 1.0: cov = 1.0
            a = opac * e * cov * ce * cb
            f = bc[k]
            c_r = cs[bi, 0] + bt * (cs[bi + 1, 0] - cs[bi, 0]) + f * sv[0]
            c_g = cs[bi, 1] + bt * (cs[bi + 1, 1] - cs[bi, 1]) + f * sv[1]
            c_b = cs[bi, 2] + bt * (cs[bi + 1, 2] - cs[bi, 2]) + f * sv[2]
            col[y, x, 0] += a * (c_r - col[y, x, 0])
            col[y, x, 1] += a * (c_g - col[y, x, 1])
            col[y, x, 2] += a * (c_b - col[y, x, 2])
            bump = (q - 0.84) / 0.11
            hd = thick * (0.4 + 0.6 * bl[k]) * (1.0 - 0.45 * tau) + ridge * thick * math.exp(-bump * bump)
            hgt[y, x] = hgt[y, x] * (1.0 - 0.6 * a) + a * hd


@njit(cache=True, fastmath=True, error_model='numpy')
def _bristles(r, bl, bc, bend):
    nb = int(2 * r / 1.7)
    if nb < 4: nb = 4
    if nb > bl.shape[0]: nb = bl.shape[0]
    w = np.random.randn(); l = np.random.rand()
    for k in range(nb):
        w = 0.6 * w + 0.8 * np.random.randn()
        l = 0.5 * l + 0.5 * np.random.rand()
        bc[k] = max(-1.5, min(1.5, w))
        bl[k] = 0.55 + 0.45 * l
        bend[k] = 0.28 * np.random.rand() ** 2
    return nb


@njit(cache=True, fastmath=True, error_model='numpy')
def _trace(src, ang, reg, x, y, dx, dy, step, nmax, c0, tol, rg, ox, oy, wander):
    H = ang.shape[0]; W = ang.shape[1]
    n = 0
    for i in range(nmax):
        x += dx * step; y += dy * step
        xi = int(x); yi = int(y)
        if xi < 1 or xi >= W - 1 or yi < 1 or yi >= H - 1: break
        if reg[yi, xi] != rg: break
        d = abs(src[yi, xi, 0] - c0[0]) + abs(src[yi, xi, 1] - c0[1]) + abs(src[yi, xi, 2] - c0[2])
        if d > tol: break
        ox[n] = x; oy[n] = y; n += 1
        a = ang[yi, xi] + wander * np.random.randn()
        nx = math.cos(a); ny = math.sin(a)
        if nx * dx + ny * dy < 0: nx = -nx; ny = -ny
        dx = 0.55 * dx + 0.45 * nx; dy = 0.55 * dy + 0.45 * ny
        m = math.sqrt(dx * dx + dy * dy) + 1e-9
        dx /= m; dy /= m
    return n


@njit(cache=True, fastmath=True, error_model='numpy')
def _layer(col, hgt, tooth, src, ang, reg, lenf, px, py, pr, seed, maxlen, tol, cjit, streak,
           opac, thick, thickv, dry, drysens, pickup, edge, ridge, wander):
    np.random.seed(seed)
    fx = np.empty(160); fy = np.empty(160); gx = np.empty(160); gy = np.empty(160)
    sx = np.empty(330); sy = np.empty(330)
    bl = np.empty(128); bc = np.empty(128); bend = np.empty(128)
    c = np.empty(3); sv = np.empty(3); c0 = np.empty(3)
    for j in range(px.shape[0]):
        x = px[j]; y = py[j]; r = pr[j]
        xi = int(x); yi = int(y)
        c0[0] = src[yi, xi, 0]; c0[1] = src[yi, xi, 1]; c0[2] = src[yi, xi, 2]
        rg = reg[yi, xi]
        step = max(1.5, 0.6 * r)
        nmax = int(maxlen * r * lenf[yi, xi] / step)
        if nmax < 1: nmax = 1
        if nmax > 150: nmax = 150
        a0 = ang[yi, xi] + wander * np.random.randn()
        dx = math.cos(a0); dy = math.sin(a0)
        nfw = int(nmax * (0.3 + 0.4 * np.random.rand())) + 1
        nf = _trace(src, ang, reg, x, y, dx, dy, step, nfw, c0, tol, rg, fx, fy, wander)
        ng = _trace(src, ang, reg, x, y, -dx, -dy, step, nmax - nfw + 1, c0, tol, rg, gx, gy, wander)
        n = 0
        for i in range(ng - 1, -1, -1):
            sx[n] = gx[i]; sy[n] = gy[i]; n += 1
        sx[n] = x; sy[n] = y; n += 1
        for i in range(nf):
            sx[n] = fx[i]; sy[n] = fy[i]; n += 1
        if n < 2:
            sx[1] = x + dx * step; sy[1] = y + dy * step; n = 2
        if np.random.rand() < 0.5:
            for i in range(n // 2):
                tx = sx[i]; sx[i] = sx[n - 1 - i]; sx[n - 1 - i] = tx
                ty = sy[i]; sy[i] = sy[n - 1 - i]; sy[n - 1 - i] = ty
        nb = _bristles(r, bl, bc, bend)
        v = 1.0 + 0.3 * cjit * np.random.randn()
        g = np.random.randn()
        h0 = np.random.randn(); h1 = np.random.randn(); h2 = np.random.randn()
        hm = 0.3 * h0 + 0.6 * h1 + 0.1 * h2
        l0 = 0.3 * c0[0] + 0.6 * c0[1] + 0.1 * c0[2]
        c[0] = c0[0] * v + cjit * (h0 - hm) * (0.25 + l0)
        c[1] = c0[1] * v + cjit * (h1 - hm) * (0.25 + l0)
        c[2] = c0[2] * v + cjit * (h2 - hm) * (0.25 + l0)
        for k in range(3):
            c[k] = min(1.0, max(0.0, c[k]))
            sv[k] = streak * (0.5 * g * (0.25 + c[k]) + 0.4 * np.random.randn())
        lum = 0.3 * c[0] + 0.6 * c[1] + 0.1 * c[2]
        th = thick * (1.0 + thickv * (lum - 0.35) * 2.2)
        if th < 0.05 * thick: th = 0.05 * thick
        _stroke(col, hgt, tooth, sx, sy, n, r, c, sv, bl[:nb], bc[:nb], bend[:nb],
                opac, th, dry, drysens, pickup, edge, ridge)


@njit(cache=True, fastmath=True, error_model='numpy')
def _one(col, hgt, tooth, sx, sy, r, c, seed, streak, opac, thick, dry, drysens, pickup, edge, ridge):
    np.random.seed(seed)
    bl = np.empty(128); bc = np.empty(128); bend = np.empty(128); sv = np.empty(3)
    nb = _bristles(r, bl, bc, bend)
    g = np.random.randn()
    for k in range(3):
        sv[k] = streak * (0.6 * g * (0.25 + c[k]) + 0.35 * np.random.randn())
    _stroke(col, hgt, tooth, sx, sy, sx.shape[0], r, c, sv, bl[:nb], bc[:nb], bend[:nb],
            opac, thick, dry, drysens, pickup, edge, ridge)


@njit(cache=True, fastmath=True, error_model='numpy')
def _ribbon(px, py, pw, W, H):
    """distance field of a tapering polyline: returns crop of (q, t, angle)."""
    n = px.shape[0]
    wm = pw.max()
    x0 = max(0, int(px.min() - wm - 2)); x1 = min(W, int(px.max() + wm + 3))
    y0 = max(0, int(py.min() - wm - 2)); y1 = min(H, int(py.max() + wm + 3))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.full((1, 1), 9.0, np.float32), np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32)
    q = np.full((y1 - y0, x1 - x0), 9.0, np.float32)
    tt = np.zeros((y1 - y0, x1 - x0), np.float32)
    aa = np.zeros((y1 - y0, x1 - x0), np.float32)
    for y in range(y0, y1):
        for x in range(x0, x1):
            best = 1e18; bi = 0; bt = 0.0; cr = 0.0
            for i in range(n - 1):
                ax = x - px[i]; ay = y - py[i]
                bx = px[i + 1] - px[i]; by = py[i + 1] - py[i]
                t = (ax * bx + ay * by) / (bx * bx + by * by + 1e-9)
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                ex = ax - t * bx; ey = ay - t * by
                d2 = ex * ex + ey * ey
                if d2 < best:
                    best = d2; bi = i; bt = t; cr = bx * ay - by * ax
            w = pw[bi] + bt * (pw[bi + 1] - pw[bi])
            d = math.sqrt(best) / (w + 1e-6)
            if d < 3.0:
                q[y - y0, x - x0] = d if cr >= 0 else -d
                tt[y - y0, x - x0] = (bi + bt) / (n - 1)
                aa[y - y0, x - x0] = math.atan2(py[bi + 1] - py[bi], px[bi + 1] - px[bi])
    return x0, y0, q, tt, aa


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def _lic(col, ang, mask, L):
    """drag the wet paint along the way the brush went."""
    H = col.shape[0]; W = col.shape[1]
    out = col.copy()
    for y in prange(H):
        for x in range(W):
            m = mask[y, x]
            if m < 0.03: continue
            n = int(L * m) + 1
            a0 = col[y, x, 0]; a1 = col[y, x, 1]; a2 = col[y, x, 2]; cnt = 1.0
            for sg in range(2):
                a = ang[y, x]
                dx = math.cos(a); dy = math.sin(a)
                if sg == 1: dx = -dx; dy = -dy
                fx = x + 0.0; fy = y + 0.0
                for k in range(n):
                    fx += dx; fy += dy
                    xi = int(fx + 0.5); yi = int(fy + 0.5)
                    if xi < 0 or xi >= W or yi < 0 or yi >= H: break
                    w = 1.0 - k / (n + 1.0)
                    a0 += w * col[yi, xi, 0]; a1 += w * col[yi, xi, 1]; a2 += w * col[yi, xi, 2]; cnt += w
                    b = ang[yi, xi]; nx = math.cos(b); ny = math.sin(b)
                    if nx * dx + ny * dy < 0: nx = -nx; ny = -ny
                    dx = nx; dy = ny
            out[y, x, 0] = a0 / cnt; out[y, x, 1] = a1 / cnt; out[y, x, 2] = a2 / cnt
    return out


# =====================================================================
#  SMALL TOOLS
# =====================================================================

def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if np.ndim(t) == 2 and np.shape(t)[-1] != 1 and (a.ndim == 1 or b.ndim == 1 or a.ndim == 3):
        t = t[..., None]
    return a + (b - a) * t

def vnoise(H, W, cell, rng, aspect=1.0):
    """smooth noise in 0..1; features `cell` px wide and cell*aspect px tall."""
    cx = max(1.0, cell); cy = max(1.0, cell * aspect)
    gh = int(H / cy) + 4; gw = int(W / cx) + 4
    g = rng.random((gh, gw)).astype(np.float32)
    im = Image.fromarray(g).resize((int(gw * cx) + 1, int(gh * cy) + 1), Image.BICUBIC)
    a = np.asarray(im)
    oy = int(cy * (1 + rng.random())); ox = int(cx * (1 + rng.random()))
    return np.clip(a[oy:oy + H, ox:ox + W], 0, 1)

def fbm(H, W, cell, rng, octaves=4, aspect=1.0, gain=0.5):
    out = np.zeros((H, W), np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        out += amp * vnoise(H, W, cell / 2 ** o, rng, aspect); tot += amp; amp *= gain
    out /= tot
    return np.clip((out - 0.5) * 1.9 + 0.5, 0, 1)

def blur(a, s):
    if a.ndim == 3:
        return np.stack([gaussian_filter(a[..., k], s) for k in range(3)], -1)
    return gaussian_filter(a, s)

def lum(c):
    return 0.3 * c[..., 0] + 0.6 * c[..., 1] + 0.1 * c[..., 2]

def spline(pts, n=40):
    pts = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    cs = CubicSpline(d, pts, bc_type='natural') if len(pts) > 2 else None
    t = np.linspace(0, d[-1], n)
    return cs(t) if cs is not None else np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], 1)


SIGNATURE = [
    [(0.43, 0.47), (0.40, 0.45), (0.35, 0.35), (0.29, 0.30), (0.21, 0.26), (0.15, 0.25), (0.08, 0.27), (0.03, 0.34), (0.00, 0.40), (0.00, 0.50)],
    [(0.00, 0.50), (0.02, 0.62), (0.05, 0.70), (0.10, 0.74), (0.17, 0.73), (0.20, 0.70), (0.23, 0.65), (0.23, 0.58), (0.22, 0.54), (0.15, 0.49), (0.01, 0.50)],
    [(0.43, 0.47), (0.46, 0.45), (0.51, 0.36), (0.55, 0.32), (0.66, 0.26), (0.72, 0.25), (0.78, 0.27), (0.83, 0.30), (0.86, 0.37), (0.86, 0.49), (0.78, 0.63), (0.71, 0.70), (0.62, 0.76), (0.31, 0.90), (0.29, 0.92), (0.29, 0.97), (0.33, 0.99), (0.42, 1.00), (0.86, 0.98)],
    [(0.43, 0.47), (0.42, 0.71)],
    [(1.13, 0.52), (1.36, 0.49), (1.46, 0.43), (1.48, 0.40), (1.47, 0.35), (1.42, 0.29), (1.36, 0.26), (1.28, 0.24), (1.20, 0.23), (1.12, 0.25), (1.08, 0.30), (1.05, 0.39), (1.05, 0.54), (1.08, 0.67), (1.14, 0.71), (1.22, 0.73), (1.50, 0.74), (1.66, 0.72), (1.74, 0.66), (1.76, 0.62), (1.78, 0.55), (1.78, 0.40), (1.75, 0.24)],
    [(2.39, 0.44), (2.35, 0.42), (2.30, 0.32), (2.22, 0.26), (2.14, 0.24), (2.07, 0.26), (2.02, 0.33), (1.97, 0.53), (1.97, 0.62), (1.99, 0.67), (2.02, 0.71), (2.07, 0.73), (2.13, 0.73), (2.17, 0.71), (2.21, 0.67), (2.23, 0.60), (2.23, 0.54), (2.19, 0.48), (2.14, 0.47), (2.10, 0.49)],
    [(2.39, 0.44), (2.40, 0.72)],
    [(2.39, 0.44), (2.41, 0.42), (2.47, 0.30), (2.52, 0.25), (2.61, 0.25), (2.65, 0.26), (2.69, 0.30)],
    [(2.69, 0.30), (2.61, 0.42), (2.56, 0.60), (2.57, 0.68), (2.60, 0.73), (2.64, 0.74), (2.71, 0.73), (2.75, 0.71), (2.79, 0.68), (2.83, 0.61), (2.83, 0.55), (2.79, 0.45), (2.76, 0.38), (2.70, 0.30)],
    [(2.97, 0.38), (2.94, 0.36), (2.91, 0.29), (2.84, 0.24), (2.76, 0.25), (2.70, 0.30)],
    [(2.97, 0.39), (2.97, 0.74)],
    [(2.98, 0.38), (3.01, 0.36), (3.07, 0.28), (3.13, 0.25), (3.20, 0.25), (3.26, 0.31)],
    [(3.26, 0.32), (3.16, 0.44), (3.13, 0.53), (3.12, 0.59), (3.14, 0.66), (3.17, 0.70), (3.23, 0.73), (3.31, 0.73), (3.36, 0.71), (3.39, 0.67), (3.40, 0.57), (3.36, 0.45), (3.27, 0.32)],
    [(3.27, 0.32), (3.36, 0.25), (3.46, 0.24), (3.57, 0.27), (3.63, 0.34), (3.65, 0.41), (3.65, 0.50), (3.62, 0.60), (3.54, 0.73)],
    [(3.88, 0.94), (3.89, 0.71), (3.94, 0.30), (3.94, 0.22), (3.90, 0.10), (3.82, 0.02), (3.71, 0.00), (3.59, 0.03), (3.44, 0.12)],
]


# =====================================================================
#  THE CANVAS
# =====================================================================

class Painting:
    def __init__(self, W, H, seed, ground=(0.60, 0.33, 0.20)):
        self.W, self.H = W, H
        self.S = W / 2400.0
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        S, rng = self.S, self.rng
        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
        self.X, self.Y = X, Y
        # coarse canvas: plain weave, uneven threads
        p = 7.0 * S if S >= 0.6 else 5.0
        wx = (vnoise(H, W, 90 * S, rng) - 0.5) * 5 * S
        wy = (vnoise(H, W, 90 * S, rng) - 0.5) * 5 * S
        wv = 0.5 + 0.5 * np.cos(2 * np.pi * (X + wx) / p) * np.cos(2 * np.pi * (Y + wy) / p)
        slub = vnoise(H, W, 5 * S, rng, aspect=30) * 0.5 + vnoise(H, W, 150 * S, rng, aspect=0.03) * 0.5
        self.tooth = np.clip(wv * (0.75 + 0.5 * slub), 0, 1).astype(np.float32)
        # toned ground, scrubbed on with a wide brush
        g = np.asarray(ground, np.float32)
        scr = fbm(H, W, 260 * S, rng, 3, aspect=0.25)
        self.col = (g[None, None, :] * (0.86 + 0.24 * scr[..., None]) * (0.93 + 0.07 * self.tooth[..., None])).astype(np.float32)
        self.ground = self.col.copy()
        self.hgt = np.zeros((H, W), np.float32)
        self.ref = np.zeros((H, W, 3), np.float32)
        self.ang = np.zeros((H, W), np.float32)
        self.lenf = np.ones((H, W), np.float32)
        self.reg = np.zeros((H, W), np.int32)
        self.detail = np.zeros((H, W), np.float32)
        # the painted field stops short of the edge, unevenly
        d = np.minimum(np.minimum(X, W - 1 - X), np.minimum(Y, H - 1 - Y))
        self.edge = d + (fbm(H, W, 120 * S, rng, 3) - 0.5) * 60 * S
        self._k = 0

    # ---- building the drawing ------------------------------------------------
    def put(self, mask, colour, ang=None, reg=None, lenf=None, detail=None, at=None):
        """lay a shape into the drawing.  at=(x0,y0) for a cropped mask."""
        if at is None:
            sl = (slice(None), slice(None))
        else:
            sl = (slice(at[1], at[1] + mask.shape[0]), slice(at[0], at[0] + mask.shape[1]))
        m = np.clip(mask, 0, 1).astype(np.float32)
        c = np.asarray(colour, np.float32)
        self.ref[sl] += m[..., None] * (c - self.ref[sl])
        hard = m > 0.5
        if ang is not None:
            self.ang[sl] = np.where(hard, ang, self.ang[sl])
        if reg is not None:
            self.reg[sl] = np.where(hard, reg, self.reg[sl])
        if lenf is not None:
            self.lenf[sl] = np.where(hard, lenf, self.lenf[sl])
        if detail is not None:
            self.detail[sl] = np.maximum(self.detail[sl], m * detail)

    def ribbon(self, pts, widths):
        pts = np.asarray(pts, np.float64); w = np.asarray(widths, np.float64)
        if w.ndim == 0: w = np.full(len(pts), float(w))
        if len(w) != len(pts): w = np.interp(np.linspace(0, 1, len(pts)), np.linspace(0, 1, len(w)), w)
        return _ribbon(np.ascontiguousarray(pts[:, 0]), np.ascontiguousarray(pts[:, 1]), w, self.W, self.H)

    # ---- painting ------------------------------------------------------------
    def layer(self, src, r, spacing=0.9, mask=None, err=None, order='value', maxlen=6.0, tol=0.22,
              cjit=0.05, streak=0.05, opac=0.95, thick=1.0, thickv=0.6, dry=0.6, drysens=0.5,
              pickup=0.06, edge=0.35, ridge=0.4, wander=0.12, sizev=0.3, margin=26):
        W, H, S, rng = self.W, self.H, self.S, self.rng
        r = r * S
        g = max(1.5, r * spacing)
        xs = np.arange(g / 2, W, g); ys = np.arange(g / 2, H, g)
        gx, gy = np.meshgrid(xs, ys)
        px = (gx + rng.uniform(-g / 2, g / 2, gx.shape)).ravel().clip(1, W - 2)
        py = (gy + rng.uniform(-g / 2, g / 2, gy.shape)).ravel().clip(1, H - 2)
        ix = px.astype(int); iy = py.astype(int)
        keep = self.edge[iy, ix] > margin * S
        if mask is not None:
            keep &= rng.random(len(px)) < mask[iy, ix]
        if err is not None:
            e = np.abs(blur(self.col, max(1.0, r * 0.5)) - src).sum(-1)
            keep &= e[iy, ix] > err / (0.35 + 1.5 * self.detail[iy, ix])
        px, py, ix, iy = px[keep], py[keep], ix[keep], iy[keep]
        if len(px) == 0: return 0
        if order == 'value':      # darks first and thin, lights last and thick
            o = np.argsort(lum(src[iy, ix]) + rng.normal(0, 0.10, len(px)))
        else:
            o = rng.permutation(len(px))
        px, py = px[o], py[o]
        pr = r * (1.0 + sizev * rng.uniform(-1, 1, len(px)))
        self._k += 1
        _layer(self.col, self.hgt, self.tooth, np.ascontiguousarray(src, np.float32), self.ang, self.reg, self.lenf,
               px, py, pr, self.seed % 100000 + 17 * self._k, maxlen, tol, cjit, streak,
               opac, thick, thickv, dry, drysens, pickup, edge, ridge, wander)
        return len(px)

    def stroke(self, pts, r, colour, streak=0.04, opac=1.0, thick=1.0, dry=0.5, drysens=0.4,
               pickup=0.0, edge=0.3, ridge=0.4, smooth_pts=True):
        """one deliberate stroke along a path (canvas pixels)."""
        pts = np.asarray(pts, float)
        r = max(0.7, r)
        ln = np.hypot(*np.diff(pts, axis=0).T).sum()
        n = int(max(2, min(300, ln / max(1.2, 0.6 * r) + 1)))
        p = spline(pts, n) if smooth_pts and len(pts) > 2 else np.stack(
            [np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pts)), pts[:, k]) for k in (0, 1)], 1)
        self._k += 1
        _one(self.col, self.hgt, self.tooth, np.ascontiguousarray(p[:, 0]), np.ascontiguousarray(p[:, 1]), float(r),
             np.asarray(colour, np.float64), self.seed % 100000 + 31 * self._k, streak, opac, thick, dry, drysens,
             pickup, edge, ridge)

    def dab(self, x, y, a, length, r, colour, **kw):
        dx, dy = math.cos(a) * length / 2, math.sin(a) * length / 2
        self.stroke([(x - dx, y - dy), (x + dx, y + dy)], r, colour, smooth_pts=False, **kw)

    def glaze(self, mask, colour, strength=0.5, cell=40):
        """a transparent film: darkens and tints, never covers."""
        S, rng = self.S, self.rng
        st = 0.7 + 0.3 * fbm(self.H, self.W, cell * S, rng, 2, aspect=4)
        m = (np.clip(mask, 0, 1) * strength * st)[..., None]
        c = np.asarray(colour, np.float32)
        self.col *= (1 - m) + m * c

    def blend(self, mask, length=14, soften=0.0):
        """work wet into wet where the edges should be lost."""
        m = np.clip(mask, 0, 1).astype(np.float32)
        self.col = _lic(self.col, self.ang, m, float(length * self.S))
        if soften > 0:
            b = blur(self.col, soften * self.S)
            self.col += (m * 0.6)[..., None] * (b - self.col)
        self.hgt *= (1 - 0.5 * m)

    def sign(self, x, y, h, colour=(0.62, 0.20, 0.10)):
        """lamp mark, then the name.  x,y = lower-right corner of the signature, canvas px."""
        S = self.S
        w = 3.94 * h
        x0 = x - w; y0 = y - h
        r = max(1.0, 0.045 * h)
        for path in SIGNATURE:
            p = [(x0 + a * h, y0 + b * h) for a, b in path]
            self.stroke(p, r, colour, streak=0.02, thick=0.5, dry=0.25, drysens=0.25, edge=0.5, smooth_pts=len(p) > 3)
        lx = x0 - 0.75 * h; ly = y0 + 0.70 * h          # the lamp: a bowl, a wick, a flame
        self.stroke([(lx - 0.30 * h, ly - 0.10 * h), (lx - 0.16 * h, ly + 0.08 * h), (lx + 0.16 * h, ly + 0.08 * h),
                     (lx + 0.30 * h, ly - 0.10 * h)], r, colour, streak=0.02, thick=0.5, dry=0.2, drysens=0.25)
        self.stroke([(lx - 0.12 * h, ly + 0.14 * h), (lx + 0.12 * h, ly + 0.14 * h)], r, colour, thick=0.5, dry=0.2, drysens=0.25)
        self.stroke([(lx, ly - 0.16 * h), (lx - 0.07 * h, ly - 0.32 * h), (lx, ly - 0.56 * h), (lx + 0.07 * h, ly - 0.32 * h),
                     (lx, ly - 0.16 * h)], r * 0.9, colour, thick=0.5, dry=0.2, drysens=0.25)

    # ---- the finished surface under a raking light ---------------------------
    def finish(self, relief=1.0, sat=1.03, weave=0.22):
        S = self.S
        h = gaussian_filter(self.hgt, 1.0 * S)
        wv = self.tooth * np.exp(-2.2 * np.clip(h, 0, 3))             # thin paint lets the canvas show
        surf = (1.5 * h + weave * wv) * S * relief
        gy, gx = np.gradient(surf)
        l = np.array([-0.55, -0.62, 0.56]); l /= np.linalg.norm(l)    # window high on the left
        nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
        shade = (-gx * l[0] - gy * l[1] + l[2]) * nz / l[2]
        hv = np.array([l[0], l[1], l[2] + 1.0]); hv /= np.linalg.norm(hv)
        spec = np.clip((-gx * hv[0] - gy * hv[1] + hv[2]) * nz, 0, 1) ** 60
        spec = (spec - spec.mean()) * np.clip(h, 0, 1.2) * 0.07
        out = self.col * (0.70 + 0.30 * shade[..., None]) + spec[..., None] * np.array([1.0, 0.97, 0.9])
        m = out.mean(-1, keepdims=True)
        out = m + (out - m) * sat
        out += (self.rng.random(out.shape[:2])[..., None] - 0.5) / 255.0
        return np.clip(out, 0, 1)


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)




# =====================================================================
#  GEOMETRY AND LIGHT
#  The place is built as solid form and lit by its own lamp and its own
#  sky.  Each picture supplies scene(x, y, z) -> (distance, material).
# =====================================================================

@njit(cache=True, fastmath=True, error_model='numpy')
def hash3(i, j, k):
    n = (i * 374761393 + j * 668265263 + k * 1274126177) & 0x7fffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0x7fffffff
    return ((n ^ (n >> 16)) & 0xffffff) / 16777216.0

@njit(cache=True, fastmath=True, error_model='numpy')
def noise3(x, y, z):
    i = math.floor(x); j = math.floor(y); k = math.floor(z)
    u = x - i; v = y - j; w = z - k
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v); w = w * w * (3 - 2 * w)
    i = int(i); j = int(j); k = int(k)
    a = hash3(i, j, k) + u * (hash3(i + 1, j, k) - hash3(i, j, k))
    b = hash3(i, j + 1, k) + u * (hash3(i + 1, j + 1, k) - hash3(i, j + 1, k))
    c = hash3(i, j, k + 1) + u * (hash3(i + 1, j, k + 1) - hash3(i, j, k + 1))
    d = hash3(i, j + 1, k + 1) + u * (hash3(i + 1, j + 1, k + 1) - hash3(i, j + 1, k + 1))
    e = a + v * (b - a); f = c + v * (d - c)
    return e + w * (f - e)

@njit(cache=True, fastmath=True, error_model='numpy')
def fbm3(x, y, z, o):
    s = 0.0; a = 0.5; t = 0.0
    for i in range(o):
        s += a * noise3(x, y, z); t += a; a *= 0.5
        x = x * 2.03 + 11.3; y = y * 2.03 + 5.7; z = z * 2.03 + 3.1
    return s / t

@njit(cache=True, fastmath=True, error_model='numpy')
def fbm2(x, z, o):
    return fbm3(x, 0.37, z, o)

@njit(cache=True, fastmath=True, error_model='numpy')
def sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0: t = 0.0
    if t > 1: t = 1.0
    return t * t * (3 - 2 * t)

@njit(cache=True, fastmath=True, error_model='numpy')
def sd_box(x, y, z, bx, by, bz):
    qx = abs(x) - bx; qy = abs(y) - by; qz = abs(z) - bz
    ox = max(qx, 0.0); oy = max(qy, 0.0); oz = max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0)

@njit(cache=True, fastmath=True, error_model='numpy')
def sd_capsule(x, y, z, ax, ay, az, bx, by, bz, r):
    px = x - ax; py = y - ay; pz = z - az
    ex = bx - ax; ey = by - ay; ez = bz - az
    h = (px * ex + py * ey + pz * ez) / (ex * ex + ey * ey + ez * ez + 1e-12)
    if h < 0: h = 0.0
    if h > 1: h = 1.0
    px -= ex * h; py -= ey * h; pz -= ez * h
    return math.sqrt(px * px + py * py + pz * pz) - r

@njit(cache=True, fastmath=True, error_model='numpy')
def smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


# =====================================================================
#  THE PICTURE
#  The sandbank below Thirunavaya, a summer in the 1960s.  Noon: the
#  sun stands almost overhead, a little to the south, and we look
#  south across the water, so every hanging cloth is seen from its
#  shaded side with the light coming through it.
# =====================================================================
from scipy.ndimage import map_coordinates

LAT, LON, MINUTE = 10.8640, 75.9830, 12 * 60 + 0
SEED = int(LAT * 1e4) * 7919 + int(LON * 1e4) * 104729 + MINUTE

CAMX, CAMY, CAMZ = 0.0, 2.6, 0.0
WATER = -0.28
ROPE = 2.12
FIREX, FIREY, FIREZ = -4.6, 0.16, 23.2
MANX, MANZ = -3.5, 23.0
POLES = np.array([(2.42, 4.95), (0.38, 6.9), (-0.45, 10.6), (-0.95, 15.6)])
# the cloths on the line: segment, from, to (fractions of the span), drop, swing, gust, phase, fold depth, folds, kind
PIECES = np.array([
    (0, 0.035, 0.500, 1.30, 0.50, 0.16, 0.6, 0.150, 1.7, 0),
    (0, 0.545, 0.975, 1.22, 0.34, 0.14, 2.4, 0.120, 2.0, 1),
    (1, 0.040, 0.470, 1.18, 0.30, 0.12, 4.0, 0.110, 1.8, 2),
    (1, 0.520, 0.960, 1.15, 0.22, 0.12, 1.1, 0.090, 2.4, 0),
    (2, 0.040, 0.340, 1.20, 0.28, 0.10, 3.0, 0.100, 1.6, 1),
    (2, 0.380, 0.660, 1.15, 0.22, 0.10, 5.0, 0.090, 1.8, 2),
    (2, 0.700, 0.970, 1.20, 0.26, 0.10, 0.3, 0.090, 1.7, 0)])
# cloth spread on the sand: x0, x1, z0, z1
_r = np.random.default_rng(SEED + 3)
ROWS = []
_x = -2.0
for _i in range(7):
    _w = _r.uniform(1.10, 1.25); _z0 = 8.2 + _r.uniform(-0.5, 0.9) + 0.25 * _i
    for _j in range(2):
        _za = _z0 + _j * 6.0 + _r.uniform(0.0, 0.5); _zb = _za + _r.uniform(4.6, 5.3)
        _zb = min(_zb, 20.5 + 0.42 * (-_x))
        if _zb - _za > 1.8: ROWS.append((_x - _w, _x, _za, _zb))
    _x -= _w + _r.uniform(0.3, 0.5)
ROWS = np.array(ROWS)
PRINTS = np.array([(-0.9 - 2.5 * (k / 25.0) ** 1.2 + 0.15 * (1 if k % 2 else -1) + 0.25 * math.sin(k / 25.0 * 5), 7.0 + 14.6 * k / 25.0)
                   for k in range(26)])


@njit(cache=True, fastmath=True, error_model='numpy')
def shore(x):
    return 24.0 - 0.42 * x + 2.2 * math.sin(x * 0.33 + 1.0) + 1.5 * math.sin(x * 0.11 + 2.0)


@njit(cache=True, fastmath=True, error_model='numpy')
def row_at(x, z):
    for k in range(ROWS.shape[0]):
        if x > ROWS[k, 0] and x < ROWS[k, 1] and z > ROWS[k, 2] and z < ROWS[k, 3]:
            return k
    return -1


@njit(cache=True, fastmath=True, error_model='numpy')
def ground_h(x, z):
    h = 0.22 * (fbm2(x * 0.16, z * 0.16, 3) - 0.5)
    if z < 30.0:
        h += 0.010 * math.sin(z * 17.0 + 5.0 * fbm2(x * 0.7, z * 0.7, 2) + x * 2.0) * sstep(20.0, 6.0, z)      # wind ripples
        if z > 6.0 and z < 23.0 and x < 0.5 and x > -5.0:
            for k in range(PRINTS.shape[0]):
                dx = x - PRINTS[k, 0]; dz = z - PRINTS[k, 1]
                if abs(dx) < 0.4 and abs(dz) < 0.5:
                    h -= 0.035 * math.exp(-(dx / 0.075) ** 2 - (dz / 0.14) ** 2)
    t = z - shore(x)
    if t > -3.0:
        h -= 1.5 * sstep(-1.0, 16.0, t) + 0.10 * sstep(-3.0, 0.0, t)
        h += 1.42 * math.exp(-((z - (76.0 + 0.10 * x)) / (4.5 + 2.5 * math.sin(x * 0.05))) ** 2) * sstep(-110.0, -40.0, x) * sstep(55.0, -5.0, x)
        h += 1.2 * math.exp(-((z - 150.0) / 6.0) ** 2) * sstep(20.0, 80.0, x)
    if z > 215.0:
        h += 3.6 * sstep(222.0, 233.0, z)
    return h


@njit(cache=True, fastmath=True, error_model='numpy')
def cloth_uv(x, y, z):
    """nearest hanging cloth: (piece, how far in front of its surface, along, drop, hem)."""
    best = 1e9; bk = -1; bal = 0.0; bu = 0.0; bh = 1.0
    if z > 17.5 or y > 2.3 or y < 0.3 or z < 2.5:
        return bk, best, bal, bu, bh
    for k in range(PIECES.shape[0]):
        sg = int(PIECES[k, 0])
        ax = POLES[sg, 0]; az = POLES[sg, 1]; bx = POLES[sg + 1, 0]; bz = POLES[sg + 1, 1]
        ln = math.sqrt((bx - ax) ** 2 + (bz - az) ** 2)
        dx = (bx - ax) / ln; dz = (bz - az) / ln
        nx = dz; nz = -dx
        if nz > 0: nx = -nx; nz = -nz
        s = ((x - ax) * dx + (z - az) * dz) / ln
        w = (x - ax) * nx + (z - az) * nz
        a0 = PIECES[k, 1]; a1 = PIECES[k, 2]
        if s < a0 - 0.15 or s > a1 + 0.15 or abs(w) > 1.6: continue
        al = (s - a0) / (a1 - a0)
        alc = min(max(al, 0.0), 1.0)
        hang = PIECES[k, 3]; ph = PIECES[k, 6]
        ry = ROPE - 0.16 * 4 * s * (1 - s) - 0.03 * abs(math.sin(3.14159 * alc * 3))
        u = ry - y
        swing = PIECES[k, 4] + PIECES[k, 5] * math.sin(6.2832 * 0.6 * alc + ph) + 0.06 * math.sin(6.2832 * 1.7 * alc + 2 * ph)
        hem = hang * (1.0 - 0.32 * swing * swing)
        uu = min(max(u / hem, 0.0), 1.0)
        nf = PIECES[k, 8]
        fold = (0.12 + 0.88 * uu ** 0.8) * (PIECES[k, 7] * math.sin(6.2832 * nf * alc + ph + 0.9 * math.sin(2.5 * uu + ph))
                                            + 0.030 * math.sin(6.2832 * nf * 2.9 * alc + 2 * ph + 1.6 * uu)
                                            + 0.010 * uu * math.sin(6.2832 * nf * 7.3 * alc + 5 * uu))
        surf = hang * swing * uu ** 1.6 + fold
        dsheet = abs(w - surf) * 0.42 - 0.004
        ds = max(a0 - s, s - a1) * ln; du = max(-u, u - hem)
        d = dsheet
        if ds > 0 or du > 0:
            d = math.sqrt(max(ds, 0.0) ** 2 + max(du, 0.0) ** 2 + max(dsheet, 0.0) ** 2)
        if d < best:
            best = d; bk = k; bal = alc; bu = u; bh = hem
    return bk, best, bal, bu, bh


@njit(cache=True, fastmath=True, error_model='numpy')
def scene(x, y, z):
    gh = ground_h(x, z)
    d = (y - gh) * 0.7; m = 1
    if z < 22.0 and x < 0.0 and y < 0.6:
        k = row_at(x, z)
        if k >= 0:
            ex = min(x - ROWS[k, 0], ROWS[k, 1] - x); ez = min(z - ROWS[k, 2], ROWS[k, 3] - z)
            lift = 0.012 * sstep(0.0, 0.05, min(ex, ez)) + 0.012 * (fbm2(x * 9.0, z * 2.2, 2) - 0.4) * sstep(0.0, 0.15, min(ex, ez))
            dd = (y - gh - lift) * 0.7
            if dd < d: d = dd; m = 30
    dw = y - WATER
    if dw < d: d = dw; m = 10
    if z > 200.0:
        # ---- the far bank: trees, palms, the small temple
        if z > 236.0:
            CS = 9.0
            ci = math.floor(x / CS)
            dc = 1e9
            for a in range(-1, 2):
                i = int(ci + a)
                for r in range(2):
                    cx = (i + 0.5 + 0.6 * (hash3(i, 3 + r, 1) - 0.5)) * CS; cz = 246.0 + 9.0 * r + 3.0 * hash3(i, 5, r)
                    if cx > -50.0 and cx < -8.0 and r == 0: continue
                    rad = 5.0 + 4.0 * hash3(i, 7, r)
                    cy = 3.5 + rad * 0.85
                    ex = (x - cx) / rad; ey = (y - cy) / (rad * 0.8); ez = (z - cz) / rad
                    dd = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * rad * 0.8
                    if dd < dc: dc = dd
            if dc < 3.0:
                dc += 0.9 - 2.6 * fbm3(x * 0.35, y * 0.35, z * 0.35, 4)
            dc *= 0.5
            if dc < d: d = dc; m = 15
        if z > 226.0 and z < 250.0:
            PS = 7.0
            ci = math.floor(x / PS)
            for a in range(-1, 2):
                i = int(ci + a)
                h1 = hash3(i, 61, 2)
                if h1 > 0.52: continue
                px = (i + 0.5 + 0.9 * (hash3(i, 63, 2) - 0.5)) * PS; pz = 233.0 + 9.0 * hash3(i, 67, 2)
                if px > -46.0 and px < -12.0 and h1 > 0.2: continue
                hh = 11.0 + 13.0 * hash3(i, 71, 2); ln = 6.0 * (hash3(i, 73, 2) - 0.5)
                tx = px + ln; ty = 3.4 + hh
                dd = sd_capsule(x, y, z, px, 3.0, pz, px + 0.35 * ln, 3.4 + 0.55 * hh, pz, 0.26)
                d2 = sd_capsule(x, y, z, px + 0.35 * ln, 3.4 + 0.55 * hh, pz, tx, ty, pz, 0.21)
                if d2 < dd: dd = d2
                if dd < d: d = dd; m = 13
                if (x - tx) ** 2 + (y - ty) ** 2 + (z - pz) ** 2 < 49.0:
                    for f in range(11):
                        an = 6.2832 * (f + hash3(i, f, 9)) / 11.0
                        fl = 4.4 + 1.6 * hash3(i, f, 11)
                        cxn = math.cos(an); szn = math.sin(an)
                        mx = tx + cxn * fl * 0.55; my = ty + 1.3 - 0.9 * hash3(i, f, 15); mz = pz + szn * fl * 0.55
                        ex = tx + cxn * fl; ey = ty - 1.0 - 3.2 * hash3(i, f, 13); ez = pz + szn * fl
                        df = min(sd_capsule(x, y, z, tx, ty, pz, mx, my, mz, 0.55), sd_capsule(x, y, z, mx, my, mz, ex, ey, ez, 0.42))
                        df = max(df, (abs(y - (my + ey) * 0.5) - 1.6) * 0.0 + df)
                        df = max(df, abs(y - (my + (ey - my) * 0.5)) * 0.0 + df)
                        if df < d: d = df; m = 14
        if x > -50.0 and x < -8.0 and z > 224.0 and z < 252.0:
            # steps to the water, the wall, two tiled roofs
            for i in range(7):
                b = sd_box(x + 29.0, y - (0.2 + 0.42 * i) * 0.5 - WATER * 0.0, z - (228.0 + 0.6 * i + 12.0), 13.0 - 0.3 * i, (0.6 + 0.42 * i) * 0.5 + 0.6, 12.0)
                if b < d: d = b; m = 12
            b = sd_box(x + 29.0, y - 5.4, z - 240.0, 12.0, 2.4, 3.0)
            if b < d: d = b; m = 16
            b = max(abs(x + 29.0) * 0.30 + (y - 12.2) * 0.954, max(7.6 - y, abs(z - 240.0) - 4.3))
            if b < d: d = b; m = 17
            b = sd_box(x + 29.0, y - 10.0, z - 243.0, 3.6, 3.0, 3.0)
            if b < d: d = b; m = 16
            b = max(abs(x + 29.0) * 0.50 + (y - 16.8) * 0.866, max(12.6 - y, abs(z - 243.0) - 4.6))
            if b < d: d = b; m = 17
            b = sd_capsule(x, y, z, -29.0, 16.6, 243.0, -29.0, 18.4, 243.0, 0.22)
            if b < d: d = b; m = 18
    if z < 18.0:
        # ---- the line: bamboo, rope, cloth
        for k in range(POLES.shape[0]):
            b = sd_capsule(x, y, z, POLES[k, 0] + 0.05, -0.2, POLES[k, 1], POLES[k, 0], 2.45, POLES[k, 1], 0.034)
            if b < 0.2:
                b -= 0.006 * math.exp(-((((y + 0.1) % 0.34) - 0.17) / 0.02) ** 2)
            if b < d: d = b; m = 3
        if y > 1.7 and y < 2.4:
            for k in range(POLES.shape[0] - 1):
                for j in range(6):
                    t0 = j / 6.0; t1 = (j + 1) / 6.0
                    b = sd_capsule(x, y, z, POLES[k, 0] + (POLES[k + 1, 0] - POLES[k, 0]) * t0, ROPE + 0.012 - 0.64 * t0 * (1 - t0), POLES[k, 1] + (POLES[k + 1, 1] - POLES[k, 1]) * t0,
                                   POLES[k, 0] + (POLES[k + 1, 0] - POLES[k, 0]) * t1, ROPE + 0.012 - 0.64 * t1 * (1 - t1), POLES[k, 1] + (POLES[k + 1, 1] - POLES[k, 1]) * t1, 0.011)
                    if b < d: d = b; m = 4
        k, dc, al, u, hem = cloth_uv(x, y, z)
        if k >= 0 and dc < d:
            d = dc; m = 20 + int(PIECES[k, 9])
    if z > 21.0 and z < 25.5 and x > -6.5 and x < -2.2 and y < 2.4:
        # ---- the washing fire: three stones, the vat, the man
        for j in range(3):
            b = math.sqrt((x - FIREX - 0.42 * (j - 1)) ** 2 + ((y - 0.10) / 0.9) ** 2 + (z - FIREZ - 0.15 * (j % 2)) ** 2) - 0.16
            if b < d: d = b; m = 6
        ex = (x - FIREX) / 0.56; ey = (y - 0.66) / 0.40; ez = (z - FIREZ) / 0.56
        b = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.40
        b = max(b, y - 0.96)
        rr = math.sqrt((x - FIREX) ** 2 + (z - FIREZ) ** 2)
        rim = math.sqrt((rr - 0.47) ** 2 + (y - 0.97) ** 2) - 0.035
        if rim < b: b = rim
        if b < d: d = b; m = 5
        b = max(rr - 0.44, abs(y - 0.95) - 0.02)                                    # suds
        hp = math.sqrt((x - FIREX - 0.05) ** 2 + ((y - 0.99) / 0.5) ** 2 + (z - FIREZ) ** 2) - 0.25
        if hp < b: b = hp
        if b < d: d = b; m = 19
        # the man, bent a little to the work
        mx = MANX; mz = MANZ
        b = min(sd_capsule(x, y, z, mx + 0.02, 0.02, mz - 0.10, mx + 0.0, 0.55, mz - 0.07, 0.058),
                sd_capsule(x, y, z, mx - 0.06, 0.02, mz + 0.10, mx - 0.02, 0.55, mz + 0.07, 0.058))
        b = min(b, min(sd_capsule(x, y, z, mx + 0.02, 0.03, mz - 0.10, mx - 0.10, 0.02, mz - 0.11, 0.04),
                       sd_capsule(x, y, z, mx - 0.06, 0.03, mz + 0.10, mx - 0.18, 0.02, mz + 0.09, 0.04)))
        to = sd_capsule(x, y, z, mx - 0.01, 1.04, mz, mx - 0.10, 1.40, mz, 0.135)
        b = min(b, to)
        b = min(b, sd_capsule(x, y, z, mx - 0.15, 1.50, mz, mx - 0.19, 1.60, mz, 0.090))
        b = min(b, sd_capsule(x, y, z, mx - 0.12, 1.42, mz - 0.17, mx - 0.36, 1.20, mz - 0.14, 0.045))
        b = min(b, sd_capsule(x, y, z, mx - 0.36, 1.20, mz - 0.14, mx - 0.60, 1.26, mz - 0.05, 0.038))
        b = min(b, sd_capsule(x, y, z, mx - 0.12, 1.40, mz + 0.17, mx - 0.34, 1.12, mz + 0.12, 0.045))
        b = min(b, sd_capsule(x, y, z, mx - 0.34, 1.12, mz + 0.12, mx - 0.55, 1.12, mz + 0.02, 0.038))
        if b < d: d = b; m = 7
        dxm = x - mx + 0.005; dzm = z - mz
        rm = math.sqrt(dxm * dxm + dzm * dzm); th = math.atan2(dzm, dxm)
        tt = min(max((y - 0.50) / 0.56, 0.0), 1.0)
        b = max((rm - (0.215 - 0.05 * tt + 0.014 * math.sin(6.0 * th + 3.0 * tt))) * 0.8, max(0.50 - y, y - 1.07))
        if b < d: d = b; m = 8
        b = sd_capsule(x, y, z, mx - 0.13, 1.50, mz + 0.09, mx + 0.04, 1.40, mz + 0.14, 0.05)
        b = min(b, sd_capsule(x, y, z, mx + 0.04, 1.40, mz + 0.14, mx + 0.13, 1.06, mz + 0.12, 0.055))
        if b < d: d = b; m = 9
        b = math.sqrt((x - mx + 0.17) ** 2 + (y - 1.635) ** 2 + (z - mz) ** 2) - 0.088
        if b < d: d = b; m = 11
        b = sd_capsule(x, y, z, mx - 0.46, 1.62, mz - 0.02, FIREX + 0.08, 0.90, FIREZ, 0.017)
        if b < d: d = b; m = 3
    if z > 70.0 and z < 95.0 and x > 4.0 and x < 18.0 and y < 3.0:
        # ---- the ferry
        ex = (x - 10.5) / 3.6; ey = (y - WATER - 0.05) / 0.36; ez = (z - 82.0) / 0.55
        b = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.36
        b = max(b, -(y - WATER + 0.25))
        if b < d: d = b; m = 40
        for j in range(3):
            px = 8.4 + 1.0 * j
            b = min(sd_capsule(x, y, z, px, WATER + 0.3, 82.0, px, WATER + 0.85, 82.0, 0.20),
                    math.sqrt((x - px) ** 2 + (y - WATER - 1.2) ** 2 + (z - 82.0) ** 2) - 0.13)
            if b < d: d = b; m = 41 + j
        b = min(sd_capsule(x, y, z, 12.3, WATER + 0.3, 82.0, 12.35, WATER + 1.75, 82.0, 0.17),
                sd_capsule(x, y, z, 13.2, WATER - 0.5, 82.0, 12.0, WATER + 2.6, 82.0, 0.045))
        if b < d: d = b; m = 44
    return d, m


@njit(cache=True, fastmath=True, error_model='numpy')
def _soft(ox, oy, oz, dx, dy, dz, maxt, k):
    res = 1.0; t = 0.03
    for i in range(70):
        h, m = scene(ox + dx * t, oy + dy * t, oz + dz * t)
        if h < 0.0006: return 0.0
        res = min(res, k * h / t)
        t += min(max(h, 0.012), 1.5)
        if t > maxt: break
    return max(res, 0.0)


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def trace(O, D, tmax, L1, LP):
    n = O.shape[0]
    T = np.full(n, 1e9, np.float32); M = np.zeros(n, np.int32); Nn = np.zeros((n, 3), np.float32)
    S1 = np.zeros(n, np.float32); S2 = np.zeros(n, np.float32); AO = np.ones(n, np.float32)
    for i in prange(n):
        ox = O[i, 0]; oy = O[i, 1]; oz = O[i, 2]; dx = D[i, 0]; dy = D[i, 1]; dz = D[i, 2]
        t = 0.02; hit = False; mm = 0
        for s in range(400):
            d, m = scene(ox + dx * t, oy + dy * t, oz + dz * t)
            if d < 0.0005 * t + 0.0003:
                hit = True; mm = m; break
            t += d
            if t > tmax: break
        if not hit: continue
        px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
        e = 0.0006 * t + 0.0010
        nx = scene(px + e, py, pz)[0] - scene(px - e, py, pz)[0]
        ny = scene(px, py + e, pz)[0] - scene(px, py - e, pz)[0]
        nz = scene(px, py, pz + e)[0] - scene(px, py, pz - e)[0]
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nl; ny /= nl; nz /= nl
        T[i] = t; M[i] = mm; Nn[i, 0] = nx; Nn[i, 1] = ny; Nn[i, 2] = nz
        occ = 0.0; sca = 1.0
        for k in range(5):
            hh = 0.02 + 0.07 * k * (1.0 + 0.02 * t)
            dd = scene(px + nx * hh, py + ny * hh, pz + nz * hh)[0]
            occ += (hh - dd) * sca; sca *= 0.75
        AO[i] = min(max(1.0 - 2.2 * occ, 0.0), 1.0)
        off = 0.02 + 0.0006 * t
        sgn = 1.0
        if mm >= 20 and mm < 30 and nx * L1[0] + ny * L1[1] + nz * L1[2] < 0:
            sgn = -1.5                                   # thin cloth: ask whether the sun reaches its other face
        qx = px + sgn * nx * off; qy = py + sgn * ny * off; qz = pz + sgn * nz * off
        S1[i] = _soft(qx, qy, qz, L1[0], L1[1], L1[2], 60.0, 22.0)
        qx = px + nx * off; qy = py + ny * off; qz = pz + nz * off
        lx = LP[0] - qx; ly = LP[1] - qy; lz = LP[2] - qz
        ll = math.sqrt(lx * lx + ly * ly + lz * lz)
        if ll < 6.0:
            S2[i] = _soft(qx, qy, qz, lx / ll, ly / ll, lz / ll, ll - 0.05, 14.0)
    return T, M, Nn, S1, S2, AO


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def cloth_coords(P3):
    n = P3.shape[0]
    out = np.zeros((n, 4), np.float32)
    for i in prange(n):
        k, dc, al, u, hem = cloth_uv(P3[i, 0], P3[i, 1], P3[i, 2])
        out[i, 0] = k; out[i, 1] = al; out[i, 2] = u; out[i, 3] = hem
    return out


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def ground_depth(P3):
    n = P3.shape[0]
    out = np.zeros(n, np.float32)
    for i in prange(n):
        out[i] = WATER - ground_h(P3[i, 0], P3[i, 2])
    return out


L1 = np.array([-0.05, 0.965, 0.26]); L1 /= np.linalg.norm(L1)
LP = np.array([FIREX, FIREY, FIREZ])
SUN_C = np.array([1.00, 0.955, 0.86])
SKY_C = np.array([0.36, 0.50, 1.00])
SAND_C = np.array([0.95, 0.82, 0.60])
FIRE_C = np.array([1.00, 0.50, 0.14])


def sky(D, tile):
    """a hot noon sky: white at the horizon, cobalt overhead, cumulus standing over the far bank."""
    e = np.clip(D[..., 1], 0.0, 1.0)
    col = lerp(np.array([0.80, 0.86, 0.90]), np.array([0.40, 0.62, 0.90]), smooth(0.0, 0.16, e)[..., None])
    col = lerp(col, np.array([0.17, 0.40, 0.82]), smooth(0.10, 0.55, e)[..., None])
    # cumulus standing over the far bank: flat violet bases, tops piled up into the light
    az = np.arctan2(D[..., 0], D[..., 2])
    def tn(u, v, off=0.0):
        return map_coordinates(tile, [(v + off) % 1024, (u + off * 0.37) % 1024], order=1, mode='wrap')
    eb = 0.035 + 0.012 * (tn(az * 300.0, az * 0.0 + 5.0) - 0.5)
    top = eb + 0.02
    for (a0, hh, ww) in ((-0.36, 0.10, 0.09), (-0.20, 0.06, 0.06), (-0.02, 0.235, 0.115), (0.10, 0.15, 0.07), (0.215, 0.07, 0.05),
                         (0.35, 0.13, 0.085), (0.47, 0.08, 0.06)):
        top = top + 0.55 * hh * np.exp(-((az - a0) / ww) ** 2)
    bil = tn(az * 700.0, D[..., 1] * 1100.0, 31.0) - 0.5 + 0.45 * (tn(az * 2400.0, D[..., 1] * 3800.0, 99.0) - 0.5)
    hgt = (D[..., 1] - eb) / np.maximum(top - eb, 1e-3)
    body = smooth(1.0, 0.86, hgt + 0.42 * bil * (0.3 + hgt)) * smooth(-0.004, 0.004, D[..., 1] - eb) * smooth(0.030, 0.045, top - eb)
    light = np.clip(0.12 + 1.0 * hgt + 0.55 * bil, 0, 1)
    cc = lerp(np.array([0.60, 0.62, 0.84]), np.array([1.0, 0.985, 0.93]), smooth(0.25, 0.75, light)[..., None])
    cc = lerp(cc, np.array([0.80, 0.83, 0.90]), (smooth(0.25, 0.0, hgt) * 0.55)[..., None])
    return lerp(col, cc, (body * 0.97)[..., None])


def shade(P3, Nn, M, S1, S2, AO, D, T, tile):
    def wn(a, b, scale):
        return map_coordinates(tile, [(b / scale * 97.0) % 1024, (a / scale * 97.0) % 1024], order=1, mode='wrap')
    x, y, z = P3[..., 0], P3[..., 1], P3[..., 2]
    n = len(x)
    alb = np.zeros((n, 3), np.float32); wet = np.zeros(n, np.float32); trans = np.zeros(n, np.float32)
    n1 = wn(x, z, 3.0); n2 = wn(x * 4.0 + y, z * 4.0 - y, 1.0); n3 = wn(x + 2 * y, z, 14.0)
    up = Nn[..., 1]
    # sand
    k = M == 1
    c = lerp(np.array([0.70, 0.58, 0.40]), np.array([0.80, 0.69, 0.50]), n3[..., None]) * (0.95 + 0.08 * n2[..., None])
    edge = y - WATER
    c = lerp(c, np.array([0.50, 0.40, 0.27]), (smooth(0.22, 0.03, edge) * 0.8)[..., None])
    c = np.where((z > 215)[..., None], np.array([0.80, 0.70, 0.52]), c)
    alb[k] = c[k]; wet[k] = smooth(0.12, 0.0, edge)[k] * 0.8
    k = M == 3; alb[k] = lerp(np.array([0.50, 0.40, 0.18]), np.array([0.72, 0.62, 0.30]), n2[..., None])[k]
    k = M == 4; alb[k] = np.array([0.55, 0.46, 0.28])
    k = M == 5; alb[k] = lerp(np.array([0.07, 0.05, 0.06]), np.array([0.22, 0.12, 0.09]), smooth(0.75, 1.0, y)[..., None])[k]; wet[k] = 0.3
    k = M == 6; alb[k] = np.array([0.20, 0.17, 0.17])
    k = M == 7; alb[k] = np.array([0.36, 0.20, 0.14])
    k = M == 8
    c = np.tile(np.array([0.93, 0.91, 0.86], np.float32), (n, 1))
    c = lerp(c, np.array([0.80, 0.56, 0.14]), (smooth(0.50, 0.505, y) * smooth(0.56, 0.55, y))[..., None])
    alb[k] = c[k]; trans[k] = 0.3
    k = M == 9; alb[k] = np.array([0.78, 0.13, 0.09]); trans[k] = 0.3
    k = M == 11; alb[k] = np.array([0.05, 0.045, 0.06])
    k = M == 12; alb[k] = lerp(np.array([0.62, 0.55, 0.50]), np.array([0.80, 0.74, 0.66]), n2[..., None])[k]
    k = M == 13; alb[k] = np.array([0.40, 0.36, 0.33])
    k = M == 14; alb[k] = lerp(np.array([0.10, 0.26, 0.08]), np.array([0.22, 0.40, 0.10]), n2[..., None])[k]; trans[k] = 0.8
    k = M == 15; alb[k] = lerp(np.array([0.05, 0.17, 0.07]), np.array([0.12, 0.28, 0.09]), n2[..., None])[k]; trans[k] = 0.3
    k = M == 16
    c = lerp(np.array([0.84, 0.82, 0.78]), np.array([0.94, 0.92, 0.86]), n2[..., None])
    door = (np.abs(x + 29.0) < 1.2) & (y < 6.3) & (z < 238)
    c = np.where(door[..., None], np.array([0.10, 0.06, 0.08]), c)
    alb[k] = c[k]
    k = M == 17; alb[k] = lerp(np.array([0.50, 0.17, 0.10]), np.array([0.74, 0.32, 0.16]), n2[..., None])[k]
    k = M == 18; alb[k] = np.array([0.85, 0.62, 0.22]); wet[k] = 0.6
    k = M == 19; alb[k] = np.array([0.92, 0.92, 0.90])
    k = M == 40; alb[k] = np.array([0.13, 0.09, 0.09])
    for j, cc in enumerate([(0.90, 0.88, 0.84), (0.70, 0.13, 0.10), (0.86, 0.66, 0.20), (0.28, 0.15, 0.12)]):
        k = M == 41 + j; alb[k] = np.array(cc)
    # cloth spread on the sand: gold across each end, a narrow line down one selvedge
    k = M == 30
    if k.any():
        c = np.tile(np.array([0.96, 0.95, 0.91], np.float32), (n, 1))
        for r in range(len(ROWS)):
            x0, x1, z0, z1 = ROWS[r]
            ins = k & (x > x0) & (x < x1) & (z > z0) & (z < z1)
            ez = np.minimum(z - z0, z1 - z); ex = x - x0
            g = np.maximum(smooth(0.135, 0.125, ez) * smooth(0.015, 0.025, ez), smooth(0.062, 0.054, ex) * smooth(0.010, 0.018, ex))
            ln = smooth(0.100, 0.094, ex) * smooth(0.076, 0.082, ex) * (r % 2)
            cr = lerp(c, np.array([0.80, 0.57, 0.15]), g[..., None])
            cr = lerp(cr, np.array([(0.10, 0.33, 0.20), (0.58, 0.12, 0.09), (0.12, 0.12, 0.30)][r % 3]), ln[..., None])
            c = np.where(ins[..., None], cr, c)
        alb[k] = c[k]
    # cloth on the line: kasavu at the hem and up each selvedge - wherever the cloth goes, it goes
    k = (M >= 20) & (M < 30)
    gold = np.zeros(n, np.float32)
    if k.any():
        cu = cloth_coords(np.ascontiguousarray(P3[k], dtype=np.float64))
        pk = cu[:, 0].astype(int); al = cu[:, 1]; u = cu[:, 2]; hem = cu[:, 3]
        kind = PIECES[np.clip(pk, 0, len(PIECES) - 1), 9].astype(int)
        sg = PIECES[np.clip(pk, 0, len(PIECES) - 1), 0].astype(int)
        wid = (PIECES[np.clip(pk, 0, len(PIECES) - 1), 2] - PIECES[np.clip(pk, 0, len(PIECES) - 1), 1]) * np.linalg.norm(POLES[sg + 1] - POLES[sg], axis=1)
        bb = hem - u; sa = np.minimum(al, 1 - al) * wid
        hw = np.where(kind == 1, 0.105, 0.075)
        g = smooth(hw, hw - 0.008, bb) * smooth(0.008, 0.016, bb)
        g = np.maximum(g, smooth(0.036, 0.030, sa) * smooth(0.005, 0.010, sa))
        st = smooth(hw + 0.032, hw + 0.027, bb) * smooth(hw + 0.012, hw + 0.017, bb) * (kind > 0)
        c = np.tile(np.array([0.96, 0.945, 0.90], np.float32), (len(al), 1))
        c = lerp(c, np.array([0.82, 0.58, 0.15]), g[..., None])
        c = lerp(c, np.array([[0, 0, 0], [0.10, 0.33, 0.20], [0.58, 0.12, 0.09]])[kind], st[..., None])
        alb[k] = c
        trans[k] = 0.62 * (1 - 0.92 * g) * (1 - 0.5 * st)
        gold[k] = g

    V = -D
    ndl = (Nn * L1).sum(-1)
    lv = LP - P3; ld = np.linalg.norm(lv, axis=-1); lv = lv / ld[..., None]
    fire_i = np.clip((Nn * lv).sum(-1) * 0.8 + 0.2, 0, 1) * S2 * 1.2 / (ld * ld + 0.15)
    amb = SKY_C * (0.45 + 0.55 * np.clip(up, 0, 1)[..., None]) * 0.70 + SAND_C * np.clip(0.25 - 0.75 * up, 0, 1)[..., None] * 0.75
    amb = amb * (0.35 + 0.65 * AO[..., None]) * 1.45
    dif = np.clip(ndl, 0, 1) * S1
    thr = np.clip(-ndl * 0.75 + 0.25, 0, 1) * (0.28 + 0.72 * S1) * trans + 0.22 * trans * np.clip(ndl, 0, 1) * (1 - S1)
    col = alb * (amb + SUN_C * 1.55 * dif[..., None] + FIRE_C * fire_i[..., None])
    col += alb * np.array([1.0, 0.90, 0.66]) * (SUN_C * 1.55 * thr[..., None])
    hv = L1 + V; hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
    sp = np.clip((Nn * hv).sum(-1), 0, 1)
    col += ((wet + 2.5 * gold) * sp ** 40 * S1 * 1.5)[..., None] * SUN_C
    col += (gold * np.clip(dif + thr, 0, 1) ** 2 * 0.25)[..., None] * np.array([1.0, 0.8, 0.3])
    return col, gold


def draw(P):
    W, H, S, rng = P.W, P.H, P.S, P.rng
    fq = np.hypot(*np.meshgrid(np.fft.fftfreq(1024), np.fft.fftfreq(1024)))
    spec = np.fft.fft2(np.random.default_rng(SEED + 1).standard_normal((1024, 1024))) * np.where(fq > 0, 1.0 / np.maximum(fq, 1 / 200.0) ** 1.4, 0) * (fq < 0.3)
    tile = np.real(np.fft.ifft2(spec)); tile = np.clip(0.5 + 0.22 * (tile - tile.mean()) / tile.std(), 0, 1).astype(np.float32)

    RW = min(W, 1400); RH = int(RW * H / W)
    TX = 0.4587; TY = TX * H / W
    fwd = np.array([0.0, -0.28 * TY, 1.0]); fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
    upv = np.cross(fwd, right)
    jj, ii = np.mgrid[0:RH, 0:RW]
    nx = ((ii + 0.5) / RW * 2 - 1) * TX; ny = (1 - (jj + 0.5) / RH * 2) * TY
    D = fwd[None, None, :] + nx[..., None] * right + ny[..., None] * upv
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    D = D.reshape(-1, 3)
    cam = np.array([CAMX, CAMY, CAMZ])
    O = np.tile(cam, (len(D), 1))
    TM = 420.0
    t0 = time.time()
    T, M, Nn, S1, S2, AO = trace(O, D, TM, L1, LP)
    print(f"  traced {time.time() - t0:.1f}s", flush=True)
    hit = M > 0
    P3 = O + D * np.minimum(T, TM)[:, None]
    skyc = sky(D, tile)
    col = skyc.copy()
    gold = np.zeros(len(D), np.float32)
    c, g = shade(P3[hit], Nn[hit], M[hit], S1[hit], S2[hit], AO[hit], D[hit], T[hit], tile)
    col[hit] = c; gold[hit] = g
    haze = lambda t: 1 - np.exp(-t * 0.0016)
    hzc = np.array([0.74, 0.82, 0.90])
    # the river: sand seen through it near the edge, the sky and the far bank lying on it further out
    wk = np.nonzero(M == 10)[0]
    if len(wk):
        dep = ground_depth(np.ascontiguousarray(P3[wk]))
        rp = 0.22 * smooth(120, 10, T[wk])
        r1 = map_coordinates(tile, [(P3[wk, 2] * 55.0) % 1024, (P3[wk, 0] * 14.0) % 1024], order=1, mode='wrap') - 0.5
        r2 = map_coordinates(tile, [(P3[wk, 2] * 17.0 + 400) % 1024, (P3[wk, 0] * 5.0) % 1024], order=1, mode='wrap') - 0.5
        Nw = np.stack([0.10 * r1 * rp, np.ones(len(wk)), 0.5 * (r1 + r2) * rp + 0.04 * r2], 1); Nw /= np.linalg.norm(Nw, axis=1, keepdims=True)
        Dr = D[wk] - 2 * (D[wk] * Nw).sum(-1, keepdims=True) * Nw
        Dr[:, 1] = np.abs(Dr[:, 1]) + 0.22 * smooth(200, 40, T[wk]); Dr /= np.linalg.norm(Dr, axis=1, keepdims=True)   # wind on the water tips it to the higher sky
        Or = P3[wk] + np.array([0, 0.01, 0])
        T2, M2, N2, S12, S22, AO2 = trace(Or, Dr, TM, L1, LP)
        rc = sky(Dr, tile)
        h2 = M2 > 0
        if h2.any():
            c2, _ = shade((Or + Dr * T2[:, None])[h2], N2[h2], M2[h2], S12[h2], S22[h2], AO2[h2], Dr[h2], T2[h2], tile)
            rc[h2] = lerp(c2 * 0.85, hzc, haze(T2[h2] + T[wk][h2])[..., None])
        cosv = np.clip(-(D[wk] * Nw).sum(-1), 0, 1)
        fr = 0.03 + 0.80 * (1 - cosv) ** 5
        bed = np.array([0.80, 0.70, 0.50]) * (S1[wk][:, None] * 0.6 + 0.4)
        body = lerp(bed, np.array([0.06, 0.34, 0.36]), (1 - np.exp(-dep * 2.2))[..., None])
        body = lerp(body, np.array([0.02, 0.15, 0.40]), (1 - np.exp(-dep * 0.9))[..., None])
        col[wk] = body * (1 - fr[:, None]) + rc * fr[:, None]
        col[wk] += (np.clip((Dr * L1).sum(-1), 0, 1) ** 900 * 5.0)[:, None] * SUN_C
    f = np.where(hit, haze(np.minimum(T, TM)), 0.0)
    col = lerp(col, hzc * 1.25, f[..., None])
    col = 1 - np.exp(-1.28 * col)                                        # the eye adjusts to the glare
    mcol = col.mean(-1, keepdims=True)
    col = np.clip(mcol + (col - mcol) * 1.22, 0, 1)
    col = col.reshape(RH, RW, 3).astype(np.float32)

    def up(a, mode=Image.BICUBIC):
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((W, H), mode))
    P.ref[:] = np.stack([up(col[..., k]) for k in range(3)], -1).clip(0, 1)
    Mi = M.reshape(RH, RW); Tn = T.reshape(RH, RW); Nr = Nn.reshape(RH, RW, 3)
    P.reg[:] = np.asarray(Image.fromarray(Mi.astype(np.int32)).resize((W, H), Image.NEAREST))
    P.mat = P.reg.copy()
    P.depth = up(np.minimum(Tn, TM), Image.BILINEAR)
    P.gold = up(gold.reshape(RH, RW), Image.BILINEAR)
    upw = np.array([0.0, 1.0, 0.0])
    tg = np.cross(Nr, np.cross(np.broadcast_to(upw, Nr.shape), Nr))
    flat = np.abs(Nr[..., 1]) > 0.9
    sxv = (tg * right).sum(-1); syv = -(tg * upv).sum(-1)
    ang = np.arctan2(syv, sxv + 1e-9)
    xs = (ii + 0.5) / RW * 2 - 1; ys = (jj + 0.5) / RH * 2 - 1
    persp = np.arctan2(ys + 0.28, xs * W / H + 1e-9)                      # on the sand the brush runs toward the vanishing point
    ang = np.where(flat & (Mi == 30), persp, ang)
    ang = np.where(flat & (Mi != 30), 0.03 + 0.25 * (fbm(RH, RW, 200 * RW / 2400, rng, 2) - 0.5), ang)
    ang = np.where((Mi == 14) | (Mi == 15), fbm(RH, RW, 40 * RW / 2400, rng, 2) * 9.0, ang)
    cl = sky(D, tile).reshape(RH, RW, 3)
    cloudy = (Mi == 0) & (lum(cl) > 0.80)
    ang = np.where(Mi == 0, 0.05 + 0.3 * (fbm(RH, RW, 300 * RW / 2400, rng, 2) - 0.5), ang)
    ang = np.where(cloudy, fbm(RH, RW, 90 * RW / 2400, rng, 2) * 7.0, ang)
    P.ang[:] = np.asarray(Image.fromarray(ang.astype(np.float32)).resize((W, H), Image.NEAREST))
    lf = np.ones((RH, RW), np.float32)
    lf[Mi == 0] = 2.0; lf[cloudy] = 0.7; lf[Mi == 1] = 1.8; lf[Mi == 10] = 2.4; lf[(Mi >= 20) & (Mi <= 30)] = 2.4
    lf[(Mi == 14) | (Mi == 15)] = 0.5
    P.lenf[:] = up(lf, Image.NEAREST)

    def project(p):
        v = np.asarray(p) - cam
        zc = v @ fwd
        return (0.5 + 0.5 * (v @ right) / zc / TX) * W, (0.5 - 0.5 * (v @ upv) / zc / TY) * H
    P.project = project
    det = np.zeros((RH, RW), np.float32)
    for mid, v in ((5, 0.9), (6, 0.8), (7, 1.0), (8, 1.0), (9, 1.0), (11, 1.0), (19, 0.9), (16, 0.7), (17, 0.7), (18, 0.8), (40, 0.8),
                   (41, 0.9), (42, 0.9), (43, 0.9), (44, 0.9), (3, 0.5), (4, 0.5), (20, 0.25), (21, 0.25), (22, 0.25), (13, 0.5)):
        det[Mi == mid] = v
    P.detail[:] = up(det, Image.BILINEAR)
    P.soft = np.clip(0.8 * (P.mat == 0) + 0.6 * smooth(150, 300, P.depth) * (P.mat != 0) + 0.5 * (P.mat == 10) * smooth(30, 90, P.depth), 0, 1)
    P.reg[P.edge < 14 * S] = -1


def paint(P):
    W, H, S, rng = P.W, P.H, P.S, P.rng
    ref = P.ref
    L = lum(ref)
    t0 = time.time()
    say = lambda s, n: print(f"  {s:<14s}{n:>7d} strokes  {time.time() - t0:5.1f}s", flush=True)

    dead = blur(ref, 28 * S)
    m = dead.mean(-1, keepdims=True)
    dead = (m + (dead - m) * 0.75) * 0.90
    say("underpaint", P.layer(dead, 56, spacing=0.7, order='random', maxlen=5, tol=0.6, cjit=0.03, streak=0.03,
                              opac=0.85, thick=0.2, thickv=0.1, dry=0.6, drysens=0.6, pickup=0.0, wander=0.2))
    say("block-in", P.layer(blur(ref, 9 * S), 28, spacing=0.62, maxlen=5, tol=0.24, cjit=0.04, streak=0.04,
                            opac=0.96, thick=0.5, thickv=0.3, dry=0.3, drysens=0.25, pickup=0.12, wander=0.14))
    P.blend(0.30 + 0.7 * P.soft, length=22, soften=5)
    say("modelling", P.layer(blur(ref, 4.0 * S), 14, spacing=0.7, err=0.07, maxlen=5.5, tol=0.18, cjit=0.04, streak=0.04,
                             opac=0.96, thick=0.6, thickv=0.3, dry=0.35, drysens=0.25, pickup=0.14, wander=0.10))
    P.blend(0.12 + 0.6 * P.soft, length=10, soften=2.0)
    say("drawing", P.layer(blur(ref, 1.8 * S), 7.0, spacing=0.75, err=0.09, maxlen=6, tol=0.14, cjit=0.03, streak=0.03,
                           opac=0.97, thick=0.6, thickv=0.3, dry=0.35, drysens=0.25, pickup=0.10, wander=0.08))
    say("small", P.layer(blur(ref, 0.9 * S), 3.4, spacing=0.8, err=0.14, mask=0.2 + 0.8 * smooth(0.1, 0.5, P.detail),
                         maxlen=5, tol=0.12, cjit=0.02, streak=0.02, opac=0.98, thick=0.5, dry=0.3, drysens=0.2,
                         pickup=0.06, wander=0.06))
    say("accents", P.layer(ref, 1.8, spacing=0.85, err=0.18, mask=smooth(0.35, 0.7, P.detail), maxlen=5, tol=0.10,
                           cjit=0.015, streak=0.015, opac=0.99, thick=0.45, dry=0.3, drysens=0.2, pickup=0.03, wander=0.04))
    P.blend(0.5 * P.soft * smooth(0.5, 0.2, P.detail), length=6, soften=1.2)

    # glaze: cobalt deepened in the water and in the shadows on the sand
    sat = ref.max(-1) - ref.min(-1)
    P.glaze(blur(smooth(0.50, 0.25, L) * smooth(0.12, 0.35, sat), 6 * S), [0.84, 0.90, 1.0], 0.30)

    # the lights: sunlit cloth and sand, loaded, laid along the fold and left
    solid = np.isin(P.mat, (1, 20, 21, 22, 30, 16, 19, 8)).astype(np.float32)
    lit = np.clip(ref * np.array([1.02, 1.01, 0.98]) + 0.01, 0, 1)
    say("lights", P.layer(lit, 7.0, spacing=1.0, mask=smooth(0.74, 0.90, L) * 0.5 * solid, maxlen=7, tol=0.10, cjit=0.015,
                          streak=0.02, opac=0.97, thick=1.2, thickv=0.3, dry=0.3, drysens=0.25, pickup=0.03, ridge=0.4))
    cloth = ((P.mat >= 20) & (P.mat <= 30)).astype(np.float32)
    say("knife", P.layer(np.clip(lit + 0.015, 0, 1), 6.0, spacing=1.4, mask=smooth(0.88, 0.97, L) * 0.45 * cloth, maxlen=8, tol=0.08,
                         cjit=0.01, streak=0.015, opac=1.0, thick=1.8, thickv=0.1, dry=0.45, drysens=0.3, pickup=0.0,
                         edge=0.10, ridge=0.7, wander=0.04, sizev=0.5))

    # gold, rationed: a few thick dabs where a fold turns to the light
    gl = P.gold * smooth(0.45, 0.75, L) * (P.mat >= 20) * (P.mat < 30)
    ys, xs = np.nonzero(gl > 0.3)
    if len(ys):
        w = gl[ys, xs]; idx = rng.choice(len(ys), size=min(70, len(ys)), replace=False, p=w / w.sum())
        for i in idx:
            P.dab(xs[i], ys[i], P.ang[ys[i], xs[i]] + np.pi / 2, rng.uniform(9, 20) * S, rng.uniform(2.0, 3.4) * S,
                  [1.0, rng.uniform(0.78, 0.88), rng.uniform(0.24, 0.38)], thick=2.6, dry=0.5, drysens=0.4, edge=0.1, ridge=1.0, streak=0.04)

    # smoke: a blue thread, leaning with the wind
    fx, fy = P.project((FIREX, 0.05, FIREZ)); mpx = (P.project((FIREX + 1.0, 0.05, FIREZ))[0] - fx)
    sp = spline([(fx - 0.5 * mpx, fy - 0.7 * mpx), (fx - 0.9 * mpx, fy - 1.9 * mpx), (fx - 2.0 * mpx, fy - 3.2 * mpx),
                 (fx - 4.2 * mpx, fy - 4.5 * mpx), (fx - 7.5 * mpx, fy - 5.4 * mpx)], 50)
    for j in range(5):
        off = np.array([rng.normal(0, 0.12), rng.normal(0, 0.10)]) * mpx
        P.stroke(sp[j * 3:] + off * np.linspace(0.2, 2.0, len(sp) - j * 3)[:, None], (0.10 + 0.07 * j) * mpx, [0.80, 0.84, 0.95], opac=0.30,
                 thick=0.2, dry=0.7, drysens=0.5, edge=0.9, streak=0.01, pickup=0.3)

    # the fire under the vat: last
    for k in range(9):
        x = fx + rng.uniform(-0.40, 0.40) * mpx
        hh = rng.uniform(0.16, 0.34) * mpx
        P.stroke([(x, fy - 0.02 * mpx), (x + rng.uniform(-0.05, 0.05) * mpx, fy - 0.5 * hh), (x + rng.uniform(-0.10, 0.02) * mpx, fy - hh)],
                 0.07 * mpx, [1.0, rng.uniform(0.36, 0.52), 0.06], thick=2.0, dry=0.3, drysens=0.2, edge=0.4)
    for k in range(6):
        x = fx + rng.uniform(-0.30, 0.30) * mpx
        hh = rng.uniform(0.10, 0.22) * mpx
        P.stroke([(x, fy - 0.02 * mpx), (x - 0.02 * mpx, fy - hh)], 0.04 * mpx, [1.0, 0.84, 0.30], thick=2.6, dry=0.2, drysens=0.1, edge=0.3)
    P.stroke([(fx - 0.05 * mpx, fy - 0.03 * mpx), (fx - 0.04 * mpx, fy - 0.15 * mpx)], 0.022 * mpx, [1.0, 0.98, 0.82], thick=3.0, dry=0.1, drysens=0.0)

    P.sign(W - 80 * S, H - 62 * S, 30 * S, colour=(0.60, 0.18, 0.09))


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0] != '-' else 1.0
    W, H = int(3200 * scale), int(2130 * scale)
    t0 = time.time()
    P = Painting(W, H, SEED)
    draw(P)
    print(f"drawing {time.time() - t0:.1f}s", flush=True)
    if '--drawing' in sys.argv:
        save(P.ref, __file__.replace('.py', '_drawing.png')); return
    paint(P)
    out = __file__.replace('.py', '.png') if scale == 1.0 else __file__.replace('.py', '_study.png')
    save(P.finish(), out)
    print(f"{out}  {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
