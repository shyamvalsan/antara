#!/usr/bin/env python3
"""
Kotka, 3 pm in December

    python kotka_3pm_december.py            # the picture, 3000 x 2250
    python kotka_3pm_december.py 0.5        # a study at half size

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
        c[0] = c0[0] * v + cjit * (h0 - hm) * (0.06 + l0)
        c[1] = c0[1] * v + cjit * (h1 - hm) * (0.06 + l0)
        c[2] = c0[2] * v + cjit * (h2 - hm) * (0.06 + l0)
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
#  A birch wood above the bay at Kotka, ten minutes after the sun has
#  gone.  Nothing is lit by the sun any more: the snow has the low fire
#  in the south-west in front of it and the violet dome over it.
#  Someone walked in to the red house, and has put the light on.
# =====================================================================
from scipy.ndimage import map_coordinates

LAT, LON, MINUTE = 60.466, 26.946, 15 * 60 + 0
SEED = int(LAT * 1e4) * 7919 + int(LON * 1e4) * 104729 + MINUTE

CAMX, CAMY, CAMZ = 0.0, 1.62, 0.0
TX, TY = 0.60, 0.45
ICE_Y = -2.75                                 # the bay
AZG = 0.40                                    # where the sun went down, radians right of the way we look
HX, HZ, HA = -12.0, 58.0, -0.67                # the house, and how it is turned
HC, HS = math.cos(HA), math.sin(HA)
HL, HD = 3.3, 2.3                             # half length, half depth
HY = -0.0011 * min(HZ, 46.0) ** 2 - 0.05                 # the snow it stands in
WALL = 2.6
RK = 0.75                                     # roof pitch
YR = HY + WALL + RK * HD + 0.10               # ridge
UW, YW = 1.0, HY + 1.50                       # the window
UD = -1.3                                     # the door

# birches: x, z, radius, lean x, lean z, height, base y, phase
_T = [(-2.3, 5.2, 0.15, 0.050, 0.004, 15.0), (-3.7, 7.6, 0.10, -0.035, 0.010, 12.0), (2.3, 7.4, 0.15, -0.045, 0.0, 14.5),
      (5.2, 9.0, 0.095, -0.065, 0.0, 12.0), (-0.1, 11.5, 0.12, 0.030, 0.0, 13.5), (1.5, 14.0, 0.15, -0.020, 0.01, 15.0),
      (-4.9, 13.0, 0.13, 0.065, 0.0, 14.0), (-8.6, 17.0, 0.16, 0.020, 0.0, 15.5), (3.2, 19.5, 0.09, 0.045, 0.0, 11.5),
      (-2.1, 21.5, 0.12, -0.040, 0.0, 13.5), (11.9, 23.0, 0.13, 0.030, 0.0, 14.5), (1.9, 25.5, 0.10, 0.020, 0.0, 12.5),
      (-11.6, 24.0, 0.14, -0.030, 0.0, 14.5), (6.9, 29.0, 0.11, -0.040, 0.0, 13.0), (-3.4, 41.0, 0.11, 0.025, 0.0, 12.5),
      (0.5, 35.0, 0.10, 0.030, 0.0, 12.0), (13.0, 36.0, 0.10, -0.030, 0.0, 12.0),
      (-6.2, 24.0, 0.115, 0.022, 0.0, 13.5), (-5.85, 33.0, 0.09, -0.028, 0.0, 12.0)]
NT = len(_T)
TREES = np.zeros((NT, 8))
_rt = np.random.default_rng(SEED + 7)
for _k, _t in enumerate(_T):
    TREES[_k, :6] = _t
    TREES[_k, 6] = -0.0011 * min(_t[1], 46.0) ** 2 - 0.35
    TREES[_k, 7] = _rt.uniform(0, 6.28)


def _axis(k, yy):
    return np.array([TREES[k, 0] + TREES[k, 3] * yy + 0.16 * math.sin(0.27 * yy + TREES[k, 7]) + 0.05 * math.sin(0.9 * yy + 2.0 * TREES[k, 7]),
                     TREES[k, 6] + yy, TREES[k, 1] + TREES[k, 4] * yy])

def _rad(k, yy):
    return TREES[k, 2] * (1 - 0.88 * min(max(yy / TREES[k, 5], 0), 1))

# the limbs that are thick enough to be form; the twigs are for the brush
_L = []; LIDX = np.zeros((NT, 2), np.int64); TLIMB = []
for _k in range(NT):
    h = TREES[_k, 5]
    nl = int(_rt.integers(4, 8))
    ys = np.sort(_rt.uniform(0.30 if _k == 0 else 0.36, 0.86, nl) ** 1.0) * h
    a0 = _rt.uniform(0, 6.28)
    LIDX[_k, 0] = len(_L); lst = []
    for j in range(nl):
        az = a0 + j * 2.4 + _rt.normal(0, 0.7)
        ph = math.radians(_rt.uniform(16, 44))
        ln = float(np.clip(_rt.uniform(0.25, 0.75) * (h - ys[j]), 1.0, 5.0))
        a = _axis(_k, ys[j])
        d = np.array([math.sin(ph) * math.cos(az), math.cos(ph), math.sin(ph) * math.sin(az)])
        r1 = (0.22 + 0.16 * _rt.random()) * _rad(_k, ys[j]) + 0.006
        pts = [a]
        bend = _rt.normal(0, 0.22, 3) + np.array([0, 0.22, 0])
        for q in range(3):
            pts.append(pts[-1] + d * ln * (0.30, 0.34, 0.36)[q])
            d = d + bend + _rt.normal(0, 0.10, 3); d /= np.linalg.norm(d)
        for q in range(3):
            _L.append((*pts[q], *pts[q + 1], r1 * (1.0, 0.72, 0.48)[q], 3.0 if q == 0 else 9.0))
        lst.append((pts, d, r1))
    LIDX[_k, 1] = len(_L); TLIMB.append(lst)
LIMBS = np.array(_L)

# the way someone walked in
_pz = np.array([2.0, 5.0, 8.0, 11.5, 15.0, 18.5, 22.0, 26.0, 31.0, 37.0, 43.0, 49.0, 54.0, 56.6, 62.0])
_px = np.array([1.0, 0.55, 0.95, 0.85, -0.5, -1.9, -3.4, -4.6, -6.6, -8.9, -10.2, -12.4, -13.8, -14.4, -15.0])
PATHX = CubicSpline(_pz, _px)(np.arange(0.0, 62.0, 0.1)).astype(np.float64)
STEP0, STRIDE, STEPEND = 2.5, 0.71, 56.2


@njit(cache=True, fastmath=True, error_model='numpy')
def path_x(z):
    f = z * 10.0
    if f < 0: f = 0.0
    if f > 617.0: f = 617.0
    i = int(f); t = f - i
    return PATHX[i] + t * (PATHX[i + 1] - PATHX[i])


@njit(cache=True, fastmath=True, error_model='numpy')
def shore_z(x):
    if x > 4.0:
        return 44.0 - 0.45 * x + 2.5 * (noise3(x * 0.13, 3.3, 1.7) - 0.5)
    return 42.2 + (4.0 - x) * 3.5


@njit(cache=True, fastmath=True, error_model='numpy')
def ground_h(x, z):
    zz = z
    if zz > 46.0: zz = 46.0
    xr = x * 0.978 + z * 0.208; zr = -x * 0.208 + z * 0.978
    # long low drifts the wind has laid across the field
    h = -0.0011 * zz * zz + 0.46 * (fbm3(xr * 0.05, 0.37, zr * 0.23, 3) - 0.5) + 0.040 * (fbm3(xr * 0.20 + 5.0, 1.9, zr * 1.0, 3) - 0.5)
    # a hollow round every trunk, and a little tail of drift behind it
    for k in range(NT):
        dx = x - TREES[k, 0]; dz = z - TREES[k, 1]
        if dx > -3.0 and dx < 2.0 and dz > -2.0 and dz < 2.0:
            rt = TREES[k, 2]
            s = 0.42 + 1.6 * rt
            h -= (0.17 + 0.5 * rt) * math.exp(-(dx * dx + dz * dz) / (s * s))
            h += 0.07 * math.exp(-((dx + 1.1) / 0.9) ** 2 - (dz / 0.42) ** 2)
    # footprints
    if z > 1.5 and z < STEPEND + 0.6:
        if abs(x - path_x(z)) < 0.55:
            k0 = int((z - STEP0) / STRIDE + 0.5)
            for k in range(k0 - 1, k0 + 2):
                if k < 0: continue
                zk = STEP0 + STRIDE * k + 0.16 * (hash3(k, 3, 7) - 0.5)
                if zk > STEPEND: continue
                sl = (path_x(zk + 0.3) - path_x(zk - 0.3)) / 0.6
                nrm = math.sqrt(1.0 + sl * sl)
                sd = 0.10
                if k % 2 == 1: sd = -0.10
                xk = path_x(zk) + sd + 0.07 * (hash3(k, 5, 9) - 0.5)
                dx = x - xk; dz = z - zk
                al = (dx * sl + dz) / nrm; ac = (dx - dz * sl) / nrm
                wd = 0.066 + 0.020 * max(-1.0, min(1.0, al / 0.16))               # a boot: broader at the ball than the heel
                q = math.sqrt((al / 0.165) ** 2 + (ac / wd) ** 2)
                h -= (0.060 + 0.025 * hash3(k, 9, 2)) * sstep(1.30, 0.30, q)
                h += 0.007 * math.exp(-((q - 1.45) / 0.40) ** 2) * max(0.0, min(1.0, 0.5 + al / 0.2))
    s = sstep(0.0, 8.0, z - shore_z(x))
    return h * (1.0 - s) + (ICE_Y - 0.4) * s


@njit(cache=True, fastmath=True, error_model='numpy')
def scene(x, y, z):
    # ---- snow, and the ice of the bay
    d = (y - ground_h(x, z)) * 0.62; m = 1
    di = y - ICE_Y
    if di < d: d = di; m = 10
    # ---- birches: leaning, tapering, each with its few real limbs
    for k in range(NT):
        yy = y - TREES[k, 6]
        ax = TREES[k, 0] + TREES[k, 3] * yy + 0.16 * math.sin(0.27 * yy + TREES[k, 7]) + 0.05 * math.sin(0.9 * yy + 2.0 * TREES[k, 7])
        az = TREES[k, 1] + TREES[k, 4] * yy
        dx = x - ax; dz = z - az
        rr = math.sqrt(dx * dx + dz * dz)
        if rr - 5.6 > d: continue
        h = TREES[k, 5]
        t = yy / h
        if t < 0: t = 0.0
        if t > 1: t = 1.0
        r = TREES[k, 2] * (1.0 - 0.88 * t) * (1.0 + 0.32 * math.exp(-max(yy, 0.0) / 0.35))
        dt = (rr - r) * 0.85
        if yy > h: dt = max(dt, (yy - h) * 0.85)
        if dt < d: d = dt; m = 3
        if True:                      # no height shortcut: a long step from below must not pass through a limb
            for j in range(LIDX[k, 0], LIDX[k, 1]):
                dl = sd_capsule(x, y, z, LIMBS[j, 0], LIMBS[j, 1], LIMBS[j, 2], LIMBS[j, 3], LIMBS[j, 4], LIMBS[j, 5], LIMBS[j, 6])
                if dl < d: d = dl; m = int(LIMBS[j, 7])
    # ---- the red house
    ex = x - HX; ez = z - HZ
    bd = max(max(abs(ex), abs(ez)) - 4.6, y - (YR + 1.4))
    if bd > 0.3:
        if bd < d: d = bd
    else:
        u = HC * ex + HS * ez; w = -HS * ex + HC * ez
        f = (y - YR + RK * abs(w)) / math.sqrt(1.0 + RK * RK)
        wl = max(sd_box(u, y - (HY + 1.5), w, HL, 4.0, HD), f + 0.02)
        if wl < d: d = wl; m = 4
        rf = max(max(f - 0.30, -(f + 0.04)), max(abs(u) - (HL + 0.40), abs(w) - (HD + 0.45))) - 0.05
        rf -= 0.035 * (noise3(u * 1.3, 2.0, w * 1.3) - 0.5)
        if rf < d: d = rf; m = 5
        ch = sd_box(u + 1.4, y - (YR + 0.1), w - 0.35, 0.27, 0.80, 0.27)
        if ch < d: d = ch; m = 6
        wn = sd_box(u - UW, y - YW, w + HD, 0.50, 0.55, 0.035)
        if wn < d: d = wn; m = 7
    # ---- spruce on the point, loaded with snow
    bd = max(max(x - 6.0, 49.0 - z), y - 15.5)
    if bd > 0.3:
        if bd < d: d = bd
    else:
        CS = 3.6
        ci = math.floor(x / CS); ck = math.floor(z / CS)
        ds = 1e9
        for a in range(-1, 2):
            for b in range(-1, 2):
                i = int(ci + a); k = int(ck + b)
                h1 = hash3(i, 23, k)
                if h1 > 0.92: continue
                cx = (i + 0.5 + 0.7 * (hash3(i, 29, k) - 0.5)) * CS; cz = (k + 0.5 + 0.7 * (hash3(i, 31, k) - 0.5)) * CS
                if cz < 53.0 or cx > 3.4 or cz > 110.0: continue
                if cx < -3.0 and cz < 67.5: continue
                if cz > shore_z(cx) - 2.5: continue
                hg = (5.5 + 12.0 * hash3(i, 37, k) ** 1.5) * (0.42 + 0.58 * sstep(4.0, -10.0, cx))
                R = 0.8 + 0.16 * hg
                y0 = -3.0
                t = (y - y0) / hg
                dx = x - cx; dz = z - cz
                rr = math.sqrt(dx * dx + dz * dz)
                if t < 0: t = 0.0
                if t < 1.0:
                    th = math.atan2(dz, dx)
                    saw = t * hg / 1.15 + 3.0 * h1
                    saw -= math.floor(saw)
                    rad = R * (1.0 - t) ** 0.85 * (0.60 + 0.40 * (1.0 - saw)) * (1.0 + 0.10 * math.sin(5.0 * th + 6.28 * h1)) + 0.05
                    dd = (rr - rad) * 0.75
                else:
                    dd = math.sqrt(rr * rr + (y - y0 - hg) ** 2)
                if dd < ds: ds = dd
        if ds < 1.2:
            ds += 0.16 - 0.45 * fbm3(x * 1.6, y * 1.6, z * 1.6, 3)
        ds *= 0.5
        if ds < d: d = ds; m = 8
    return d, m


# ---- the tracer: nearest surface, its normal, how open it is to the sky, and what each light can see of it
@njit(cache=True, fastmath=True, error_model='numpy')
def _soft(ox, oy, oz, dx, dy, dz, maxt, k):
    res = 1.0; t = 0.03
    for i in range(70):
        h, m = scene(ox + dx * t, oy + dy * t, oz + dz * t)
        if h < 0.0006: return 0.0
        res = min(res, k * h / t)
        t += min(max(h, 0.012), 0.6)
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
        for s in range(320):
            d, m = scene(ox + dx * t, oy + dy * t, oz + dz * t)
            if d < 0.0007 * t + 0.0003:
                hit = True; mm = m; break
            t += d
            if t > tmax: break
        if not hit: continue
        px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
        e = 0.0008 * t + 0.0012
        nx = scene(px + e, py, pz)[0] - scene(px - e, py, pz)[0]
        ny = scene(px, py + e, pz)[0] - scene(px, py - e, pz)[0]
        nz = scene(px, py, pz + e)[0] - scene(px, py, pz - e)[0]
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nl; ny /= nl; nz /= nl
        T[i] = t; M[i] = mm; Nn[i, 0] = nx; Nn[i, 1] = ny; Nn[i, 2] = nz
        occ = 0.0; sca = 1.0
        for k in range(5):
            hh = 0.03 + 0.13 * k
            dd = scene(px + nx * hh, py + ny * hh, pz + nz * hh)[0]
            occ += (hh - dd) * sca; sca *= 0.75
        AO[i] = min(max(1.0 - 1.5 * occ, 0.0), 1.0)
        qx = px + nx * 0.02; qy = py + ny * 0.02; qz = pz + nz * 0.02
        S1[i] = _soft(qx, qy, qz, L1[0], L1[1], L1[2], 45.0, 7.0)
        lx = LP[0] - qx; ly = LP[1] - qy; lz = LP[2] - qz
        ll = math.sqrt(lx * lx + ly * ly + lz * lz)
        S2[i] = _soft(qx, qy, qz, lx / ll, ly / ll, lz / ll, ll - 0.05, 18.0)
    return T, M, Nn, S1, S2, AO


L1 = np.array([math.sin(AZG), 0.075, math.cos(AZG)]); L1 /= np.linalg.norm(L1)      # the glow, low in front
_wu, _ww = UW, -HD - 2.6
LP = np.array([HX + HC * _wu - HS * _ww, YW - 0.45, HZ + HS * _wu + HC * _ww])       # the lamp, just outside its window
WINC = np.array([HX + HC * UW + HS * HD, YW, HZ + HS * UW - HC * HD])
LAMP_C = np.array([1.00, 0.56, 0.16])
GLOW_C = np.array([1.00, 0.50, 0.30])
SKY_C = np.array([0.185, 0.215, 0.560])
KT = 1.7

_SE = np.array([0.0, 0.030, 0.070, 0.125, 0.20, 0.30, 0.42, 0.62])
_SG = np.array([(0.90, 0.33, 0.20), (1.00, 0.52, 0.22), (1.00, 0.72, 0.38), (0.88, 0.82, 0.58), (0.60, 0.68, 0.70),
                (0.37, 0.40, 0.64), (0.215, 0.20, 0.46), (0.135, 0.115, 0.32)])
_SF = np.array([(0.62, 0.36, 0.44), (0.72, 0.45, 0.47), (0.72, 0.54, 0.57), (0.60, 0.56, 0.66), (0.45, 0.48, 0.68),
                (0.32, 0.34, 0.60), (0.20, 0.19, 0.45), (0.135, 0.115, 0.32)])


def ts(tile, u, v):
    return map_coordinates(tile, [np.mod(v, 1024), np.mod(u, 1024)], order=1, mode='wrap')


def lin(c):
    """from a colour as the eye will have it to the light that makes it."""
    return -np.log(1 - 0.96 * np.clip(c, 0, 1)) / KT


def sky(D, tile, shore=True):
    """the sky ten minutes after sunset, in any direction (as the eye has it)."""
    e = np.clip(D[..., 1], 0, 1)
    az = np.arctan2(D[..., 0], D[..., 2]) - AZG
    g = np.exp(-(az / 0.85) ** 2)
    cg = np.stack([np.interp(e, _SE, _SG[:, k]) for k in range(3)], -1)
    cf = np.stack([np.interp(e, _SE, _SF[:, k]) for k in range(3)], -1)
    col = lerp(cf, cg, g[..., None])
    # two or three long bars of cloud lying in the glow, lit from underneath
    b = ts(tile, az * 80.0 + 300.0, e * 2300.0 + 90.0)
    b2 = ts(tile, az * 80.0 + 300.0, (e - 0.010) * 2300.0 + 90.0)
    env = smooth(0.045, 0.085, e) * smooth(0.27, 0.15, e)
    mk = smooth(0.655, 0.75, b) * env; mk2 = smooth(0.655, 0.75, b2)
    plum = lerp(np.array([0.36, 0.30, 0.50]), np.array([0.56, 0.30, 0.38]), g[..., None])
    col = lerp(col, plum, (0.10 * mk)[..., None])
    if shore:   # the far shore of the bay: low, dark, a few islands
        es = 0.007 + 0.016 * smooth(0.40, 0.72, ts(tile, az * 260.0 + 40.0, 0 * az + 77.0)) + 0.004 * ts(tile, az * 1500.0, 0 * az + 300.0)
        ms = smooth(es + 0.0015, es - 0.0015, D[..., 1])
        col = lerp(col, lerp(np.array([0.20, 0.13, 0.30]), col, 0.22), ms[..., None])
    return np.clip(col, 0, 1)


def shade(P3, Nn, M, S1, S2, AO, D, T, tile):
    """what colour each visible point is, under this light (as light, before the eye)."""
    x, y, z = P3[..., 0], P3[..., 1], P3[..., 2]
    n = len(x)
    alb = np.zeros((n, 3), np.float32); sheen = np.zeros(n, np.float32); emit = np.zeros((n, 3), np.float32)
    up = Nn[..., 1]
    n1 = ts(tile, x * 31.0 + z * 9.0, z * 37.0 - x * 5.0)
    n2 = ts(tile, x * 140.0 + y * 60.0, z * 140.0 - y * 45.0)
    snow = lerp(np.array([0.86, 0.88, 0.93]), np.array([0.93, 0.93, 0.95]), n1[..., None])
    # snow
    k = M == 1
    alb[k] = snow[k]; sheen[k] = 0.50
    # ice, blown half bare
    k = M == 10
    bare = smooth(0.52, 0.68, ts(tile, x * 2.2 + 50.0, z * 0.5 + 9.0))
    alb[k] = lerp(snow, np.array([0.50, 0.60, 0.74]), (bare * 0.55)[..., None])[k]; sheen[k] = (0.60 + 0.3 * bare)[k]
    # birch bark: white, with its black marks
    k = M == 3
    th = np.arctan2(Nn[..., 2], Nn[..., 0])
    yy = y + 0.0011 * np.minimum(z, 46.0) ** 2
    c = lerp(np.array([0.80, 0.77, 0.72]), np.array([0.92, 0.89, 0.84]), n2[..., None])
    c = lerp(c, np.array([0.86, 0.72, 0.62]), (smooth(0.55, 0.8, ts(tile, th * 20.0 + x * 40.0, yy * 25.0)) * 0.35)[..., None])
    lent = smooth(0.72, 0.78, ts(tile, th * 16.0 + x * 37.0 + z * 11.0, yy * 330.0))
    scar = smooth(0.80, 0.84, ts(tile, th * 12.0 + z * 13.0 + x * 7.0, yy * 110.0 + x * 13.0))
    base = smooth(1.1, 0.10, yy - 0.9 * ts(tile, th * 30.0 + x * 20.0, z * 20.0 + yy * 6.0)) * smooth(0.36, 0.52, ts(tile, th * 75.0 + x * 50.0, yy * 45.0))
    scar = scar * smooth(0.38, 0.62, ts(tile, th * 5.0 + x * 9.0, yy * 9.0 + z * 4.0))
    near = smooth(9.0, 7.0, z)
    dk = np.clip(0.70 * lent + 0.75 * scar + 0.75 * base, 0, 1) * (1 - 0.35 * near)
    c = lerp(c, lerp(np.array([0.085, 0.07, 0.085]), np.array([0.20, 0.17, 0.21]), near[..., None]), dk[..., None])
    alb[k] = c[k]
    k = M == 9                                                          # young wood
    alb[k] = lerp(np.array([0.20, 0.11, 0.13]), np.array([0.34, 0.22, 0.22]), n2[..., None])[k]
    # the house: falu red boards, white corners and frames
    ex = x - HX; ez = z - HZ
    u = HC * ex + HS * ez; w = -HS * ex + HC * ez
    k = M == 4
    c = lerp(np.array([0.235, 0.070, 0.070]), np.array([0.315, 0.095, 0.085]), ts(tile, u * 260.0 + w * 260.0, y * 12.0)[..., None])
    front = w < -HD + 0.03
    white = (np.abs(np.abs(u) - HL) < 0.10) & front
    white |= (np.abs(np.abs(w) - HD) < 0.10) & (np.abs(u) > HL - 0.03)
    door = front & (np.abs(u - UD) < 0.46) & (y < HY + 2.02)
    c = np.where(white[..., None], np.array([0.58, 0.56, 0.60]), c)
    c = np.where(door[..., None], np.array([0.13, 0.08, 0.10]), c)
    c = np.where((y < HY + 0.35)[..., None], np.array([0.22, 0.20, 0.24]), c)       # stone footing
    alb[k] = c[k]
    k = M == 5
    alb[k] = snow[k]; sheen[k] = 0.15
    k = M == 6
    alb[k] = np.where((y > YR + 0.84)[..., None], snow, np.array([0.34, 0.13, 0.10]))[k]
    k = M == 7
    emit[k] = np.array([2.4, 1.35, 0.42])
    alb[k] = np.array([0.1, 0.08, 0.05])
    # spruce, and what it is carrying
    k = M == 8
    load = smooth(0.50, 0.78, up + 0.45 * (ts(tile, x * 60.0 + y * 30.0, z * 60.0) - 0.5))
    c = lerp(np.array([0.030, 0.055, 0.050]), np.array([0.055, 0.085, 0.065]), n2[..., None])
    alb[k] = lerp(c, snow, (load * 0.75)[..., None])[k]

    # --- light: no sun.  A cool dome overhead, a broad warm glow low in front, one lamp.
    V = -D
    ndl1 = (Nn * L1).sum(-1)
    lv = LP - P3; ld = np.linalg.norm(lv, axis=-1); lv = lv / ld[..., None]
    ndl2 = (Nn * lv).sum(-1)
    lamp_i = 1.15 / (0.55 + (ld / 6.5) ** 2)
    dome = 0.22 + 0.78 * (0.5 + 0.5 * up)
    amb = SKY_C * dome[..., None] * (0.25 + 0.75 * AO[..., None]) * 1.35
    amb = amb + np.array([0.10, 0.12, 0.27]) * ((0.5 - 0.5 * up) * AO)[..., None]            # the snow gives light back upward
    wrap = np.clip((ndl1 + 0.07) / 1.07, 0, 1)
    dif1 = wrap * (0.40 + 0.60 * S1) * (0.55 + 0.45 * AO)
    dif2 = np.clip(ndl2 * 0.85 + 0.15, 0, 1) * S2 * lamp_i * np.where(M == 4, 0.22, np.where(M == 5, 0.2, 1.0))
    col = alb * (amb + GLOW_C * 1.45 * dif1[..., None] + LAMP_C * dif2[..., None]) + emit
    # snow seen along its surface gives back the sky beyond it
    R = D - 2 * (D * Nn).sum(-1, keepdims=True) * Nn
    Rs = R.copy(); Rs[..., 1] = np.abs(Rs[..., 1]) + 0.05
    Rs /= np.linalg.norm(Rs, axis=-1, keepdims=True)
    fres = (1 - np.clip((V * Nn).sum(-1), 0, 1)) ** 3.2
    col += (sheen * fres * (0.4 + 0.6 * AO))[..., None] * lin(sky(Rs, tile, shore=False))
    return col


def draw(P):
    W, H, S, rng = P.W, P.H, P.S, P.rng
    fq = np.hypot(*np.meshgrid(np.fft.fftfreq(1024), np.fft.fftfreq(1024)))
    spec = np.fft.fft2(np.random.default_rng(SEED + 1).standard_normal((1024, 1024))) * np.where(fq > 0, 1.0 / np.maximum(fq, 1 / 200.0) ** 1.4, 0) * (fq < 0.3)
    tile = np.real(np.fft.ifft2(spec)); tile = np.clip(0.5 + 0.22 * (tile - tile.mean()) / tile.std(), 0, 1).astype(np.float32)

    RW = min(W, 1200); RH = int(RW * H / W)
    fwd = np.array([0.0, 0.06, 1.0]); fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
    upv = np.cross(fwd, right)
    cam = np.array([CAMX, CAMY, CAMZ])
    jj, ii = np.mgrid[0:RH, 0:RW]
    nx = ((ii + 0.5) / RW * 2 - 1) * TX; ny = (1 - (jj + 0.5) / RH * 2) * TY
    D = fwd[None, None, :] + nx[..., None] * right + ny[..., None] * upv
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    D = D.reshape(-1, 3)
    O = np.tile(cam, (len(D), 1))
    t0 = time.time()
    T, M, Nn, S1, S2, AO = trace(O, D, 170.0, L1, LP)
    print(f"  traced {time.time() - t0:.1f}s", flush=True)
    # the bay goes on further than it is worth marching: the rest of it is a plane
    far = (M == 0) & (D[:, 1] < -0.0012)
    T[far] = (ICE_Y - CAMY) / D[far, 1]; M[far] = 10; Nn[far] = (0, 1, 0); S1[far] = 1; S2[far] = 1; AO[far] = 1
    hit = M > 0
    P3 = O + D * np.where(hit, T, 0.0)[:, None]
    skyc = sky(D, tile)
    col = np.zeros_like(skyc)
    col[hit] = shade(P3[hit], Nn[hit], M[hit], S1[hit], S2[hit], AO[hit], D[hit], T[hit], tile)
    # cold air: distance takes things toward the colour of the sky low down behind them
    Dh = D.copy(); Dh[:, 1] = 0.035; Dh /= np.linalg.norm(Dh, axis=-1, keepdims=True)
    hz = lin(0.55 * sky(Dh, tile, shore=False) + 0.45 * np.array([0.30, 0.31, 0.58]))
    f = np.where(hit, 1 - np.exp(-np.where(hit, T, 0.0) * 0.0050), 0.0)
    col = lerp(col, hz, f[..., None])
    # the lamp in the air just outside the glass
    tc = np.clip(((WINC - O) * D).sum(-1), 0, np.where(hit, T, 1e4))
    dmin = np.linalg.norm(O + D * tc[:, None] - WINC, axis=-1)
    col += (0.07 * np.exp(-dmin / 1.3))[:, None] * LAMP_C
    col = 1 - np.exp(-KT * col)                                          # the eye adjusts
    col[~hit] = skyc[~hit]
    mcol = col.mean(-1, keepdims=True)
    col = np.clip(mcol + (col - mcol) * 1.12, 0, 1)
    col = col.reshape(RH, RW, 3).astype(np.float32)

    def up(a, mode=Image.BICUBIC):
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((W, H), mode))
    P.ref[:] = np.stack([up(col[..., k]) for k in range(3)], -1).clip(0, 1)
    Mi = M.reshape(RH, RW); Tn = np.where(hit, T, 1e4).reshape(RH, RW); Nr = Nn.reshape(RH, RW, 3)
    P.reg[:] = np.asarray(Image.fromarray(Mi.astype(np.int32)).resize((W, H), Image.NEAREST))
    P.mat = P.reg.copy()
    P.depth = up(np.minimum(Tn, 1e4), Image.NEAREST)
    # the brush follows the fall of each surface: up a trunk, along a drift
    upw = np.array([0.0, 1.0, 0.0])
    tg = np.cross(Nr, np.cross(np.broadcast_to(upw, Nr.shape), Nr))
    flat = np.abs(Nr[..., 1]) > 0.80
    tg[flat] = right
    sxv = (tg * right).sum(-1); syv = -(tg * upv).sum(-1)
    ang = np.arctan2(syv, sxv + 1e-9)
    ang = np.where((Mi == 1) | (Mi == 10), 0.10 * (fbm(RH, RW, 200 * RW / 2400, rng, 2) - 0.5) * 2, ang)
    ang = np.where(Mi == 8, 1.57 + 1.3 * (fbm(RH, RW, 40 * RW / 2400, rng, 2) - 0.5), ang)
    ang = np.where(Mi == 0, 0.0 + 0.09 * (fbm(RH, RW, 300 * RW / 2400, rng, 2) - 0.5), ang)
    ang = np.where(Mi == 5, 0.0, ang)
    P.ang[:] = np.asarray(Image.fromarray(ang.astype(np.float32)).resize((W, H), Image.NEAREST))
    lf = np.ones((RH, RW), np.float32)
    lf[Mi == 1] = 2.4; lf[Mi == 10] = 2.6; lf[Mi == 0] = 2.0; lf[Mi == 3] = 2.0; lf[Mi == 8] = 0.6
    P.lenf[:] = up(lf, Image.NEAREST)

    def project(p):
        v = np.asarray(p, float) - cam
        zc = v @ fwd
        return (0.5 + 0.5 * (v @ right) / zc / TX) * W, (0.5 - 0.5 * (v @ upv) / zc / TY) * H
    P.project = project
    fx, fy = project(WINC)
    P.flame = (fx, fy)
    X, Y = P.X, P.Y
    dfl = np.hypot(X - fx, Y - fy) / W
    det = np.zeros((RH, RW), np.float32)
    for mid, v in ((4, 1.0), (5, 0.9), (6, 1.0), (7, 1.0), (9, 0.6)):
        det[Mi == mid] = v
    det[Mi == 3] = (0.35 + 0.35 * smooth(30.0, 8.0, Tn))[Mi == 3]
    hollow = np.clip(2.2 * (1 - AO.reshape(RH, RW)), 0, 0.8) * (Mi == 1)
    det = np.maximum(det, hollow)
    P.detail[:] = np.maximum(up(det, Image.BILINEAR), 0.8 * np.exp(-(dfl / 0.03) ** 2))
    P.soft = np.clip(0.8 * smooth(45.0, 140.0, P.depth) * (P.mat != 0) + 0.75 * (P.mat == 0) + 0.3 * (P.mat == 8), 0, 1)
    P.soft = P.soft * (1 - 0.7 * smooth(0.2, 0.6, P.detail))
    P.reg[P.edge < 14 * S] = -1
    P.camv = (cam, fwd, right, upv)


# ---- the crowns: too fine to be form, so they are drawn.  A limb throws branches, a branch throws
#      twigs, and the last twigs of a birch give up and hang.
def _grow(rng, p0, d, L, lvl, out, sc):
    n = int(rng.integers(4, 8))
    L = L * float(np.exp(rng.normal(0, 0.38)))
    kink = int(rng.integers(1, n)) if rng.random() < 0.55 else -1
    pts = [p0]; dirs = []
    droop = (0.0, 0.05, 0.20, 0.62)[lvl]
    for i in range(n):
        d = d + np.array([0.0, -droop * (0.4 + i / n), 0.0]) + rng.normal(0, 0.10 + 0.03 * lvl, 3)
        if i == kink: d = d + rng.normal(0, 0.55, 3)
        d = d / np.linalg.norm(d)
        pts.append(pts[-1] + d * L / n * rng.uniform(0.6, 1.4)); dirs.append(d)
    out.append((np.array(pts), lvl))
    if lvl >= 3: return
    nc = int(rng.integers((4, 3, 2)[lvl - 1], (8, 6, 5)[lvl - 1]) * sc + rng.random())
    for c in range(nc):
        t = rng.uniform(0.15, 1.0) ** 0.8 * n
        i = min(int(t), n - 1)
        b = pts[i] + (pts[i + 1] - pts[i]) * (t - i)
        nd = dirs[i] + rng.normal(0, 0.75, 3) + np.array([0.0, 0.25 if lvl == 1 else -0.25, 0.0])
        nd /= np.linalg.norm(nd)
        if lvl == 2:
            ln = rng.uniform(0.45, 1.5)
        else:
            ln = L * rng.uniform(0.40, 0.80)
        _grow(rng, b, nd, ln, lvl + 1, out, sc)


def crowns(P):
    W, H, S = P.W, P.H, P.S
    rng = np.random.default_rng(SEED + 11)
    cam, fwd, right, upv = P.camv
    ref = P.ref; L = lum(ref)
    order = np.argsort(-TREES[:, 1])
    nst = 0
    veil = np.zeros((H // 4 + 1, W // 4 + 1), np.float32)
    todo = []
    for k in order:
        zt = TREES[k, 1]
        sc = 1.0 if zt < 16 else (0.8 if zt < 28 else 0.62)
        out = []
        for (lp, d1, r1) in TLIMB[k]:
            n1 = int(rng.integers(4, 9))
            for c in range(n1):
                t = rng.uniform(0.2, 1.0) * 3
                q = min(int(t), 2)
                p = lp[q] + (lp[q + 1] - lp[q]) * (t - q)
                dd = lp[q + 1] - lp[q]; dd = dd / np.linalg.norm(dd)
                nd = dd + rng.normal(0, 0.65, 3) + np.array([0, 0.15, 0]); nd /= np.linalg.norm(nd)
                _grow(rng, p, nd, rng.uniform(0.7, 2.0), 1, out, sc)
            _grow(rng, lp[3], d1, rng.uniform(1.0, 2.0), 1, out, sc)
        h = TREES[k, 5]
        for c in range(int(rng.integers(9, 14))):                         # the leader
            yy = h * rng.uniform(0.55, 1.0)
            p = _axis(k, yy)
            az = rng.uniform(0, 6.28); ph = rng.uniform(0.5, 1.2)
            nd = np.array([math.sin(ph) * math.cos(az), math.cos(ph), math.sin(ph) * math.sin(az)])
            _grow(rng, p, nd, rng.uniform(0.7, 1.6) * (1.25 - yy / h) * 1.6, 1, out, sc)
        _grow(rng, _axis(k, h), np.array([TREES[k, 3], 1.0, 0.0]) / math.hypot(TREES[k, 3], 1.0), 1.0, 2, out, sc)
        todo.append((k, out))
    # a thin veil where the twigs are too many to count, then the twigs
    strokes = []
    for k, out in todo:
        for pts, lvl in out:
            v = pts - cam
            zc = v @ fwd
            if zc.min() < 0.5: continue
            sx = (0.5 + 0.5 * (v @ right) / zc / TX) * W; sy = (0.5 - 0.5 * (v @ upv) / zc / TY) * H
            dist = np.linalg.norm(v, axis=1)
            ix = np.clip(sx.astype(int), 0, W - 1); iy = np.clip(sy.astype(int), 0, H - 1)
            ok = (sx > 2) & (sx < W - 3) & (sy > 2) & (sy < H - 3)
            ok &= (P.depth[iy, ix] > dist - 0.6) & (P.edge[iy, ix] > 24 * S)
            if ok.sum() < 3: continue
            i0 = np.argmax(ok); i1 = len(ok) - np.argmax(ok[::-1])
            if not ok[i0:i1].all():
                i1 = i0 + np.argmin(ok[i0:])
            if i1 - i0 < 3: continue
            strokes.append((sx[i0:i1], sy[i0:i1], dist[i0:i1].mean(), lvl, k))
            np.add.at(veil, (iy[i0:i1] // 4, ix[i0:i1] // 4), (1.0 if lvl >= 2 else 0.4) * min(1.0, 14.0 / dist.mean()))
    veil = gaussian_filter(veil, 14 * S / 4 * 3)
    veil = np.asarray(Image.fromarray(veil).resize((W, H), Image.BILINEAR))
    skyish = (P.mat == 0).astype(np.float32)
    P.glaze(np.clip(veil * 2.2, 0, 1) * blur(skyish, 3 * S), [0.74, 0.62, 0.80], 0.55, cell=60)
    for sx, sy, dist, lvl, k in strokes:
        wm = (0.0, 0.042, 0.021, 0.011)[lvl] * (TREES[k, 2] / 0.13) ** 0.6
        r = 0.5 * wm / dist * W / (2 * TX)
        j = len(sx) // 2
        bg = ref[int(sy[j]), int(sx[j])]; bl = L[int(sy[j]), int(sx[j])]
        haze = 1 - math.exp(-dist * 0.016)
        c = np.array([0.20, 0.10, 0.21]) if lvl < 3 else np.array([0.27, 0.13, 0.24])
        c = c * (0.85 + 0.3 * rng.random())
        c = lerp(c, 0.5 * bg + 0.5 * np.array([0.40, 0.33, 0.52]), 0.05 + 0.60 * haze)
        op = min(1.0, 0.70 + 0.30 * r / (0.9 * S)) * (0.35 + 0.65 * smooth(0.14, 0.34, bl))
        if op < 0.12: continue
        pts = np.stack([sx, sy], 1)
        P.stroke(pts, max(0.95 * S, r), np.clip(c, 0, 1), streak=0.02, opac=op, thick=0.35 + 0.25 * (lvl == 1),
                 dry=0.3 if lvl < 3 else 0.5, drysens=0.10, edge=0.5, ridge=0.2, smooth_pts=rng.random() < 0.4)
        nst += 1
    return nst


def paint(P):
    W, H, S, rng = P.W, P.H, P.S, P.rng
    ref = P.ref
    L = lum(ref)
    t0 = time.time()
    say = lambda s, n: print(f"  {s:<14s}{n:>7d} strokes  {time.time() - t0:5.1f}s", flush=True)

    # 1. underpainting: the big shapes, thin, a little lower in key and greyer
    dead = blur(ref, 30 * S)
    m = dead.mean(-1, keepdims=True)
    dead = (m + (dead - m) * 0.7) * 0.92
    say("underpaint", P.layer(dead, 60, spacing=0.7, order='random', maxlen=5, tol=0.6, cjit=0.05, streak=0.04,
                              opac=0.85, thick=0.2, thickv=0.1, dry=0.6, drysens=0.6, pickup=0.0, wander=0.25))
    # 2. block-in with a broad flat brush, edge to edge
    say("block-in", P.layer(blur(ref, 10 * S), 30, spacing=0.62, maxlen=5.5, tol=0.30, cjit=0.06, streak=0.05,
                            opac=0.96, thick=0.5, thickv=0.4, dry=0.35, drysens=0.3, pickup=0.12, wander=0.14))
    P.blend(0.35 + 0.65 * P.soft, length=24, soften=5)
    # 3. modelling
    say("modelling", P.layer(blur(ref, 4.5 * S), 15, spacing=0.7, err=0.075, maxlen=5.5, tol=0.22, cjit=0.04, streak=0.04,
                             opac=0.96, thick=0.6, thickv=0.5, dry=0.4, drysens=0.3, pickup=0.14, wander=0.10))
    P.blend(0.15 + 0.6 * P.soft, length=12, soften=2.0)
    say("drawing", P.layer(blur(ref, 2.0 * S), 7.5, spacing=0.75, err=0.10, maxlen=5, tol=0.17, cjit=0.03, streak=0.03,
                           opac=0.97, thick=0.6, thickv=0.5, dry=0.4, drysens=0.3, pickup=0.10, wander=0.08))
    say("small", P.layer(blur(ref, 1.0 * S), 3.6, spacing=0.8, err=0.14, mask=0.20 + 0.80 * smooth(0.1, 0.5, P.detail),
                         maxlen=5, tol=0.14, cjit=0.03, streak=0.03, opac=0.98, thick=0.5, dry=0.35, drysens=0.25,
                         pickup=0.06, wander=0.06))
    say("accents", P.layer(ref, 1.9, spacing=0.85, err=0.18, mask=smooth(0.35, 0.7, P.detail), maxlen=5, tol=0.12,
                           cjit=0.02, streak=0.02, opac=0.99, thick=0.45, dry=0.3, drysens=0.2, pickup=0.03, wander=0.05))
    P.blend(0.5 * P.soft * smooth(0.5, 0.2, P.detail), length=6, soften=1.2)
    # the sky and the open snow are quiet: work them together while they are wet
    quiet = ((P.mat == 0) * 0.5 + np.isin(P.mat, (1, 10)) * 0.70) * smooth(0.45, 0.1, P.detail)
    P.blend(blur(quiet.astype(np.float32), 2 * S), length=34, soften=3.5)
    P.blend(smooth(0.80, 1.0, blur((P.mat == 0).astype(np.float32), 3 * S)), length=40, soften=5.0)

    # 4. glaze: the violet of a northern afternoon, thin, into every dark
    P.glaze(blur(smooth(0.24, 0.05, L), 8 * S), [0.74, 0.76, 1.0], 0.40)

    # 5. the snow lights, loaded, laid once, long and level along the drifts
    warm = ref[..., 0] - ref[..., 2]
    snowy = np.isin(P.mat, (1, 5, 10)).astype(np.float32)
    Lb = blur(L, 40 * S)
    lm = np.clip(smooth(0.04, 0.11, L - Lb) + 0.8 * smooth(0.05, 0.20, warm) * smooth(0.45, 0.60, L), 0, 1) * snowy
    lit = np.clip(ref * np.array([1.05, 1.01, 0.97]) + 0.012, 0, 1)
    say("lights", P.layer(lit, 7.0, spacing=1.0, mask=lm * 0.26, maxlen=10, tol=0.13, cjit=0.02, streak=0.05, opac=0.94,
                          thick=1.3, thickv=0.4, dry=0.35, drysens=0.25, pickup=0.05, ridge=0.6, wander=0.04))
    bark = (P.mat == 3).astype(np.float32)
    say("bark lights", P.layer(lit, 4.5, spacing=1.0, mask=bark * smooth(0.50, 0.66, L) * 0.5, maxlen=4, tol=0.12, cjit=0.03,
                               streak=0.05, opac=0.96, thick=1.4, thickv=0.3, dry=0.4, drysens=0.3, pickup=0.03, ridge=0.6))
    # 6. the band of fire along the horizon: the knife, drawn level
    fire = smooth(0.90, 0.99, blur((P.mat == 0).astype(np.float32), 9 * S)) * smooth(0.22, 0.48, warm) * smooth(0.50, 0.62, L)
    say("fire", P.layer(np.clip(ref * np.array([1.03, 1.0, 0.96]), 0, 1), 11.0, spacing=1.1, mask=fire * 0.5, maxlen=12, tol=0.10,
                        cjit=0.01, streak=0.02, opac=0.9, thick=0.7, thickv=0.1, dry=0.3, drysens=0.2, pickup=0.10,
                        edge=0.6, ridge=0.1, wander=0.02, sizev=0.3))
    skym = smooth(0.80, 1.0, blur((P.mat == 0).astype(np.float32), 3 * S))
    P.blend(skym * 0.8, length=26, soften=3.0)
    P.hgt *= (1 - 0.6 * skym)
    say("knife", P.layer(np.clip(lit * 1.03 + 0.015, 0, 1), 4.5, spacing=1.5, mask=smooth(0.68, 0.84, L) * 0.4 * snowy, maxlen=8,
                         tol=0.14, cjit=0.02, streak=0.07, opac=1.0, thick=2.2, thickv=0.2, dry=0.5, drysens=0.3,
                         pickup=0.0, edge=0.08, ridge=1.0, wander=0.03, sizev=0.5))

    # 7. the crowns
    say("twigs", crowns(P))

    # 8. the window, last: four panes
    cam, fwd, right, upv = P.camv
    def wpt(du, dy):
        return P.project((WINC[0] + HC * du, WINC[1] + dy, WINC[2] + HS * du))
    x0, y0 = wpt(0.0, 0.50); x1, y1 = wpt(0.0, -0.50)
    hw = abs(wpt(0.46, 0.0)[0] - wpt(-0.46, 0.0)[0]) * 0.5
    P.stroke([(x0, y0), (x1, y1)], hw, [1.0, 0.66, 0.20], thick=1.2, dry=0.1, drysens=0.05, streak=0.02, edge=0.3, ridge=0.0, smooth_pts=False)
    P.stroke([(x0, y0 + 0.14 * (y1 - y0)), (x1, y1 - 0.14 * (y1 - y0))], hw * 0.60, [1.0, 0.86, 0.46], thick=1.8, dry=0.1, drysens=0.0,
             streak=0.0, edge=0.7, ridge=0.0, smooth_pts=False)
    P.sign(W - 70 * S, H - 62 * S, 30 * S)


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0] != '-' else 1.0
    W, H = int(3000 * scale), int(2250 * scale)
    t0 = time.time()
    P = Painting(W, H, SEED)
    draw(P)
    print(f"drawing {time.time() - t0:.1f}s", flush=True)
    if '--drawing' in sys.argv:
        save(P.ref, __file__.replace('.py', '_drawing.png')); return
    paint(P)
    out = __file__.replace('.py', '.png') if scale == 1.0 else __file__.replace('.py', '_study.png')
    save(P.finish(relief=0.8), out)
    print(f"{out}  {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
