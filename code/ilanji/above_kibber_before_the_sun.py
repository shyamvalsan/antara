#!/usr/bin/env python3
"""
Above Kibber, before the sun reaches the pass

    python above_kibber_before_the_sun.py        # the picture, 3200 x 2100
    python above_kibber_before_the_sun.py 0.5    # a study at half size

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
#  Spiti, on the shoulder above Kibber, a clear dawn at 4200 m.
#  The sun is still behind the eastern ridge.  It has found the highest
#  snow and the top of one bare ridge; everything else is lit by the
#  sky alone.  One window in the village is awake.
#
#  The land is too big for metres: one unit here is a hundred of them.
# =====================================================================
from scipy.ndimage import map_coordinates, label

LAT, LON, MINUTE = 32.332, 78.009, 6 * 60 + 12
SEED = int(LAT * 1e4) * 7919 + int(LON * 1e4) * 104729 + MINUTE

EYE = 5.00                                   # the eye, 1.8 m above the shoulder
PITCH = -0.022
TX, TY = 0.50, 0.50 * 2100 / 3200
_s = np.array([0.95, 0.080, -0.22]); SUN = _s / np.linalg.norm(_s)      # toward the sun: east, low, a little behind us
SUNX, SUNY, SUNZ = float(SUN[0]), float(SUN[1]), float(SUN[2])

PEAKS = np.array([  # x, z, summit, half-width
    (-11.0, 62.0, 19.0, 8.0), (3.0, 68.0, 17.4, 7.0), (-25.0, 61.0, 16.6, 7.0), (14.0, 73.0, 17.8, 7.5),
    (27.0, 78.0, 17.0, 8.0), (-38.0, 68.0, 17.2, 8.0), (-3.0, 60.0, 14.6, 4.0), (8.5, 64.0, 14.8, 4.0)])
CHX, CHZ, CHS, CHROT = -0.125, 0.37, 1.00, 0.55          # the chorten: place, scale, turn
POLX, POLZ, POLH = 0.005, 0.335, 0.020                   # the stake the flags run down to
VCX0, VCZ0 = 0.70, 6.30                                  # Kibber


@njit(cache=True, fastmath=True, error_model='numpy')
def noised(x, z):
    i = math.floor(x); j = math.floor(z)
    u = x - i; v = z - j
    du = 6.0 * u * (1.0 - u); dv = 6.0 * v * (1.0 - v)
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v)
    i = int(i); j = int(j)
    a = hash3(i, 3, j); b = hash3(i + 1, 3, j); c = hash3(i, 3, j + 1); d = hash3(i + 1, 3, j + 1)
    k = a - b - c + d
    return a + (b - a) * u + (c - a) * v + k * u * v, du * ((b - a) + k * v), dv * ((c - a) + k * u)


@njit(cache=True, fastmath=True, error_model='numpy')
def erode(x, z, o):
    """noise that has been rained on: where it is steep the fine detail is washed away."""
    a = 0.0; b = 1.0; dx = 0.0; dz = 0.0
    for i in range(o):
        n, nx, nz = noised(x, z)
        dx += nx; dz += nz
        a += b * n / (1.0 + dx * dx + dz * dz)
        b *= 0.5
        t = 1.6 * x - 1.2 * z; z = 1.2 * x + 1.6 * z; x = t
    return a


@njit(cache=True, fastmath=True, error_model='numpy')
def gorge_x(z):
    return -1.9 - 0.16 * z + 0.9 * math.sin(z * 0.17 + 0.8)


@njit(cache=True, fastmath=True, error_model='numpy')
def spur(x, z, ax, az, bx, bz, w):
    ex = bx - ax; ez = bz - az
    t = ((x - ax) * ex + (z - az) * ez) / (ex * ex + ez * ez)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    dx = x - ax - t * ex; dz = z - az - t * ez
    return math.exp(-(dx * dx + dz * dz) / (w * w)) * (1.0 - 0.55 * t)


@njit(cache=True, fastmath=True, error_model='numpy')
def gully(x, z, along_x):
    """0 in the bed of a gully, 1 on the crest of a rib.  the ribs run down the fall line."""
    if along_x:
        u = z * 0.40 + 1.3 * noise3(x * 0.11, 3.3, z * 0.11); v = x * 0.055
    else:
        u = x * 0.40 + 1.3 * noise3(x * 0.11, 8.3, z * 0.11); v = z * 0.055
    a = abs(2.0 * noise3(u, v, 7.7) - 1.0)
    b = abs(2.0 * noise3(u * 2.3 + 5.0, v * 1.7, 2.2) - 1.0)
    return min(1.0, 1.25 * (0.72 * a + 0.28 * b))


@njit(cache=True, fastmath=True, error_model='numpy')
def macro(x, z):
    return macro_r(x, z, 1.0)


@njit(cache=True, fastmath=True, error_model='numpy')
def macro_r(x, z, rib):
    """the big shapes: valley, village hill, east wall and its spurs, west flank, far range, gorge;
    then the ribs and gullies water has cut down every flank."""
    zc = min(max(z, -8.0), 46.0)
    h = 3.55 + 0.050 * zc
    amp = 1.15
    h += 0.80 * math.exp(-((x - 0.8) / 2.6) ** 2 - ((z - 8.0) / 1.9) ** 2)
    amp *= 1.0 - 0.85 * math.exp(-((x - 0.7) / 3.4) ** 2 - ((z - 5.6) / 2.8) ** 2)
    # the eastern wall, with the pass in it
    xe = x - 1.6 * math.sin(z * 0.12 + 0.4) - 0.05 * z
    e = sstep(4.0, 23.0, xe)
    e = math.pow(e, 1.2)
    h += 7.5 * e * (1.0 - 0.26 * math.exp(-((z - 36.0) / 7.0) ** 2))
    amp += 1.5 * e
    # spurs coming down off it across the middle distance
    h += 1.5 * spur(x, z, 10.0, 11.0, 2.6, 13.5, 2.2)
    sp2 = spur(x, z, 16.0, 20.0, -1.5, 27.0, 3.2); sp3 = spur(x, z, 21.0, 31.0, 4.0, 40.0, 3.6)
    h += 3.2 * sp2 + 2.6 * sp3
    # the flank west of the gorge
    gx = gorge_x(z)
    w = sstep(0.8, 12.0, gx - x)
    h += (1.4 + 0.07 * zc) * w
    amp += 1.1 * w
    ra = min(1.0, e + w) * (1.0 - 0.9 * math.exp(-((x - 0.7) / 3.6) ** 2 - ((z - 5.6) / 3.0) ** 2))
    # one bare ridge standing in front of the snow
    rz = z - 38.0 - 0.22 * (x + 14.0); rx = x + 14.0
    rb = math.exp(-(rz / 2.8) ** 2 - (rx / 6.5) ** 2)
    h += 3.9 * rb
    amp += 1.6 * rb
    rc = 0.8 * rb + 0.9 * min(1.0, 1.6 * (sp2 + sp3))
    # the far range
    f = sstep(42.0, 58.0, z - 0.22 * x)
    if f > 0.0:
        pk = 0.0
        for i in range(PEAKS.shape[0]):
            r = math.sqrt((x - PEAKS[i, 0]) ** 2 + (z - PEAKS[i, 1]) ** 2)
            v = (PEAKS[i, 2] - 9.5) * (1.0 - r / PEAKS[i, 3])
            if v > pk: pk = v
        h = h * (1.0 - f) + f * (9.5 + pk)
        amp = amp * (1.0 - f) + f * 4.2
        ra *= 1.0 - f
        rc = rc * (1.0 - f) + 5.0 * f * (1.0 - f)
    if rib > 0.0:
        if ra > 0.01: h += rib * 1.45 * ra * (gully(x, z, True) - 0.45)
        if rc > 0.01: h += rib * 1.45 * rc * (gully(x, z, False) - 0.45)
        amp *= 1.0 - 0.45 * min(1.0, ra + rc)
    # the gorge
    dg = abs(x - gx)
    fl = 1.0 + 0.045 * zc
    if h > fl and z < 60.0:
        g = math.exp(-(dg / (1.15 + 0.03 * zc)) ** 4) * (1.0 - sstep(40.0, 54.0, z))
        h -= (h - fl) * 0.93 * g
        amp *= 1.0 - 0.5 * g
    return h, amp


@njit(cache=True, fastmath=True, error_model='numpy')
def terrace_w(x, z):
    t = math.exp(-((x - 0.6) / 2.5) ** 2 - ((z - 4.9) / 1.5) ** 2)
    return sstep(0.30, 0.60, t)


TQ = 0.038

@njit(cache=True, fastmath=True, error_model='numpy')
def height(x, z, o):
    hm, amp = macro(x, z)
    h = hm + amp * 1.3 * (erode(x * 0.11 + 3.7, z * 0.11 + 1.3, o) - 0.6)
    tw = terrace_w(x, z)
    if tw > 0.0:                                         # barley terraces: the slope cut into gentle steps
        q = h / TQ + 1.6 * (fbm2(x * 0.9, z * 0.9, 2) - 0.5)
        fq = math.floor(q)
        hs = (fq + sstep(0.72, 1.0, q - fq)) * TQ
        h += tw * (hs - (q * TQ))
    return h


@njit(cache=True, fastmath=True, error_model='numpy')
def path_x(z):
    """where the path lies on the shoulder: in at the bottom edge, past the chorten, on toward the village."""
    if z < 0.20: return -0.012 - 0.33 * (z - 0.085)
    if z < 0.30: return -0.050 - 0.20 * (z - 0.20)
    if z < 0.40: return -0.070 + 0.20 * (z - 0.30)
    if z < 0.47: return -0.050 + 0.71 * (z - 0.40)
    return 0.0 + 1.0 * (z - 0.47)


@njit(cache=True, fastmath=True, error_model='numpy')
def shoulder(x, z):
    """the near ground we stand on."""
    ze = 0.52 + 0.30 * x + 0.05 * math.sin(x * 9.0 + 1.0) + 0.02 * math.sin(x * 23.0)
    s = (EYE - 0.018) - 0.19 * z - 0.11 * x
    s += 0.030 * (fbm2(x * 9.0 + 2.0, z * 9.0, 3) - 0.5) + 0.004 * (fbm2(x * 70.0, z * 70.0, 2) - 0.5)
    over = z - ze
    if over > 0.0:
        s -= 1.5 * math.pow(over, 1.3)
    return s


def _village():
    """no two houses alike, none square to its neighbour; they crowd the middle and thin out up the hill."""
    r = np.random.default_rng(SEED + 2)
    base = []
    tries = 0
    while len(base) < 31 and tries < 6000:
        tries += 1
        a = r.uniform(0, 6.2832); rr = math.sqrt(r.random())
        cz = 6.22 + 0.60 * rr * math.sin(a)
        cx = VCX0 + 0.98 * rr * math.cos(a) * (1.0 - 0.42 * (cz - 5.62) / 1.2) + 0.10 * (cz - 6.2)
        big = r.random() < 0.22
        hw = r.uniform(0.090, 0.135) if big else r.uniform(0.042, 0.085)
        hd = r.uniform(0.050, 0.078) if big else r.uniform(0.036, 0.060)
        ht = r.uniform(0.060, 0.135)
        rot = r.normal(0, 0.13)
        if r.random() < 0.16: rot = r.choice([-1.0, 1.0]) * r.uniform(0.35, 0.75)
        ok = True
        for (ox, oz, ow, od) in [(b[0], b[2], b[3], b[5]) for b in base]:
            if abs(cx - ox) < 0.74 * (hw + ow) and abs(cz - oz) < 0.74 * (hd + od): ok = False; break
        if not ok: continue
        by = min(height(cx - hw, cz - hd, 9), height(cx + hw, cz - hd, 9), height(cx, cz, 9)) - 0.004
        base.append((cx, by, cz, hw, ht, hd, math.cos(rot), math.sin(rot)))
    out = []
    for b in base:
        out.append(b)
        if r.random() < 0.36:                            # a room on the roof
            cx, by, cz, hw, ht, hd, c, s_ = b
            sgn = r.choice([-1.0, 1.0])
            out.append((cx + sgn * hw * 0.42 * c, by, cz + hd * 0.35, hw * 0.52, ht + r.uniform(0.028, 0.045), hd * 0.6, c, s_))
    # the old house at the top of the village, bigger than the rest
    cx, cz = VCX0 + 0.27, 7.02
    out.append((cx, height(cx, cz - 0.08, 9) - 0.004, cz, 0.15, 0.16, 0.075, math.cos(0.06), math.sin(0.06)))
    return np.array(out)

HOUSES = _village()
NH = HOUSES.shape[0]
VLO = np.array([HOUSES[:, 0].min() - 0.13, HOUSES[:, 1].min() - 0.05, HOUSES[:, 2].min() - 0.10])
VHI = np.array([HOUSES[:, 0].max() + 0.13, (HOUSES[:, 1] + HOUSES[:, 4]).max() + 0.02, HOUSES[:, 2].max() + 0.10])
VCX, VCY, VCZ = (VLO + VHI) / 2
VHX, VHY, VHZ = (VHI - VLO) / 2
# the window that is awake: a front-row house near the middle
_front = np.nonzero((np.abs(HOUSES[:, 0] - VCX0 + 0.1) < 0.5) & (np.abs(HOUSES[:, 7]) < 0.2) & (HOUSES[:, 3] > 0.05) & (HOUSES[:, 4] < 0.125))[0]
KW = int(_front[np.argmin(HOUSES[_front, 2])])
WLX, WLY = 0.30 * HOUSES[KW, 3], 0.62 * HOUSES[KW, 4]
_c, _sn = HOUSES[KW, 6], HOUSES[KW, 7]
def _hworld(k, lx, ly, lz):
    c, s = HOUSES[k, 6], HOUSES[k, 7]
    return np.array([HOUSES[k, 0] + c * lx - s * lz, HOUSES[k, 1] + ly, HOUSES[k, 2] + s * lx + c * lz])
WIN = _hworld(KW, WLX, WLY, -HOUSES[KW, 5])
LPW = _hworld(KW, WLX, WLY - 0.008, -HOUSES[KW, 5] - 0.030)
CHY = shoulder(CHX, CHZ) - 0.003
POLY = shoulder(POLX, POLZ)
CHC, CHSN = math.cos(CHROT), math.sin(CHROT)


@njit(cache=True, fastmath=True, error_model='numpy')
def scene(x, y, z):
    hm, amp = macro(x, z)
    sh = -1e9
    if z < 6.0 and z > -3.0 and abs(x) < 8.0:
        sh = shoulder(x, z)
    top = hm + 0.95 * amp
    if sh > top: top = sh
    if y - top > 0.6:
        return (y - top) * 0.5 + 0.1, 1
    h = height(x, z, 9)
    m = 1
    if sh > h: h = sh; m = 6
    d = (y - h) * 0.5
    # ---- Kibber: whitewashed boxes stepped down the slope
    dv = sd_box(x - VCX, y - VCY, z - VCZ, VHX, VHY, VHZ)
    if dv < d:
        if dv > 0.10:
            d = dv; m = 2
        else:
            for k in range(NH):
                ex = x - HOUSES[k, 0]; ez = z - HOUSES[k, 2]
                lx = HOUSES[k, 6] * ex + HOUSES[k, 7] * ez; lz = -HOUSES[k, 7] * ex + HOUSES[k, 6] * ez
                ht = HOUSES[k, 4]
                dd = sd_box(lx, y - HOUSES[k, 1] - (ht - 0.06) * 0.5, lz, HOUSES[k, 3], (ht + 0.06) * 0.5, HOUSES[k, 5])
                if dd < d: d = dd; m = 2
    # ---- the near shoulder: its stones, the chorten, the pole
    if z < 1.3 and m == 6 and y - h < 0.05:
        ci = math.floor(x / 0.045); ck = math.floor(z / 0.045)
        for a in range(-1, 2):
            for b in range(-1, 2):
                i = int(ci + a); k = int(ck + b)
                if hash3(i, 51, k) > 0.30: continue
                cx = (i + 0.2 + 0.6 * hash3(i, 53, k)) * 0.045; cz = (k + 0.2 + 0.6 * hash3(i, 57, k)) * 0.045
                if abs(cx - path_x(cz)) < 0.016: continue                   # the path is kept clear
                rad = 0.0022 + 0.0065 * hash3(i, 59, k) ** 2
                gy = shoulder(cx, cz)
                ex = (x - cx) / rad; ey = (y - gy - 0.05 * rad) / (rad * 0.55); ez = (z - cz) / (rad * (0.8 + 0.5 * hash3(i, 61, k)))
                dd = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0 - 0.45 * (noise3(x * 420.0, y * 420.0, z * 420.0) - 0.5)) * rad * 0.42
                if dd < d: d = dd; m = 4
    ex = x - CHX; ez = z - CHZ
    if abs(ex) < 0.10 and abs(ez) < 0.10 and y < CHY + 0.14:
        r2 = math.sqrt(ex * ex + ez * ez)
        if r2 > 0.06:
            dd = r2 - 0.035
            if dd < d: d = dd; m = 3
        else:
            lx = (CHC * ex + CHSN * ez) / CHS; lz = (-CHSN * ex + CHC * ez) / CHS; py = (y - CHY) / CHS
            dd = sd_box(lx, py - 0.004, lz, 0.0215, 0.010, 0.0215)
            b = sd_box(lx, py - 0.0165, lz, 0.0175, 0.0030, 0.0175)
            if b < dd: dd = b
            b = sd_box(lx, py - 0.0225, lz, 0.0145, 0.0030, 0.0145)
            if b < dd: dd = b
            b = sd_box(lx, py - 0.0285, lz, 0.0115, 0.0030, 0.0115)
            if b < dd: dd = b
            rr = math.sqrt(lx * lx + lz * lz)
            b = max(math.sqrt(rr * rr + (py - 0.0415) ** 2) - 0.0122, 0.0315 - py)  # the dome
            b = max(b, (rr - (0.0075 + 0.33 * (py - 0.0315))) * 0.9)
            if b < dd: dd = b
            b = sd_box(lx, py - 0.0560, lz, 0.0048, 0.0028, 0.0048)                 # harmika
            if b < dd: dd = b
            rs = 0.0037 * (1.0 - (py - 0.0585) / 0.021) + 0.0005
            rs *= 1.0 + 0.16 * math.sin(py * 2700.0)
            b = max((rr - rs) * 0.8, max(0.0585 - py, py - 0.0780))                 # the thirteen rings
            if b < dd: dd = b
            dd *= CHS
            if dd < d: d = dd; m = 3
    else:                                                # outside its box a ray may not step past it
        bd = sd_box(ex, y - CHY - 0.05, ez, 0.10, 0.09, 0.10)
        if bd < d: d = min(d, max(bd, 0.03))
    ex = x - POLX; ez = z - POLZ
    if abs(ex) < 0.03 and abs(ez) < 0.03 and y < POLY + POLH + 0.02:
        dd = sd_capsule(x, y, z, POLX, POLY - 0.01, POLZ, POLX + 0.0020, POLY + POLH, POLZ + 0.001, 0.00038)
        if dd < d: d = dd; m = 5
    else:
        bd = sd_box(ex, y - POLY - 0.01, ez, 0.03, POLH + 0.01, 0.03)
        if bd < d: d = min(d, max(bd, 0.015))
    return d, m


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def sunreach(P, Nn):
    """cast shadow of the land on itself: march the heightfield toward the sun, far past the reach of the tracer,
    and with a penumbra as narrow as a sun this low deserves."""
    n = P.shape[0]
    out = np.ones(n, np.float32)
    for i in prange(n):
        x = P[i, 0] + Nn[i, 0] * 0.06; y = P[i, 1] + Nn[i, 1] * 0.06 + 0.04; z = P[i, 2] + Nn[i, 2] * 0.06
        res = 1.0; t = 0.35
        for s in range(150):
            yy = y + SUNY * t
            if yy > 21.0 or t > 190.0: break
            dh = yy - height(x + SUNX * t, z + SUNZ * t, 9 if t < 8.0 else 6)
            if dh < 0.0:
                res = 0.0; break
            v = 26.0 * dh / t
            if v < res: res = v
            t += 0.25 + 0.035 * t
        out[i] = res
    return out


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def landinfo(P):
    """for each point of land: how much sky it sees, whether it lies in a gully or on a rib,
    and which way is downhill on the big form (no stones, no noise)."""
    n = P.shape[0]
    vis = np.ones(n, np.float32); gul = np.ones(n, np.float32); fall = np.zeros((n, 2), np.float32)
    for i in prange(n):
        x = P[i, 0]; y = P[i, 1] + 0.03; z = P[i, 2]
        occ = 0.0
        for a in range(8):
            dx = math.cos(a * 0.7854 + 0.3); dz = math.sin(a * 0.7854 + 0.3)
            hm = 0.0
            r = 0.35
            for k in range(5):
                sl = (height(x + dx * r, z + dz * r, 5) - y) / r
                if sl > hm: hm = sl
                r *= 2.1
            occ += hm / math.sqrt(1.0 + hm * hm)
        vis[i] = 1.0 - occ / 8.0
        e = 0.35
        gx = (macro_r(x + e, z, 0.0)[0] - macro_r(x - e, z, 0.0)[0]) / (2 * e)
        gz = (macro_r(x, z + e, 0.0)[0] - macro_r(x, z - e, 0.0)[0]) / (2 * e)
        fall[i, 0] = gx; fall[i, 1] = gz
        xe = x - 1.6 * math.sin(z * 0.12 + 0.4) - 0.05 * z
        ee = math.pow(sstep(4.0, 23.0, xe), 1.2)
        w = sstep(0.8, 12.0, gorge_x(z) - x)
        f = sstep(42.0, 58.0, z - 0.22 * x)
        ra = min(1.0, ee + w) * (1.0 - f)
        g1 = gully(x, z, True); g2 = gully(x, z, False)
        t = min(1.0, ra)
        gul[i] = g1 * t + g2 * (1.0 - t)
    return vis, gul, fall

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


L1 = SUN.copy()
LP = LPW.copy()
SUN_C = np.array([1.00, 0.35, 0.10]) * 2.5          # the first light is red-gold
LAMP_C = np.array([1.00, 0.50, 0.14])
SKY_A = np.array([0.25, 0.37, 1.00])                # what the open sky gives the shadow
BNC_C = np.array([0.34, 0.15, 0.20])                # what the lit snow hands back, and the ground
BNC_D = np.array([-0.25, 0.25, 0.93])
PATH = np.array([(0.004, 0.035), (-0.012, 0.085), (-0.034, 0.15), (-0.050, 0.20), (-0.066, 0.26), (-0.070, 0.30), (-0.064, 0.35),
                 (-0.050, 0.40), (-0.028, 0.44), (0.0, 0.47), (0.06, 0.53), (0.20, 0.70), (0.45, 1.0), (0.80, 1.9), (0.75, 3.2), (0.40, 4.4),
                 (0.22, 5.2), (0.42, 5.9), (0.66, 6.4), (0.95, 7.0), (1.5, 7.9), (2.4, 9.2), (3.4, 11.5), (5.2, 14.5), (6.6, 18.5),
                 (9.2, 22.5), (11.0, 27.0), (14.5, 31.0), (17.5, 35.0), (22.0, 37.5), (28.0, 38.0)])


def sky(D, tile):
    """a clear dawn: cobalt overhead, turquoise and pale gold along the eastern ridge."""
    e = D[..., 1]
    az = np.arctan2(D[..., 0], D[..., 2])
    daz = np.abs(az - math.atan2(SUN[0], SUN[2]))
    g = smooth(2.75, 1.05, daz)
    col = lerp(np.array([0.045, 0.115, 0.43]), np.array([0.17, 0.33, 0.70]), smooth(0.50, 0.08, e)[..., None])
    col = lerp(col, np.array([0.36, 0.52, 0.78]), (0.6 * smooth(0.20, 0.02, e))[..., None])
    col = lerp(col, np.array([0.44, 0.70, 0.80]), (g * smooth(0.44, 0.15, e) * 0.95)[..., None])
    col = lerp(col, np.array([1.00, 0.86, 0.55]), (g ** 1.3 * smooth(0.33, 0.10, e))[..., None])
    return col


def shade(P3, Nn, M, S1, S2, AO, D, T, tile, vis, gul):
    """what colour each visible point is, under this light."""
    def wn(a, b, scale):
        return map_coordinates(tile, [(b / scale * 97.0) % 1024, (a / scale * 97.0) % 1024], order=1, mode='wrap')
    x, y, z = P3[..., 0], P3[..., 1], P3[..., 2]
    n = len(x)
    alb = np.zeros((n, 3), np.float32); emis = np.zeros((n, 3), np.float32); snow = np.zeros(n, np.float32)
    up = Nn[..., 1]
    nA = wn(x, z, 3.0); nB = wn(x * 0.45 + 0.6 * y, z, 0.6); nC = wn(x, z + y, 0.08); nD = wn(x * 1.3 + y, z * 1.3, 0.012)

    # ---- the bare land: buff and rust in bands, violet-grey where it has slid
    k = M == 1
    sc = y * 1.0 + 2.2 * nA + 0.05 * x
    bn = wn(x * 0.02 + z * 0.015, sc, 0.22)
    c = lerp(np.array([0.54, 0.44, 0.32]), np.array([0.42, 0.22, 0.15]), smooth(0.42, 0.66, bn)[..., None])
    c = lerp(c, np.array([0.37, 0.34, 0.38]), (smooth(0.55, 0.85, nB) * 0.6)[..., None])
    c = lerp(c, np.array([0.58, 0.47, 0.34]), (smooth(0.55, 0.95, gul) * 0.35)[..., None])       # the ribs are bare bedrock, warmer
    fan = smooth(0.45, 0.10, gul) * smooth(0.62, 0.88, up)
    c = lerp(c, np.array([0.62, 0.58, 0.54]), (np.maximum(fan * 0.85, smooth(0.78, 0.95, up) * 0.40))[..., None])   # scree in the gullies and fans at their feet lie paler
    c = c * (0.88 + 0.24 * nC[..., None]) * (0.78 + 0.22 * smooth(0.35, 0.7, up))[..., None]
    # barley terraces under the village
    tw = np.exp(-((x - 0.6) / 2.5) ** 2 - ((z - 4.9) / 1.5) ** 2); tw = smooth(0.30, 0.60, tw)
    tid = np.floor(y / TQ)
    hsh = np.modf(np.sin(tid * 12.9898 + np.floor(x * 1.7 + 3 * nA) * 78.233) * 43758.5453)[0] % 1.0
    fld = lerp(np.array([0.50, 0.44, 0.27]), np.array([0.33, 0.37, 0.24]), (hsh < 0.42)[..., None].astype(np.float32))
    fld = lerp(fld, np.array([0.36, 0.27, 0.20]), (hsh > 0.78)[..., None].astype(np.float32))
    fld = fld * (0.70 + 0.25 * nD[..., None])
    fld = lerp(np.array([0.20, 0.17, 0.18]), fld * 1.15, smooth(0.80, 0.93, up)[..., None])           # the risers are dry stone
    c = lerp(c, fld, (tw * 0.92)[..., None])
    # snow: above a line on the far range, lying where the wind lets it
    far = smooth(40.0, 52.0, z - 0.22 * x)
    sline = 15.2 - 4.6 * far
    sv = y - sline + 2.0 * (nB - 0.5) + 0.7 * (nC - 0.5)
    sn = smooth(-0.6, 0.5, sv) * smooth(0.26, 0.52, up + 0.3 * (nC - 0.5) + 0.12 * smooth(0.0, 4.0, sv))
    c = lerp(c, np.array([0.93, 0.94, 0.96]), sn[..., None])
    snow[k] = sn[k]
    alb[k] = c[k]
    # ---- the near shoulder: dark earth, dry grass, a little frost
    k = (M == 6)
    c = lerp(np.array([0.17, 0.135, 0.115]), np.array([0.27, 0.22, 0.17]), nD[..., None])
    frost = smooth(0.45, 0.75, wn(x, z, 0.02)) * smooth(0.90, 0.99, up)
    c = lerp(c, np.array([0.40, 0.42, 0.46]), (frost * 0.35)[..., None])
    tuft = smooth(0.52, 0.70, wn(x * 9.0, z * 9.0, 0.012)) * smooth(0.45, 0.65, wn(x, z, 0.03))
    c = lerp(c, np.array([0.33, 0.27, 0.13]), (tuft * 0.7)[..., None])
    alb[k] = c[k]
    k = (M == 4)
    alb[k] = (lerp(np.array([0.36, 0.35, 0.37]), np.array([0.55, 0.56, 0.60]), wn(x * 6.0 + y * 9.0, z * 6.0, 0.012)[..., None]) * (0.75 + 0.25 * smooth(0.2, 0.8, up))[..., None])[k]
    # ---- the path, worn pale
    k = (M == 1) | (M == 6)
    if k.any():
        px = x[k]; pz = z[k]
        dmin = np.full(px.shape, 1e9)
        for i in range(len(PATH) - 1):
            ax, az_ = PATH[i]; bx, bz = PATH[i + 1]
            ex = bx - ax; ez = bz - az_
            tt = np.clip(((px - ax) * ex + (pz - az_) * ez) / (ex * ex + ez * ez), 0, 1)
            dmin = np.minimum(dmin, np.hypot(px - ax - tt * ex, pz - az_ - tt * ez))
        wig = 0.6 * (nC[k] - 0.5)
        wdt = np.maximum(np.where(M[k] == 6, 0.0070, 0.008), 0.0013 * T[k]) * (1.0 + wig)
        pm = smooth(1.0, 0.45, dmin / wdt) * (0.85 - 0.45 * snow[k]) * np.exp(-T[k] / 22.0)
        pc = np.where((M[k] == 6)[:, None], np.array([0.56, 0.53, 0.50]), np.array([0.66, 0.58, 0.47]))
        alb[k] = lerp(alb[k], pc, pm[..., None])

    # ---- houses: whitewash, a maroon frieze under the roof, black-framed windows, brushwood stacked along the parapet
    k = M == 2
    if k.any():
        ki = np.nonzero(k)[0]
        hx, hy, hz = x[ki], y[ki], z[ki]
        best = np.full(len(ki), 1e9); loc = np.zeros((len(ki), 3)); dim = np.ones((len(ki), 3)); hid = np.zeros(len(ki), np.int32)
        for h in range(NH):
            cx, by, cz, hw, ht, hd, c_, s_ = HOUSES[h]
            ex = hx - cx; ez = hz - cz
            lx = c_ * ex + s_ * ez; lz = -s_ * ex + c_ * ez; ly = hy - by
            q = np.abs(np.maximum(np.maximum(np.abs(lx) - hw, np.abs(lz) - hd), ly - ht))
            u_ = q < best
            best[u_] = q[u_]; hid[u_] = h
            loc[u_] = np.stack([lx, ly, lz], 1)[u_]; dim[u_] = (hw, ht, hd)
        lx, ly, lz = loc.T; hw, ht, hd = dim.T
        c_ = HOUSES[hid, 6]; s_ = HOUSES[hid, 7]
        nlx = c_ * Nn[ki, 0] + s_ * Nn[ki, 2]; nlz = -s_ * Nn[ki, 0] + c_ * Nn[ki, 2]
        roof = up[ki] > 0.6
        front = np.abs(nlz) >= np.abs(nlx)
        a = np.where(front, lx, lz); fw = np.where(front, hw, hd)
        wash = np.array([0.95, 0.93, 0.89]) * (0.88 + 0.20 * nC[ki][..., None]) * (0.9 + 0.1 * smooth(0.0, 0.03, ly))[..., None]
        c = wash.copy()
        ns = np.maximum(2, np.round(ht / 0.042))
        fy = ly / ht * ns; st = np.floor(fy); fy = fy - st
        fa = (a + fw) / 0.046 + hid * 0.37; col_ = np.floor(fa); fa = fa - col_
        wh = np.modf(np.abs(np.sin(col_ * 12.9898 + st * 78.233 + hid * 37.719) * 43758.5453))[0]
        win = (fy > 0.36) & (fy < 0.70) & (fa > 0.32) & (fa < 0.68) & (st >= 1) & (wh < 0.70) & (np.abs(a) < fw - 0.012) & ~roof
        door = (st == 0) & (fy < 0.75) & (np.abs(a - 0.2 * fw) < 0.008) & front & (nlz < 0) & ~roof & (hid % 2 == 0)
        c[win | door] = np.array([0.075, 0.06, 0.10])
        fr = (ly > ht - 0.011) & ~roof
        c[fr] = np.array([0.46, 0.105, 0.095])
        c[(ly > ht - 0.0032) & ~roof] = np.array([0.13, 0.09, 0.10])
        rc = np.array([0.43, 0.37, 0.31]) * (0.85 + 0.3 * nC[ki][..., None])
        rim = (np.abs(lx) > hw - 0.007) | (np.abs(lz) > hd - 0.007)
        rc[rim] = np.array([0.15, 0.115, 0.12])
        c[roof] = rc[roof]
        lit = (hid == KW) & front & (nlz < 0) & (np.abs(lx - WLX) < 0.009) & (np.abs(ly - WLY) < 0.0075) & ~roof
        c[lit] = np.array([1.0, 0.6, 0.2])
        alb[ki] = c
        e_ = np.zeros((len(ki), 3)); e_[lit] = np.array([3.2, 1.25, 0.28]); emis[ki] = e_
    # ---- the chorten: lime wash gone grey at the foot, red ochre on the harmika and the rings
    k = M == 3
    if k.any():
        py = (y - CHY) / CHS
        c = np.array([0.95, 0.94, 0.91]) * (0.93 + 0.09 * wn(x * 40.0, (z + y) * 40.0, 0.012)[..., None])
        c = lerp(c, np.array([0.60, 0.56, 0.53]), (0.45 * smooth(0.010, 0.0, py))[..., None])
        c = lerp(c, np.array([0.62, 0.20, 0.11]), (smooth(0.0525, 0.0535, py) * 0.95)[..., None])
        c = lerp(c, np.array([0.70, 0.46, 0.16]), (smooth(0.0590, 0.0600, py) * (0.5 + 0.5 * np.sin(py * 2700.0)) * 0.7)[..., None])
        alb[k] = c[k]
    k = M == 5
    alb[k] = np.array([0.16, 0.12, 0.10])

    # ---- light
    ndl1 = np.clip((Nn * L1).sum(-1), 0, 1)
    sun = np.clip(ndl1 * 1.15, 0, 1) * S1
    lv = LP - P3; ld = np.linalg.norm(lv, axis=-1); lv = lv / ld[..., None]
    ndl2 = (Nn * lv).sum(-1)
    lamp_i = 0.0048 / (ld * ld + 0.0016)             # a painter's lamp: it reaches further than a real one
    ao = 0.20 + 0.80 * AO
    ao = np.where(M == 3, 0.62 + 0.38 * AO, np.where(M == 2, 0.36 + 0.64 * AO, ao))     # small white things are not buried by the hill they stand on
    land = (M == 1)
    skyv = np.where(land, 0.42 + 0.58 * smooth(0.55, 0.95, vis), np.where(M == 6, 0.66, 1.0))       # how much of the sky a slope can see
    upf = np.where(land | (M == 6), 0.26 + 0.74 * np.clip(up, 0, 1) ** 1.5, 0.50 + 0.50 * np.clip(up, 0, 1))
    warmv = np.where(land, 0.65 + 0.60 * smooth(0.95, 0.50, vis), 1.0)       # the gullies keep what the rock hands back
    bn_ = np.clip((Nn * (BNC_D / np.linalg.norm(BNC_D))).sum(-1) * 0.6 + 0.4, 0, 1)
    amb = (SKY_A * (upf * skyv)[..., None] * 1.00 + BNC_C * ((0.35 + 0.65 * bn_) * warmv)[..., None] * 0.55) * ao[..., None]
    white = (M == 2) | (M == 3)
    east = np.clip((Nn * np.array([0.93, 0.20, -0.30])).sum(-1), 0, 1)                 # the pale gold of the eastern sky on walls turned to it
    amb = amb + (white * east * np.where(M == 3, 0.065, 0.035))[..., None] * np.array([1.0, 0.62, 0.36])
    dif2 = np.clip(ndl2 * 0.85 + 0.15, 0, 1) * S2 * lamp_i
    amb = amb * np.where(M == 3, 1.50, np.where(M == 2, 1.32, 1.0))[..., None]      # lime wash drinks the sky
    amb = amb * (1.0 - 0.85 * smooth(0.0, 0.5, sun))[..., None]        # where the sun lands the eye stops seeing the blue
    col = alb * (amb + SUN_C * sun[..., None] + LAMP_C * dif2[..., None]) + emis
    # snow in shadow keeps a little of the sky inside it
    col += (snow * 0.05 * ao)[..., None] * np.array([0.3, 0.5, 1.0])
    k = M == 3                                           # lime wash in shadow is a cool white, not a blue
    g = col[k].mean(-1, keepdims=True)
    col[k] = g + (col[k] - g) * 0.62
    return col, sun, snow


def draw(P):
    W, H, S, rng = P.W, P.H, P.S, P.rng
    fq = np.hypot(*np.meshgrid(np.fft.fftfreq(1024), np.fft.fftfreq(1024)))
    spec = np.fft.fft2(np.random.default_rng(SEED + 1).standard_normal((1024, 1024))) * np.where(fq > 0, 1.0 / np.maximum(fq, 1 / 200.0) ** 1.4, 0) * (fq < 0.3)
    tile = np.real(np.fft.ifft2(spec)); tile = np.clip(0.5 + 0.22 * (tile - tile.mean()) / tile.std(), 0, 1).astype(np.float32)

    RW = min(W, 1200); RH = int(round(RW * H / W))
    fwd = np.array([0.0, PITCH, 1.0]); fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
    upv = np.cross(fwd, right)
    CAM = np.array([0.0, EYE, 0.0])
    P.cam = (fwd, right, upv, TX, TY)
    jj, ii = np.mgrid[0:RH, 0:RW]
    nx = ((ii + 0.5) / RW * 2 - 1) * TX; ny = (1 - (jj + 0.5) / RH * 2) * TY
    D = fwd[None, None, :] + nx[..., None] * right + ny[..., None] * upv
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    D = D.reshape(-1, 3)
    O = np.tile(CAM, (len(D), 1))
    TMAX = 150.0
    t0 = time.time()
    T, M, Nn, S1, S2, AO = trace(O, D, TMAX, L1, LP)
    print(f"  traced {time.time() - t0:.1f}s", flush=True)
    hit = M > 0
    P3 = O + D * np.minimum(T, TMAX)[:, None]
    sr = sunreach(P3[hit], Nn[hit].astype(np.float64))
    land = (M[hit] == 1) | (M[hit] == 6)
    S1[hit] = np.where(land, sr, np.minimum(S1[hit], sr))        # the land shadows itself; built things take both
    print(f"  shadows {time.time() - t0:.1f}s  lit {np.mean(S1[hit] > 0.5):.3f}  far lit {np.mean(S1[hit & (T > 45)] > 0.5):.3f}", flush=True)
    skyc = sky(D, tile)
    col = skyc.copy()
    sunm = np.zeros(len(D), np.float32); snowm = np.zeros(len(D), np.float32)
    vis, gul, fall = landinfo(P3[hit])
    print(f"  land {time.time() - t0:.1f}s  vis {vis.min():.2f} {np.median(vis):.2f} {vis.max():.2f}", flush=True)
    c, sun, snow = shade(P3[hit], Nn[hit], M[hit], S1[hit], S2[hit], AO[hit], D[hit], T[hit], tile, vis, gul)
    col[hit] = c; sunm[hit] = sun; snowm[hit] = snow
    # thin air: distance takes very little, and what it takes it gives to the blue
    Tc = np.minimum(T, TMAX)
    f = np.where(hit, 1 - np.exp(-Tc * 0.0042), 0.0)
    haze = np.array([0.16, 0.25, 0.60]) * 0.75 + 0.12 * skyc
    col = col * (1 - 0.45 * f[..., None]) + haze * f[..., None]
    # the lit window in cold air
    tc = np.clip(((LP - O) * D).sum(-1), 0, Tc)
    dmin = np.linalg.norm(O + D * tc[:, None] - LP, axis=-1)
    col += (0.00013 / (dmin ** 2 + 0.00022) + 0.07 * np.exp(-dmin / 0.06))[:, None] * LAMP_C * np.array([1.0, 0.9, 0.7])
    col = 1 - np.exp(-1.75 * col)                                        # the eye adjusts
    mcol = col.mean(-1, keepdims=True)
    col = np.clip(mcol + (col - mcol) * 1.27, 0, 1)
    col = col.reshape(RH, RW, 3).astype(np.float32)

    def up(a, mode=Image.BICUBIC):
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((W, H), mode))
    P.ref[:] = np.stack([up(col[..., k]) for k in range(3)], -1).clip(0, 1)
    Mi = M.reshape(RH, RW); Tn = Tc.reshape(RH, RW); Nr = Nn.reshape(RH, RW, 3)
    # a stroke may not cross from one ridge to the one behind it
    lt = np.log(Tn + 0.05)
    gy, gx = np.gradient(lt)
    edge = np.hypot(gx, gy) > 0.045
    lab, _ = label(~edge)
    lab[edge] = 0
    reg = lab.astype(np.int32) * 8 + Mi
    P.reg[:] = np.asarray(Image.fromarray(reg.astype(np.int32)).resize((W, H), Image.NEAREST))
    P.mat = np.asarray(Image.fromarray(Mi.astype(np.int32)).resize((W, H), Image.NEAREST))
    P.depth = up(Tn, Image.BILINEAR)
    P.sunm = np.clip(up(sunm.reshape(RH, RW), Image.BILINEAR), 0, 1)
    P.snow = np.clip(up(snowm.reshape(RH, RW), Image.BILINEAR), 0, 1)
    # the brush follows the fall line of each slope
    upw = np.array([0.0, 1.0, 0.0])
    tg = np.cross(Nr, np.cross(np.broadcast_to(upw, Nr.shape), Nr))
    flat = np.abs(Nr[..., 1]) > 0.93
    tg[flat] = right
    # on the land the brush goes straight down the fall line of the big form, and nowhere else
    fl = np.zeros((RH * RW, 2), np.float32); fl[hit] = fall
    fl = fl.reshape(RH, RW, 2)
    lm_ = gaussian_filter((Mi == 1).astype(np.float32), 15.0) + 1e-4               # straighten it: one direction per flank
    fl = np.stack([gaussian_filter(fl[..., k] * (Mi == 1), 15.0) / lm_ for k in range(2)], -1)
    g2 = (fl ** 2).sum(-1)
    tl = np.stack([-fl[..., 0], -g2, -fl[..., 1]], -1)
    tl[g2 < 0.012] = right
    tg = np.where((Mi == 1)[..., None], tl, tg)
    tg[(Mi == 6) | (Mi == 4)] = right
    tg[(Mi == 3) & (np.abs(Nr[..., 1]) < 0.93) & (P3.reshape(RH, RW, 3)[..., 1] < CHY + 0.052 * CHS)] = right   # along the tiers, not across them
    sxv = (tg * right).sum(-1); syv = -(tg * upv).sum(-1)
    ang = np.arctan2(syv, sxv + 1e-9)
    ang = np.where(Mi == 0, -0.10 + 0.16 * (fbm(RH, RW, 500 * RW / 2400, rng, 2) - 0.5), ang)
    P.ang[:] = np.asarray(Image.fromarray(ang.astype(np.float32)).resize((W, H), Image.NEAREST))
    lf = np.full((RH, RW), 2.3, np.float32)
    lf[Mi == 0] = 2.2; lf[Mi == 2] = 0.8; lf[Mi == 3] = 0.8; lf[Mi == 4] = 0.6; lf[Mi == 6] = 1.2
    P.lenf[:] = up(lf, Image.NEAREST)
    X, Y = P.X, P.Y
    def project(p):
        v = np.asarray(p, float) - CAM
        zc = v @ fwd
        return (0.5 + 0.5 * (v @ right) / zc / TX) * W, (0.5 - 0.5 * (v @ upv) / zc / TY) * H, zc
    P.project = project
    fx, fy, _ = project(WIN)
    P.flame = (fx, fy)
    dfl = np.hypot(X - fx, Y - fy) / W
    det = np.zeros((RH, RW), np.float32)
    det[Mi == 2] = 1.0; det[Mi == 3] = 0.55; det[Mi == 5] = 0.8; det[Mi == 4] = 0.3
    lite = sunm.reshape(RH, RW)
    det = np.maximum(det, 0.55 * smooth(0.05, 0.4, gaussian_filter(lite, 2.0)))       # the lit planes are found, hard
    vx, vy, _ = project([VCX0, 4.2, 6.0])
    P.detail[:] = np.maximum(np.maximum(up(det, Image.BILINEAR), 0.8 * np.exp(-(dfl / 0.05) ** 2)),
                             0.45 * np.exp(-((X - vx) / (0.20 * W)) ** 2 - ((Y - vy) / (0.10 * W)) ** 2))
    P.soft = np.clip(1.0 * (P.mat == 0) + 0.45 * (P.mat == 1) * (1 - smooth(0.02, 0.3, P.sunm)) * (1 - P.detail)
                     + 0.10 * (P.mat == 6), 0, 1).astype(np.float32)
    P.reg[P.edge < 14 * S] = -1


def _cloth(P, TL, TR, BL, BR, base, rng, torn, text):
    """one prayer flag: folded first, coloured second.  corners in canvas px."""
    S = P.S
    TL, TR, BL, BR = (np.asarray(v, float) for v in (TL, TR, BL, BR))
    wpx = np.hypot(*(TR - TL)); hpx = 0.5 * (np.hypot(*(BL - TL)) + np.hypot(*(BR - TR)))
    if wpx < 3 or hpx < 3: return
    ns = int(np.clip(wpx / (6.0 * S), 3, 9))
    nf = rng.uniform(0.7, 1.35); ph = rng.uniform(0, 6.28)
    rad = max(0.9, 0.85 * wpx / ns)
    base = np.asarray(base, float)
    side = rng.choice([-1, 1])
    ridge = []
    for j in range(ns):
        s = (j + 0.5) / ns
        fold = math.sin(6.283 * nf * s + ph)                         # +1 a ridge turned to the sky, -1 a valley
        top = TL + (TR - TL) * s
        ln = 1.0
        if torn > 0:                                                 # worn from the free edge inward
            e = s if side > 0 else 1 - s
            ln = 1.0 - torn * max(0.0, e - 0.35) / 0.65 * rng.uniform(0.6, 1.2)
        ln *= 1.0 - 0.06 * rng.random()
        bot = top + ((BL + (BR - BL) * s) - top) * ln
        perp = np.array([-(bot - top)[1], (bot - top)[0]]); perp /= (np.hypot(*perp) + 1e-9)
        mid = 0.5 * (top + bot) + perp * fold * 0.05 * wpx
        v = 0.80 + 0.20 * fold
        c = base * v + (1 - v) * 0.10 * np.array([0.25, 0.3, 0.7])
        if fold < -0.3:                                              # the valley takes colour from the cloth facing it
            c = c * np.array([0.92, 0.94, 1.06])
        P.stroke([top, mid, bot], rad, np.clip(c, 0, 1), streak=0.035, opac=0.97, thick=0.7 + 0.5 * max(fold, 0), dry=0.3,
                 drysens=0.2, edge=0.35, ridge=0.5)
        ridge.append((fold, top, mid, bot, ln))
    # light held inside thin cloth: one saturated film where a single layer faces the east
    f, top, mid, bot, ln = max(ridge, key=lambda r: -abs(r[0]))
    glow = np.clip(base ** 0.75 * 1.12 + np.array([0.06, 0.03, 0.0]), 0, 1)
    P.stroke([top + 0.2 * (mid - top), mid, bot + 0.25 * (mid - bot)], rad * 0.9, glow, streak=0.02, opac=0.55, thick=0.5,
             dry=0.5, drysens=0.3, edge=0.6)
    # the lit ridge, loaded, laid once
    for f, top, mid, bot, ln in ridge:
        if f > 0.72:
            hi = np.clip(base * 1.22 + np.array([0.07, 0.09, 0.13]), 0, 1)
            P.stroke([top + 0.06 * (bot - top), mid, bot + 0.12 * (top - bot)], max(0.8, rad * 0.42), hi, streak=0.02, opac=0.95,
                     thick=1.5, dry=0.45, drysens=0.25, edge=0.3, ridge=0.8)
    # the printed prayer rides the folds
    if text and wpx > 34 * S:
        ink = np.clip(base * 0.42, 0, 1)
        for row in (0.26, 0.42, 0.58, 0.74):
            for f, top, mid, bot, ln in ridge:
                if ln < row + 0.1 or rng.random() < 0.2: continue
                p = top + (bot - top) * (row / ln) * ln + (mid - 0.5 * (top + bot)) * (1 - abs(2 * row - 1))
                d = (TR - TL) / wpx
                P.stroke([p - d * rad * 0.7, p + d * rad * 0.7], max(0.7, 0.028 * hpx), ink, streak=0.0, opac=0.6, thick=0.3,
                         dry=0.4, drysens=0.3, edge=0.5, smooth_pts=False)


FLAG_C = np.array([(0.16, 0.28, 0.62), (0.60, 0.65, 0.80), (0.58, 0.15, 0.16), (0.15, 0.37, 0.29), (0.70, 0.54, 0.18)])


def _string(P, A, B, sag, fw, rng, k0=0):
    """a line of flags from A to B (world), far end first."""
    S = P.S
    A = np.asarray(A, float); B = np.asarray(B, float)
    L = np.linalg.norm(B - A)
    n = int(L / (fw * 1.22))
    def at(t):
        p = A + (B - A) * t
        p[1] -= sag * 4 * t * (1 - t)
        return p
    def vis(p):
        x, y, zc = P.project(p)
        if zc <= 0.03 or x < 2 or x >= P.W - 2 or y < 2 or y >= P.H - 2: return True
        return P.depth[int(y), int(x)] > np.linalg.norm(p - np.array([0.0, EYE, 0.0])) - 0.006
    pts = [at(t) for t in np.linspace(0, 1, 60)]
    scr = [P.project(p) for p in pts]
    for i in range(len(pts) - 1):
        if scr[i][2] > 0.03 and scr[i + 1][2] > 0.03 and vis(pts[i]) and vis(pts[i + 1]):
            P.stroke([scr[i][:2], scr[i + 1][:2]], max(0.8, 1.1 * S * 0.3 / scr[i][2] * 0.25 + 0.6 * S),
                     [0.10, 0.09, 0.15], streak=0.0, opac=0.9, thick=0.4, dry=0.2, drysens=0.15, edge=0.5, smooth_pts=False)
    order = range(n) if np.linalg.norm(A - np.array([0, EYE, 0])) > np.linalg.norm(B - np.array([0, EYE, 0])) else range(n - 1, -1, -1)
    for i in order:
        if rng.random() < 0.07: continue                                   # one gone altogether
        sz = rng.uniform(0.86, 1.0)
        t0 = (i + 0.5 - 0.40 * sz) / n; t1 = (i + 0.5 + 0.40 * sz) / n
        a = at(t0); b = at(t1)
        if not vis(0.5 * (a + b)): continue
        if min(P.project(a)[2], P.project(b)[2]) < 0.035: continue
        w1 = rng.uniform(0.0, 0.75); w2 = np.clip(w1 + rng.normal(0, 0.25), 0.0, 0.95)
        wind = np.array([0.75, 0.22, -0.30]) * fw                           # the dawn wind comes down the valley
        da = np.array([0, -fw * 1.12, 0]) * (1 - 0.35 * w1) + wind * w1
        db = np.array([0, -fw * 1.12, 0]) * (1 - 0.35 * w2) + wind * w2
        col = FLAG_C[(i + k0) % 5].copy()
        worn = rng.uniform(0.0, 0.5)
        g = col.mean()
        col = col + (np.array([0.42, 0.46, 0.60]) * (0.5 + g) - col) * worn * 0.6     # bleached toward the shadow's own colour
        col *= rng.uniform(0.88, 1.06)
        torn = rng.uniform(0.35, 0.8) if rng.random() < 0.33 else 0.0
        TLp = P.project(a)[:2]; TRp = P.project(b)[:2]; BLp = P.project(a + da)[:2]; BRp = P.project(b + db)[:2]
        _cloth(P, TLp, TRp, BLp, BRp, np.clip(col, 0, 1), rng, torn, True)


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
                              opac=0.85, thick=0.2, thickv=0.1, dry=0.6, drysens=0.6, pickup=0.0, wander=0.10))
    # 2. block-in with a broad flat brush, edge to edge
    say("block-in", P.layer(blur(ref, 10 * S), 30, spacing=0.62, maxlen=5.0, tol=0.30, cjit=0.03, streak=0.04,
                            opac=0.96, thick=0.5, thickv=0.4, dry=0.35, drysens=0.3, pickup=0.12, wander=0.08))
    P.blend(0.30 + 0.70 * P.soft, length=24, soften=5)
    # 3. modelling
    say("modelling", P.layer(blur(ref, 4.5 * S), 15, spacing=0.7, err=0.075, maxlen=5.0, tol=0.22, cjit=0.03, streak=0.04,
                             opac=0.96, thick=0.6, thickv=0.5, dry=0.4, drysens=0.3, pickup=0.14, wander=0.07))
    P.blend(0.12 + 0.6 * P.soft, length=10, soften=2.0)
    # the sky once more, clean, with a wide brush and little on it
    skym = (P.mat == 0).astype(np.float32)
    say("sky", P.layer(blur(ref, 6 * S), 26, spacing=0.6, mask=skym, order='random', maxlen=7, tol=0.20, cjit=0.012, streak=0.02,
                       opac=0.9, thick=0.35, thickv=0.2, dry=0.3, drysens=0.3, pickup=0.25, wander=0.05))
    P.blend(skym * (1 - smooth(0.2, 0.6, P.detail)), length=34, soften=7)
    say("drawing", P.layer(blur(ref, 2.0 * S), 7.5, spacing=0.75, err=0.095, maxlen=5, tol=0.17, cjit=0.028, streak=0.035,
                           opac=0.97, thick=0.6, thickv=0.5, dry=0.4, drysens=0.3, pickup=0.10, wander=0.09))
    say("small", P.layer(blur(ref, 1.0 * S), 3.6, spacing=0.8, err=0.14, mask=0.22 + 0.78 * smooth(0.1, 0.5, P.detail),
                         maxlen=5, tol=0.14, cjit=0.03, streak=0.03, opac=0.98, thick=0.5, dry=0.35, drysens=0.25,
                         pickup=0.06, wander=0.07))
    say("accents", P.layer(ref, 1.9, spacing=0.85, err=0.17, mask=smooth(0.35, 0.7, P.detail), maxlen=5, tol=0.12,
                           cjit=0.02, streak=0.02, opac=0.99, thick=0.45, dry=0.3, drysens=0.2, pickup=0.03, wander=0.05))
    P.blend(0.5 * P.soft * smooth(0.5, 0.2, P.detail), length=6, soften=1.2)

    # 4. glaze: cobalt-violet, thin, into the near dark and the gorge
    P.glaze(blur(smooth(0.17, 0.04, L) * ((P.mat == 1) | (P.mat == 6) | (P.mat == 4)), 8 * S), [0.76, 0.78, 1.0], 0.25)

    # 5. the lights, loaded, laid once: only where the sun has arrived
    sunm = smooth(0.12, 0.45, P.sunm) * (P.mat == 1)
    lit = np.clip(ref * np.array([1.04, 1.01, 0.96]) + 0.01, 0, 1)
    say("lights", P.layer(lit, 6.0, spacing=1.0, mask=sunm * smooth(0.35, 0.6, L) * 0.6, maxlen=4.5, tol=0.14, cjit=0.04,
                          streak=0.05, opac=0.97, thick=1.5, thickv=0.4, dry=0.3, drysens=0.25, pickup=0.04, ridge=0.6))
    # the knife, on sunlit snow and nowhere else
    say("knife", P.layer(np.clip(lit * 1.04 + 0.02, 0, 1), 5.0, spacing=1.35, mask=sunm * smooth(0.3, 0.7, P.snow) * smooth(0.5, 0.75, L) * 0.6,
                         maxlen=7, tol=0.16, cjit=0.02, streak=0.07, opac=1.0, thick=2.4, thickv=0.2, dry=0.5, drysens=0.3,
                         pickup=0.0, edge=0.08, ridge=1.0, wander=0.04, sizev=0.5))

    CAM = np.array([0.0, EYE, 0.0])
    def visible(p, tol=0.05):
        x, y, zc = P.project(p)
        if zc <= 0 or x < 30 * S or x >= W - 30 * S or y < 30 * S or y >= H - 30 * S: return False
        return abs(P.depth[int(y), int(x)] - np.linalg.norm(np.asarray(p) - CAM)) < tol * (1 + zc)

    # 6. the shepherd and his sheep, already on the path below the village
    fr = np.random.default_rng(SEED + 9)
    spot = None
    for i in range(len(PATH) - 1):
        for t in np.linspace(0, 1, 30):
            px, pz = PATH[i] + (PATH[i + 1] - PATH[i]) * t
            if pz < 3.0 or pz > 6.1: continue
            p = np.array([px, height(px, pz, 9), pz])
            if visible(p, 0.02) and visible(p + np.array([-0.06, 0, -0.05]), 0.03):
                spot = (i, t, p); break
        if spot: break
    if spot:
        i, t, p = spot
        dirp = PATH[i + 1] - PATH[i]; dirp = dirp / np.linalg.norm(dirp)
        def ground(px, pz): return np.array([px, height(px, pz, 9), pz])
        sx, sy, zc = P.project(p)
        u = 0.5 * H / (zc * TY) * 0.01                                     # canvas px per metre there
        for j in range(7):                                                 # sheep first, strung along the path ahead of him
            off = 0.035 + 0.022 * j + fr.uniform(-0.006, 0.006)
            q = ground(p[0] + dirp[0] * off + fr.uniform(-0.012, 0.012), p[2] + dirp[1] * off + fr.uniform(-0.004, 0.004))
            if not visible(q, 0.03): continue
            qx, qy, _ = P.project(q)
            cs = np.array([0.50, 0.53, 0.68]) * fr.uniform(0.85, 1.1) if fr.random() < 0.8 else np.array([0.16, 0.14, 0.20])
            P.dab(qx, qy - 0.55 * u, fr.uniform(-0.3, 0.3), 1.3 * u, max(0.9, 0.42 * u), cs, thick=1.0, dry=0.2, drysens=0.1, edge=0.4)
        P.stroke([(sx, sy), (sx + 0.05 * u, sy - 1.25 * u)], max(1.0, 0.30 * u), [0.30, 0.11, 0.14], thick=1.1, dry=0.15, drysens=0.1,
                 edge=0.4, smooth_pts=False)
        P.dab(sx + 0.06 * u, sy - 1.55 * u, 0.0, 0.3 * u, max(0.8, 0.20 * u), [0.42, 0.30, 0.30], thick=1.0, dry=0.1, drysens=0.0)

    # 7. a thread of hearth smoke, from the old house at the top where it can be seen against the hill
    KS = NH - 1
    top = _hworld(KS, -0.45 * HOUSES[KS, 3], HOUSES[KS, 4] + 0.004, 0.0)
    sx, sy, zc = P.project(top)
    u = 0.5 * H / (zc * TY) * 0.01
    pts = [(sx, sy)]
    x_, y_ = sx, sy
    for i in range(1, 15):
        y_ -= 4.2 * u
        x_ -= (0.10 * i ** 1.5) * u + fr.normal(0, 0.25) * u
        pts.append((x_, y_))
    pts = np.array(pts)
    for k in range(5):
        j0 = k * 2; seg = pts[j0:j0 + 6] + fr.normal(0, 0.25 * u, (len(pts[j0:j0 + 6]), 2))
        if len(seg) < 3: continue
        P.stroke(seg, (0.8 + 0.55 * k) * u, np.array([0.66, 0.70, 0.88]) + 0.02 * k, streak=0.02, opac=0.72 - 0.09 * k, thick=0.25,
                 dry=0.75, drysens=0.5, pickup=0.25, edge=0.8)

    # 8. cloth: the flags, far end first
    POLT = np.array([POLX + 0.0020, POLY + POLH - 0.001, POLZ + 0.001])
    CHT = np.array([CHX + 0.003, CHY + 0.0585 * CHS, CHZ - 0.006])
    Bl = np.array([-0.46, shoulder(-0.46, 0.30) + 0.012, 0.30])
    fw = 0.0078
    _string(P, CHT, Bl, 0.006, fw, fr, 3)
    _string(P, CHT, POLT, 0.010, fw, fr, 0)

    # 9. the window, last: a wash of warmth on the wall, then the flame itself
    fx, fy = P.flame
    zc = P.project(WIN)[2]
    u = 0.5 * H / (zc * TY) * 0.01
    ww, wh = 1.25 * u, 1.05 * u
    P.stroke([(fx - ww * 0.5, fy), (fx + ww * 0.5, fy)], max(1.2, wh * 0.62), [1.0, 0.50, 0.10], thick=1.4, dry=0.15, drysens=0.05,
             streak=0.03, edge=0.35, smooth_pts=False)
    P.stroke([(fx - ww * 0.28, fy + 0.05 * wh), (fx + ww * 0.30, fy + 0.05 * wh)], max(1.0, wh * 0.40), [1.0, 0.78, 0.26], thick=2.0,
             dry=0.15, drysens=0.05, streak=0.02, edge=0.35, smooth_pts=False)
    P.stroke([(fx - ww * 0.06, fy + 0.10 * wh), (fx + ww * 0.10, fy + 0.10 * wh)], max(0.8, wh * 0.20), [1.0, 0.96, 0.78], thick=2.8,
             dry=0.1, drysens=0.0, streak=0.0, edge=0.3, smooth_pts=False)
    P.sign(W - 70 * S, H - 62 * S, 30 * S)


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0] != '-' else 1.0
    W, H = int(3200 * scale), int(2100 * scale)
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
