#!/usr/bin/env python3
"""
Hanle, the night the sky went red
Stanzin Dolkar

    python hanle_the_night_the_sky_went_red.py              # the picture, 4200 x 2800
    python hanle_the_night_the_sky_went_red.py 0.5          # a study at half size
    python hanle_the_night_the_sky_went_red.py 0.5 out.png  # a study, written where you say

The night of 10 to 11 May 2024, about two in the morning, from a shoulder
of rock below the telescope at 4,500 m, latitude 32.78 N, looking due
north.  Local sidereal time is close to 17 h: the pole star stands 32
degrees up, the bowl of the Plough is high on the left with its two
pointers, the Little Bear hangs over the pole, Cassiopeia is low on the
right and the outer Milky Way climbs out of it through Cepheus.

The red is oxygen at 630 nm, far to the north and very high: an arc and
a diffuse glow 300 km up, 500 to 1,500 km away, with faint rays along
the field lines and a little violet where the tops are in sunlight.  It
is integrated along each line of sight over a curved Earth, which is
what makes it an arch.  There is no green curtain at this latitude; the
thin green low down is the ordinary airglow.

The seventy brightest northern stars are put where they stood at that
hour.  Everything fainter is sampled from a disc galaxy seen from
inside, with dust in shells.  The ground is a heightfield marched
bearing by bearing and then put back through the lens; the river is a
mirror with a little wind on it and ice at its edges; the dome, the
gompa and the house are boxes, drums and spheres.

Nothing is photographed and nothing is downloaded.  Everything is seeded.
Needs numpy, scipy, numba, pillow.  Three to four minutes at full size
on eight cores, a little longer the first time while numba compiles.
"""
import sys, os, math, time
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
W = int(round(4200 * S)); H = int(round(2800 * S))
F = 2100.0 * S                       # focal length in pixels: 90 degrees across
YH = 0.70 * H                        # the horizon row
PITCH = math.atan((YH - H / 2) / F)
CP = math.cos(PITCH); SP = math.sin(PITCH)
SEED = 20240511
FREF = 4.6                           # what a sixth-magnitude star puts on the print
D2R = math.pi / 180.0
NS = int(math.ceil(YH)) + 2          # rows that hold sky

LAT = 32.78 * D2R
LST = 17.03                          # hours

NJ = dict(cache=True, fastmath=True)


# ---------------------------------------------------------------------
#  where things are tonight
# ---------------------------------------------------------------------
def radec_dir(ra_h, dec_deg):
    """Unit vector (east, north, up) of a place on the sky at this hour."""
    ha = (LST - ra_h) * 15.0 * D2R
    d = dec_deg * D2R
    return np.array([-math.cos(d) * math.sin(ha),
                     math.cos(LAT) * math.sin(d) - math.sin(LAT) * math.cos(d) * math.cos(ha),
                     math.sin(LAT) * math.sin(d) + math.cos(LAT) * math.cos(d) * math.cos(ha)])


_gx = radec_dir(17.7611, -29.008)            # toward the centre of the galaxy
_gz = radec_dir(12.8567, 27.128)             # toward its north pole
_gz = _gz - (_gz @ _gx) * _gx; _gz /= np.linalg.norm(_gz)
_gy = np.cross(_gz, _gx)
GAL = np.stack([_gx, _gy, _gz]).astype(np.float64)

# right ascension (h), declination (deg), magnitude, B-V
BRIGHT = np.array([
    # the Great Bear
    [11.062, 61.75, 1.79, 1.07], [11.031, 56.38, 2.37, -0.02], [11.897, 53.69, 2.44, 0.00],
    [12.257, 57.03, 3.31, 0.08], [12.900, 55.96, 1.77, -0.02], [13.399, 54.93, 2.27, 0.02],
    [13.420, 54.99, 4.01, 0.16], [13.792, 49.31, 1.86, -0.19], [8.504, 60.72, 3.36, 0.85],
    [9.525, 63.06, 3.67, 0.33], [9.850, 59.04, 3.80, 0.29], [9.548, 51.68, 3.17, 0.46],
    [8.987, 48.04, 3.14, 0.19], [9.060, 47.16, 3.60, 0.00], [10.285, 42.91, 3.45, 0.03],
    [10.372, 41.50, 3.05, 1.59], [11.161, 44.50, 3.01, 1.14], [11.767, 47.78, 3.71, 1.18],
    [11.308, 33.09, 3.48, 1.40], [11.303, 31.53, 3.79, 0.59],
    # the Little Bear
    [2.530, 89.264, 1.98, 0.60], [14.845, 74.16, 2.08, 1.47], [15.345, 71.83, 3.05, 0.05],
    [17.537, 86.59, 4.36, 0.02], [16.766, 82.04, 4.23, 0.89], [15.734, 77.79, 4.32, 0.04],
    [16.292, 75.76, 4.95, 0.37],
    # Cassiopeia
    [0.675, 56.54, 2.23, 1.17], [0.153, 59.15, 2.27, 0.34], [0.945, 60.72, 2.47, -0.15],
    [1.430, 60.24, 2.68, 0.13], [1.907, 63.67, 3.38, -0.15], [0.818, 57.82, 3.44, 0.57],
    [0.616, 53.90, 3.66, -0.20], [2.484, 67.40, 4.52, 0.12], [2.057, 72.42, 3.98, -0.01],
    # Cepheus
    [21.310, 62.59, 2.44, 0.22], [21.478, 70.56, 3.23, -0.22], [23.656, 77.63, 3.21, 1.03],
    [22.181, 58.20, 3.35, 1.57], [20.755, 61.84, 3.43, 0.92], [22.828, 66.20, 3.52, 1.05],
    [22.486, 58.42, 4.00, 0.60], [21.725, 58.78, 4.08, 2.35], [22.250, 57.04, 4.19, 0.28],
    [20.493, 62.99, 4.22, 0.20],
    # the Dragon
    [17.943, 51.49, 2.23, 1.52], [17.507, 52.30, 2.79, 0.98], [16.400, 61.51, 2.74, 0.91],
    [19.209, 67.66, 3.07, 1.00], [17.146, 65.71, 3.17, -0.12], [15.415, 58.97, 3.29, 1.16],
    [18.351, 72.73, 3.57, 0.49], [14.073, 64.38, 3.65, -0.05], [17.892, 56.87, 3.75, 1.18],
    [12.558, 69.79, 3.87, -0.13], [11.523, 69.33, 3.84, 1.62], [19.803, 70.27, 3.83, 0.89],
    [16.031, 58.57, 4.01, 0.52], [18.346, 71.34, 4.22, 0.38],
    # the Swan, the Lizard, the Herdsman, the Hounds, the Giraffe
    [20.690, 45.28, 1.25, 0.09], [20.370, 40.26, 2.20, 0.68], [19.750, 45.13, 2.87, -0.03],
    [19.495, 51.73, 3.79, 0.15], [19.285, 53.37, 3.77, 0.96], [20.227, 46.74, 3.79, 1.28],
    [21.082, 43.93, 3.72, 1.65], [22.522, 50.28, 3.77, 0.01], [14.420, 51.85, 4.05, 0.50],
    [14.273, 46.09, 4.18, 0.08], [12.934, 38.32, 2.90, -0.12], [12.562, 41.36, 4.26, 0.59],
    [5.057, 60.44, 4.03, 0.92], [4.901, 66.34, 4.29, 0.03], [17.658, 46.01, 3.80, -0.18],
    [16.329, 46.31, 3.89, -0.15],
])

# dust shells: distance (kpc), opacity, scale height seen from here, sparseness, frequency, octaves, tilt
SK = np.array([0.13, 0.20, 0.32, 0.50, 0.80, 1.30, 2.10, 3.40, 5.50])
AK = np.array([2.0, 2.1, 2.3, 2.5, 2.8, 3.4, 4.2, 5.5, 7.0])
HK = np.array([0.013, 0.019, 0.027, 0.036, 0.048, 0.058, 0.062, 0.065, 0.065])
TK = np.array([0.16, 0.14, 0.10, 0.06, 0.02, 0.0, 0.0, 0.0, 0.0])
FK = 2.6 * (SK / 0.13) ** 0.55
OK = np.array([8, 8, 8, 7, 7, 7, 6, 6, 6])
BK = np.array([0.075, 0.065, 0.05, 0.03, 0.012, 0.0, 0.0, 0.0, 0.0])

# thin dark filaments that leave the band: l0, b0, l1, b1 (degrees), width, opacity
STRAND = np.array([[113.5, 1.6, 105.0, 10.5, 0.24, 2.0],
                   [118.0, 2.4, 114.5, 8.6, 0.16, 1.1]])


def strand_centre(k, t):
    """(l, b) in degrees along filament k at parameter t."""
    l0, b0, l1, b1 = STRAND[k, :4]
    nx, ny = -(b1 - b0), (l1 - l0)
    n = math.hypot(nx, ny); nx /= n; ny /= n
    w = 0.55 * math.sin(t * 5.3 + 1.1 + 2.0 * k) + 0.25 * math.sin(t * 13.0 + 0.4)
    return l0 + (l1 - l0) * t + nx * w, b0 + (b1 - b0) * t + ny * w


# ---------------------------------------------------------------------
#  noise
# ---------------------------------------------------------------------
@njit(inline='always', **NJ)
def _h3(ix, iy, iz, sd):
    n = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (sd * 2654435761)
    n &= 0xFFFFFFFF
    n = ((n ^ (n >> 15)) * 2246822519) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 3266489917) & 0xFFFFFFFF
    n ^= n >> 16
    return n / 4294967296.0


@njit(**NJ)
def noise3(x, y, z, sd):
    fx = math.floor(x); fy = math.floor(y); fz = math.floor(z)
    ix = int(fx); iy = int(fy); iz = int(fz)
    tx = x - fx; ty = y - fy; tz = z - fz
    ux = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
    uy = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
    uz = tz * tz * tz * (tz * (tz * 6 - 15) + 10)
    a = _h3(ix, iy, iz, sd); b = _h3(ix + 1, iy, iz, sd)
    c = _h3(ix, iy + 1, iz, sd); d = _h3(ix + 1, iy + 1, iz, sd)
    e = _h3(ix, iy, iz + 1, sd); f = _h3(ix + 1, iy, iz + 1, sd)
    g = _h3(ix, iy + 1, iz + 1, sd); h = _h3(ix + 1, iy + 1, iz + 1, sd)
    ab = a + (b - a) * ux; cd = c + (d - c) * ux
    ef = e + (f - e) * ux; gh = g + (h - g) * ux
    v0 = ab + (cd - ab) * uy; v1 = ef + (gh - ef) * uy
    return 2.0 * (v0 + (v1 - v0) * uz) - 1.0


@njit(**NJ)
def noise2(x, y, sd):
    fx = math.floor(x); fy = math.floor(y)
    ix = int(fx); iy = int(fy)
    tx = x - fx; ty = y - fy
    ux = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
    uy = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
    a = _h3(ix, iy, 0, sd); b = _h3(ix + 1, iy, 0, sd)
    c = _h3(ix, iy + 1, 0, sd); d = _h3(ix + 1, iy + 1, 0, sd)
    ab = a + (b - a) * ux; cd = c + (d - c) * ux
    return 2.0 * (ab + (cd - ab) * uy) - 1.0


@njit(**NJ)
def fbm3(x, y, z, o, sd):
    s = 0.0; a = 0.5
    for i in range(o):
        s += a * noise3(x, y, z, sd + i)
        nx = 1.60 * y + 1.20 * z; ny = -1.60 * x + 0.72 * y - 0.96 * z; nz = -1.20 * x - 0.96 * y + 1.28 * z
        x = nx + 3.1; y = ny + 1.7; z = nz + 5.3
        a *= 0.5
    return s


@njit(**NJ)
def ridged3(x, y, z, o, sd):
    s = 0.0; a = 0.5; w = 1.0; nrm = 0.0
    for i in range(o):
        r = 1.0 - abs(noise3(x, y, z, sd + i))
        r = r * r
        s += a * r * w; nrm += a
        w = min(1.0, r * 1.6 + 0.15)
        nx = 1.64 * y + 1.23 * z; ny = -1.64 * x + 0.738 * y - 0.984 * z; nz = -1.23 * x - 0.984 * y + 1.312 * z
        x = nx + 3.1; y = ny + 1.7; z = nz + 5.3
        a *= 0.56
    return s / nrm


@njit(**NJ)
def fbm2(x, y, o, sd):
    s = 0.0; a = 0.5
    for i in range(o):
        s += a * noise2(x, y, sd + i)
        nx = 1.6 * x + 1.2 * y; ny = -1.2 * x + 1.6 * y
        x = nx + 7.3; y = ny + 2.9
        a *= 0.5
    return s


@njit(**NJ)
def rfbm2(x, y, o, sd):
    s = 0.0; a = 0.5; w = 1.0
    for i in range(o):
        r = 1.0 - abs(noise2(x, y, sd + i))
        r = r * r * w
        s += a * r
        w = min(1.0, max(0.0, r * 1.8))
        nx = 1.62 * x + 1.21 * y; ny = -1.21 * x + 1.62 * y
        x = nx + 7.3; y = ny + 2.9
        a *= 0.5
    return s


@njit(inline='always', **NJ)
def sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------
#  the lens
# ---------------------------------------------------------------------
@njit(inline='always', **NJ)
def pix_dir(px, py, W, H, F, CPc, SPc):
    u = (px - 0.5 * W) / F; v = (0.5 * H - py) / F
    dx = u; dy = CPc - v * SPc; dz = SPc + v * CPc
    n = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx * n, dy * n, dz * n


@njit(inline='always', **NJ)
def dir_pix(dx, dy, dz, W, H, F, CPc, SPc):
    t = dy * CPc + dz * SPc
    return 0.5 * W + F * dx / t, 0.5 * H - F * (-dy * SPc + dz * CPc) / t


@njit(inline='always', **NJ)
def bil(img, fx, fy, c):
    Hh = img.shape[0]; Ww = img.shape[1]
    if fx < 0.0: fx = 0.0
    if fx > Ww - 1.001: fx = Ww - 1.001
    if fy < 0.0: fy = 0.0
    if fy > Hh - 1.001: fy = Hh - 1.001
    x0 = int(fx); y0 = int(fy)
    tx = fx - x0; ty = fy - y0
    return (img[y0, x0, c] * (1 - tx) + img[y0, x0 + 1, c] * tx) * (1 - ty) + \
           (img[y0 + 1, x0, c] * (1 - tx) + img[y0 + 1, x0 + 1, c] * tx) * ty


# ---------------------------------------------------------------------
#  the dust, and the light behind it
# ---------------------------------------------------------------------
@njit(**NJ)
def cloud(x, y, z, sd, o, th):
    wx = fbm3(x * 0.7 + 11.3, y * 0.7 + 5.1, z * 0.7 + 2.9, 3, sd + 101)
    wy = fbm3(x * 0.7 + 3.7, y * 0.7 + 19.1, z * 0.7 + 7.7, 3, sd + 202)
    wz = fbm3(x * 0.7 + 8.9, y * 0.7 + 1.3, z * 0.7 + 13.1, 3, sd + 303)
    x += 0.60 * wx; y += 0.60 * wy; z += 0.60 * wz
    m = fbm3(x, y, z, 4, sd)
    r = ridged3(x * 2.3 + 0.35 * wy, y * 2.3 + 0.35 * wz, z * 2.3 + 0.35 * wx, o, sd + 7)
    v = m * 1.6 + 1.0 * (r - 0.50)
    d = sstep(th, th + 0.78, v)
    return d ** 1.5 * (0.55 + 0.9 * r) * 1.6


@njit(parallel=True, **NJ)
def sky_pass(NSr, W, H, F, CPc, SPc, G, SKa, AKa, FKa, OKa, BKa, HKa, TKa, STR, NEB, mw, e0m, taum, taun):
    K = SKa.shape[0]
    zsun = 0.015
    for j in prange(NSr):
        tk = np.zeros(K)
        for i in range(W):
            dx, dy, dz = pix_dir(i + 0.5, j + 0.5, W, H, F, CPc, SPc)
            if dz < -0.01:
                continue
            X = dx * G[0, 0] + dy * G[0, 1] + dz * G[0, 2]
            Y = dx * G[1, 0] + dy * G[1, 1] + dz * G[1, 2]
            Z = dx * G[2, 0] + dy * G[2, 1] + dz * G[2, 2]
            cb = math.sqrt(max(1e-9, 1.0 - Z * Z))
            ldeg = math.atan2(Y, X) * 57.29578; bdeg = math.asin(Z) * 57.29578
            # ---- shells of dust
            tn = 0.0; tt = 0.0
            for k in range(K):
                s = SKa[k]
                zz = s * (Z - BKa[k] * cb * 0.5) + zsun
                R = math.sqrt((8.0 - s * X) ** 2 + (s * Y) ** 2)
                prof = math.exp(-abs(zz) / HKa[k]) * math.exp(-(max(R, 3.0) - 8.0) / 4.5)
                a = AKa[k] * prof
                if k < 5:
                    a *= 0.85
                if a < 0.004:
                    tk[k] = 0.0
                    continue
                f = FKa[k]
                c = cloud(X * f + 13.7 * k, Y * f - 7.1 * k, Z * f + 3.3 * k, 1000 + 37 * k, OKa[k], TKa[k])
                tk[k] = a * (c + (0.04 if k < 5 else 0.09))
            # the filaments
            for q in range(STR.shape[0]):
                l0 = STR[q, 0]; b0 = STR[q, 1]; l1 = STR[q, 2]; b1 = STR[q, 3]
                ex = l1 - l0; ey = b1 - b0; L2 = ex * ex + ey * ey
                t = ((ldeg - l0) * ex + (bdeg - b0) * ey) / L2
                if t > -0.1 and t < 1.1:
                    Ln = math.sqrt(L2)
                    nx = -ey / Ln; ny = ex / Ln
                    wob = 0.55 * math.sin(t * 5.3 + 1.1 + 2.0 * q) + 0.25 * math.sin(t * 13.0 + 0.4)
                    dd = (ldeg - l0) * nx + (bdeg - b0) * ny - wob
                    dd += 0.16 * noise2(ldeg * 1.3 + 7.0, bdeg * 1.3, 57) + 0.07 * noise2(ldeg * 4.0, bdeg * 4.0 + 2.0, 58)
                    wd = STR[q, 4] * (1.0 + 0.55 * noise2(t * 7.0, 3.3 + q, 55)) * (0.75 + 0.6 * t)
                    g = 1.0 / (1.0 + (dd / wd) ** 2) ** 1.6
                    fr = max(0.0, 0.55 + 0.75 * noise2(ldeg * 1.1, bdeg * 1.1, 56) + 0.3 * noise2(ldeg * 5.0, bdeg * 5.0, 59))
                    tk[1] += STR[q, 5] * g * fr * sstep(-0.08, 0.06, t) * sstep(1.08, 0.75, t)
            for k in range(K):
                tt += tk[k]
                if k < 5: tn += tk[k]
            # ---- the light, marched out through the shells
            T0 = 1.0; T1 = 1.0; T2 = 1.0
            a0 = 0.0; a1 = 0.0; a2 = 0.0; e0 = 0.0
            kk = 0; s = 0.01
            mott = 1.0 + 0.22 * fbm3(X * 46.0, Y * 46.0, Z * 46.0, 4, 31) + 0.10 * fbm3(X * 300.0, Y * 300.0, Z * 300.0, 3, 32)
            while s < 22.0:
                ds = 0.015 + 0.045 * s
                if ds > 0.30: ds = 0.30
                sm = s + 0.5 * ds
                while kk < K and SKa[kk] < sm:
                    tv = tk[kk]
                    T0 *= math.exp(-0.80 * tv); T1 *= math.exp(-tv); T2 *= math.exp(-1.22 * tv)
                    kk += 1
                xg = sm * X - 8.0; yg = sm * Y; zg = sm * Z + zsun
                R = math.sqrt(xg * xg + yg * yg)
                az = abs(zg)
                if az > 2.4 or R > 19.0:
                    break
                arm = noise3(xg * 0.8 + 3.1, yg * 0.8 + 7.7, zg * 3.0, 77)
                arm2 = noise3(xg * 2.6 + 1.1, yg * 2.6 + 4.2, zg * 8.0, 78)
                rd = math.exp(-(R - 8.0) / 2.6) * (0.90 * math.exp(-az / 0.22) + 0.07 * math.exp(-az / 0.60))
                rd *= 1.35 * (1.0 + 0.68 * arm + 0.36 * arm2) * mott
                mb = math.sqrt((xg / 1.35) ** 2 + (yg / 0.90) ** 2 + (zg / 0.68) ** 2)
                rb = 54.0 * math.exp(-2.2 * mb ** 1.4) * (0.6 + 0.4 * mott)
                a0 += (rd * 0.80 + rb * 1.08) * T0 * ds
                a1 += (rd * 0.92 + rb * 0.90) * T1 * ds
                a2 += (rd * 1.10 + rb * 0.66) * T2 * ds
                e0 += (rd + rb) * ds
                s += ds
            # ---- a few knots of hydrogen, in front, muted
            for q in range(NEB.shape[0]):
                dl = ldeg - NEB[q, 0]; db = bdeg - NEB[q, 1]
                rr = (dl * dl + db * db) / (NEB[q, 2] * NEB[q, 2])
                if rr < 6.0:
                    sh = math.exp(-rr) * max(0.0, 0.55 + 1.3 * fbm3(X * 150.0 + q, Y * 150.0, Z * 150.0, 4, 90 + q))
                    sh *= NEB[q, 3] * math.exp(-0.35 * tn)
                    a0 += sh * 1.00; a1 += sh * 0.30; a2 += sh * 0.36
            mw[j, i, 0] = a0; mw[j, i, 1] = a1; mw[j, i, 2] = a2
            e0m[j, i] = e0; taum[j, i] = tt; taun[j, i] = tn


# ---------------------------------------------------------------------
#  stars, one at a time
# ---------------------------------------------------------------------
def bb_rgb(T):
    """Black-body colour, linear, unit luminance."""
    t = np.asarray(T, np.float64) / 100.0
    r = np.where(t <= 66, 255.0, 329.698727446 * np.maximum(t - 60, 1e-3) ** -0.1332047592)
    g = np.where(t <= 66, 99.4708025861 * np.log(np.maximum(t, 1)) - 161.1195681661,
                 288.1221695283 * np.maximum(t - 60, 1e-3) ** -0.0755148492)
    b = np.where(t >= 66, 255.0, np.where(t <= 19, 0.0,
                 138.5177312231 * np.log(np.maximum(t - 10, 1e-3)) - 305.0447927307))
    c = np.stack([r, g, b], -1).clip(0, 255) / 255.0
    c = c ** 2.2
    l = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
    return c / l[..., None]


@njit(**NJ)
def splat(img, xs, ys, fl, col, hs):
    Hh = img.shape[0]; Ww = img.shape[1]
    n = xs.shape[0]
    for q in range(n):
        f = fl[q]
        x = xs[q]; y = ys[q]
        # core, skirt, and for the few that blaze a small soft halo
        rad = 2
        if f > 0.6: rad = 4
        if f > 8.0: rad = int(min(70.0 * hs, 5.0 + 2.6 * hs * f ** 0.42))
        x0 = int(math.floor(x)) - rad; x1 = int(math.floor(x)) + rad + 1
        y0 = int(math.floor(y)) - rad; y1 = int(math.floor(y)) + rad + 1
        if x1 < 0 or y1 < 0 or x0 >= Ww or y0 >= Hh:
            continue
        s1 = 0.60; s2 = 1.35 * max(1.0, hs ** 0.5); r0 = 3.2 * hs
        k1 = 0.84 / (6.2832 * s1 * s1); k2 = 0.13 / (6.2832 * s2 * s2)
        k3 = 0.03 * 0.9 / (3.14159 * r0 * r0)
        for yy in range(max(0, y0), min(Hh, y1 + 1)):
            for xx in range(max(0, x0), min(Ww, x1 + 1)):
                ddx = xx + 0.5 - x; ddy = yy + 0.5 - y
                r2 = ddx * ddx + ddy * ddy
                v = k1 * math.exp(-r2 / (2 * s1 * s1))
                if rad > 2:
                    v += k2 * math.exp(-r2 / (2 * s2 * s2))
                if rad > 4:
                    v += k3 * (1.0 + r2 / (r0 * r0)) ** -1.9 * sstep(rad * rad, 0.3 * rad * rad, r2)
                v *= f
                img[yy, xx, 0] += v * col[q, 0]
                img[yy, xx, 1] += v * col[q, 1]
                img[yy, xx, 2] += v * col[q, 2]


def make_stars(e0, tau, taun, rng):
    N = 2_300_000
    a1, a2, m_lo, m_b, m_hi = 0.50, 0.40, 4.35, 8.0, 14.0
    n1 = 10 ** (a1 * m_b) - 10 ** (a1 * m_lo)
    n2 = a1 / a2 * 10 ** (a1 * m_b) * (10 ** (a2 * (m_hi - m_b)) - 1.0)
    e0n = (e0 / np.percentile(e0, 99.7)).clip(0, 1) ** 0.62
    out_x = []; out_y = []; out_m = []
    got = 0
    while got < N:
        n = 3_000_000
        u = rng.random(n)
        first = rng.random(n) < n1 / (n1 + n2)
        m = np.where(first,
                     np.log10(u * n1 + 10 ** (a1 * m_lo)) / a1,
                     m_b + np.log10(1.0 + u * (10 ** (a2 * (m_hi - m_b)) - 1.0)) / a2)
        x = rng.random(n) * W; y = rng.random(n) * (YH + 1.0)
        ix = np.minimum(x.astype(np.int32), W - 1); iy = np.minimum(y.astype(np.int32), NS - 1)
        uu = (x - W / 2) / F; vv = (H / 2 - y) / F
        sol = (1 + uu * uu + vv * vv) ** -1.5
        cm = np.interp(m, [4.0, 7.0, 10.5], [0.55, 0.22, 0.10])
        wgt = sol * (cm + (1 - cm) * e0n[iy, ix])
        keep = rng.random(n) < wgt
        out_x.append(x[keep]); out_y.append(y[keep]); out_m.append(m[keep]); got += keep.sum()
    x = np.concatenate(out_x)[:N]; y = np.concatenate(out_y)[:N]; m = np.concatenate(out_m)[:N]
    # a few clusters, loose and tight, where the band is
    for q in range(14):
        cx = rng.random() * W; cy = rng.random() * YH * 0.9
        ix = int(cx); iy = int(cy)
        if e0n[iy, ix] < 0.45 or tau[iy, ix] > 1.2:
            continue
        nn = int(rng.integers(50, 200)); sg = rng.uniform(4, 12) * S
        rr = np.abs(rng.standard_cauchy(nn)).clip(0, 6) * sg * 0.6
        th = rng.random(nn) * 6.2832
        x = np.concatenate([x, cx + rr * np.cos(th)]); y = np.concatenate([y, cy + rr * np.sin(th)])
        m = np.concatenate([m, rng.uniform(7.4, 12.5, nn)])
    n = x.size
    ix = np.clip(x.astype(np.int32), 0, W - 1); iy = np.clip(y.astype(np.int32), 0, NS - 1)
    # temperature: most are cool and faint; a few are hot and blaze
    T = np.where(rng.random(n) < 0.72, rng.normal(4700, 750, n), rng.normal(6500, 1100, n))
    hot = rng.random(n) < np.interp(m, [3, 8, 12], [0.30, 0.07, 0.02])
    T = np.where(hot, 8000 + rng.exponential(5000, n), T)
    giant = (~hot) & (m < 6.0) & (rng.random(n) < 0.35)
    T = np.where(giant, rng.normal(4000, 350, n), T).clip(2900, 30000)
    col = bb_rgb(T)
    col = 1.0 + (col - 1.0) * 0.62           # hold the colour back
    # dust in front of each star
    behind = rng.random(n) < np.interp(m, [3, 6, 9.5], [0.25, 0.6, 0.93])
    ts = taun[iy, ix] * behind * rng.random(n) ** 0.35 + (tau - taun)[iy, ix] * rng.random(n) ** 2.2 * behind
    ext = np.exp(-ts[:, None] * np.array([0.80, 1.0, 1.22])[None, :])
    fl = FREF * 10 ** (-0.4 * (m - 6.0)) * (1.0 + 0.45 * np.clip(5.0 - m, 0, 4))
    col = col * ext
    return x, y, fl, col


def dir_to_pixel(d):
    t = d[1] * CP + d[2] * SP
    return W / 2 + F * d[0] / t, H / 2 - F * (-d[1] * SP + d[2] * CP) / t, t


def galactic_dir(ldeg, bdeg):
    l = ldeg * D2R; b = bdeg * D2R
    g = np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)])
    return GAL.T @ g


# ---------------------------------------------------------------------
#  the red
# ---------------------------------------------------------------------
AY0 = 760.0          # how far north the arc stands, km
DIPC = 0.62          # cotangent of the field's dip up there


@njit(**NJ)
def aurora_los(dx, dy, dz, ns):
    """Red line and violet, integrated along one line of sight over a curved Earth.
    Structure is laid out at the feet of the field lines, so rays lean as they must."""
    Re = 6371.0
    if dz < 0.0:
        dz = 0.0
    hd = math.sqrt(dx * dx + dy * dy) + 1e-9
    s0 = -Re * dz + math.sqrt(Re * Re * dz * dz + 2 * Re * 150.0 + 150.0 * 150.0)
    s1 = -Re * dz + math.sqrt(Re * Re * dz * dz + 2 * Re * 900.0 + 900.0 * 900.0)
    ds = (s1 - s0) / ns
    ex = dx / hd; ey = dy / hd
    stepx = 1.5 * abs(ds * dx) + 1e-3
    w1 = min(1.0, 46.0 / stepx); w2 = min(1.0, 17.0 / stepx); w3 = min(1.0, 8.5 / stepx)
    red = 0.0; vio = 0.0
    for k in range(ns):
        s = s0 + (k + 0.5) * ds
        h = math.sqrt(Re * Re + 2 * Re * s * dz + s * s) - Re
        g = Re * math.atan2(s * hd, Re + s * dz)
        gx = g * ex; gy = g * ey
        yf = gy + (h - 300.0) * DIPC
        if yf < 200.0:
            continue
        wv = noise2(gx / 820.0 + 1.3, yf / 900.0 + 0.2, 201)
        yc = AY0 + 85.0 * wv + 40.0 * noise2(gx / 260.0, 3.3, 202)
        d1 = yf - yc
        a = 1.45 * math.exp(-(d1 / 125.0) ** 2)
        a += 0.85 * sstep(-60.0, 380.0, d1) * math.exp(-max(0.0, d1 - 380.0) / 1700.0)
        d3 = yf - (yc - 235.0 + 45.0 * noise2(gx / 400.0 + 5.0, 1.1, 203))
        a += 0.13 * math.exp(-(d3 / 48.0) ** 2)
        if a < 1e-3:
            continue
        a *= 1.0 + 0.20 * noise2(yf / 85.0 + 2.0 * wv, gx / 1500.0, 204)       # faint bands
        ew = 0.30 + 0.80 * math.exp(-((gx + 130.0) / 540.0) ** 2)
        ew *= 1.0 + 0.30 * noise2(gx / 330.0 + 2.0, yf / 800.0, 205)
        env = sstep(-0.25, 0.45, noise2(gx / 210.0 + 9.0, yf / 700.0, 206))
        r1 = max(0.0, noise2(gx / 46.0 + 1.0, yf / 520.0, 211))
        r2 = max(0.0, noise2(gx / 17.0 + 4.0, yf / 300.0, 213))
        r3 = max(0.0, noise2(gx / 8.5 + 7.0, yf / 200.0, 215))
        ray = env * (0.40 * w1 * r1 * r1 + 1.3 * w2 * r2 * r2 + 0.45 * w3 * r3 * r3)
        sc = 50.0 * (1.0 + 1.0 * ray)
        u = (h - 262.0) / sc
        hr = math.exp(1.0 - u - math.exp(-u))
        if u > 0.0:
            hr /= 1.0 + (u / 3.0) ** 4
        uv = (h - 350.0) / 66.0
        hv = math.exp(1.0 - uv - math.exp(-uv))
        if uv > 0.0:
            hv /= 1.0 + (uv / 2.3) ** 4
        red += a * ew * (1.0 + 1.6 * ray) * hr
        vio += a * ew * (0.06 + 0.50 * ray) * hv
    return red * ds / 1000.0, vio * ds / 1000.0


@njit(parallel=True, **NJ)
def aurora_pass(Hd, Wd, step, W, H, F, CPc, SPc, out):
    for j in prange(Hd):
        for i in range(Wd):
            dx, dy, dz = pix_dir((i + 0.5) * step, (j + 0.5) * step, W, H, F, CPc, SPc)
            r, v = aurora_los(dx, dy, dz, 110)
            out[j, i, 0] = r; out[j, i, 1] = v


AUR_R = np.array([1.00, 0.030, 0.032])      # what 630 nm does to the print
AUR_V = np.array([0.42, 0.11, 1.00])        # the sunlit tops
AUR_GAIN = 0.80


# ---------------------------------------------------------------------
#  air: extinction, airglow, the sky's own floor
# ---------------------------------------------------------------------
@njit(parallel=True, **NJ)
def air_pass(NSr, W, H, F, CPc, SPc, mw, st, aur, out, EXPO, AG, AR, AV):
    Re = 6371.0
    for j in prange(NSr):
        for i in range(W):
            dx, dy, dz = pix_dir(i + 0.5, j + 0.5, W, H, F, CPc, SPc)
            sa = max(dz, 0.0)
            adeg = math.asin(sa) * 57.29578
            X = 1.0 / (sa + 0.50572 * (adeg + 6.07995) ** -1.6364)
            t0 = math.exp(-0.050 * X); t1 = math.exp(-0.068 * X); t2 = math.exp(-0.100 * X)
            c2 = 1.0 - sa * sa
            # green line, 95 km
            q = Re / (Re + 95.0)
            V = 1.0 / math.sqrt(1.0 - q * q * c2)
            tt = -Re * sa + math.sqrt(Re * Re * sa * sa + 2 * Re * 95.0 + 95.0 * 95.0)
            gx = tt * dx; gy = tt * dy
            wv = fbm2(gx / 420.0, gy / 420.0, 3, 61)
            ph = (gx * 0.82 + gy * 0.57) / 58.0 + 4.5 * wv
            bands = math.sin(6.2832 * ph)
            ph2 = (gx * -0.35 + gy * 0.94) / 130.0 + 1.8 * fbm2(gx / 600.0 + 9.0, gy / 600.0, 2, 62)
            bands2 = math.sin(6.2832 * ph2)
            patch = fbm2(gx / 260.0 + 4.0, gy / 260.0 + 1.0, 4, 63)
            grn = 0.0056 * 0.50 * V * (1.0 + 0.06 * bands * (0.5 + 0.5 * bands2) + 0.10 * bands2 + 0.45 * patch)
            bg = 0.0038 * (1.0 + 0.10 * V)
            ar = aur[j, i, 0] * AG; av = aur[j, i, 1] * AG
            r = grn * 0.40 + bg * 0.36 + ar * AR[0] + av * AV[0]
            g = grn * 1.00 + bg * 0.60 + ar * AR[1] + av * AV[1]
            b = grn * 0.52 + bg * 1.00 + ar * AR[2] + av * AV[2]
            # what the air removes it partly gives back
            m = (r + g + b) / 3.0
            r = r * t0 + (0.55 * r + 0.45 * m) * (1 - t0) * 0.50
            g = g * t1 + (0.55 * g + 0.45 * m) * (1 - t1) * 0.50
            b = b * t2 + (0.55 * b + 0.45 * m) * (1 - t2) * 0.46
            out[j, i, 0] = (mw[j, i, 0] * EXPO + st[j, i, 0]) * t0 + r
            out[j, i, 1] = (mw[j, i, 1] * EXPO + st[j, i, 1]) * t1 + g
            out[j, i, 2] = (mw[j, i, 2] * EXPO + st[j, i, 2]) * t2 + b


# ---------------------------------------------------------------------
#  the ground, in metres.  x east, y north, z above the river.
#  I am at the origin, forty metres up.
# ---------------------------------------------------------------------
EZG = 40.0                                   # the rock under my feet
DHX, DHY, DHH = 211.0, 583.0, 90.0           # the telescope's hill
DH_C = math.cos(38.0 * D2R); DH_S = math.sin(38.0 * D2R)
CRX, CRY, CRH = -1450.0, 2850.0, 112.0       # the gompa's crag
CHX0, CHY0, CHC = -1500.0, 60.0, 3.0         # the river grid: origin and cell
SNOW0 = 640.0

# the road up to the telescope, in the hill's own coordinates (along, across)
ROAD = np.array([[-205.0, 62.0], [-165.0, -52.0], [-128.0, 48.0], [-96.0, -40.0], [-66.0, 34.0],
                 [-40.0, -22.0], [-18.0, 14.0], [-4.0, 2.0]])


def spline(pts, step=4.0):
    P = np.array(pts, np.float64)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


def river_segments(rng):
    """The river and its side water as short segments with a half-width at each end."""
    main = spline([(-300, 3900), (-120, 3200), (-430, 2600), (-90, 2050), (-390, 1600), (-30, 1250),
                   (-310, 960), (40, 760), (-235, 575), (-25, 440), (-200, 335), (-70, 238),
                   (120, 190), (330, 214), (560, 165), (950, 200)])
    braid = spline([(-120, 3200), (160, 2700), (350, 2150), (110, 1700), (310, 1330), (120, 1040), (40, 760)])
    oxbow = spline([(-330, 300), (-420, 345), (-470, 430), (-440, 520), (-365, 560)])
    back = spline([(330, 214), (420, 290), (560, 330), (700, 300), (820, 330)])
    segs = []
    for (P, w0, taper) in ((main, 13.0, False), (braid, 7.5, False), (oxbow, 9.0, True), (back, 6.0, True)):
        n = len(P)
        s = np.concatenate([[0], np.cumsum(np.hypot(*(P[1:] - P[:-1]).T))])
        wn = gaussian_filter(rng.normal(0, 1, n), 9.0); wn /= (wn.std() + 1e-9)
        w = w0 * (1.0 + 0.30 * wn)
        if taper:
            w *= np.clip(np.minimum(s, s[-1] - s) / 60.0, 0.05, 1.0) ** 0.7
        else:
            w *= 1.0 + 0.5 * np.exp(-((P[:, 1] - 215.0) / 120.0) ** 2)   # the near reach is wide and shallow
        for k in range(n - 1):
            segs.append([P[k, 0], P[k, 1], P[k + 1, 0], P[k + 1, 1], w[k], w[k + 1]])
    return np.array(segs, np.float64), main


@njit(parallel=True, **NJ)
def chan_grid(SEG, CHg):
    for j in prange(CHg.shape[0]):
        y = CHY0 + j * CHC
        for i in range(CHg.shape[1]):
            x = CHX0 + i * CHC
            best = 500.0
            for k in range(SEG.shape[0]):
                ax = SEG[k, 0]; ay = SEG[k, 1]
                lim = best + 40.0
                if abs(x - ax) > lim or abs(y - ay) > lim:
                    continue
                ex = SEG[k, 2] - ax; ey = SEG[k, 3] - ay
                L2 = ex * ex + ey * ey + 1e-9
                tt = ((x - ax) * ex + (y - ay) * ey) / L2
                if tt < 0.0: tt = 0.0
                if tt > 1.0: tt = 1.0
                d = math.sqrt((x - ax - tt * ex) ** 2 + (y - ay - tt * ey) ** 2) - (SEG[k, 4] + (SEG[k, 5] - SEG[k, 4]) * tt)
                if d < best:
                    best = d
            CHg[j, i] = best


@njit(inline='always', **NJ)
def chan_val(CH, x, y):
    """Distance to the nearest bank, metres; negative in the channel."""
    fx = (x - CHX0) / CHC; fy = (y - CHY0) / CHC
    if fx < 0.0 or fy < 0.0 or fx >= CH.shape[1] - 1.001 or fy >= CH.shape[0] - 1.001:
        return 500.0
    x0 = int(fx); y0 = int(fy); tx = fx - x0; ty = fy - y0
    return (CH[y0, x0] * (1 - tx) + CH[y0, x0 + 1] * tx) * (1 - ty) + (CH[y0 + 1, x0] * (1 - tx) + CH[y0 + 1, x0 + 1] * tx) * ty


@njit(**NJ)
def grit(x, y, sc, sd):
    """Fine texture on a turned grid, so that nothing lines up with the view."""
    xr = (0.84 * x - 0.54 * y) / sc; yr = (0.54 * x + 0.84 * y) / sc
    return noise2(xr + 11.0, yr, sd) + 0.55 * noise2(-1.9 * yr + 5.0, 2.1 * xr + 0.8 * yr, sd + 1) + 0.3 * noise2(4.3 * xr - 1.0 * yr, 1.2 * xr + 4.1 * yr, sd + 2)


@njit(**NJ)
def rim_r(az):
    q = az / 0.74
    q = q * q
    if q > 1.0: q = 1.0
    return 5.5 + 4.4 * q + 0.55 * noise2(az * 2.6, 0.5, 311) + 0.25 * noise2(az * 8.0, 1.5, 312) + 0.10 * noise2(az * 31.0, 2.5, 319)


@njit(**NJ)
def platform(x, y, r, lod):
    """The rock I stand on: bedded, broken at the edge, falling away."""
    az = math.atan2(x, y)
    rim = rim_r(az)
    h = EZG + 0.08 * noise2(x / 2.3, y / 2.3, 313)
    # bedding: the rock breaks in low steps
    sd = (0.42 * x + 0.91 * y) / 0.70 + 0.9 * noise2(x / 2.9, y / 2.9, 314) + 0.25 * noise2(x / 0.6, y / 0.6, 315)
    fl = math.floor(sd)
    h -= (fl + sstep(0.70, 0.97, sd - fl)) * 0.035
    if lod < 0.05:
        # blocks sitting on it, more of them toward the edge
        near = sstep(3.6, 1.2, rim - r) * sstep(2.6, 3.6, r)
        for lev in range(3):
            c = 0.82
            if lev == 1: c = 0.33
            if lev == 2: c = 0.12
            cx = math.floor(x / c); cy = math.floor(y / c)
            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    ix = int(cx) + ox; iy = int(cy) + oy
                    p = (0.42 if lev == 0 else (0.60 if lev == 1 else 0.55)) * near
                    if _h3(ix, iy, lev, 80) > p:
                        continue
                    bx = (ix + 0.2 + 0.6 * _h3(ix, iy, lev, 81)) * c
                    by = (iy + 0.2 + 0.6 * _h3(ix, iy, lev, 82)) * c
                    rad = c * (0.26 + 0.20 * _h3(ix, iy, lev, 83))
                    ang = 3.1416 * _h3(ix, iy, lev, 84)
                    ca = math.cos(ang); sa = math.sin(ang)
                    ddx = x - bx; ddy = y - by
                    ex = (ddx * ca + ddy * sa) / (rad * 1.3); ey = (-ddx * sa + ddy * ca) / (rad * 0.85)
                    d2 = ex * ex + ey * ey
                    d2 *= 1.0 + 0.40 * noise2(x * 2.2 / c + ix, y * 2.2 / c + iy, 85) + 0.2 * noise2(x * 6.0 / c, y * 6.0 / c + iy, 91)
                    if d2 >= 1.0:
                        continue
                    hh = rad * (0.40 + 0.55 * _h3(ix, iy, lev, 86))
                    # angular, not river-worn: a tilted flat top and a short steep side
                    tilt = 0.35 * (ex * (_h3(ix, iy, lev, 88) - 0.5) + ey * (_h3(ix, iy, lev, 89) - 0.5))
                    top = hh * min(1.0 + tilt, 1.35 * (1.0 - d2 ** 1.25)) - 0.15 * hh
                    top += 0.02 * c * noise2(x * 9.0 / c, y * 9.0 / c, 87)
                    hb = EZG - 0.035 * ((0.42 * bx + 0.91 * by) / 0.70) + top
                    if hb > h:
                        h = hb
        # cracks and grain
        cr = rfbm2(x / 0.9 + 3.0, y / 0.9, 3, 316)
        h -= 0.05 * sstep(0.62, 0.86, cr)
        h += 0.012 * fbm2(x / 0.11, y / 0.11, 3, 317)
        if lod < 0.004:
            h += 0.0035 * fbm2(x / 0.018, y / 0.018, 3, 318)
    if r > rim:
        h -= 1.45 * (r - rim) + 0.5 * sstep(0.0, 0.5, r - rim)
    return h


@njit(**NJ)
def dome_hill(x, y, lod):
    dx = x - DHX; dy = y - DHY
    if abs(dx) > 900.0 or abs(dy) > 1100.0:
        return 0.0, 0.0
    u = dx * DH_S + dy * DH_C; v = dx * DH_C - dy * DH_S
    lu = 122.0 if u < 0 else 400.0
    lv = 100.0 if v < 0 else 215.0
    m = math.exp(-(abs(u) / lu) ** 2.3 - (abs(v) / lv) ** 2.1)
    if m < 0.004:
        return 0.0, m
    h = DHH * m
    flank = 4.0 * m * (1.0 - m)
    oc = 6 if lod < 0.8 else (5 if lod < 3.0 else 3)
    cr = rfbm2(x / 82.0 + 5.1, y / 82.0 + 0.3, oc, 321)
    h += flank * 13.0 * (cr - 0.50)
    # ribs and gullies down the fall line
    rad = math.sqrt(dx * dx + dy * dy) + 1.0
    wq = 0.8 * noise2(x / 70.0, y / 70.0, 322)
    rib = rfbm2(dx / rad * 2.6 + rad * 0.004 + 7.0 + wq, dy / rad * 2.6 + 2.0 - wq, 4, 323)
    h += flank * 5.0 * (rib - 0.45) * (0.5 + 0.6 * noise2(x / 55.0, y / 55.0 + 4.0, 324))
    # outcrops: steps where the rock comes through the scree
    oc_ = fbm2(x / 30.0 + 1.0, y / 30.0, 3, 325)
    h += flank * 5.5 * sstep(0.10, 0.30, oc_)
    if lod < 1.2:
        # bedding comes out of the slope as ledges; blocks lie below them
        led = h / 3.4 + 0.9 * noise2(x / 45.0, y / 45.0, 328) + 0.12 * grit(x, y, 9.0, 329)
        h += flank * 0.75 * (sstep(0.50, 0.95, led - math.floor(led)) - 0.5) * (0.35 + 0.65 * sstep(-0.2, 0.3, noise2(x / 60.0 + 4.0, y / 60.0, 330)))
        h += math.sqrt(m) * 0.55 * fbm2(x / 4.5, y / 4.5, 3, 326)
        h += flank * 0.55 * (rfbm2(x / 1.6, y / 1.6, 3, 327) - 0.4)
        if lod < 0.5:
            h += flank * 0.45 * sstep(0.25, 0.85, grit(x, y, 2.4, 392)) * (1.0 - lod / 0.5)
    # the top was levelled for the building
    w = sstep(34.0, 15.0, rad)
    h = h * (1.0 - w) + (DHH - 1.2) * w
    return h, m


@njit(**NJ)
def crag(x, y, lod):
    dx = x - CRX; dy = y - CRY
    r2 = dx * dx + dy * dy
    if r2 > 800.0 * 800.0:
        return 0.0, 0.0
    r = math.sqrt(r2)
    m = math.exp(-(r / 92.0) ** 2.2)
    h = CRH * m + 20.0 * math.exp(-(r / 290.0) ** 2) + 34.0 * math.exp(-((dx + 190.0) / 170.0) ** 2 - ((dy - 60.0) / 110.0) ** 2)
    if m > 0.004:
        cr = rfbm2(x / 60.0 + 2.1, y / 60.0, 5, 331)
        h += 4.0 * m * (1.0 - m) * 22.0 * (cr - 0.5)
        w = sstep(40.0, 20.0, r)
        h = h * (1.0 - w) + (CRH - 2.0) * w
    return h, m


@njit(**NJ)
def range_mask(x, y):
    if y < 1500.0 and abs(x) < 1500.0:
        return 0.0
    xa = x + 0.10 * y
    wl = sstep(2400.0, 7800.0, abs(xa) + 900.0 * noise2(x / 4200.0 + 3.0, y / 4200.0, 131))
    wf = sstep(9000.0, 22000.0, y + 2500.0 * noise2(x / 6000.0, y / 6000.0 + 5.0, 132))
    return 1.0 - (1.0 - wl) * (1.0 - wf)


@njit(**NJ)
def terrain(CH, x, y, lod):
    """Height above the river.  lod is the width of a pixel on the ground there, metres."""
    r2 = x * x + y * y
    if r2 < 55.0 * 55.0:
        r = math.sqrt(r2)
        hp = platform(x, y, r, lod)
        if hp > 3.0:
            return hp
    # the valley floor
    h = 1.25 + 0.80 * noise2(x / 700.0 + 3.1, y / 700.0 + 1.7, 301) + 0.34 * noise2(x / 90.0, y / 90.0, 302)
    c = chan_val(CH, x, y)
    if lod < 3.0:
        h += 0.10 * noise2(x / 9.0, y / 9.0, 303)
        if lod < 0.45:
            # tussocks, bigger where the ground is wet; small stones everywhere
            fz = 1.0 - lod / 0.45
            h += fz * (0.035 + 0.11 * sstep(170.0, 15.0, c)) * (sstep(-0.25, 0.75, grit(x, y, 1.0, 308)) - 0.4)
            if lod < 0.15:
                h += (1.0 - lod / 0.15) * 0.03 * sstep(0.1, 0.8, grit(x, y, 0.3, 380))
    if c < 400.0:
        cp = c if c > 0.0 else 0.0
        h -= 0.95 * math.exp(-(cp / 150.0) ** 2)
        # low places near the river hold water
        h -= 1.5 * sstep(130.0, 15.0, c) * max(0.0, fbm2(x / 46.0 + 7.0, y / 60.0 + 2.0, 4, 305))
        if c < 5.0:
            h -= 1.6 * sstep(5.0, -7.0, c)
            # gravel bars in the wide reaches
            if c < -9.0:
                h += 1.75 * sstep(0.10, 0.26, fbm2(x / 44.0 + 2.0, y / 27.0, 4, 306)) * sstep(-9.0, -15.0, c)
    hd, md = dome_hill(x, y, lod)
    if hd > 0.0:
        h += hd
    hc, mc = crag(x, y, lod)
    h += hc
    M = range_mask(x, y)
    if M > 0.0:
        oc = int(math.log2(5600.0 / (1.5 * lod + 1e-3)))
        if oc < 3: oc = 3
        if oc > 11: oc = 11
        r = rfbm2(x / 5600.0 + 11.3, y / 5600.0 + 4.1, oc, 141)
        h += 110.0 * M + 1500.0 * r * sstep(0.0, 0.45, M) * (0.22 + 0.78 * M)
        o2 = oc - 2 if oc > 4 else 2
        h += 380.0 * (rfbm2(x / 1600.0 + 3.3, y / 1600.0 + 8.1, o2, 151) - 0.30) * sstep(0.0, 0.30, M)
    return h


# ---------------------------------------------------------------------
#  light on the ground
# ---------------------------------------------------------------------
C_COOL = np.array([0.0112, 0.0122, 0.0144]) * 3.0     # the rest of the sky on a flat white card
C_RED = np.array([0.0620, 0.0055, 0.0090]) * 3.8      # the red on a card that faces it
_la = np.array([-0.12, 0.92, 0.37]); LA = _la / np.linalg.norm(_la)
HAZE = np.array([0.0150, 0.0050, 0.0075])
MIST = np.array([0.0260, 0.0070, 0.0105])
LAMPC = np.array([1.0, 0.55, 0.16])


@njit(**NJ)
def road_dist(u, v):
    best = 1e9
    for k in range(ROAD.shape[0] - 1):
        ax = ROAD[k, 0]; ay = ROAD[k, 1]
        ex = ROAD[k + 1, 0] - ax; ey = ROAD[k + 1, 1] - ay
        tt = ((u - ax) * ex + (v - ay) * ey) / (ex * ex + ey * ey)
        if tt < 0.0: tt = 0.0
        if tt > 1.0: tt = 1.0
        d = (u - ax - tt * ex) ** 2 + (v - ay - tt * ey) ** 2
        if d < best: best = d
    return math.sqrt(best)


@njit(**NJ)
def shade_land(CH, x, y, h, t, lf, LAMP):
    e = max(0.010, 0.7 * lf)
    hx = max(terrain(CH, x + e, y, lf), 0.0) - max(terrain(CH, x - e, y, lf), 0.0)
    hy = max(terrain(CH, x, y + e, lf), 0.0) - max(terrain(CH, x, y - e, lf), 0.0)
    nx = -hx; ny = -hy; nz = 2.0 * e
    l = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
    nx *= l; ny *= l; nz *= l
    # hollows are darker
    ao = 1.0
    r1 = max(0.05, 5.0 * lf)
    for q in range(2):
        rr = r1 * (1.0 if q == 0 else 5.5)
        hs = (max(terrain(CH, x + rr, y, lf * 2), 0.0) + max(terrain(CH, x - rr, y, lf * 2), 0.0)
              + max(terrain(CH, x, y + rr, lf * 2), 0.0) + max(terrain(CH, x, y - rr, lf * 2), 0.0)) * 0.25
        oc = (hs - h) / rr
        if oc > 0.0:
            ao *= 1.0 - min(0.72, 1.5 * oc)
        else:
            ao *= 1.0 + min(0.14, -0.5 * oc)
    cool = (0.30 + 0.70 * (0.5 + 0.5 * nz) + 0.30 * max(0.0, -ny)) * ao
    rd = (nx * LA[0] + ny * LA[1] + nz * LA[2] + 0.30) / 1.30
    red = (rd ** 1.5 if rd > 0.0 else 0.0) * ao
    slope = 1.0 - nz
    r2 = x * x + y * y
    # ---- what it is made of
    if r2 < 55.0 * 55.0 and h > 3.0:
        st = noise2((0.42 * x + 0.91 * y) / 0.19, (0.91 * x - 0.42 * y) / 2.5, 341)
        fine = noise2(x / 0.035, y / 0.035, 342)
        lich = sstep(0.30, 0.55, noise2(x / 0.33, y / 0.33, 343) + 0.4 * noise2(x / 0.07, y / 0.07, 344))
        v = 0.86 + 0.20 * st + 0.14 * fine
        ar = 0.200 * v; ag = 0.172 * v; ab = 0.152 * v
        ar += (0.30 - ar) * 0.55 * lich; ag += (0.30 - ag) * 0.55 * lich; ab += (0.23 - ab) * 0.55 * lich
    else:
        hd, md = dome_hill(x, y, lf)
        hc, mc = crag(x, y, lf)
        M = range_mask(x, y)
        big = noise2(x / 420.0 + 2.0, y / 420.0, 331)
        mid = noise2(x / 37.0, y / 37.0 + 5.0, 332)
        fine = 0.6 * grit(x, y, 1.6, 383) if lf < 0.6 else 0.0
        fine2 = 0.6 * grit(x, y, 0.3, 386) if lf < 0.12 else 0.0
        c = chan_val(CH, x, y)
        # the plain: dust, dark turf by the water, pale salt where it dries
        turf = sstep(-0.15, 0.35, 0.5 * big + 0.5 * mid + 0.75 * sstep(150.0, 15.0, c) - 0.30)
        salt = sstep(0.26, 0.31, 1.6 * fbm2(x / 210.0 + 9.0, y / 210.0, 5, 335) + 0.04 * fine - 0.4 * turf) * sstep(25.0, 110.0, c) * (0.75 + 0.25 * mid)
        ar = 0.25; ag = 0.215; ab = 0.175
        ar += (0.070 - ar) * turf; ag += (0.076 - ag) * turf; ab += (0.046 - ab) * turf
        ar += (0.62 - ar) * salt; ag += (0.61 - ag) * salt; ab += (0.59 - ab) * salt
        if c < 6.0:
            g = sstep(6.0, 1.0, c) * sstep(0.02, 0.25, h)
            ar += (0.33 - ar) * g; ag += (0.31 - ag) * g; ab += (0.29 - ab) * g
        v = 1.0 + 0.22 * mid + 0.24 * fine * (1.0 + turf) + 0.16 * fine2
        if lf < 0.45:
            # frost on the heads of the tussocks, black between them
            hm = sstep(-0.25, 0.75, grit(x, y, 1.0, 308))
            fz = (1.0 - lf / 0.45) * (0.35 + 0.65 * turf)
            v *= 1.0 - 0.32 * fz * (1.0 - hm)
            fr = 0.36 * fz * hm * hm
            ar += (0.33 - ar) * fr; ag += (0.34 - ag) * fr; ab += (0.33 - ab) * fr
            if lf < 0.15:
                pb = sstep(0.1, 0.8, grit(x, y, 0.3, 380)) * (1.0 - lf / 0.15) * (1.0 - turf)
                ar += (0.30 - ar) * 0.6 * pb; ag += (0.29 - ag) * 0.6 * pb; ab += (0.28 - ab) * 0.6 * pb
        v *= 0.50 + 0.50 * sstep(0.0, 0.22, h)                   # wet at the edge
        ar *= v; ag *= v; ab *= v
        rockw = max(sstep(0.008, 0.07, md), sstep(0.006, 0.05, mc))
        if rockw > 0.0:
            st = noise2(x / 14.0 + 0.3 * h, h / 1.6, 345)
            v = 0.82 + 0.30 * st + 0.20 * fine + 0.12 * fine2
            rr_ = 0.170 * v; rg_ = 0.150 * v; rb_ = 0.132 * v
            scree = sstep(0.40, 0.12, slope) * sstep(0.02, 0.10, slope) * (0.6 + 0.4 * mid)
            rr_ += 0.085 * scree; rg_ += 0.080 * scree; rb_ += 0.072 * scree
            if lf < 0.6:
                # dry grass in tufts on the scree, dark; pale lichen on the blocks
                tf = sstep(0.2, 0.7, grit(x, y, 0.9, 395)) * sstep(0.45, 0.2, slope) * (1.0 - lf / 0.6)
                rr_ *= 1.0 - 0.55 * tf; rg_ *= 1.0 - 0.52 * tf; rb_ *= 1.0 - 0.60 * tf
                bl = sstep(0.25, 0.85, grit(x, y, 2.4, 392)) * (1.0 - lf / 0.6)
                rr_ += 0.06 * bl; rg_ += 0.06 * bl; rb_ += 0.055 * bl
            if md > 0.01:
                dxh = x - DHX; dyh = y - DHY
                u = dxh * DH_S + dyh * DH_C; vv = dxh * DH_C - dyh * DH_S
                if u < 20.0 and u > -230.0 and abs(vv) < 90.0:
                    dr = road_dist(u, vv)
                    rw = sstep(2.6, 1.5, dr)
                    rr_ += (0.36 - rr_) * rw; rg_ += (0.335 - rg_) * rw; rb_ += (0.30 - rb_) * rw
            ar += (rr_ - ar) * rockw; ag += (rg_ - ag) * rockw; ab += (rb_ - ab) * rockw
        if M > 0.0:
            w = sstep(0.0, 0.10, M)
            st = noise2(x / 1100.0, h / 80.0, 346)
            rr_ = 0.175 + 0.07 * st; rg_ = 0.158 + 0.06 * st; rb_ = 0.150 + 0.055 * st
            sl_ = SNOW0 + 240.0 * noise2(x / 1700.0, y / 1700.0, 347)
            gl = rfbm2(x / 520.0, y / 520.0, 4, 348)
            sn = sstep(0.0, 300.0, h - sl_) * sstep(0.46, 0.20, slope)
            sn = max(sn * (0.55 + 0.45 * sstep(0.2, 0.5, gl)), 0.8 * sstep(0.50, 0.24, gl) * sstep(-330.0, -40.0, h - sl_) * sstep(0.5, 0.2, slope))
            rr_ += (0.80 - rr_) * sn; rg_ += (0.82 - rg_) * sn; rb_ += (0.86 - rb_) * sn
            ar += (rr_ - ar) * w; ag += (rg_ - ag) * w; ab += (rb_ - ab) * w
    r = ar * (cool * C_COOL[0] + red * C_RED[0])
    g = ag * (cool * C_COOL[1] + red * C_RED[1])
    b = ab * (cool * C_COOL[2] + red * C_RED[2])
    # the lamp
    lx = LAMP[0] - x; ly = LAMP[1] - y; lz = LAMP[2] - h
    d2 = lx * lx + ly * ly + lz * lz
    if d2 < 90.0 * 90.0:
        d = math.sqrt(d2)
        lx /= d; ly /= d; lz /= d
        cs = nx * lx + ny * ly + nz * lz
        front = -(lx * LAMP[3] + ly * LAMP[4])
        if cs > 0.0 and front > 0.0:
            li = LAMP[6] * cs * front * front / (d2 + 1.5) * sstep(90.0, 50.0, d)
            r += ar * li * LAMPC[0]; g += ag * li * LAMPC[1]; b += ab * li * LAMPC[2]
    return r, g, b


@njit(**NJ)
def shade_water(CH, U, x, y, hraw, dx, dy, dz, K, jit, LAMP, W, H, F, CPc, SPc, ez):
    hd = math.sqrt(dx * dx + dy * dy); ex = dx / hd; ey = dy / hd
    # what stands on the far bank along this bearing
    hor = 0.0
    q = 10.0
    for k in range(30):
        hh = terrain(CH, x + ex * q, y + ey * q, q / 500.0)
        s = hh / q
        if s > hor: hor = s
        q *= 1.33
    # a little wind, in patches
    pn = fbm2(x / 55.0 + 2.0, y / 130.0 + 1.0, 3, 351)
    rough = sstep(0.02, 0.32, pn)
    sgv = 0.0045 + 0.020 * rough
    sgh = 0.30 * sgv
    sw = 0.0030 * (noise2(x / 5.0 + 0.3 * y, y / 2.2, 352) + 0.5 * noise2(x / 1.9, y / 0.8 + 3.0, 353))
    c0 = 0.0; c1 = 0.0; c2 = 0.0
    for k in range(K):
        qv = (k + jit) / K
        if qv < 0.03: qv = 0.03
        if qv > 0.97: qv = 0.97
        g = 0.55 * math.log(qv / (1.0 - qv))
        g2 = 2.0 * _h3(int(x * 37.0) + k, int(y * 37.0), k, 7) - 1.0
        sa = sw + sgv * g; sb = sgh * g2
        nx = -(sa * ex - sb * ey); ny = -(sa * ey + sb * ex); nz = 1.0
        l = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
        nx *= l; ny *= l; nz *= l
        dn = dx * nx + dy * ny + dz * nz
        rx = dx - 2.0 * dn * nx; ry = dy - 2.0 * dn * ny; rz = dz - 2.0 * dn * nz
        if rz < 0.002: rz = 0.002
        rs = rz / math.sqrt(rx * rx + ry * ry)
        if rs < hor:
            c0 += 0.0016; c1 += 0.0011; c2 += 0.0015
        else:
            fx, fy = dir_pix(rx, ry, rz, W, H, F, CPc, SPc)
            c0 += bil(U, fx - 0.5, fy - 0.5, 0); c1 += bil(U, fx - 0.5, fy - 0.5, 1); c2 += bil(U, fx - 0.5, fy - 0.5, 2)
    c0 /= K; c1 /= K; c2 /= K
    se = -dz
    Rf = 0.02 + 0.98 * (1.0 - se) ** 4
    Rf *= 1.0 - 0.30 * rough
    dp = 1.0 - Rf
    r = c0 * Rf + 0.0011 * dp; g = c1 * Rf + 0.0015 * dp; b = c2 * Rf + 0.0021 * dp
    # the lamp's path: the facets that would send it to me
    lx = LAMP[0] - x; ly = LAMP[1] - y; lz = LAMP[2]
    d2 = lx * lx + ly * ly + lz * lz
    if d2 < 200.0 * 200.0:
        d = math.sqrt(d2)
        lx /= d; ly /= d; lz /= d
        front = -(lx * LAMP[3] + ly * LAMP[4])
        if front > 0.0:
            hx = lx - dx; hy = ly - dy; hz = lz - dz
            sx = hx / hz; sy = hy / hz
            sa = sx * ex + sy * ey - sw; sb = -sx * ey + sy * ex
            sv = 0.022 + 1.2 * sgv; sh = sv * 0.55
            gl = math.exp(-0.5 * ((sa / sv) ** 2 + (sb / sh) ** 2)) / (sv * sh)
            rp = 0.5 + 0.5 * noise2(x / 0.9 + 0.4 * noise2(y / 0.6, 1.0, 355), y / 0.35, 354)
            gl *= (0.25 + 1.5 * rp * rp) * LAMP[6] * front * 0.012 / (d2 + 4.0)
            r += gl * LAMPC[0]; g += gl * LAMPC[1]; b += gl * LAMPC[2]
    # ice: a shelf along the banks, and the still ponds skinned over.  It does not mirror; it is a pale card.
    fi = sstep(-0.11, -0.03, hraw) * (0.62 + 0.40 * noise2(x / 7.0, y / 7.0, 371))
    fp = sstep(-0.55, -0.20, hraw) * (0.50 + 0.9 * fbm2(x / 50.0 + 3.0, y / 50.0, 3, 372))
    if fp > fi: fi = fp
    fi = sstep(0.43, 0.53, fi + 0.05 * grit(x, y, 2.2, 389))
    if fi > 0.0:
        al = 0.52 + 0.10 * noise2(x / 2.6, y / 2.6, 374) + 0.06 * noise2(x / 0.4, y / 0.4, 375)
        al *= 1.0 - 0.35 * sstep(0.80, 0.93, rfbm2(x / 6.0, y / 6.0, 3, 376))        # cracks
        ir = al * (C_COOL[0] + 0.37 * C_RED[0]) + 0.22 * r
        ig = al * 1.04 * (C_COOL[1] + 0.37 * C_RED[1]) + 0.22 * g
        ib = al * 1.10 * (C_COOL[2] + 0.37 * C_RED[2]) + 0.22 * b
        r += (ir - r) * fi; g += (ig - g) * fi; b += (ib - b) * fi
    return r, g, b


# ---------------------------------------------------------------------
#  the ground as a panorama: bearing across, slope up
# ---------------------------------------------------------------------
@njit(parallel=True, **NJ)
def pano_march(CH, T, az0, daz, s_top, dsl, ez):
    NR = T.shape[0]; NA = T.shape[1]
    for i in prange(NA):
        az = az0 + (i + 0.5) * daz
        sa = math.sin(az); ca = math.cos(az)
        jr = NR - 1
        t = 1.2; tp = t; sp = -1e9
        while t < 90000.0 and jr >= 0:
            h = terrain(CH, sa * t, ca * t, t * daz)
            if h < 0.0: h = 0.0
            s = (h - ez) / t
            while jr >= 0:
                sj = s_top - (jr + 0.5) * dsl
                if sj > s:
                    break
                if sj > sp:
                    T[jr, i] = tp + (t - tp) * (sj - sp) / (s - sp)
                else:
                    T[jr, i] = tp
                jr -= 1
            tp = t; sp = s
            t += max(0.004, (0.00045 if t < 1200.0 else 0.0011) * t)


@njit(parallel=True, **NJ)
def pano_shade(CH, U, T, P, az0, daz, s_top, dsl, ez, LAMP, W, H, F, CPc, SPc):
    NR = T.shape[0]; NA = T.shape[1]
    for i in prange(NA):
        az = az0 + (i + 0.5) * daz
        sa = math.sin(az); ca = math.cos(az)
        for j in range(NR):
            t0 = T[j, i]
            if t0 > 1e20:
                continue
            tn = t0 * 1.01
            if j > 0 and T[j - 1, i] < 1e20:
                tn = T[j - 1, i]
            dt = tn - t0
            if dt > 0.10 * t0: dt = 0.0                # an edge, not a surface: do not reach past it
            lf = t0 * daz
            n = int(dt / (1.5 * lf))
            if n < 1: n = 1
            if n > 8: n = 8
            K = 16 // n
            if K < 4: K = 4
            a0 = 0.0; a1 = 0.0; a2 = 0.0
            for k in range(n):
                tk = t0 + dt * ((k + 0.5) / n - 0.5)
                x = sa * tk; y = ca * tk
                h = terrain(CH, x, y, lf)
                if h <= 0.0:
                    l = 1.0 / math.sqrt(tk * tk + ez * ez)
                    r, g, b = shade_water(CH, U, x, y, h, x * l, y * l, -ez * l, K, _h3(i, j, k, 5), LAMP, W, H, F, CPc, SPc, ez)
                    h = 0.0
                else:
                    r, g, b = shade_land(CH, x, y, h, tk, lf, LAMP)
                # the air between, and what lies in the valley on a cold night
                f = math.exp(-tk / 70000.0)
                dzr = ez - h
                if abs(dzr) < 0.5: dzr = 0.5
                tm = 10.0 * (math.exp(-max(h, 0.0) / 10.0) - math.exp(-ez / 10.0)) / dzr * tk / 3400.0
                tm *= 0.7 + 0.5 * noise2(x / 900.0 + 3.0, y / 1400.0, 361)
                fm = math.exp(-tm)
                r = (r * f + HAZE[0] * (1 - f)) * fm + MIST[0] * (1 - fm)
                g = (g * f + HAZE[1] * (1 - f)) * fm + MIST[1] * (1 - fm)
                b = (b * f + HAZE[2] * (1 - f)) * fm + MIST[2] * (1 - fm)
                a0 += r; a1 += g; a2 += b
            P[j, i, 0] = a0 / n; P[j, i, 1] = a1 / n; P[j, i, 2] = a2 / n; P[j, i, 3] = 1.0


@njit(parallel=True, **NJ)
def pano_to_frame(P, az0, daz, s_top, dsl, r0, W, H, F, CPc, SPc, out):
    """Back through the lens, four samples to the pixel."""
    for j in prange(r0, H):
        for i in range(W):
            a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0
            for sj in range(2):
                for si in range(2):
                    dx, dy, dz = pix_dir(i + 0.25 + 0.5 * si, j + 0.25 + 0.5 * sj, W, H, F, CPc, SPc)
                    az = math.atan2(dx, dy)
                    sl = dz / math.sqrt(dx * dx + dy * dy)
                    fx = (az - az0) / daz - 0.5; fy = (s_top - sl) / dsl - 0.5
                    if fy < 0.0:
                        continue
                    a0 += bil(P, fx, fy, 0); a1 += bil(P, fx, fy, 1); a2 += bil(P, fx, fy, 2); a3 += bil(P, fx, fy, 3)
            out[j - r0, i, 0] = a0 * 0.25; out[j - r0, i, 1] = a1 * 0.25
            out[j - r0, i, 2] = a2 * 0.25; out[j - r0, i, 3] = a3 * 0.25


# ---------------------------------------------------------------------
#  things people built: boxes, drums, a dome
#  box:  cx cy cos sin a b ztop zbot batter mat ground id
#  drum: cx cy z0 z1 R mat
#  dome: cx cy cz R slit_azimuth slit_half_width
# ---------------------------------------------------------------------
@njit(**NJ)
def hit_box(B, i, ox, oy, oz, dx, dy, dz):
    c = B[i, 2]; s = B[i, 3]; a = B[i, 4]; b = B[i, 5]; k = B[i, 8]
    rx = ox - B[i, 0]; ry = oy - B[i, 1]
    lx = rx * c + ry * s; ly = -rx * s + ry * c; lz = oz - B[i, 6]
    ex = dx * c + dy * s; ey = -dx * s + dy * c; ez = dz
    zb = B[i, 7] - B[i, 6]
    tin = -1e30; tout = 1e30; face = -1
    for f in range(6):
        if f == 0:
            den = ex + k * ez; num = a - (lx + k * lz)
        elif f == 1:
            den = -ex + k * ez; num = a - (-lx + k * lz)
        elif f == 2:
            den = ey + k * ez; num = b - (ly + k * lz)
        elif f == 3:
            den = -ey + k * ez; num = b - (-ly + k * lz)
        elif f == 4:
            den = ez; num = -lz
        else:
            den = -ez; num = lz - zb
        if den > 1e-12:
            tt = num / den
            if tt < tout:
                tout = tt
        elif den < -1e-12:
            tt = num / den
            if tt > tin:
                tin = tt; face = f
        elif num < 0.0:
            return -1.0, -1
        if tin > tout:
            return -1.0, -1
    if tin < 1e-4:
        return -1.0, -1
    return tin, face


@njit(**NJ)
def hit_drum(C, i, ox, oy, oz, dx, dy, dz):
    rx = ox - C[i, 0]; ry = oy - C[i, 1]
    a = dx * dx + dy * dy
    b = rx * dx + ry * dy
    c = rx * rx + ry * ry - C[i, 4] * C[i, 4]
    disc = b * b - a * c
    if disc < 0.0:
        return -1.0, 0
    sq = math.sqrt(disc)
    t = (-b - sq) / a
    z = oz + dz * t
    if t > 1e-4 and z >= C[i, 2] and z <= C[i, 3]:
        return t, 0
    # the underside or the top
    zc = C[i, 2] if dz > 0.0 else C[i, 3]
    if abs(dz) > 1e-9:
        t = (zc - oz) / dz
        if t > 1e-4:
            px = rx + dx * t; py = ry + dy * t
            if px * px + py * py <= C[i, 4] * C[i, 4]:
                return t, 1
    return -1.0, 0


@njit(inline='always', **NJ)
def in_slit(qx, qy, qz, R, saz, hw):
    ax = math.sin(saz); ay = math.cos(saz)
    lat = qx * ay - qy * ax
    fwd = qx * ax + qy * ay
    return abs(lat) < hw and fwd > -0.30 * R and qz > 0.10 * R


@njit(**NJ)
def hit_dome(D, i, ox, oy, oz, dx, dy, dz):
    """0 nothing, 1 the shell from outside, 2 the inside seen through the slit."""
    rx = ox - D[i, 0]; ry = oy - D[i, 1]; rz = oz - D[i, 2]
    R = D[i, 3]
    b = rx * dx + ry * dy + rz * dz
    c = rx * rx + ry * ry + rz * rz - R * R
    disc = b * b - c
    if disc < 0.0:
        return -1.0, 0
    sq = math.sqrt(disc)
    t1 = -b - sq; t2 = -b + sq
    if t1 < 1e-4:
        return -1.0, 0
    qx = rx + dx * t1; qy = ry + dy * t1; qz = rz + dz * t1
    if qz < -0.9:
        return -1.0, 0
    if not in_slit(qx, qy, qz, R, D[i, 4], D[i, 5]):
        return t1, 1
    qx = rx + dx * t2; qy = ry + dy * t2; qz = rz + dz * t2
    if in_slit(qx, qy, qz, R, D[i, 4], D[i, 5]):
        return -1.0, 0
    if qz < -0.9:
        # the floor, or rather the telescope standing on it
        return t2, 2
    return t2, 2


@njit(**NJ)
def sky_seen(rx, ry, rz):
    """What a mirror facing this way would show: the red if it looks north, else the plain night."""
    if rz < 0.0:
        return 0.0025, 0.0020, 0.0024
    a, v = aurora_los(rx, ry, rz, 20)
    a *= AUR_GAIN; v *= AUR_GAIN
    return (0.004 + a * AUR_R[0] + v * AUR_V[0], 0.006 + a * AUR_R[1] + v * AUR_V[1], 0.009 + a * AUR_R[2] + v * AUR_V[2])


@njit(**NJ)
def card(nx, ny, nz, ao):
    cool = (0.30 + 0.70 * (0.5 + 0.5 * nz) + 0.30 * max(0.0, -ny)) * ao
    rd = (nx * LA[0] + ny * LA[1] + nz * LA[2] + 0.30) / 1.30
    red = (rd ** 1.5 if rd > 0.0 else 0.0) * ao
    return cool, red


@njit(**NJ)
def shade_box(B, i, face, x, y, z, LAMP, LIT):
    c = B[i, 2]; s = B[i, 3]; a = B[i, 4]; b = B[i, 5]; k = B[i, 8]
    mat = int(B[i, 9]); bid = int(B[i, 11])
    rx = x - B[i, 0]; ry = y - B[i, 1]
    lx = rx * c + ry * s; ly = -rx * s + ry * c
    dtop = B[i, 6] - z
    hgt = z - B[i, 10]
    inv = 1.0 / math.sqrt(1.0 + k * k)
    if face == 0:
        nlx = inv; nly = 0.0; nlz = k * inv; sc = ly; half = b
    elif face == 1:
        nlx = -inv; nly = 0.0; nlz = k * inv; sc = -ly; half = b
    elif face == 2:
        nlx = 0.0; nly = inv; nlz = k * inv; sc = -lx; half = a
    elif face == 3:
        nlx = 0.0; nly = -inv; nlz = k * inv; sc = lx; half = a
    elif face == 4:
        nlx = 0.0; nly = 0.0; nlz = 1.0; sc = lx; half = a
    else:
        nlx = 0.0; nly = 0.0; nlz = -1.0; sc = lx; half = a
    nx = nlx * c - nly * s; ny = nlx * s + nly * c; nz = nlz
    er = 0.0; eg = 0.0; eb = 0.0
    wn = noise2(sc * 0.33 + bid * 7.1, dtop * 0.21 + face * 3.3, 401)
    wn2 = noise2(sc * 2.1 + bid * 3.1, dtop * 1.7 + face * 1.3, 402)
    v = 0.84 + 0.16 * wn + 0.07 * wn2
    if mat == 0 or mat == 1:           # lime on mud brick; the dukhang is red
        if mat == 0:
            ar = 0.78 * v; ag = 0.77 * v; ab = 0.74 * v
        else:
            ar = 0.42 * v; ag = 0.17 * v; ab = 0.11 * v
        if face < 4:
            if dtop < 0.9:
                ar = 0.20; ag = 0.06; ab = 0.05
            elif dtop > 1.6 and (dtop - 1.6) % 3.0 < 1.5 and (sc + half) % 3.4 > 1.2 and (sc + half) % 3.4 < 2.2 \
                    and _h3(bid, face, int((sc + half) / 3.4) + 16 * int((dtop - 1.6) / 3.0), 403) < 0.75:
                ar = 0.03; ag = 0.03; ab = 0.03
        else:
            ar = 0.22; ag = 0.20; ab = 0.17
    elif mat == 2:                     # the house: whitewash, fuel stacked along the roof edge
        ar = 0.74 * v; ag = 0.72 * v; ab = 0.68 * v
        if face < 4:
            if dtop < 0.45:
                ar = 0.050 + 0.03 * wn2; ag = 0.042 + 0.03 * wn2; ab = 0.034 + 0.02 * wn2
            ar *= 1.0 - 0.35 * sstep(0.9, 0.0, hgt); ag *= 1.0 - 0.38 * sstep(0.9, 0.0, hgt); ab *= 1.0 - 0.42 * sstep(0.9, 0.0, hgt)
            if face == 3 and bid == int(LIT[0]):
                wz = hgt
                for q in range(3):
                    wc = LIT[1] + (q - 1) * LIT[4]
                    if q == 0:
                        # the door
                        if abs(sc - wc) < 0.50 and wz < 1.95:
                            ar = 0.035; ag = 0.028; ab = 0.022
                    elif abs(sc - wc) < 0.62 and wz > LIT[2] - 0.62 and wz < LIT[2] + 0.62:
                        ar = 0.030; ag = 0.028; ab = 0.028            # the black surround
                        if abs(sc - wc) < 0.42 and abs(wz - LIT[2]) < 0.45:
                            ar = 0.012; ag = 0.012; ab = 0.014
                            if q == 1:
                                fl = math.exp(-((wz - LIT[2] + 0.15) / 0.40) ** 2 - ((sc - wc) / 0.30) ** 2)
                                e0 = LIT[3] * (0.30 + 1.0 * fl)
                                if abs(sc - wc) < 0.035 or abs(wz - LIT[2] - 0.08) < 0.035:
                                    e0 *= 0.22
                                er = e0 * LAMPC[0]; eg = e0 * LAMPC[1]; eb = e0 * LAMPC[2]
        else:
            ar = 0.20; ag = 0.18; ab = 0.15
    elif mat == 3:                     # dry stone
        v2 = 0.75 + 0.45 * noise2(sc * 2.3 + bid, dtop * 3.1, 404)
        ar = 0.19 * v2; ag = 0.17 * v2; ab = 0.15 * v2
    elif mat == 4:                     # the telescope's rooms: pale panels, a dark plinth, few windows
        ar = 0.56 * v; ag = 0.57 * v; ab = 0.58 * v
        if face < 4:
            if hgt < 0.7:
                ar = 0.16; ag = 0.16; ab = 0.16
            elif dtop < 0.35:
                ar = 0.30; ag = 0.31; ab = 0.33
            elif hgt > 1.5 and hgt < 2.7 and (sc + half) % 4.2 > 1.5 and (sc + half) % 4.2 < 2.9:
                ar = 0.022; ag = 0.024; ab = 0.030
            if abs(((sc + half) % 2.1) - 1.05) > 1.02:
                ar *= 0.8; ag *= 0.8; ab *= 0.8
        else:
            ar = 0.25; ag = 0.25; ab = 0.26
    else:                              # iron
        ar = 0.07; ag = 0.07; ab = 0.075
    ao = 0.55 + 0.45 * sstep(0.0, 3.5, hgt)
    if face == 4: ao = 1.0
    cool, red = card(nx, ny, nz, ao)
    r = ar * (cool * C_COOL[0] + red * C_RED[0]) + er
    g = ag * (cool * C_COOL[1] + red * C_RED[1]) + eg
    bb = ab * (cool * C_COOL[2] + red * C_RED[2]) + eb
    if er == 0.0:
        lxx = LAMP[0] - x; lyy = LAMP[1] - y; lzz = LAMP[2] - z
        d2 = lxx * lxx + lyy * lyy + lzz * lzz
        if d2 < 60.0 * 60.0 and d2 > 0.5:
            d = math.sqrt(d2)
            cs = (nx * lxx + ny * lyy + nz * lzz) / d
            front = -(lxx * LAMP[3] + lyy * LAMP[4]) / d
            if cs > 0.0 and front > 0.0:
                li = LAMP[6] * cs * front * front / (d2 + 1.5)
                r += ar * li * LAMPC[0]; g += ag * li * LAMPC[1]; bb += ab * li * LAMPC[2]
    return r, g, bb


@njit(**NJ)
def shade_metal(nx, ny, nz, dx, dy, dz, ar, ag, ab, ao, gloss):
    cool, red = card(nx, ny, nz, ao)
    r = ar * (cool * C_COOL[0] + red * C_RED[0])
    g = ag * (cool * C_COOL[1] + red * C_RED[1])
    b = ab * (cool * C_COOL[2] + red * C_RED[2])
    dn = dx * nx + dy * ny + dz * nz
    rx = dx - 2.0 * dn * nx; ry = dy - 2.0 * dn * ny; rz = dz - 2.0 * dn * nz
    fr = gloss * (0.35 + 0.65 * (1.0 - min(1.0, abs(dn))) ** 3)
    e0, e1, e2 = sky_seen(rx, ry, rz)
    return r + fr * e0 * ao, g + fr * e1 * ao, b + fr * e2 * ao


@njit(parallel=True, **NJ)
def objects_pass(x0, y0, x1, y1, ss, W, H, F, CPc, SPc, ez, B, C, D, LAMP, LIT, T, az0, daz, s_top, dsl, out):
    for j in prange(y0, y1):
        for i in range(x0, x1):
            a0 = 0.0; a1 = 0.0; a2 = 0.0; cov = 0.0
            for sj in range(ss):
                for si in range(ss):
                    dx, dy, dz = pix_dir(i + (si + 0.5) / ss, j + (sj + 0.5) / ss, W, H, F, CPc, SPc)
                    best = 1e30; kind = -1; idx = -1; face = 0
                    for q in range(B.shape[0]):
                        t, f = hit_box(B, q, 0.0, 0.0, ez, dx, dy, dz)
                        if t > 0.0 and t < best:
                            best = t; kind = 0; idx = q; face = f
                    for q in range(C.shape[0]):
                        t, f = hit_drum(C, q, 0.0, 0.0, ez, dx, dy, dz)
                        if t > 0.0 and t < best:
                            best = t; kind = 1; idx = q; face = f
                    for q in range(D.shape[0]):
                        t, f = hit_dome(D, q, 0.0, 0.0, ez, dx, dy, dz)
                        if t > 0.0 and t < best:
                            best = t; kind = 2; idx = q; face = f
                    if kind < 0:
                        continue
                    # is the ground in front of it?
                    az = math.atan2(dx, dy); hd = math.sqrt(dx * dx + dy * dy)
                    fi = int((az - az0) / daz); fj = int((s_top - dz / hd) / dsl)
                    if fi >= 0 and fi < T.shape[1] and fj >= 0 and fj < T.shape[0]:
                        if T[fj, fi] < best * hd * 0.999:
                            continue
                    x = dx * best; y = dy * best; z = ez + dz * best
                    if kind == 0:
                        r, g, b = shade_box(B, idx, face, x, y, z, LAMP, LIT)
                    elif kind == 1:
                        qx = x - C[idx, 0]; qy = y - C[idx, 1]
                        R = C[idx, 4]
                        hgt = z - C[idx, 2]
                        if face == 1:
                            r, g, b = shade_metal(0.0, 0.0, -1.0 if dz > 0 else 1.0, dx, dy, dz, 0.10, 0.10, 0.11, 0.6, 0.0)
                        else:
                            nx = qx / R; ny = qy / R
                            th = math.atan2(qx, qy)
                            v = 0.90 + 0.10 * noise2(th * 3.0, hgt * 0.4, 411)
                            al = 0.60 * v
                            if abs((th * R) % 1.9 - 0.95) > 0.90: al *= 0.80          # panel joints
                            if C[idx, 5] > 0.5: al = 0.16                            # the gallery and its rail
                            elif hgt < 0.8: al = 0.20
                            ao = 0.60 + 0.40 * sstep(0.0, 4.0, hgt)
                            if C[idx, 5] < 0.5 and C[idx, 3] - z < 1.2: ao *= 0.55 + 0.45 * (C[idx, 3] - z) / 1.2
                            r, g, b = shade_metal(nx, ny, 0.0, dx, dy, dz, al, al, al * 1.03, ao, 0.10)
                    else:
                        qx = x - D[idx, 0]; qy = y - D[idx, 1]; qz = z - D[idx, 2]
                        R = D[idx, 3]
                        if face == 2:
                            # inside: black paint, and a little of the night that comes in with the slit
                            nx = -qx / R; ny = -qy / R; nz = -qz / R
                            ax = math.sin(D[idx, 4]); ay = math.cos(D[idx, 4])
                            see = max(0.0, nx * ax + ny * ay) * 0.5 + 0.12
                            r = 0.05 * see * C_COOL[0]; g = 0.05 * see * C_COOL[1]; b = 0.055 * see * C_COOL[2]
                        else:
                            nx = qx / R; ny = qy / R; nz = qz / R
                            ax = math.sin(D[idx, 4]); ay = math.cos(D[idx, 4])
                            lat = qx * ay - qy * ax; fw = qx * ax + qy * ay
                            # gores run up from the ring to the crown, parallel to the slit
                            al = 0.50 + 0.05 * noise2(lat * 0.9, fw * 0.9 + qz, 412)
                            gg = (lat / (R * 0.19)) % 1.0
                            if gg < 0.06 or gg > 0.94: al *= 0.80
                            hh = (math.asin(max(-1.0, min(1.0, fw / math.sqrt(R * R - lat * lat + 1e-6)))) * 3.3) % 1.0
                            if hh < 0.05: al *= 0.86
                            ao = 0.75 + 0.25 * sstep(-0.9, 2.0, qz)
                            if abs(lat) < D[idx, 5] + 0.34 and fw > -0.36 * R and qz > 0.06 * R:
                                al = 0.30                                         # the slit's frame
                            if abs(lat) < D[idx, 5] + 0.22 and fw < -0.30 * R and fw > -0.80 * R:
                                al *= 0.72                                        # the shutter, run back over the crown
                            r, g, b = shade_metal(nx, ny, nz, dx, dy, dz, al, al, al * 1.04, ao, 0.42)
                    f = math.exp(-best / 70000.0)
                    tm = 10.0 * (math.exp(-max(z, 0.0) / 10.0) - math.exp(-ez / 10.0)) / max(0.5, abs(ez - z)) * best / 3400.0
                    fm = math.exp(-tm)
                    r = (r * f + HAZE[0] * (1 - f)) * fm + MIST[0] * (1 - fm)
                    g = (g * f + HAZE[1] * (1 - f)) * fm + MIST[1] * (1 - fm)
                    b = (b * f + HAZE[2] * (1 - f)) * fm + MIST[2] * (1 - fm)
                    a0 += r; a1 += g; a2 += b; cov += 1.0
            q = 1.0 / (ss * ss)
            out[j - y0, i - x0, 0] = a0 * q; out[j - y0, i - x0, 1] = a1 * q
            out[j - y0, i - x0, 2] = a2 * q; out[j - y0, i - x0, 3] = cov * q


def build_things(CH, main):
    """The telescope on its hill, the gompa on its crag, one house by the river."""
    B = []; C = []; D = []
    ids = [0]

    def ground(x, y):
        return max(0.0, terrain(CH, x, y, 0.3))

    def box(x, y, a, b, hgt, rot=0.0, k=0.0, mat=0, ztop=None, zbot=None, sink=2.5):
        th = rot * D2R
        c, s = math.cos(th), math.sin(th)
        gs = [ground(x + du * c - dv * s, y + du * s + dv * c) for du in (-a, 0, a) for dv in (-b, 0, b)]
        g = max(gs)
        zt = g + hgt if ztop is None else ztop
        ids[0] += 1
        B.append([x, y, c, s, a, b, zt, min(gs) - sink if zbot is None else zbot, k, mat, g, ids[0]])
        return len(B) - 1

    # ---- the telescope
    g = ground(DHX, DHY)
    R = 6.4
    C.append([DHX, DHY, g - 2.0, g + 8.6, R, 0.0])                    # the drum
    C.append([DHX, DHY, g + 5.6, g + 5.85, R + 1.15, 1.0])            # the gallery
    C.append([DHX, DHY, g + 6.75, g + 6.85, R + 1.12, 1.0])           # its rail
    D.append([DHX, DHY, g + 8.6 + 0.9, R + 0.12, 222.0 * D2R, 1.45])
    th = 38.0
    ux, uy = DH_S, DH_C; vx, vy = DH_C, -DH_S
    box(DHX + 12.5 * vx + 2.0 * ux, DHY + 12.5 * vy + 2.0 * uy, 7.5, 4.6, 4.6, rot=-th + 90.0, mat=4)      # the control rooms
    box(DHX + 21.5 * vx + 4.0 * ux, DHY + 21.5 * vy + 4.0 * uy, 3.0, 3.4, 3.3, rot=-th + 90.0, mat=4)
    # a small dome farther along the ridge, and the weather mast
    sx, sy = DHX + 30.0 * ux - 7.0 * vx, DHY + 30.0 * uy - 7.0 * vy
    gsm = ground(sx, sy)
    C.append([sx, sy, gsm - 2.0, gsm + 2.9, 2.0, 0.0])
    D.append([sx, sy, gsm + 3.1, 2.1, 300.0 * D2R, 0.45])
    mx, my = DHX - 15.0 * vx - 3.0 * ux, DHY - 15.0 * vy - 3.0 * uy
    box(mx, my, 0.09, 0.09, 11.0, mat=5)
    gm = ground(mx, my)
    box(mx, my, 0.75, 0.06, 0.0, rot=20.0, mat=5, ztop=gm + 10.2, zbot=gm + 10.05)

    # ---- the gompa
    def cbox(u, v, a, b, storeys, rot=0.0, mat=0, lift=0.0):
        box(CRX + u, CRY + v, a, b, 3.0 * storeys + 1.4 + lift, rot=rot, k=0.07, mat=mat, sink=6.0)
    cbox(0, 2, 9.0, 7.0, 4, mat=1)
    cbox(-0.5, 3.0, 4.6, 3.8, 5, mat=1, lift=1.0)
    cbox(-17.5, 0.5, 8.0, 6.0, 3, rot=3)
    cbox(-31.0, -1.5, 6.4, 5.0, 2, rot=-4)
    cbox(16.5, -0.5, 7.2, 6.0, 3, rot=-3)
    cbox(29.5, -2.5, 6.2, 5.2, 3, rot=4)
    cbox(41.0, -6.0, 5.2, 4.4, 2, rot=-6)
    cbox(-9.0, -13.5, 7.0, 4.4, 2, rot=-2)
    cbox(8.5, -15.0, 6.2, 4.2, 2, rot=3)
    cbox(22.5, -14.0, 5.0, 4.0, 2, rot=-5)
    cbox(-24.5, -12.5, 5.6, 4.0, 2, rot=6)
    cbox(1.0, -25.0, 5.2, 3.6, 2, rot=-3)
    cbox(-14.5, -24.0, 4.4, 3.4, 1, rot=4)
    cbox(15.5, -26.0, 4.2, 3.2, 1, rot=-7)
    cbox(-42.0, -6.0, 3.8, 3.2, 1, rot=-8)

    # ---- the house, on the bank, its window toward me
    hx, hy, best = -70.0, 300.0, 1e18
    for yy in np.arange(240.0, 460.0, 3.0):                           # dry ground, with open water between it and me
        for xx in np.arange(-260.0, 80.0, 3.0):
            d = math.hypot(xx, yy); ux, uy = -xx / d, -yy / d
            if min(terrain(CH, xx + a_, yy + b_, 0.6) for a_ in (-7, 0, 7) for b_ in (-6, 0, 6)) < 0.35:
                continue
            wet = [terrain(CH, xx + ux * q, yy + uy * q, 0.6) < -0.3 for q in range(12, 44, 4)]
            if not all(wet):
                continue
            sc_ = (xx + 70.0) ** 2 + (yy - 300.0) ** 2
            if sc_ < best:
                best = sc_; hx = xx; hy = yy
    hn = math.hypot(hx, hy)
    rot = math.degrees(math.atan2(-hx / hn, hy / hn)) + 9.0           # box +y points away from me
    c, s = math.cos(rot * D2R), math.sin(rot * D2R)
    hi = box(hx, hy, 4.6, 3.1, 3.3, rot=rot, k=0.03, mat=2)
    box(hx - 6.6 * c - 0.6 * (-s), hy - 6.6 * s - 0.6 * c, 2.2, 2.5, 2.4, rot=rot, k=0.03, mat=2)
    # the pen: four low walls of stone
    px, py = hx + 12.5 * c + 2.0 * (-s), hy + 12.5 * s + 2.0 * c
    for (du, dv, a_, b_) in ((0, -5.0, 6.5, 0.35), (0, 5.0, 6.5, 0.35), (-6.5, 0, 0.35, 5.0), (6.5, 0, 0.35, 5.0)):
        box(px + du * c - dv * s, py + du * s + dv * c, a_, b_, 1.25, rot=rot, mat=3, sink=1.0)
    gh_ = B[hi][10]
    wsc = 1.55; wz = 1.75
    ly = -(3.1 + 0.03 * 1.5) - 0.30
    wx = hx + wsc * c - ly * s; wy = hy + wsc * s + ly * c
    LAMP = np.array([wx, wy, gh_ + wz, s, -c, 0.0, 14.0], np.float64)   # place, the way the window faces, strength
    LIT = np.array([float(B[hi][11]), wsc, wz, 18.0, 2.3], np.float64)
    return np.array(B, np.float64), np.array(C, np.float64), np.array(D, np.float64), LAMP, LIT


# ---------------------------------------------------------------------
#  signature: a compass-drawn star and the altitude
# ---------------------------------------------------------------------
def sign(im8):
    ss = 4
    bw = int(250 * S) + 8; bh = int(74 * S) + 8
    ov = Image.new('L', (bw * ss, bh * ss), 0)
    d = ImageDraw.Draw(ov)
    R = 24 * S * ss; cx = 34 * S * ss; cy = bh * ss / 2
    lw = max(2, int(round(1.5 * S * ss)))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=95, width=max(1, lw // 2))
    pts = [(cx + R * math.cos(-math.pi / 2 + k * math.pi / 4), cy + R * math.sin(-math.pi / 2 + k * math.pi / 4)) for k in range(8)]
    for k in range(8):
        d.line([pts[k], pts[(k + 3) % 8]], fill=255, width=lw)
    d.ellipse([cx - lw, cy - lw, cx + lw, cy + lw], fill=255)
    glyph = {
        '4': [[(0.72, 1.6), (0.72, 0.0), (0.0, 1.08), (1.0, 1.08)]],
        '5': [[(0.9, 0.0), (0.17, 0.0), (0.1, 0.72), (0.5, 0.62), (0.85, 0.82), (0.92, 1.15), (0.7, 1.5), (0.4, 1.6), (0.08, 1.45)]],
        '0': [[(0.5, 0.0), (0.2, 0.2), (0.08, 0.8), (0.2, 1.4), (0.5, 1.6), (0.8, 1.4), (0.92, 0.8), (0.8, 0.2), (0.5, 0.0)]],
        'm': [[(0.0, 1.6), (0.0, 0.72)], [(0.0, 0.98), (0.22, 0.74), (0.48, 0.80), (0.56, 1.0), (0.56, 1.6)],
              [(0.56, 0.98), (0.78, 0.74), (1.04, 0.80), (1.12, 1.0), (1.12, 1.6)]],
    }
    sc = 15 * S * ss; x = cx + R + 20 * S * ss; y0 = cy - 0.8 * sc
    for ch in '4500 m':
        if ch == ' ':
            x += 0.6 * sc; continue
        for line in glyph[ch]:
            d.line([(x + px * sc, y0 + py * sc) for px, py in line], fill=235, width=lw, joint='curve')
        x += (1.55 if ch != 'm' else 1.3) * sc
    ov = ov.resize((bw, bh), Image.LANCZOS)
    a = np.asarray(ov, np.float32) / 255.0 * 0.50
    x0 = int(58 * S); y0 = H - bh - int(46 * S)
    reg = im8[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
    ink = np.array([186, 170, 158], np.float32)
    im8[y0:y0 + bh, x0:x0 + bw] = (reg * (1 - a[..., None]) + ink * a[..., None]).round().astype(np.uint8)


# ---------------------------------------------------------------------
def main():
    T0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(__file__))[0]
    outp = os.path.join(here, stem + ('.png' if S == 1.0 else '_study.png'))
    if len(sys.argv) > 2:
        outp = sys.argv[2]
    rng = np.random.default_rng(SEED)

    def tick(msg):
        print(f'{time.time() - T0:6.1f}s  {msg}', flush=True)

    # --- galaxy
    NEB = np.array([[99.3, 3.7, 1.2, 0.55], [118.2, 5.0, 0.9, 0.40], [123.1, -6.2, 0.4, 0.50],
                    [111.5, 0.8, 0.45, 0.55], [107.2, -0.9, 0.5, 0.40], [103.0, 2.6, 0.5, 0.30]])
    mw = np.zeros((NS, W, 3), np.float32); e0 = np.zeros((NS, W), np.float32)
    tau = np.zeros((NS, W), np.float32); taun = np.zeros((NS, W), np.float32)
    sky_pass(NS, W, H, F, CP, SP, GAL, SK, AK, FK, OK, BK, HK, TK, STRAND, NEB, mw, e0, tau, taun)
    tick('galaxy')

    # --- stars
    x, y, fl, col = make_stars(e0, tau, taun, rng)
    bx = []; by = []; bf = []; bc = []
    for ra, dec, m, bv in BRIGHT:
        px, py, zc = dir_to_pixel(radec_dir(ra, dec))
        if zc > 0.1:
            bx.append(px); by.append(py); bf.append(FREF * 10 ** (-0.4 * (m - 6.0)) * (1.0 + 0.45 * min(4.0, max(0.0, 5.0 - m))))
            bc.append(4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62)))
    # one more
    el, eb = strand_centre(0, 0.46)
    px, py, zc = dir_to_pixel(galactic_dir(el, eb))
    bx.append(px); by.append(py); bf.append(FREF * 10 ** (-0.4 * (3.6 - 6.0)) * 1.63); bc.append(5300.0)
    x = np.concatenate([x, bx]); y = np.concatenate([y, by]); fl = np.concatenate([fl, bf])
    col = np.vstack([col, 1.0 + (bb_rgb(np.array(bc)) - 1.0) * 0.62])
    st = np.zeros((NS, W, 3), np.float32)
    order = np.argsort(y, kind='stable')
    splat(st, x[order], y[order], (fl * S * S)[order], col[order].astype(np.float64), max(S, 0.35))
    tick(f'{x.size} stars')

    # --- the red
    step = 2
    Hd = (NS + step - 1) // step; Wd = (W + step - 1) // step
    au = np.zeros((Hd, Wd, 2), np.float32)
    aurora_pass(Hd, Wd, float(step), W, H, F, CP, SP, au)
    aur = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(au[..., c]), 'F').resize((Wd * step, Hd * step), Image.BICUBIC))[:NS, :W]
                    for c in range(2)], -1)
    aur = np.maximum(aur, 0).astype(np.float32)
    tick('aurora')

    # --- air
    sky = np.zeros((NS, W, 3), np.float32)
    air_pass(NS, W, H, F, CP, SP, mw, st, aur, sky, 0.012, AUR_GAIN, AUR_R, AUR_V)
    del mw, st, aur
    tick('air')

    # --- the ground
    SEG, mainline = river_segments(rng)
    CH = np.zeros((int(4100 / CHC), int(3000 / CHC)), np.float32)
    chan_grid(SEG, CH)
    ez = terrain(CH, 0.0, 0.0, 0.001) + 1.7
    B, C, D, LAMP, LIT = build_things(CH, mainline)
    tick('river')
    PX = 2.0; PY = 2.0
    daz = 1.0 / (F * PX); dsl = 1.0 / (F * PY)
    az0 = -0.815; s_top = 0.24
    NA = int(1.63 / daz); NR = int(0.62 / dsl)
    T = np.full((NR, NA), 1e30, np.float32)
    pano_march(CH, T, az0, daz, s_top, dsl, ez)
    tick('ground marched')
    P = np.zeros((NR, NA, 4), np.float32)
    pano_shade(CH, sky, T, P, az0, daz, s_top, dsl, ez, LAMP, W, H, F, CP, SP)
    tick('ground lit')
    r0 = int(YH - 0.25 * F)
    gnd = np.zeros((H - r0, W, 4), np.float32)
    pano_to_frame(P, az0, daz, s_top, dsl, r0, W, H, F, CP, SP, gnd)
    del P
    img = np.zeros((H, W, 3), np.float32)
    img[:NS] = sky
    del sky
    img[r0:] = img[r0:] * (1 - gnd[..., 3:4]) + gnd[..., :3]
    del gnd

    # --- what people built
    def frame_box(pts, pad):
        xs = []; ys = []
        for p in pts:
            px, py, zc = dir_to_pixel(np.array([p[0], p[1], p[2] - ez]) / np.linalg.norm([p[0], p[1], p[2] - ez]))
            xs.append(px); ys.append(py)
        return (max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)), min(W, int(max(xs) + pad) + 1), min(H, int(max(ys) + pad) + 1))
    groups = [
        [(DHX + dx_, DHY + dy_, DHH + dz_) for dx_ in (-45, 45) for dy_ in (-45, 45) for dz_ in (-6, 22)],
        [(CRX + dx_, CRY + dy_, CRH + dz_) for dx_ in (-60, 60) for dy_ in (-40, 40) for dz_ in (-30, 24)],
        [(LAMP[0] + dx_, LAMP[1] + dy_, LAMP[2] + dz_) for dx_ in (-30, 30) for dy_ in (-30, 30) for dz_ in (-4, 4)],
    ]
    for gpts in groups:
        x0, y0, x1, y1 = frame_box(gpts, 6)
        ob = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        objects_pass(x0, y0, x1, y1, 4, W, H, F, CP, SP, ez, B, C, D, LAMP, LIT, T, az0, daz, s_top, dsl, ob)
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - ob[..., 3:4]) + ob[..., :3]
    lp = dir_to_pixel(np.array([LAMP[0], LAMP[1], LAMP[2] - ez]))
    tick(f'built things; lamp at ({lp[0] / S:.0f}, {lp[1] / S:.0f}) in full-size pixels')
    del T
    # a little of the lamp in the cold air round it
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = ((xx - lp[0]) ** 2 + (yy - lp[1]) ** 2) / (S * S)
    halo = 0.26 * np.exp(-d2 / (2 * 5.0 ** 2)) + 0.045 * np.exp(-d2 / (2 * 18.0 ** 2)) + 0.0075 / (1 + d2 / 60.0 ** 2) ** 1.5
    del d2, xx, yy
    for c in range(3):
        img[..., c] += halo * LAMPC[c]
    del halo

    # --- the print
    # a breath of glow, the way an eye gives it
    for c in range(3):
        g1 = gaussian_filter(img[..., c], 3.0 * S); g2 = gaussian_filter(img[..., c], 26.0 * S)
        img[..., c] = img[..., c] * 0.915 + g1 * 0.055 + g2 * 0.030
    img = np.maximum(img, 0)
    img = 1.5 * img ** 1.14
    # the shoulder of the curve, half of it applied to the brightest channel so the red stays red
    mx = img.max(-1, keepdims=True) + 1e-6
    img = 0.5 * img * (1 + img / 16.0) / (1 + img) + 0.5 * img * (1 + mx / 16.0) / (1 + mx)
    del mx
    # hold the colour back from what a camera would give
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    img = lum[..., None] + (img - lum[..., None]) * 0.94
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1.0 - 0.16 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - 0.55 * H) / (H / 2)) ** 2) ** 1.2 * 0.5
    img *= vig[..., None]
    del xx, yy
    img += np.array([0.0009, 0.0009, 0.0013], np.float32)
    out = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.maximum(img, 1e-9) ** (1 / 2.4) - 0.055)
    # a very fine grain, so the blacks are full of something
    g = rng.normal(0, 1, (H, W)).astype(np.float32)
    g = g * 0.8 + gaussian_filter(g, 0.8) * 0.9
    out += g[..., None] * 0.0052 + rng.normal(0, 0.0022, (H, W, 3)).astype(np.float32)
    out8 = (out.clip(0, 1) * 255 + rng.random((H, W, 3)).astype(np.float32)).clip(0, 255).astype(np.uint8)
    sign(out8)
    Image.fromarray(out8).save(outp, optimize=True)
    tick(f'saved {outp}  {W}x{H}')


if __name__ == '__main__':
    main()
