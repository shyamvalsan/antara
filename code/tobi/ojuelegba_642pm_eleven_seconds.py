"""Ojuelegba, 6:42 pm, eleven seconds

Tobi Adeyemi (rider 417).  Acrylic, car spray and sign-writer's enamel on
salvaged shuttering ply.

Model, views, cut, paint - in that order.
  MODEL  the junction under the flyover: road, kerbs, the bridge on its piers,
         blocks with balconies and signboards, danfos close on both sides, a
         tanker, okadas, poles, wires, the traffic light - solids, in metres.
  VIEWS  ahead, ahead again five metres on, left mirror, right mirror, down at
         the tank, up at the wires, and across at a door rail - true
         perspective views of that one model, each plane counted down to two
         or three flat tones by the sodium lamps.
  CUT    the panel is cut along straight lines that are in those views (the
         sill of a bus, the edge of the bridge, a pier, a wire), plus the arcs
         of the mirrors.
  PAINT  plywood first, then taped flat fills, spray, enamel line, letters.

usage:  python ojuelegba_642pm_eleven_seconds.py [scale] [out.png] [mode]
        scale 1.0 = 2000 x 3000 px; 0.5 for a study.  mode: full | views | layout
"""
import sys, math
import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import hough_line, hough_line_peaks, ProjectiveTransform, warp

WAYBILL = "LOS-417-1842-30611"                      # the delivery he was on
SEED = int("".join(ch for ch in WAYBILL if ch.isdigit())) % (2 ** 31)
TITLE = "Ojuelegba, 6:42 pm, eleven seconds"

# =====================================================================
#  ENGINE 1: solids
# =====================================================================
JIT = dict(cache=True, fastmath=True, error_model='numpy')


@njit(**JIT)
def hash3(i, j, k):
    n = (i * 374761393 + j * 668265263 + k * 1274126177) & 0x7fffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0x7fffffff
    return ((n ^ (n >> 16)) & 0xffffff) / 16777216.0


@njit(**JIT)
def sd_box(x, y, z, bx, by, bz):
    qx = abs(x) - bx; qy = abs(y) - by; qz = abs(z) - bz
    ox = max(qx, 0.0); oy = max(qy, 0.0); oz = max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0)


@njit(**JIT)
def sd_box2(x, y, bx, by):
    qx = abs(x) - bx; qy = abs(y) - by
    ox = max(qx, 0.0); oy = max(qy, 0.0)
    return math.sqrt(ox * ox + oy * oy) + min(max(qx, qy), 0.0)


@njit(**JIT)
def sd_cap(x, y, z, ax, ay, az, bx, by, bz, r):
    px = x - ax; py = y - ay; pz = z - az
    ex = bx - ax; ey = by - ay; ez = bz - az
    h = (px * ex + py * ey + pz * ez) / (ex * ex + ey * ey + ez * ez + 1e-12)
    if h < 0: h = 0.0
    if h > 1: h = 1.0
    px -= ex * h; py -= ey * h; pz -= ez * h
    return math.sqrt(px * px + py * py + pz * pz) - r


@njit(**JIT)
def sd_cylx(x, y, z, r, hw):
    """a wheel: axis along x."""
    a = math.sqrt(y * y + z * z) - r; b = abs(x) - hw
    return min(max(a, b), 0.0) + math.sqrt(max(a, 0.0) ** 2 + max(b, 0.0) ** 2)


@njit(**JIT)
def sd_cylz(x, y, z, r, hl):
    a = math.sqrt(x * x + y * y) - r; b = abs(z) - hl
    return min(max(a, b), 0.0) + math.sqrt(max(a, 0.0) ** 2 + max(b, 0.0) ** 2)


# materials
SKY, ROAD, LANE, KERB, CONC, WALLA, WALLB, WALLC, DARK, SIGN = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
DYEL, DBLK, GLASS, TYRE, HUB, CAB, TANK, BTANK, JACKET, HELMET = 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
POLE, REDR, WATER, BRAKE, HEAD, SKIN, CHROME, RAIL, SUN, PLATE, SHIRT = 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31
GLOVE, TROUS, FAR, SIGN2, LAMP, KERBW, WALLD, GREEN = 32, 33, 34, 35, 36, 37, 38, 39
NMAT = 40


@njit(**JIT)
def danfo(lx, ly, lz, flag):
    """a VW-type bus.  lx to its right, lz to its front.  flag 1: side door open, 2: and a conductor in it."""
    d = sd_box(lx, ly - 1.17, lz, 0.84, 0.80, 2.15) - 0.03
    m = DYEL
    if ly > 1.20:                                       # the nose leans back above the waist
        pl = ((lz - 2.18) + (ly - 1.20) * 0.28) * 0.96
        if pl > d: d = pl
    ax = abs(lx)
    if d < 0.03:
        side = ax > 0.835; back = lz < -2.14; front = lz > 1.95 and not side
        if ly < 0.47: m = DBLK
        elif (ly > 0.80 and ly < 0.925) or (ly > 1.01 and ly < 1.135): m = DBLK      # the two stripes
        elif ly > 1.40 and ly < 1.82:
            if side and lz > -1.95 and lz < 1.72:
                f = (lz + 1.95) / 1.223; f -= math.floor(f)
                if f > 0.09: m = GLASS
            elif back and ax < 0.70: m = GLASS
            elif front and ax < 0.74 and ly < 1.86: m = GLASS
        if back:
            if ax > 0.60 and ax < 0.80 and ly > 0.50 and ly < 0.76: m = BRAKE
            if ax < 0.27 and ly > 0.53 and ly < 0.70: m = PLATE
        if front:
            hx = ax - 0.56; hy = ly - 0.66
            if hx * hx + hy * hy < 0.0150: m = HEAD
            if ax < 0.27 and ly > 0.50 and ly < 0.62: m = PLATE
    if flag > 0:                                        # door opening on the right side
        dd = sd_box(lx - 0.80, ly - 1.14, lz - 0.55, 0.50, 0.66, 0.52)
        if -dd > d:
            d = -dd; m = DARK
        dp = sd_box(lx - 0.905, ly - 1.14, lz + 0.56, 0.022, 0.66, 0.52)       # the door, slid back
        if dp < d:
            d = dp; m = DYEL
            if (ly > 0.80 and ly < 0.925) or (ly > 1.01 and ly < 1.135): m = DBLK
            elif ly > 1.40 and ly < 1.76 and abs(lz + 0.56) < 0.42: m = GLASS
        dr = sd_cap(lx, ly, lz, 0.90, 0.62, 0.17, 0.90, 1.74, 0.17, 0.024)     # the rail
        if dr < d: d = dr; m = RAIL
        if flag > 1:                                    # the conductor: hanging out, one hand back on the rail
            t = sd_cap(lx, ly, lz, 0.76, 1.00, 0.62, 1.00, 1.48, 0.68, 0.17)
            if t < d: d = t; m = SHIRT
            t = sd_cap(lx, ly, lz, 0.70, 0.48, 0.55, 0.76, 0.98, 0.62, 0.10)
            if t < d: d = t; m = TROUS
            a1 = sd_cap(lx, ly, lz, 1.04, 1.50, 0.56, 0.99, 1.36, 0.32, 0.052)
            a2 = sd_cap(lx, ly, lz, 0.99, 1.36, 0.36, 0.93, 1.49, 0.19, 0.046)
            hd = math.sqrt((lx - 1.08) ** 2 + (ly - 1.80) ** 2 + (lz - 0.74) ** 2) - 0.115
            h1 = math.sqrt((lx - 0.925) ** 2 + (ly - 1.50) ** 2 + (lz - 0.17) ** 2) - 0.066
            t = min(min(a1, a2), min(hd, h1))
            if t < d: d = t; m = SKIN
    b = sd_box(lx, ly - 0.44, abs(lz) - 2.20, 0.86, 0.075, 0.05)               # bumpers
    if b < d: d = b; m = DBLK
    wz = abs(lz) - 1.36                                                         # wheels
    w = sd_cylx(ax - 0.75, ly - 0.33, wz, 0.33, 0.11)
    if w < d:
        d = w; m = TYRE
        if (ly - 0.33) ** 2 + wz * wz < 0.031: m = HUB
    return d, m


@njit(**JIT)
def truck(lx, ly, lz, flag):
    """a tanker.  front at +z."""
    ax = abs(lx)
    d = sd_box(lx, ly - 1.95, lz - 3.55, 1.20, 1.15, 0.95) - 0.03; m = CAB
    if d < 0.03:
        if lz > 4.45:
            if ly > 2.05 and ly < 2.72 and ax < 1.08: m = GLASS
            elif ly < 1.30: m = DBLK
            hx = ax - 0.86; hy = ly - 1.08
            if hx * hx + hy * hy < 0.020: m = HEAD
        elif ax > 1.19 and ly > 2.05 and ly < 2.75 and lz > 3.0 and lz < 4.25: m = GLASS
    t = sd_cylz(lx, ly - 2.10, lz + 1.05, 1.12, 3.35) - 0.03
    if t < d:
        d = t; m = TANK
        f = (lz + 4.4) / 2.2; f -= math.floor(f)
        if f < 0.07: m = DBLK
        if lz < -4.38 and lx * lx + (ly - 2.10) ** 2 < 0.30: m = SIGN
    c = sd_box(lx, ly - 0.82, lz + 0.4, 1.05, 0.13, 4.25)
    if c < d: d = c; m = DBLK
    c = sd_box(lx, ly - 0.95, lz + 4.60, 1.20, 0.16, 0.05)
    if c < d:
        d = c; m = DBLK
        if ax > 0.72: m = BRAKE
        if ax < 0.30: m = PLATE
    for k in range(3):
        if k == 0: wz = lz - 3.35
        elif k == 1: wz = lz + 2.35
        else: wz = lz + 3.55
        w = sd_cylx(ax - 1.0, ly - 0.52, wz, 0.52, 0.20)
        if w < d:
            d = w; m = TYRE
            if (ly - 0.52) ** 2 + wz * wz < 0.07: m = HUB
    return d, m


@njit(**JIT)
def moto(lx, ly, lz, flag):
    """a small motorcycle.  flag 1: with rider, 2: my own (forearms and knees only), 3: rider and a box."""
    ax = abs(lx)
    wz = abs(lz) - 0.66
    q = math.sqrt((ly - 0.31) ** 2 + wz * wz) - 0.255
    d = math.sqrt(q * q + lx * lx) - 0.058; m = TYRE
    t = sd_cylx(lx, ly - 0.31, wz, 0.085, 0.07)
    if t < d: d = t; m = HUB
    t = sd_box(lx, ly - 0.44, lz - 0.02, 0.13, 0.15, 0.22) - 0.02
    if t < d: d = t; m = DBLK
    t = sd_cap(lx, ly, lz, 0.0, 0.31, -0.66, 0.0, 0.66, -0.12, 0.04)
    if t < d: d = t; m = DBLK
    t = sd_cap(lx, ly, lz, 0.0, 0.82, 0.04, 0.0, 0.87, 0.36, 0.145)             # the tank
    if t < d: d = t; m = BTANK
    t = sd_cap(lx, ly, lz, 0.0, 0.80, -0.62, 0.0, 0.80, -0.16, 0.115)           # seat
    if t < d: d = t; m = DBLK
    t = sd_cap(ax, ly, lz, 0.075, 0.31, 0.66, 0.075, 1.00, 0.44, 0.03)           # forks
    if t < d: d = t; m = CHROME
    t = sd_cap(lx, ly, lz, 0.0, 0.60, 0.73, 0.0, 0.52, 0.93, 0.07)              # front mudguard
    t2 = sd_cap(lx, ly, lz, 0.0, 0.60, 0.73, 0.0, 0.60, 0.55, 0.07)
    if t2 < t: t = t2
    if t < d: d = t; m = BTANK
    t = sd_cap(lx, ly, lz, -0.36, 1.04, 0.40, 0.36, 1.04, 0.40, 0.020)           # handlebar
    if t < d: d = t; m = CHROME
    t = sd_cap(ax, ly, lz, 0.27, 1.04, 0.40, 0.40, 1.04, 0.40, 0.030)            # grips
    if t < d: d = t; m = DBLK
    t = math.sqrt(lx * lx + (ly - 0.93) ** 2 + (lz - 0.55) ** 2) - 0.095          # headlamp
    if t < d:
        d = t; m = CHROME
        if lz > 0.60: m = HEAD
    t = sd_box(lx, ly - 0.76, lz + 0.80, 0.06, 0.04, 0.04)
    if t < d: d = t; m = BRAKE
    if flag == 4:
        t = sd_cap(lx, ly, lz, 0.0, 0.95, -0.30, 0.0, 1.40, -0.12, 0.175)
        if t < d: d = t; m = JACKET
        t = math.sqrt(lx * lx + (ly - 1.68) ** 2 + (lz + 0.05) ** 2) - 0.150
        if t < d: d = t; m = HELMET
    if flag == 2 or flag == 4:
        t = sd_cylz(lx, lz - 0.30, ly - 1.06, 0.075, 0.03)                       # the clock (axis up)
        if t < d: d = t; m = HUB
        t = sd_cap(ax, ly, lz, 0.25, 1.30, -0.06, 0.335, 1.08, 0.37, 0.055)      # forearms
        if t < d: d = t; m = JACKET
        t = sd_cap(ax, ly, lz, 0.30, 1.07, 0.38, 0.39, 1.07, 0.41, 0.058)        # gloves
        if t < d: d = t; m = GLOVE
        t = sd_cap(ax, ly, lz, 0.14, 0.90, -0.32, 0.215, 0.74, 0.14, 0.085)      # thighs, knees
        if t < d: d = t; m = TROUS
    elif flag == 1 or flag == 3:
        t = sd_cap(lx, ly, lz, 0.0, 0.95, -0.30, 0.0, 1.40, -0.12, 0.175)
        if t < d: d = t; m = JACKET
        t = math.sqrt(lx * lx + (ly - 1.68) ** 2 + (lz + 0.05) ** 2) - 0.150
        if t < d: d = t; m = HELMET
        t = sd_cap(ax, ly, lz, 0.20, 1.40, -0.10, 0.34, 1.08, 0.37, 0.055)
        if t < d: d = t; m = JACKET
        t = sd_cap(ax, ly, lz, 0.30, 1.07, 0.38, 0.39, 1.07, 0.41, 0.058)
        if t < d: d = t; m = GLOVE
        t = sd_cap(ax, ly, lz, 0.13, 0.90, -0.30, 0.21, 0.72, 0.14, 0.085)
        t2 = sd_cap(ax, ly, lz, 0.21, 0.72, 0.14, 0.21, 0.30, 0.02, 0.065)
        if t2 < t: t = t2
        if t < d: d = t; m = TROUS
        if flag == 3:
            t = sd_box(lx, ly - 1.12, lz + 0.72, 0.23, 0.21, 0.21) - 0.01
            if t < d: d = t; m = SIGN
    return d, m


@njit(**JIT)
def vehicles(x, y, z, d, m, VEH, P):
    for k in range(VEH.shape[0] + 1):
        if k < VEH.shape[0]:
            typ = int(VEH[k, 0]); cx = VEH[k, 1]; cz = VEH[k, 2]; dr = int(VEH[k, 3]); fl = int(VEH[k, 4])
        else:
            if P[2] < 0.5: break
            typ = 2; cx = P[0]; cz = P[1]; dr = 0; fl = 2
            if P[2] > 1.5: fl = 4                       # for the sun: all of me
        if dr == 0: lx = x - cx; lz = z - cz
        elif dr == 1: lx = -(x - cx); lz = -(z - cz)
        elif dr == 2: lx = -(z - cz); lz = x - cx
        else: lx = z - cz; lz = -(x - cx)
        if typ == 0: bd = sd_box(lx - 0.45, y - 1.0, lz, 1.75, 1.5, 2.75)
        elif typ == 1: bd = sd_box(lx, y - 1.6, lz, 1.75, 2.1, 5.3)
        else: bd = sd_box(lx, y - 0.95, lz, 0.90, 1.40, 1.45)
        if bd < 0.0:
            if typ == 0: dd, mm = danfo(lx, y, lz, fl)
            elif typ == 1: dd, mm = truck(lx, y, lz, fl)
            else: dd, mm = moto(lx, y, lz, fl)
            if dd < d: d = dd; m = mm
        else:
            bd += 0.30                                  # outside: never step further than into the padding
            if bd < d: d = bd
    return d, m


# =====================================================================
#  THE MODEL: Ojuelegba, coming up to the flyover.
#  x to my right, y up, z the way I am going.  Metres.
# =====================================================================
VEH = np.array([           # type (0 danfo 1 tanker 2 okada), x, z, facing (0 = +z, 2 = +x), flag
    (0, -1.90, 4.6, 0, 2),      # on my left, door open, conductor hanging
    (0, 2.30, 1.3, 0, 0),       # on my right
    (0, 2.75, 12.5, 0, 0),      # ahead, braking
    (1, 0.30, 19.5, 0, 0),      # the tanker, stopped under the bridge: the gap closing
    (2, 0.55, 9.0, 0, 1),       # the other bike, in the gap
    (0, -2.5, -8.5, 0, 0),      # behind, left
    (2, 1.75, -5.0, 0, 1),      # behind, right
    (0, 3.9, -10.5, 0, 0),
    (0, -3.6, 15.5, 0, 0),
], dtype=np.float64)
BLD = np.array([           # x0, x1, z0, z1, height, material, balconies
    (-26.0, -9.5, -36.0, -5.0, 10.0, WALLA, 1),
    (-26.0, -9.5, -3.0, 14.0, 13.2, WALLB, 1),
    (-26.0, -9.5, 38.0, 62.0, 16.4, WALLC, 0),
    (9.5, 26.0, -30.0, 1.0, 13.2, WALLC, 1),
    (9.5, 26.0, 3.0, 15.0, 9.8, WALLA, 1),
    (9.5, 26.0, 37.0, 66.0, 12.8, WALLB, 0),
], dtype=np.float64)
SGN = np.array([           # signboards: centre x, y, z, half sizes, material
    (-9.25, 4.9, 6.0, 0.12, 0.85, 3.6, SIGN),
    (9.25, 11.2, 9.0, 0.12, 1.3, 4.2, SIGN2),
    (9.25, 4.6, -9.0, 0.12, 0.8, 3.0, SIGN),
    (-9.25, 8.0, 46.0, 0.12, 1.2, 4.5, SIGN2),
    (9.25, 6.0, 44.0, 0.12, 1.5, 4.0, SIGN),
], dtype=np.float64)
POLES = np.array([         # x, z, height, type (0 street light with arm toward the road, 1 pole with a cross-arm)
    (6.95, 6.0, 9.0, 0), (-6.95, -6.0, 9.0, 0), (6.95, 42.0, 9.0, 0), (-6.95, 14.5, 9.0, 0),
    (7.7, -2.0, 8.4, 1), (-7.7, 9.0, 8.4, 1), (7.7, 17.0, 8.4, 1), (-7.7, -13.0, 8.4, 1),
], dtype=np.float64)
BRK = np.array([(2.75, 10.2, 0.70), (0.30, 14.85, 0.95)])      # braking: red on the road behind them
LIGHT = np.array([1.0, 3.0,
                  4.1, 8.9, 6.0, 16.5,
                  -4.1, 8.9, -6.0, 16.5,
                  -4.1, 8.9, 14.5, 12.0])
TMAX = 240.0
FLY_Z0, FLY_Z1, FLY_Y = 20.0, 32.0, 6.3


@njit(**JIT)
def scene(x, y, z, P):
    ax = abs(x)
    # ---- the road, the walkways
    d = y; m = ROAD
    if d < 0.03:
        if ax < 6.3 and z > 12.4 and z < 15.4:                      # zebra
            f = (x + 6.3) / 1.4; f -= math.floor(f)
            if f < 0.5: m = LANE
        elif abs(x - 0.25) < 0.075 and z < 11.0:
            g = z / 6.0; g -= math.floor(g)
            if g < 0.42: m = LANE
        for k in range(BRK.shape[0]):
            if z < BRK[k, 1] - 0.35 and z > BRK[k, 1] - 3.6 and abs(abs(x - BRK[k, 0]) - BRK[k, 2]) < 0.22: m = REDR
    if ax > 6.0:
        t = sd_box2(ax - 16.5, y - 0.11, 10.0, 0.11)
        if t < d:
            d = t; m = CONC
            if ax < 6.95:
                g = z / 2.0; g -= math.floor(g)
                m = KERB
                if g < 0.5: m = KERBW
    # ---- buildings with balconies, windows, signboards
    if ax > 7.5:
        for k in range(BLD.shape[0]):
            cx = 0.5 * (BLD[k, 0] + BLD[k, 1]); cz = 0.5 * (BLD[k, 2] + BLD[k, 3]); hh = BLD[k, 4]
            hx = 0.5 * (BLD[k, 1] - BLD[k, 0]); hz = 0.5 * (BLD[k, 3] - BLD[k, 2])
            bd = sd_box(x - cx, y - hh * 0.5, z - cz, hx + 1.5, hh * 0.5 + 0.2, hz + 0.2)
            if bd < 0.0:
                t = sd_box(x - cx, y - hh * 0.5, z - cz, hx, hh * 0.5, hz)
                mm = int(BLD[k, 5])
                if t < 0.03 and ax < 9.6 and y < hh - 0.7 and y > 0.4:
                    yy = y / 3.2; fl = math.floor(yy); yy -= fl
                    zz = (z - BLD[k, 2]) / 3.4; fz = math.floor(zz); zz -= fz
                    if yy > 0.34 and yy < 0.80 and zz > 0.28 and zz < 0.74:
                        mm = DARK
                        if hash3(int(fl), int(fz), k + 3) < 0.30: mm = LAMP
                if BLD[k, 6] > 0.5 and y > 2.4 and y < hh - 1.6:
                    yb = y - 3.2 * math.floor(y / 3.2 + 0.5)
                    t2 = sd_box(ax - 8.95, yb - 0.38, z - cz, 0.62, 0.50, hz - 0.6)
                    if t2 < t: t = t2; mm = int(BLD[k, 5])
                if t < d: d = t; m = mm
            else:
                bd += 0.3
                if bd < d: d = bd
        for k in range(SGN.shape[0]):
            t = sd_box(x - SGN[k, 0], y - SGN[k, 1], z - SGN[k, 2], SGN[k, 3], SGN[k, 4], SGN[k, 5])
            if t < d: d = t; m = int(SGN[k, 6])
    else:
        t = 7.5 - ax + 0.3
        if t < d: d = t
    # ---- the flyover: deck, parapets, hammerhead piers
    bd = sd_box2(y - 4.5, z - 26.0, 5.0, 6.6)
    if bd < 0.0:
        t = sd_box2(y - (FLY_Y + 0.75), z - 26.0, 0.75, 6.0)
        t2 = sd_box2(y - (FLY_Y + 1.95), abs(z - 26.0) - 5.82, 0.45, 0.18)
        if t2 < t: t = t2
        xr = ax - 9.0
        if ax > 18.0: xr = ax - 27.0
        t2 = sd_box(xr, y - (FLY_Y - 0.5), z - 26.0, 1.4, 0.5, 5.2)
        if t2 < t: t = t2
        t2 = sd_box(xr, y - 2.7, abs(z - 26.0) - 3.0, 0.55, 2.8, 0.55)
        if t2 < t: t = t2
        if t < d: d = t; m = CONC
    else:
        bd += 0.3
        if bd < d: d = bd
    # ---- poles, lamps, the traffic light
    if y < 10.5:
        for k in range(POLES.shape[0]):
            px = POLES[k, 0]; pz = POLES[k, 1]; ph = POLES[k, 2]
            bd = sd_box(x - px, y - 5.0, z - pz, 3.4, 5.4, 1.5)
            if bd < 0.0:
                rr = math.sqrt((x - px) ** 2 + (z - pz) ** 2)
                t = max(rr - 0.11, y - ph); mm = POLE
                if POLES[k, 3] < 0.5:
                    sg = -1.0
                    if px < 0: sg = 1.0
                    t2 = sd_cap(x, y, z, px, ph - 0.1, pz, px + sg * 2.6, ph + 0.25, pz, 0.06)
                    if t2 < t: t = t2
                    t2 = sd_box(x - (px + sg * 2.85), y - (ph + 0.22), z - pz, 0.42, 0.08, 0.17)
                    if t2 < t:
                        t = t2
                        if y < ph + 0.2: mm = LAMP
                else:
                    t2 = sd_box(x - px, y - (ph - 0.5), z - pz, 0.07, 0.07, 1.2)
                    if t2 < t: t = t2
                    t2 = sd_box(x - px, y - (ph - 1.3), z - pz, 0.07, 0.07, 0.8)
                    if t2 < t: t = t2
                if t < d: d = t; m = mm
            else:
                bd += 0.3
                if bd < d: d = bd
        bd = sd_box(x - 4.4, y - 3.2, z - 17.6, 2.6, 3.4, 0.6)
        if bd < 0.0:
            t = max(math.sqrt((x - 6.4) ** 2 + (z - 17.6) ** 2) - 0.09, y - 5.9)
            t2 = sd_cap(x, y, z, 6.4, 5.8, 17.6, 2.9, 5.8, 17.6, 0.06)
            if t2 < t: t = t2
            mm = POLE
            t2 = sd_box(x - 3.0, y - 5.25, z - 17.6, 0.24, 0.66, 0.16)
            if t2 < t:
                t = t2
                if z < 17.5:
                    if (x - 3.0) ** 2 + (y - 5.66) ** 2 < 0.024: mm = BRAKE
                    elif (x - 3.0) ** 2 + (y - 5.25) ** 2 < 0.024 or (x - 3.0) ** 2 + (y - 4.84) ** 2 < 0.024: mm = DARK
            if t < d: d = t; m = mm
        else:
            bd += 0.3
            if bd < d: d = bd
    else:
        t = y - 10.2
        if t < d: d = t
    return vehicles(x, y, z, d, m, VEH, P)


@njit(**JIT)
def far(ox, oy, oz, dx, dy, dz):
    """beyond the model: the dusk.  -> t, material, tone"""
    el = math.asin(max(-1.0, min(1.0, dy)))
    az = math.atan2(dx, dz)
    if el < 0.075 + 0.05 * math.cos(az - 0.5): return 1e9, SKY, 0
    if el < 0.36 + 0.10 * math.cos(az - 0.5): return 1e9, SKY, 1
    return 1e9, SKY, 2


# =====================================================================
#  ENGINE 2: the tracer.  Nearest surface, its material, and its tone:
#  0 lit, 1 half (glancing), 2 shade or cast shadow.  Nothing in between.
# =====================================================================
@njit(**JIT)
def _open(ox, oy, oz, dx, dy, dz, maxt, P):
    t = 0.04
    for i in range(220):
        h, m = scene(ox + dx * t, oy + dy * t, oz + dz * t, P)
        if h < 0.0012: return False
        t += max(h, 0.012)
        if t > maxt: break
    return True


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def trace(O, D, P0, LG, tmax):
    n = D.shape[0]
    P = P0.copy(); PS = P0.copy()                       # own bike: 1 seen and casting, 3 casting only
    if P0[2] > 1.5: P[2] = 0.0
    if P0[2] > 0.5: PS[2] = 2.0
    T = np.full(n, 1e9, np.float32); M = np.zeros(n, np.uint8); TN = np.zeros(n, np.uint8)
    for i in prange(n):
        ox = O[0]; oy = O[1]; oz = O[2]; dx = D[i, 0]; dy = D[i, 1]; dz = D[i, 2]
        t = 0.05; hit = False; mm = 0
        for s in range(700):
            d, m = scene(ox + dx * t, oy + dy * t, oz + dz * t, P)
            if d < 0.0006 * t + 0.0004:
                hit = True; mm = m; break
            t += d
            if t > tmax: break
        if not hit:
            tf, mf, tnf = far(ox, oy, oz, dx, dy, dz)
            T[i] = tf; M[i] = mf; TN[i] = tnf
            continue
        px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
        e = 0.0006 * t + 0.0010
        nx = scene(px + e, py, pz, P)[0] - scene(px - e, py, pz, P)[0]
        ny = scene(px, py + e, pz, P)[0] - scene(px, py - e, pz, P)[0]
        nz = scene(px, py, pz + e, P)[0] - scene(px, py, pz - e, P)[0]
        nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nl; ny /= nl; nz /= nl
        T[i] = t; M[i] = mm
        off = 0.012 + 0.0012 * t
        qx = px + nx * off; qy = py + ny * off; qz = pz + nz * off
        tone = 2
        if mm == BRAKE or mm == HEAD or mm == LAMP:
            tone = 0
        elif LG[0] < 0.5:
            ndl = nx * LG[1] + ny * LG[2] + nz * LG[3]
            if ndl > 0.02 and _open(qx, qy, qz, LG[1], LG[2], LG[3], 140.0, PS):
                tone = 1
                if ndl > 0.30: tone = 0
        else:
            nl_ = int(LG[1])
            for k in range(nl_):
                lx = LG[2 + 4 * k] - qx; ly = LG[3 + 4 * k] - qy; lz = LG[4 + 4 * k] - qz; rg = LG[5 + 4 * k]
                ll = math.sqrt(lx * lx + ly * ly + lz * lz)
                if ll > rg: continue
                lx /= ll; ly /= ll; lz /= ll
                ndl = nx * lx + ny * ly + nz * lz
                if ndl > 0.04 and _open(qx, qy, qz, lx, ly, lz, ll - 0.25, PS):
                    tk = 1
                    if ndl > 0.34 and ll < rg * 0.62: tk = 0
                    if tk < tone: tone = tk
        TN[i] = tone
    return T, M, TN


# ---- cameras and placements ------------------------------------------------
def basis(v):
    yw = math.radians(v['yaw']); p = math.radians(v['pitch'])
    f = np.array([math.sin(yw) * math.cos(p), math.sin(p), math.cos(yw) * math.cos(p)])
    r = np.array([math.cos(yw), 0.0, -math.sin(yw)])
    u = np.cross(f, r)
    rl = math.radians(v.get('roll', 0.0))
    if rl:
        r, u = r * math.cos(rl) + u * math.sin(rl), u * math.cos(rl) - r * math.sin(rl)
    return f, r, u, math.tan(math.radians(v['fov']) / 2)


def view_trace(v, U, V):
    """trace view v at displayed coordinates (U, V); V up, |V| <= 1 is the nominal frame height."""
    f, r, u, tn = basis(v)
    if v.get('flip'): U = -U
    D = f[None, :] + (U[:, None] * r[None, :] + V[:, None] * u[None, :]) * tn
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    P = np.array(v.get('own', (0.0, 0.0, 0.0)), dtype=np.float64)
    return trace(np.array(v['pos'], dtype=np.float64), np.ascontiguousarray(D), P, LIGHT, TMAX)


def px_to_uv(pl, X, Y, W, H):
    unit = pl['s'] * min(W, H) / 2
    dx = (X - pl['cx'] * W) / unit; dy = -(Y - pl['cy'] * H) / unit
    a = math.radians(pl.get('rot', 0.0))
    return dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)


def uv_to_px(pl, U, V, W, H):
    unit = pl['s'] * min(W, H) / 2
    a = math.radians(pl.get('rot', 0.0))
    dx = U * math.cos(a) + V * math.sin(a); dy = -U * math.sin(a) + V * math.cos(a)
    return pl['cx'] * W + dx * unit, pl['cy'] * H - dy * unit


def project(pl, v, P3, W, H):
    """world points -> panel pixels through a placement.  also returns depth along the ray."""
    f, r, u, tn = basis(v)
    rel = np.asarray(P3, dtype=np.float64) - np.array(v['pos'])[None, :]
    zc = rel @ f
    zs = np.where(zc > 0.05, zc, np.nan)
    U = (rel @ r) / (zs * tn); V = (rel @ u) / (zs * tn)
    if v.get('flip'): U = -U
    X, Y = uv_to_px(pl, U, V, W, H)
    return X, Y, np.linalg.norm(rel, axis=1)


# ---- small noise helpers ------------------------------------------------------
def vnoise(H, W, cy, cx, rng):
    """smooth value noise in [0,1], cell size (cy, cx) pixels."""
    gh = int(H / max(cy, 1e-6)) + 4; gw = int(W / max(cx, 1e-6)) + 4
    gh = min(gh, H + 4); gw = min(gw, W + 4)
    g = rng.random((gh, gw)).astype(np.float32)
    zy = (H + 3 * cy) / gh if gh < H + 4 else 1.0
    out = ndi.zoom(g, (max(H / (gh - 3), 1.0), max(W / (gw - 3), 1.0)), order=3, mode='reflect')
    oy = int(rng.integers(0, max(out.shape[0] - H, 0) + 1)); ox = int(rng.integers(0, max(out.shape[1] - W, 0) + 1))
    out = out[oy:oy + H, ox:ox + W]
    if out.shape != (H, W):
        out = np.pad(out, ((0, H - out.shape[0]), (0, W - out.shape[1])), mode='edge')
    return np.clip(out, 0, 1)


def fbm(H, W, cy, cx, rng, octs=3):
    s = np.zeros((H, W), np.float32); a = 1.0; t = 0.0
    for o in range(octs):
        s += a * vnoise(H, W, max(cy / 2 ** o, 1.0), max(cx / 2 ** o, 1.0), rng); t += a; a *= 0.5
    return s / t


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# =====================================================================
#  ENGINE 3: the ground.  Salvaged shuttering ply: rotary-cut face veneer,
#  a knot or two, old screw holes, cement haze, the ghost of an old sign.
# =====================================================================
def plywood(H, W, S, rng, knots, holes, ghost=None):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lam = 30.0 * S                                               # mean band spacing, px
    f = yy / lam + 7.5 * fbm(H, W, 150 * S, 1500 * S, rng, 2) + 1.3 * fbm(H, W, 34 * S, 520 * S, rng, 2)
    f += 0.25 * xx / (W) * 6.0 * (vnoise(H, W, 900 * S, 2500 * S, rng) - 0.5)
    kdark = np.zeros((H, W), np.float32)
    for (kx, ky, kr) in knots:
        ex = (xx - kx * W) / (kr * S * 1.9); ey = (yy - ky * H) / (kr * S)
        r2 = ex * ex + ey * ey
        f += 4.2 * np.exp(-r2 / 5.5) * np.sign(ey + 1e-6) * np.minimum(np.abs(ey), 1.0)
        kdark = np.maximum(kdark, np.exp(-(r2 ** 2.2) / 0.30) + 0.45 * np.exp(-((np.sqrt(r2) - 1.12) / 0.10) ** 2))
    g = f - np.floor(f)
    band = np.floor(f).astype(np.int64)
    amp = (0.55 + 0.45 * ((band * 2654435761 % 1000) / 1000.0)).astype(np.float32)
    lw = (g ** 2.6) * amp                                        # latewood: slow darkening, sharp return
    lw = lw * sstep(1.0, 0.93, g) + 0.0
    fib = vnoise(H, W, 1.4 * S, 240 * S, rng) - 0.5
    fib2 = vnoise(H, W, 3.5 * S, 700 * S, rng) - 0.5
    pores = (vnoise(H, W, 1.2 * S, 16 * S, rng) > 0.80).astype(np.float32) * (1 - lw) * (vnoise(H, W, 60 * S, 300 * S, rng) > 0.35)
    pores = ndi.gaussian_filter(pores, 0.5 * S)
    tint = fbm(H, W, 500 * S, 1100 * S, rng, 2) - 0.5
    early = np.array([0.800, 0.635, 0.415], np.float32); late = np.array([0.600, 0.415, 0.240], np.float32)
    col = early[None, None, :] + (late - early)[None, None, :] * np.clip(lw * 0.85 + 0.25 * fib2 + 0.12, 0, 1)[..., None]
    col *= (1.0 + 0.07 * fib + 0.10 * tint - 0.16 * pores)[..., None]
    col[..., 2] *= (1.0 - 0.10 * tint)
    # weather and cement: shuttering ply carries a grey haze, heavier in drifts
    haze = sstep(0.50, 0.80, fbm(H, W, 380 * S, 700 * S, rng, 3)) * (0.55 + 0.45 * vnoise(H, W, 5 * S, 60 * S, rng))
    haze *= (1 - 0.6 * lw)
    grey = np.array([0.74, 0.73, 0.69], np.float32)
    col = col * (1 - 0.26 * haze[..., None]) + grey * 0.26 * haze[..., None]
    if ghost is not None:                                        # whatever was painted here before
        gm, gc = ghost
        ga = gm * (0.26 - 0.20 * lw) * (0.5 + 0.5 * vnoise(H, W, 40 * S, 90 * S, rng))
        col = col * (1 - ga[..., None]) + np.array(gc, np.float32) * ga[..., None]
    knot_c = np.array([0.27, 0.16, 0.09], np.float32)
    col = col * (1 - 0.85 * kdark[..., None]) + knot_c * 0.85 * kdark[..., None]
    hgt = 0.55 * lw - 0.8 * pores - 0.5 * kdark + 0.10 * fib
    for (hx, hy, hr) in holes:                                   # screw holes, with a little rust below
        cx = hx * W; cy = hy * H; r = hr * S
        dd = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ang = np.arctan2(yy - cy, xx - cx)
        rag = r * (1 + 0.07 * np.sin(ang * 5 + hx * 40) + 0.05 * np.sin(ang * 9 + hy * 70))
        hole = sstep(rag + 0.8 * S, rag - 0.8 * S, dd)
        crush = sstep(r * 2.3, r * 1.2, dd) * (1 - hole)
        stain = np.exp(-((xx - cx) / (r * 1.6)) ** 2) * sstep(-r, r, yy - cy) * np.exp(-np.maximum(yy - cy, 0) / (r * 9)) * 0.38
        stain = np.maximum(stain, 0.30 * np.exp(-(dd / (r * 2.6)) ** 2))
        col = col * (1 - stain[..., None]) + np.array([0.42, 0.20, 0.09], np.float32) * stain[..., None]
        col = col * (1 + 0.10 * crush[..., None])
        col = col * (1 - hole[..., None]) + np.array([0.07, 0.05, 0.04], np.float32) * hole[..., None]
        hgt = hgt - 0.6 * crush - 4.0 * hole
    return np.clip(col, 0, 1), hgt.astype(np.float32), lw.astype(np.float32)


# =====================================================================
#  ENGINE 4: cutting the panel
# =====================================================================
def find_lines(M, TN, T, npk=22):
    """strong straight lines in a low-res view, from its material / tone / depth edges.
    -> list of ((u0, v0), (u1, v1), strength) in view coordinates."""
    h, w = M.shape
    L = M.astype(np.int32) * 4 + TN
    e = np.zeros((h, w), bool)
    e[:, 1:] |= L[:, 1:] != L[:, :-1]; e[1:, :] |= L[1:, :] != L[:-1, :]
    lt = np.log(np.minimum(T, 5000.0))
    e[:, 1:] |= np.abs(lt[:, 1:] - lt[:, :-1]) > 0.25; e[1:, :] |= np.abs(lt[1:, :] - lt[:-1, :]) > 0.25
    th = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    hs, an, ds = hough_line(e, theta=th)
    acc, ang, dist = hough_line_peaks(hs, an, ds, min_distance=6, min_angle=4, threshold=0.25 * hs.max(), num_peaks=npk)
    out = []
    for a, t, r in zip(acc, ang, dist):
        c, s = math.cos(t), math.sin(t)
        x0, y0 = r * c, r * s
        p0 = (x0 - 1000 * s, y0 + 1000 * c); p1 = (x0 + 1000 * s, y0 - 1000 * c)
        uv = [((p[0] - w / 2) / (h / 2), -(p[1] - h / 2) / (h / 2)) for p in (p0, p1)]
        out.append((uv[0], uv[1], float(a)))
    return out


def snap_cut(hint, cands, pl, W, H):
    """move a hinted cut onto the nearest strong line found in the view placed by pl."""
    (hx0, hy0), (hx1, hy1) = hint
    h0 = np.array([hx0 * W, hy0 * H]); h1 = np.array([hx1 * W, hy1 * H])
    ha = math.atan2(h1[1] - h0[1], h1[0] - h0[0])
    best = None
    for (a, b, st) in cands:
        ax, ay = uv_to_px(pl, a[0], a[1], W, H); bx, by = uv_to_px(pl, b[0], b[1], W, H)
        ca = math.atan2(by - ay, bx - ax)
        da = abs((ca - ha + math.pi / 2) % math.pi - math.pi / 2)
        n = np.array([-(by - ay), bx - ax]); n /= np.linalg.norm(n)
        d0 = abs((h0 - np.array([ax, ay])) @ n); d1 = abs((h1 - np.array([ax, ay])) @ n)
        if da > math.radians(6.0) or max(d0, d1) > 0.055 * min(W, H): continue
        cost = math.degrees(da) + 60 * (d0 + d1) / min(W, H)
        if best is None or cost < best[0]:
            p = np.array([ax, ay]); dv = np.array([bx - ax, by - ay]); dv /= np.linalg.norm(dv)
            q0 = p + dv * ((h0 - p) @ dv); q1 = p + dv * ((h1 - p) @ dv)
            best = (cost, q0, q1)
    if best is None: return h0, h1, False
    return best[1], best[2], True


def build_pieces(W, H, cuts, views_lr, places):
    """split the panel.  each cut divides one named region into two named regions."""
    lab = np.zeros((H, W), np.int32); names = {'all': 0}; nid = 1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    geo = []; nsnap = 0; nline = 0
    for c in cuts:
        multi = isinstance(c['in'], (list, tuple))
        if multi: mask = np.isin(lab, [names[k] for k in c['in']])
        else:
            par = names.pop(c['in'])
            mask = lab == par
        if 'line' in c:
            nline += 1
            if c.get('snap'):
                p0, p1, ok = snap_cut(c['line'], views_lr[places[c['snap']]['view']]['lines'], places[c['snap']], W, H)
                nsnap += ok
                if not ok: print('   cut into', c['a'], '/', c['b'], ': no line found near the hint, kept as drawn')
            else:
                p0 = np.array([c['line'][0][0] * W, c['line'][0][1] * H]); p1 = np.array([c['line'][1][0] * W, c['line'][1][1] * H])
            dx, dy = p1 - p0
            sd = (xx - p0[0]) * dy - (yy - p0[1]) * dx           # sign
            if abs(dx) >= abs(dy): a_side = sd * np.sign(dx) > 0        # a = above
            else: a_side = sd * np.sign(dy) > 0                           # a = left
            a_side = ~a_side if abs(dx) < abs(dy) else a_side
            geo.append(('line', p0, p1))
        elif 'circle' in c:
            cx, cy, r = c['circle']
            a_side = (xx - cx * W) ** 2 + (yy - cy * H) ** 2 < (r * min(W, H)) ** 2
            geo.append(('circle', cx * W, cy * H, r * min(W, H)))
        else:                                                    # rounded box: cx, cy, half w, half h, radius, angle
            cx, cy, hw, hh, rr, ang = c['round']
            u = min(W, H); a = math.radians(ang)
            ex = (xx - cx * W) * math.cos(a) + (yy - cy * H) * math.sin(a)
            ey = -(xx - cx * W) * math.sin(a) + (yy - cy * H) * math.cos(a)
            qx = np.abs(ex) - (hw - rr) * u; qy = np.abs(ey) - (hh - rr) * u
            sdv = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - rr * u
            a_side = sdv < 0
            geo.append(('round', c['round']))
        if multi:                                                # a shape laid over several regions
            lab[mask & a_side] = nid; names[c['a']] = nid; nid += 1
            continue
        ia, ib = nid, nid + 1; nid += 2
        lab[mask & a_side] = ia; lab[mask & ~a_side] = ib
        names[c['a']] = ia; names[c['b']] = ib
    print('   %d straight cuts, %d of them snapped to lines found in the views' % (nline, nsnap))
    return lab, names


# =====================================================================
#  ENGINE 5: paint
# =====================================================================
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
FONT_N = "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf"


def text_mask(txt, font=FONT_B, px=220, track=0.04):
    f = ImageFont.truetype(font, px)
    wds = [f.getlength(ch) for ch in txt]
    tw = int(sum(wds) + track * px * (len(txt) - 1)) + 8
    asc, desc = f.getmetrics()
    im = Image.new('L', (tw, asc + desc), 0)
    d = ImageDraw.Draw(im); x = 4.0
    for ch, wd in zip(txt, wds):
        d.text((x, 0), ch, font=f, fill=255); x += wd + track * px
    a = np.asarray(im, np.float32) / 255.0
    rows = np.where(a.max(axis=1) > 0.5)[0]
    return a[rows[0]:rows[-1] + 1]


def warp_quad(src, quad, W, H):
    """lay image src into the panel so its corners (TL, TR, BR, BL) land on quad (pixels)."""
    quad = np.asarray(quad, np.float64)
    if not np.all(np.isfinite(quad)): return None, None
    x0 = int(max(math.floor(quad[:, 0].min()) - 2, 0)); x1 = int(min(math.ceil(quad[:, 0].max()) + 2, W))
    y0 = int(max(math.floor(quad[:, 1].min()) - 2, 0)); y1 = int(min(math.ceil(quad[:, 1].max()) + 2, H))
    if x1 <= x0 or y1 <= y0: return None, None
    sh, sw = src.shape
    tf = ProjectiveTransform()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tf.estimate(quad - np.array([x0, y0]), np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64))
    out = warp(src, tf, output_shape=(y1 - y0, x1 - x0), order=1, mode='constant', cval=0.0)
    return out.astype(np.float32), (y0, y1, x0, x1)


class Panel:
    """the painted surface: colour, relief, gloss."""

    def __init__(self, W, H, S, rng, ply, hgt, lw):
        self.W, self.H, self.S, self.rng = W, H, S, rng
        self.col = ply.copy(); self.hgt = hgt.copy(); self.lw = lw
        self.gloss = np.full((H, W), 0.04, np.float32)
        self.cover = np.zeros((H, W), np.float32)                # how much paint lies on the ply
        # brush fields: fine bristle streaks and broad stroke bands, in three directions
        fine = vnoise(H, W, 1.6 * S, 340 * S, rng) - 0.5; bandn = vnoise(H, W, 95 * S, 1300 * S, rng) - 0.5
        self.br_h = 0.65 * fine + 0.35 * bandn * 2
        fine = vnoise(H, W, 340 * S, 1.6 * S, rng) - 0.5; bandn = vnoise(H, W, 1300 * S, 95 * S, rng) - 0.5
        self.br_v = 0.65 * fine + 0.35 * bandn * 2
        big = int(1.5 * max(W, H))
        fine = vnoise(big, big, 1.6 * S, 340 * S, rng) - 0.5; bandn = vnoise(big, big, 95 * S, 1300 * S, rng) - 0.5
        rot = ndi.rotate(0.65 * fine + 0.70 * bandn, 33.0, reshape=False, order=1)
        oy = (big - H) // 2; ox = (big - W) // 2
        self.br_d = rot[oy:oy + H, ox:ox + W]
        self.patch = fbm(H, W, 260 * S, 260 * S, rng, 2) - 0.5
        self.speck = rng.random((H, W)).astype(np.float32)
        self.speck2 = ndi.gaussian_filter(rng.random((H, W)).astype(np.float32), 0.8 * S)
        self.speck2 = (self.speck2 - self.speck2.mean()) / (self.speck2.std() + 1e-6)

    def lay(self, alpha, colour, thick=0.5, gloss=0.2, level=0.5):
        """put paint down.  alpha (H,W) in [0,1]; colour a 3-vector or an (H,W,3) image."""
        a = alpha[..., None]
        self.col = self.col * (1 - a) + (colour if np.ndim(colour) == 3 else np.asarray(colour, np.float32)[None, None, :]) * a
        self.hgt = self.hgt * (1 - level * alpha) + thick * alpha
        self.gloss = self.gloss * (1 - alpha) + gloss * alpha
        self.cover = np.maximum(self.cover, alpha)

    def fill(self, sl, mask, rgb, hide, finish, direction, edge_in, body=0.80):
        """flat acrylic inside a taped shape (a crop sl of the panel).  hide: hiding power of each colour."""
        S = self.S
        br = {'h': self.br_h, 'v': self.br_v, 'd': self.br_d}[direction][sl]
        lw = self.lw[sl]; patch = self.patch[sl]
        if finish == 'thin':                                     # one dragged coat: the ply ghosts through
            op = np.clip(hide * (body - 0.42 * lw + 0.50 * br + 0.20 * patch), 0.0, 0.95); thick = 0.18; lev = 0.15
        elif finish == 'spray':
            op = np.clip(hide + 0.06, 0, 0.985); thick = 0.22; lev = 0.25
        else:                                                    # two brushed coats
            op = np.clip(hide * (1.0 - 0.45 * lw * (1 - hide) + 0.05 * br + 0.04 * patch), 0, 0.985)
            thick = 0.50; lev = 0.55
        if finish == 'spray': th = thick * (1 + 0.08 * self.speck2[sl])
        else: th = thick * (1 + 0.9 * br)
        ridge = np.exp(-(edge_in / (1.3 * S)) ** 2)              # paint piles against the tape
        th = th + (0.55 if finish != 'spray' else 0.25) * ridge
        c = rgb * (1.0 + (0.030 * br[..., None] if finish != 'spray' else 0.0))
        a = mask * op
        self.col[sl] = self.col[sl] * (1 - a[..., None]) + c * a[..., None]
        self.hgt[sl] = self.hgt[sl] * (1 - lev * mask) + th * mask
        self.gloss[sl] = self.gloss[sl] * (1 - mask) + (0.30 if finish == 'spray' else 0.18) * mask
        self.cover[sl] = np.maximum(self.cover[sl], a)

    def overspray(self, region, piece, colour, reach, strength=0.55):
        """a sprayed shape with no tape on one side: a soft, speckled skirt beyond the piece."""
        S = self.S
        g = ndi.gaussian_filter(region.astype(np.float32), reach * S)
        g2 = ndi.gaussian_filter(region.astype(np.float32), reach * S * 0.35)
        a = np.clip(2.3 * g, 0, 1) ** 1.2 * 0.6 + np.clip(2.2 * g2, 0, 1) ** 1.5 * 0.4
        a = a * (~piece)
        drops = (self.speck < 0.25 + 1.3 * a).astype(np.float32)             # droplets thin out with distance
        a = strength * a * (0.35 + 0.65 * drops)
        self.lay(a.astype(np.float32), colour, thick=0.10, gloss=0.25, level=0.05)

    def run(self, x, y, length, width, colour):
        """a run: too much paint in one place, and gravity."""
        S = self.S; H, W = self.H, self.W
        L = int(length * S); w = width * S
        y0 = int(y); y1 = min(y0 + L + int(4 * w), H); x0 = max(int(x - 6 * w), 0); x1 = min(int(x + 6 * w), W)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        t = (yy - y0) / max(L, 1)
        cxx = x + 1.6 * S * np.sin(t * 5.0) * t
        ww = w * (0.55 + 0.25 * (1 - t)) * (t <= 1)
        body = sstep(ww + 0.7, ww - 0.7, np.abs(xx - cxx)) * (t <= 1)
        bulb = sstep(w * 1.15 + 0.7, w * 1.15 - 0.7, np.sqrt((xx - cxx) ** 2 + (yy - (y0 + L)) ** 2))
        a = np.maximum(body, bulb).astype(np.float32)
        full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = a
        self.lay(full * 0.97, colour, thick=1.1, gloss=0.45, level=0.2)

    def enamel(self, alpha, colour=(0.068, 0.066, 0.075), thick=1.25):
        self.lay(np.clip(alpha, 0, 1).astype(np.float32), colour, thick=thick, gloss=0.85, level=0.75)

    def finish(self):
        """raking daylight over the relief; enamel shines, acrylic does not."""
        S = self.S
        h = ndi.gaussian_filter(self.hgt, 0.6 * S)
        gy, gx = np.gradient(h)
        gx /= S ** 0; gy /= S ** 0
        lam = -(gx * (-0.62) + gy * (-0.78))                                   # light from upper left
        lam = np.clip(lam * 0.40 * S, -0.16, 0.16)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        sheen = np.exp(-(((xx / self.W - 0.30) / 0.55) ** 2 + ((yy / self.H - 0.18) / 0.5) ** 2))
        spec = self.gloss * (0.045 * sheen + 0.30 * np.clip(lam, 0, 1))
        out = self.col * (1.0 + lam[..., None] * (0.55 + 0.45 * (1 - self.gloss[..., None]))) + spec[..., None]
        fall = 1.0 - 0.035 * ((xx / self.W - 0.42) ** 2 + (yy / self.H - 0.40) ** 2) * 2
        return np.clip(out * fall[..., None], 0, 1)


def line_alpha(dist, width, S, rng_field, wob=0.22):
    """an enamel line of a given width laid along distance-zero; the hand makes it swell and thin."""
    w = 0.5 * width * S * (1.0 + wob * rng_field)
    return sstep(w + 0.75, w - 0.75, dist)


def draw_polys(W, H, polys, width):
    """rasterise polylines (lists of (x, y)) into a soft mask."""
    ss = 2
    im = Image.new('L', (W * ss, H * ss), 0); d = ImageDraw.Draw(im)
    for p in polys:
        pts = [(float(x) * ss, float(y) * ss) for x, y in p if np.isfinite(x) and np.isfinite(y)]
        if len(pts) > 1: d.line(pts, fill=255, width=max(int(round(width * ss)), 1), joint='curve')
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255.0


def stencil_417(S, rng):
    """the plate: a stencil, so every island of card needs its bridges."""
    px = int(150 * S)
    m = text_mask("417", font="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", px=px, track=0.10)
    h, w = m.shape
    cols = np.where(m.max(axis=0) > 0.5)[0]
    runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
    bw = max(int(0.055 * h), 2)
    for r in runs:
        c = (r[0] + r[-1]) // 2
        m[:, c - bw // 2:c - bw // 2 + bw] = 0
    hb = int(0.52 * h); m[hb:hb + max(bw * 2 // 3, 1), runs[0][0]:runs[0][-1]] *= 0   # the 4's cross-bar
    pad = int(0.42 * h); bd = max(int(0.085 * h), 2); rr = int(0.36 * h)
    Hh = h + 2 * pad; Ww = w + 2 * pad
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    qx = np.abs(xx - Ww / 2) - (Ww / 2 - rr - bd); qy = np.abs(yy - Hh / 2) - (Hh / 2 - rr - bd)
    sd = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - rr
    ring = (np.abs(sd + bd / 2) < bd / 2).astype(np.float32)
    for fx in (0.27, 0.73):
        c = int(fx * Ww); ring[:, c - bw // 2:c - bw // 2 + bw] = 0
    c = int(0.5 * Hh); ring[c - bw // 2:c - bw // 2 + bw, :] = 0
    out = ring; out[pad:pad + h, pad:pad + w] = np.maximum(out[pad:pad + h, pad:pad + w], m)
    return out


# =====================================================================
#  THE PICTURE
# =====================================================================
C = {  # his colours, and the darker values of them that shade needs
    'yel': (0.965, 0.770, 0.085), 'yel2': (0.880, 0.640, 0.060), 'yel3': (0.640, 0.430, 0.075),
    'blk': (0.075, 0.075, 0.085),
    'wht': (0.915, 0.890, 0.805), 'wht2': (0.770, 0.800, 0.790),
    'blu': (0.560, 0.690, 0.755), 'blu2': (0.395, 0.520, 0.610), 'blu3': (0.215, 0.305, 0.400), 'blu4': (0.130, 0.185, 0.260),
    'red': (0.800, 0.130, 0.095), 'red2': (0.540, 0.090, 0.085),
    'rust': (0.640, 0.285, 0.110), 'org': (0.905, 0.455, 0.085), 'rust2': (0.385, 0.185, 0.100),
    'grn': (0.050, 0.455, 0.265), 'grn2': (0.035, 0.290, 0.190),
}
CN = list(C.keys()); CI = {k: i + 1 for i, k in enumerate(CN)}
CRGB = np.array([(0, 0, 0)] + [C[k] for k in CN], np.float32)
HIDE = {'yel': 0.86, 'yel2': 0.90, 'yel3': 0.94, 'blk': 0.985, 'wht': 0.92, 'wht2': 0.93, 'blu': 0.95, 'blu2': 0.96,
        'blu3': 0.975, 'blu4': 0.98, 'red': 0.90, 'red2': 0.95, 'rust': 0.95, 'org': 0.88, 'rust2': 0.97, 'grn': 0.95, 'grn2': 0.97}
CHIDE = np.array([0.0] + [HIDE[k] for k in CN], np.float32)

DUSK = {  # material -> (under the sodium lamp, half, dusk)
    SKY: ('org', 'blu', 'blu'), ROAD: ('rust', 'blu2', 'blu3'), LANE: ('yel', 'wht2', 'blu2'), REDR: ('red', 'red', 'red2'),
    CONC: ('org', 'blu2', 'blu3'), KERB: ('blk', 'blk', 'blk'), KERBW: ('wht', 'wht2', 'blu2'),
    WALLA: ('org', 'wht2', 'blu2'), WALLB: ('rust', 'rust', 'rust2'), WALLC: ('blu', 'blu2', 'blu3'),
    DARK: ('blk', 'blk', 'blu4'), LAMP: ('yel', 'yel', 'yel'), SIGN: ('red', 'red', 'red2'), SIGN2: ('grn', 'grn', 'grn2'),
    POLE: ('blk', 'blk', 'blk'), DYEL: ('yel', 'yel', 'yel2'), DBLK: ('blk', 'blk', 'blk'), GLASS: ('blu', 'blu3', 'blu4'),
    TYRE: ('blk', 'blk', 'blk'), HUB: ('wht', 'wht2', 'blu2'), CAB: ('red', 'red2', 'red2'), TANK: ('wht', 'wht2', 'blu2'),
    BTANK: ('red', 'red', 'red2'), JACKET: ('blu2', 'blu3', 'blu4'), HELMET: ('wht', 'wht2', 'blu2'), BRAKE: ('red', 'red', 'red'),
    HEAD: ('wht', 'wht', 'wht'), SKIN: ('rust', 'rust2', 'rust2'), CHROME: ('wht', 'wht2', 'blu2'), RAIL: ('wht', 'wht2', 'blu'),
    PLATE: ('wht', 'wht2', 'wht2'), SHIRT: ('grn', 'grn', 'grn2'), GLOVE: ('blk', 'blk', 'blk'), TROUS: ('blu3', 'blu4', 'blk'),
}


def scheme_table(sch, over=None):
    t = np.zeros((NMAT, 3), np.uint8)
    d = dict(sch)
    if over: d.update(over)
    for m, cs in d.items():
        t[m] = [CI[c] for c in cs]
    return t


PANEL = (2000, 3000)
SCHEME = DUSK
EYE = (0.35, 1.56, 0.0)
OWN = (0.35, 0.0, 3.0)
VIEWS = {   # seven looks in eleven seconds
    'ahead': dict(pos=EYE, yaw=2, pitch=4, fov=70, own=OWN),
    'later': dict(pos=(0.45, 1.56, 5.0), yaw=-4, pitch=7, fov=58, own=(0.45, 5.0, 3.0)),
    'lmir': dict(pos=(-0.09, 1.22, 0.30), yaw=180 + 10, pitch=2.0, fov=30, flip=True, own=OWN),
    'rmir': dict(pos=(0.79, 1.22, 0.30), yaw=180 - 13, pitch=1.0, fov=32, flip=True, own=OWN),
    'down': dict(pos=(0.35, 1.62, -0.12), yaw=-10, pitch=-52, fov=74, own=(0.35, 0.0, 1.0)),
    'up': dict(pos=(-1.7, 1.56, 21.2), yaw=24, pitch=58, fov=84, own=OWN),
    'door': dict(pos=EYE, yaw=-17, pitch=-1, fov=30, own=OWN),
}
def place(view, s, anchor, at, rot=0.0, over=None):
    """a view laid on the panel at scale s so that an anchor (a world point, or a (u, v)) lands at `at`."""
    W, H = PANEL
    pl = dict(view=view, s=s, cx=0.0, cy=0.0, rot=rot, over=over)
    if len(anchor) == 3:
        X, Y, _ = project(pl, VIEWS[view], np.array([anchor]), W, H); X = X[0]; Y = Y[0]
    else:
        X, Y = uv_to_px(pl, anchor[0], anchor[1], W, H)
    pl['cx'] = at[0] - X / W; pl['cy'] = at[1] - Y / H
    return pl


def wl(p, a, b):
    """the panel line on which a world line falls, seen through placement p."""
    W, H = PANEL
    X, Y, _ = project(PLACES[p], VIEWS[PLACES[p]['view']], np.array([a, b]), W, H)
    return ((X[0] / W, Y[0] / H), (X[1] / W, Y[1] / H))


def wpt(p, pt):
    W, H = PANEL
    X, Y, _ = project(PLACES[p], VIEWS[PLACES[p]['view']], np.array([pt]), W, H)
    return X[0] / W, Y[0] / H


VP = (0.35, 1.56, 1e5)
VPL = (0.45, 1.56, 1e5)
V0 = (0.50, 0.455)
PLACES = {
    'A': place('ahead', 1.5, VP, V0),
    'A0': place('ahead', 0.8, VP, V0),
    'A1': place('ahead', 1.0, VP, V0),
    'L3': place('later', 1.8, VPL, V0),
    'A2': place('ahead', 2.3, VP, V0),
    'L': place('later', 1.35, VPL, V0),
    'U': place('up', 1.3, (0.0, 0.0), (0.50, 0.19), rot=-9.0),
    'DR': place('door', 1.0, (-0.98, 1.50, 4.77), (0.262, 0.478)),
    'LM': place('lmir', 0.74, (0.0, 0.0), (0.205, 0.665)),
    'RM': place('rmir', 0.70, (0.0, 0.0), (0.80, 0.64)),
    'D': place('down', 0.74, (0.0, -0.2), (0.50, 0.905)),
}
CUTS = [
    dict(**{'in': 'all'}, line=wl('A', (-1.03, 0.40, 3.0), VP), a='nw', b='se'),                 # the sill of the bus on my left
    dict(**{'in': 'nw'}, line=wl('A', (1.43, 0.40, 3.0), VP), a='left', b='top'),                # the sill of the bus on my right
    dict(**{'in': 'se'}, line=wl('A', (1.43, 0.40, 3.0), VP), a='bottom', b='right'),
    dict(**{'in': 'top'}, line=wl('L', (-8.0, FLY_Y, FLY_Z0), (8.0, FLY_Y, FLY_Z0)), a='topup', b='under0'),  # the edge of the bridge
    dict(**{'in': 'topup'}, line=wl('U', (-8.0, FLY_Y, FLY_Z0), (8.0, FLY_Y, FLY_Z0)), a='tsky', b='tund'),   # the same edge, looking up
    dict(**{'in': 'tund'}, line=wl('A1', (6.4, 1.0, 17.6), (6.4, 5.5, 17.6)), a='tun', b='tr'),               # the traffic light's pole
    dict(**{'in': 'under0'}, line=wl('A1', (6.4, 1.0, 17.6), (6.4, 5.5, 17.6)), a='under', b='underr'),
    dict(**{'in': 'left'}, line=wl('A', (-1.03, 2.0, 3.0), VP), a='lroof', b='lbus'),                        # its roof
    dict(**{'in': 'right'}, line=wl('A', (1.43, 2.0, 3.0), VP), a='rroof', b='rbus0'),
    dict(**{'in': 'rbus0'}, line=wl('A', (2.48, 0.4, 10.3), (2.48, 2.0, 10.3)), a='rgap', b='rbus'),          # the edge of its plate, carried up
    dict(**{'in': 'lbus'}, line=wl('DR', (-0.995, 0.5, 4.22), (-0.995, 1.8, 4.22)), a='lside0', b='ldoor'),   # the door, slid back
    dict(**{'in': 'lside0'}, line=wl('A', (-1.03, 1.135, 3.0), VP), a='lside', b='lstripe'),                  # the top of its stripes
    dict(**{'in': 'bottom'}, line=wl('D', (-0.15, 1.04, 0.54), (0.85, 1.04, 0.54)), a='tip0', b='cock0'),     # just past my handlebar
    dict(**{'in': 'tip0'}, line=wl('A', (0.25, 0.0, 3.0), VP), a='tipl', b='tipr'),                           # the centre line
    dict(**{'in': 'cock0'}, line=wl('D', (1.43, 0.40, -1.0), (1.43, 0.40, 2.0)), a='cock1', b='cockr'),       # the same sills, looking down
    dict(**{'in': 'cock1'}, line=wl('D', (-1.03, 0.40, -1.0), (-1.03, 0.40, 2.0)), a='cockl', b='cock'),
    dict(**{'in': ['lside', 'lstripe', 'ldoor', 'cock', 'cockl', 'tipl']}, round=(0.205, 0.665, 0.185, 0.115, 0.065, -6.0), a='lmirror'),
    dict(**{'in': ['rbus', 'rgap', 'cock', 'cockr', 'tipr']}, round=(0.80, 0.64, 0.175, 0.11, 0.06, 5.0), a='rmirror'),
]
B = dict(finish='brush', dir='h', lines=2)
PIECES = {   # painted in this order
    'tsky': dict(B, place='A0', finish='thin', dir='d', body=1.12),
    'tun': dict(B, place='U', dir='d'),
    'tr': dict(B, place='A1', dir='v'),
    'under': dict(B, place='L'),
    'underr': dict(B, place='A', dir='v'),
    'lroof': dict(B, place='L'),
    'rroof': dict(B, place='A', dir='v'),
    'lside': dict(B, place='A'),
    'lstripe': dict(B, place='A2'),
    'ldoor': dict(B, place='DR', dir='v'),
    'rgap': dict(B, place='L', dir='v'),
    'rbus': dict(B, place='A', finish='spray', over=dict(col='yel', into=['rroof'], reach=18, strength=0.8)),
    'tipl': dict(B, place='L3', dir='v'),
    'tipr': dict(finish='bare', place=None),
    'cockl': dict(B, place='A2', dir='d'),
    'cockr': dict(B, place='L', dir='d'),
    'cock': dict(B, place='D'),
    'lmirror': dict(B, place='LM'),
    'rmirror': dict(B, place='RM'),
}
LOCK = {'nw|se': 13.0, 'lside|lstripe': 6.0, 'tun|tr': 7.0, 'under|underr': 7.0}
LOCK_W = 10.0
NOLINE = [(ROAD, LANE), (ROAD, REDR), (DYEL, DBLK), (KERB, KERBW)]
LETTERS = [
    dict(text="NO CONDITION IS PERMANENT", place='A', pieces=['lside'], on=[DYEL], col='red', shadow='blk', off=(4, 4),
         quad3d=[(-1.025, 1.385, 0.2), (-1.025, 1.385, 4.55), (-1.025, 1.175, 4.55), (-1.025, 1.175, 0.2)]),
    dict(text="GOD'S TIME", place='A', pieces=['rbus'], on=[DYEL], col='blk', shadow='wht', off=(3, 3),
         quad3d=[(2.02, 1.375, 10.27), (3.95, 1.375, 10.27), (3.95, 1.175, 10.27), (2.02, 1.175, 10.27)]),
    dict(text="KJA 6", place='A', pieces=['rbus'], on=[PLATE], col='blk', font=FONT_N,
         quad3d=[(2.52, 0.69, 10.27), (3.25, 0.69, 10.27), (3.25, 0.55, 10.27), (2.52, 0.55, 10.27)]),
]


def _wires():
    out = []
    def cat(a, b, sag, n=60):
        a = np.array(a, float); b = np.array(b, float)
        t = np.linspace(0, 1, n)
        p = a[None, :] + (b - a)[None, :] * t[:, None]
        p[:, 1] -= sag * 4 * t * (1 - t)
        return p
    U = [(7.7, 7.9, -2.0), (-7.7, 7.9, 9.0), (7.7, 7.9, 17.0), (-7.7, 7.9, -13.0)]
    for off in (-1.1, 0.0, 1.1):
        out.append(cat((U[3][0], U[3][1], U[3][2] + off), (U[0][0], U[0][1], U[0][2] + off), 1.0))
        out.append(cat((U[0][0], U[0][1], U[0][2] + off), (U[1][0], U[1][1], U[1][2] + off), 1.2))
        out.append(cat((U[1][0], U[1][1], U[1][2] + off), (U[2][0], U[2][1], U[2][2] + off), 0.9))
    out.append(cat((7.7, 7.1, -2.0), (7.7, 7.1, 17.0), 0.8)); out.append(cat((-7.7, 7.1, -13.0), (-7.7, 7.1, 9.0), 0.8))
    out.append(cat((7.7, 7.1, 17.0), (9.5, 9.0, 12.0), 0.3)); out.append(cat((-7.7, 7.1, 9.0), (-9.5, 11.5, 4.0), 0.3))
    out.append(cat((7.7, 7.9, 17.0), (-6.95, 9.0, 14.5), 0.7)); out.append(cat((-7.7, 7.9, 9.0), (6.95, 9.0, 6.0), 1.1))
    out.append(cat((-9.5, 8.5, 6.0), (7.7, 7.9, 17.0), 1.3)); out.append(cat((-7.7, 7.9, 9.0), (9.5, 9.0, 13.0), 1.0))
    out.append(cat((-7.7, 7.1, 9.3), (7.7, 7.1, 17.3), 1.5)); out.append(cat((-7.7, 6.9, 8.7), (7.7, 6.9, 16.7), 1.7))
    out.append(cat((-9.5, 10.0, 12.0), (9.5, 8.0, 14.5), 1.2)); out.append(cat((-6.95, 8.8, 14.5), (7.7, 7.5, 17.0), 0.8))
    out.append(cat((-7.7, 7.9, 9.0), (-7.7, 7.6, 19.9), 0.7)); out.append(cat((7.7, 7.9, 17.0), (7.7, 7.6, 19.9), 0.3))
    out.append(cat((-7.7, 7.1, 9.0), (-9.5, 7.6, 19.9), 0.6)); out.append(cat((-9.5, 6.5, 13.5), (9.5, 6.0, 4.0), 1.4))
    for (t, cx, cz, dr, fl) in VEH:                       # spokes
        if t != 2: continue
        for wz in (cz - 0.66, cz + 0.66):
            for k in range(9):
                a = k * math.pi / 9
                out.append(np.array([(cx, 0.31 + 0.25 * math.sin(a) * u, wz + 0.25 * math.cos(a) * u) for u in np.linspace(-1, 1, 14)]))
    return out


THIN3D = _wires()
STALKS = [
    dict(pts=[(0.275, 0.735), (0.300, 0.775), (0.318, 0.812)], w=5.0),
    dict(pts=[(0.715, 0.708), (0.70, 0.76), (0.672, 0.815)], w=5.0),
]
RUNS = [(0.836, 0.527, 170, 3.2, 'red')]
KNOTS = [(0.60, 0.70, 22), (0.30, 0.08, 20)]
HOLES = [(0.04, 0.028, 8.5), (0.96, 0.028, 8.5), (0.04, 0.972, 8.5), (0.96, 0.972, 8.5), (0.555, 0.63, 7.0)]
GHOST = None
SIGNED = dict(at=(0.855, 0.95), h=0.040, col='wht')


# =====================================================================
#  ENGINE 6: putting it together
# =====================================================================
def render_view_lr(v, h=300, ar=1.6):
    w = int(h * ar)
    yy, xx = np.mgrid[0:h, 0:w]
    U = ((xx + 0.5 - w / 2) / (h / 2)).ravel().astype(np.float64); V = (-(yy + 0.5 - h / 2) / (h / 2)).ravel().astype(np.float64)
    T, M, TN = view_trace(v, U, V)
    return T.reshape(h, w), M.reshape(h, w), TN.reshape(h, w)


def contact_sheet(path, h=420):
    tab = scheme_table(SCHEME)
    ims = []
    for name, v in VIEWS.items():
        T, M, TN = render_view_lr(v, h)
        im = Image.fromarray((CRGB[tab[M, TN]] * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for (a, b, st) in find_lines(M, TN, T)[:12]:
            w = M.shape[1]
            d.line([(a[0] * h / 2 + w / 2, -a[1] * h / 2 + h / 2), (b[0] * h / 2 + w / 2, -b[1] * h / 2 + h / 2)], fill=(255, 0, 255), width=1)
        d.text((6, 4), name, fill=(255, 0, 255))
        ims.append(im)
    w = ims[0].size[0]
    sheet = Image.new('RGB', (w * 3, h * ((len(ims) + 2) // 3)), (40, 40, 40))
    for i, im in enumerate(ims): sheet.paste(im, ((i % 3) * w, (i // 3) * h))
    sheet.save(path)


def bbox(m, pad, H, W):
    r = np.where(m.any(axis=1))[0]; c = np.where(m.any(axis=0))[0]
    if len(r) == 0: return None
    return (slice(max(r[0] - pad, 0), min(r[-1] + pad + 1, H)), slice(max(c[0] - pad, 0), min(c[-1] + pad + 1, W)))


def make(scale, out, mode):
    import time
    t0 = time.time()
    W = int(round(PANEL[0] * scale)); H = int(round(PANEL[1] * scale)); S = scale
    rng = np.random.default_rng(SEED)
    print(TITLE, '  %dx%d  waybill %s' % (W, H, WAYBILL))
    # ---- 1. the views, small, and the straight lines in them
    VL = {}
    for name, v in VIEWS.items():
        T, M, TN = render_view_lr(v, 300)
        VL[name] = dict(lines=find_lines(M, TN, T))
    # ---- 2. the cut
    lab, names = build_pieces(W, H, CUTS, VL, PLACES)
    for n in names: assert n in PIECES, 'piece %s has no view' % n
    print('   %d pieces' % len(names))
    # ---- 3. each piece gets its view, traced exactly where it falls
    Mg = np.zeros((H, W), np.uint8); Tg = np.full((H, W), 1e9, np.float32)
    ci = {}
    mg = int(14 * S) + 3
    for p, pl in PLACES.items():
        idsp = [names[n] for n in names if PIECES[n].get('place') == p]
        if not idsp: continue
        mask = np.isin(lab, idsp)
        dil = ndi.maximum_filter(mask, size=2 * mg + 1)
        iy, ix = np.nonzero(dil)
        U, V = px_to_uv(pl, ix + 0.5, iy + 0.5, W, H)
        T, M, TN = view_trace(VIEWS[pl['view']], U.astype(np.float64), V.astype(np.float64))
        tab = scheme_table(SCHEME, pl.get('over'))
        c = np.zeros((H, W), np.uint8); c[iy, ix] = tab[M, TN]; ci[p] = c
        own = mask[iy, ix]
        Mg[iy[own], ix[own]] = M[own]; Tg[iy[own], ix[own]] = T[own]
        print('   traced %-3s %8d rays  %.0fs' % (p, len(ix), time.time() - t0))
    yy, xx = np.mgrid[0:H, 0:W]
    if mode == 'layout':
        rgb = np.zeros((H, W, 3), np.float32) + np.array([0.80, 0.66, 0.45], np.float32)
        for n, i in names.items():
            if PIECES[n].get('place'):
                m = lab == i; rgb[m] = CRGB[ci[PIECES[n]['place']][m]]
        e = np.zeros((H, W), bool); e[:, 1:] |= lab[:, 1:] != lab[:, :-1]; e[1:, :] |= lab[1:, :] != lab[:-1, :]
        rgb[ndi.binary_dilation(e, iterations=max(int(3 * S), 1))] = 0.05
        im = Image.fromarray((rgb * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
        for n, i in names.items():
            if not (lab == i).any(): print('   EMPTY', n); continue
            cy, cx = ndi.center_of_mass(lab == i)
            d.text((cx - 12, cy - 5), n, fill=(255, 0, 255))
        im.save(out); return
    # ---- 4. the ground
    ghost = None
    if GHOST:
        gm = np.zeros((H, W), np.float32)
        for g in GHOST['items']:
            tm = text_mask(g['text'], font=FONT_N, px=int(g['h'] * H))
            gy = int(g['at'][1] * H); gx = int(g['at'][0] * W)
            hh = min(tm.shape[0], H - gy); ww = min(tm.shape[1], W - gx)
            if hh > 0 and ww > 0: gm[gy:gy + hh, gx:gx + ww] = np.maximum(gm[gy:gy + hh, gx:gx + ww], tm[:hh, :ww])
        if 'band' in GHOST:
            y0, y1 = GHOST['band']; gm[int(y0 * H):int(y1 * H), :] = np.maximum(gm[int(y0 * H):int(y1 * H), :], 0.55)
        ghost = (gm, GHOST['col'])
    ply, hgt, lw = plywood(H, W, S, rng, KNOTS, HOLES, ghost)
    pan = Panel(W, H, S, rng, ply, hgt, lw)
    print('   ply down  %.0fs' % (time.time() - t0))
    de = np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy)).astype(np.float32)
    limit = sstep(0.0, 1.6, de - (2.0 * S + 9 * S * sstep(0.60, 0.90, fbm(H, W, 90 * S, 90 * S, rng, 2))))
    bn = vnoise(H, W, 2.0 * S, 120 * S, rng)
    sgn = np.where(vnoise(H, W, 300 * S, 300 * S, rng) > 0.5, 1, -1)
    sh = np.rint(sgn * 8 * S * np.clip((bn - 0.76) / 0.24, 0, 1)).astype(np.int64)
    labb = lab[yy, np.clip(xx + sh, 0, W - 1)]                    # paint creeps under tape along the grain
    wf = (vnoise(H, W, 70 * S, 70 * S, rng) - 0.5) * 2
    wf2 = (vnoise(H, W, 38 * S, 38 * S, rng) - 0.5) * 2
    # ---- 5. flat fills, piece by piece, each taped
    LV = np.zeros((H, W), np.uint8)
    for n, sp in PIECES.items():
        if n not in names: continue
        i = names[n]
        LV[lab == i] = sp.get('lines', 2) if sp.get('place') else 0
        if not sp.get('place') or sp['finish'] == 'bare': continue
        m = labb == i
        sl = bbox(m, 4, H, W)
        if sl is None: continue
        mc = m[sl]
        cc = ci[sp['place']][sl]
        mc = mc & (cc > 0)
        e_in = ndi.distance_transform_edt(mc).astype(np.float32)
        ma = ndi.gaussian_filter(mc.astype(np.float32), 0.55)
        rgb = CRGB[cc]
        wgt = ndi.gaussian_filter((cc > 0).astype(np.float32), 0.6) + 1e-6
        rgb = np.stack([ndi.gaussian_filter(rgb[..., k], 0.6) / wgt for k in range(3)], axis=-1)
        hide = CHIDE[cc]
        if sp.get('only'): ma = ma * np.isin(cc, [CI[k] for k in sp['only']])
        if sp.get('skip'): ma = ma * ~np.isin(cc, [CI[k] for k in sp['skip']])
        pan.fill(sl, ma * limit[sl], rgb, hide, sp['finish'], sp.get('dir', 'h'), e_in, sp.get('body', 0.80))
    print('   fills  %.0fs' % (time.time() - t0))
    # ---- 6. spray skirts and runs
    for n, sp in PIECES.items():
        if n in names and sp.get('over'):
            o = sp['over']; i = names[n]
            region = (lab == i) & (ci[sp['place']] == CI[o['col']])
            into = np.isin(lab, [names[k] for k in o['into']])
            pan.overspray(region, ~into, C[o['col']], o.get('reach', 20), o.get('strength', 0.55))
    for (fx, fy, ln, wd, cn) in RUNS:
        pan.run(fx * W, fy * H, ln, wd, C[cn])
    # ---- 7. enamel line.  thin and scratchy first, then the drawing, then the lock
    polys = []
    for n, sp in PIECES.items():
        if n not in names or not sp.get('place') or not THIN3D: continue
        i = names[n]; pl = PLACES[sp['place']]; v = VIEWS[pl['view']]
        for poly in THIN3D:
            X, Y, Dp = project(pl, v, poly, W, H)
            cur = []
            for x, y, dp in zip(X, Y, Dp):
                ok = np.isfinite(x) and 0 <= x < W and 0 <= y < H
                if ok:
                    ok = lab[int(y), int(x)] == i and Tg[int(y), int(x)] > dp - 0.35
                if ok: cur.append((x, y))
                else:
                    if len(cur) > 1: polys.append(cur)
                    cur = []
            if len(cur) > 1: polys.append(cur)
    scr = vnoise(H, W, 3.0 * S, 9.0 * S, rng)
    if polys:
        a = draw_polys(W, H, polys, 2.6 * S)
        pan.enamel(a * (0.2 + 0.8 * sstep(0.10, 0.30, scr - 0.25 * lw)) * limit, thick=0.5)
    for st in STALKS:
        a = draw_polys(W, H, [[(x * W, y * H) for x, y in st['pts']]], st['w'] * S)
        a = a * (0.25 + 0.75 * sstep(0.02, 0.16, scr - 0.15 * lw))
        pan.enamel(a * limit, thick=0.7)
    NL = np.zeros((NMAT, NMAT), bool)
    for a_, b_ in NOLINE: NL[a_, b_] = NL[b_, a_] = True
    lt = np.log(np.minimum(Tg, 4000.0))
    near = np.zeros((H, W), bool); mid = np.zeros((H, W), bool)
    for ax in (0, 1):
        sa = (slice(None), slice(0, W - 1)) if ax == 1 else (slice(0, H - 1), slice(None))
        sb = (slice(None), slice(1, W)) if ax == 1 else (slice(1, H), slice(None))
        same = lab[sa] == lab[sb]
        tmin = np.minimum(Tg[sa], Tg[sb])
        dT = np.abs(lt[sa] - lt[sb]) > 0.10
        dm = (Mg[sa] != Mg[sb]) & ~NL[Mg[sa], Mg[sb]] & (LV[sa] >= 2)
        e = same & (LV[sa] >= 1) & (dT | dm) & (tmin < LINE_FAR)
        near[sa] |= e & (tmin < LINE_NEAR); mid[sa] |= e & (tmin >= LINE_NEAR)
    a = np.zeros((H, W), np.float32)
    if near.any(): a = np.maximum(a, line_alpha(ndi.distance_transform_edt(~near).astype(np.float32) - 0.5, LINE_W[0], S, wf2, 0.30))
    if mid.any(): a = np.maximum(a, line_alpha(ndi.distance_transform_edt(~mid).astype(np.float32) - 0.5, LINE_W[1], S, wf2, 0.30))
    pan.enamel(a * limit, thick=0.8)
    # the lock: thick line along the cuts
    wmap = np.zeros((H, W), np.float32)
    inv = {i: n for n, i in names.items()}
    nid = max(names.values()) + 1
    LW = np.full((nid, nid), LOCK_W, np.float32)
    for k, wv in LOCK.items():
        a_, b_ = k.split('|')
        if a_ in names and b_ in names: LW[names[a_], names[b_]] = LW[names[b_], names[a_]] = wv
    for ax in (0, 1):
        sa = (slice(None), slice(0, W - 1)) if ax == 1 else (slice(0, H - 1), slice(None))
        sb = (slice(None), slice(1, W)) if ax == 1 else (slice(1, H), slice(None))
        df = lab[sa] != lab[sb]
        wv = np.where(df, LW[lab[sa], lab[sb]], 0)
        wmap[sa] = np.maximum(wmap[sa], wv)
    Bm = wmap > 0
    dist, ind = ndi.distance_transform_edt(~Bm, return_indices=True)
    wn = wmap[ind[0], ind[1]]
    dist = dist.astype(np.float32) - 0.5 + 0.9 * S * wf2
    a = sstep(0.75, -0.75, dist - 0.5 * wn * S * (1 + 0.20 * wf))
    pan.enamel(a * limit, thick=1.5)
    print('   line  %.0fs' % (time.time() - t0))
    # ---- 8. letters: sign-painter's block capitals with a drop shadow, always cut off
    for L in LETTERS:
        tm = text_mask(L['text'], font=L.get('font', FONT_B), px=int(max(60, min(400, 260 * S))), track=L.get('track', 0.05))
        if 'quad3d' in L:
            pl = PLACES[L['place']]
            X, Y, _ = project(pl, VIEWS[pl['view']], np.array(L['quad3d']), W, H)
            quad = np.stack([X, Y], axis=1)
        else:
            hpx = L['h'] * H; wpx = hpx * tm.shape[1] / tm.shape[0] * L.get('stretch', 1.0)
            a_ = math.radians(L.get('rot', 0.0)); shr = L.get('shear', 0.0)
            ex = np.array([math.cos(a_), -math.sin(a_)]); ey = np.array([math.sin(a_) + shr, math.cos(a_)])
            o = np.array([L['at'][0] * W, L['at'][1] * H])
            quad = np.array([o, o + ex * wpx, o + ex * wpx + ey * hpx, o + ey * hpx])
        wm, bb = warp_quad(tm, quad, W, H)
        if wm is None: continue
        y0, y1, x0, x1 = bb
        wm = sstep(0.40, 0.60, ndi.gaussian_filter(wm, 0.7 * S) + 0.07 * wf2[y0:y1, x0:x1])
        clip = np.isin(lab[y0:y1, x0:x1], [names[k] for k in L['pieces'] if k in names])
        clip = ndi.gaussian_filter(clip.astype(np.float32), 0.6)
        if L.get('on'): clip = clip * np.isin(Mg[y0:y1, x0:x1], L['on'])
        full = np.zeros((H, W), np.float32)
        if L.get('shadow'):
            dx, dy = [int(round(v * S)) for v in L.get('off', (5, 5))]
            sm = np.roll(np.roll(wm, dy, axis=0), dx, axis=1)
            full[y0:y1, x0:x1] = sm * clip
            pan.enamel(full * limit, C[L['shadow']], thick=0.7)
            full[:] = 0
        full[y0:y1, x0:x1] = wm * clip
        pan.enamel(full * limit, C[L['col']], thick=0.9)
    # ---- 9. the plate
    st = stencil_417(S * SIGNED['h'] / 0.070, rng)
    sh_, sw_ = st.shape
    y0 = int(SIGNED['at'][1] * H - sh_ / 2); x0 = int(SIGNED['at'][0] * W - sw_ / 2)
    full = np.zeros((H, W), np.float32); full[y0:y0 + sh_, x0:x0 + sw_] = st[:min(sh_, H - y0), :min(sw_, W - x0)]
    soft = np.clip(ndi.gaussian_filter(full, 1.5 * S) * 1.12, 0, 1)
    drops = (pan.speck < 0.5).astype(np.float32)
    halo = ndi.gaussian_filter(full, 9 * S) * (1 - soft) * 0.55 * (0.3 + 0.7 * drops)
    card = np.zeros((H, W), np.float32); pd = int(26 * S)
    card[max(y0 - pd, 0):y0 + sh_ + pd, max(x0 - pd, 0):x0 + sw_ + pd] = 1
    fog = (ndi.gaussian_filter(card, 14 * S) * (1 - card)) * 0.22 * (0.4 + 0.6 * (pan.speck > 0.5))
    a = np.clip(soft * (0.90 + 0.10 * pan.speck2.clip(-1, 1)) + halo + fog, 0, 1)
    pan.lay((a * limit).astype(np.float32), C[SIGNED['col']], thick=0.25, gloss=0.35, level=0.1)
    # ---- 10. light
    img = pan.finish()
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
    print('   done  %.0fs  ->' % (time.time() - t0), out)


LINE_FAR, LINE_NEAR, LINE_W = 110.0, 16.0, (5.0, 2.6)


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    out = sys.argv[2] if len(sys.argv) > 2 else __file__.replace('.py', '.png')
    mode = sys.argv[3] if len(sys.argv) > 3 else 'full'
    if mode == 'views': contact_sheet(out)
    else: make(scale, out, mode)


if __name__ == '__main__':
    main()
