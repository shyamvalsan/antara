"""
Vav, the seventh storey down, noon in May

Ruhi Vaghela.

Survey.  A corridor vav of the roofed kind, running east to west, entered from the east.  Clear width
between the side walls 5.80 m.  Storeys of 3.40 m.  Flights of twenty risers, 0.17 m by 0.40 m, with a
landing of 2.60 m under each pavilion.  Each pavilion is two rows of four supports (two free pillars
0.42 m square, two wall pilasters) carrying beams at every storey; the side bays are floored, the middle
bay is left open from the stair to the roof and crossed only by the tie beams.  The lens stands on the
tie beam of the fourth pavilion, third storey down, eye 8.60 m below the roof soffit.  Ahead: the fifth
pavilion, the sixth, then the last flight into the tank.  Tank floor 23.80 m below the soffit, 24.30 m
below the ground; water standing 0.80 m deep.  The roof is cut by one slot, 1.50 m by 24.4 m, over the
north half of the nave.  The sun is 10 degrees off the zenith, to the north-west of it, about forty
minutes after noon.

Usage:  python vav_the_seventh_storey_down_noon_in_may.py [scale=1.0] [out.png] [draft|final]
        scale 1.0 is 2700 x 3600.

Built as signed-distance solids in metres and lit by calculation:
  sun     - one hard directional light through the slot
  sky     - the slot as a white area light
  bounce  - what the sun lights becomes many small lamps, each with its shadow (first bounce);
            the later bounces are gathered into a lattice of irradiance probes
  dust    - marched through the shaft, with the shadows of the beams in it
  water   - a plane: mirror by Fresnel, green body below, the steps seen through it
"""
import sys, math, time, warnings
import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaWarning
warnings.simplefilter('ignore', NumbaWarning)
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

SEED = 19850512

# ------------------------------------------------------------------ the survey (metres)
WX = 3.30                       # half the clear width, wall to wall
S = 3.40                        # storey
RI, TR = 0.17, 0.40             # riser, tread
FL, LL = 8.0, 2.6               # flight, landing
PZ = FL + LL                    # one storey of plan
Z0 = 0.20                       # first riser of the flight under the lens
YA = -4 * S                     # the landing under the lens (fourth storey)
YBOT = -7 * S                   # tank floor
ZEND = 38.0                     # the well wall
ZBACK = -8.0
ROOF = 0.5                      # roof slab thickness; ground is at +0.5
WY = -23.0                      # water
SLOPE = S / PZ
CSL = 1.0 / math.sqrt(1.0 + SLOPE * SLOPE)
AR = 0.012                      # the nosings are worn round
PX = 2.20                       # pillar axis from the centre line
PIN = PX - 0.21                 # the nave face of the galleries
GP = 2.12                       # pillar to pillar along the well
ZG0 = Z0 - 1.3 + 1.06           # a row of pillars
BY, BZ = -3 * S, ZG0 + 4 * GP   # where the pigeons sit
SX0, SX1, SZ0, SZ1 = 0.50, 1.90, 11.4, 37.4          # the slot
SXC, SXH, SZC, SZH = (SX0 + SX1) * 0.5, (SX1 - SX0) * 0.5, (SZ0 + SZ1) * 0.5, (SZ1 - SZ0) * 0.5
_sl = math.sqrt(0.13 ** 2 + 1.0 + 0.03 ** 2)
SUNX, SUNY, SUNZ = 0.13 / _sl, 1.0 / _sl, -0.03 / _sl   # towards the sun
E0R, E0G, E0B = 1.00, 0.93, 0.80                      # sun, on a surface square to it
LSR, LSG, LSB = 0.080, 0.094, 0.112                   # radiance of the May sky: white, a little blue
CAMX, CAMY, CAMZ = -0.25, -8.60, 0.15
FOC = 2250.0                                          # pixels per unit tangent at scale 1
VTOP = 0.60
# the woman
XW, YW, ZW = -0.575, YA - 2 * S - RI * 12, Z0 + 2 * PZ + TR * 11 + 0.15
# air
FOG = 0.0035                    # scattering per metre
BEAM = 0.0085                    # the same dust, seen against the sun
# water
SIGR, SIGG, SIGB = 1.9, 0.75, 1.35

M_STONE, M_VEIL, M_SKIRT, M_SKIN, M_BRASS, M_BIRD, M_WATER = 1, 2, 3, 4, 5, 6, 20
NJ = dict(cache=True, fastmath=True, error_model='numpy')


# ------------------------------------------------------------------ small tools
@njit(**NJ)
def hash3(i, j, k):
    n = (i * 374761393 + j * 668265263 + k * 1274126177) & 0x7fffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0x7fffffff
    return ((n ^ (n >> 16)) & 0xffffff) / 16777216.0

@njit(**NJ)
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

@njit(**NJ)
def fbm3(x, y, z, o):
    s = 0.0; a = 0.5; t = 0.0
    for i in range(o):
        s += a * noise3(x, y, z); t += a; a *= 0.5
        x = x * 2.03 + 11.3; y = y * 2.03 + 5.7; z = z * 2.03 + 3.1
    return s / t

@njit(**NJ)
def sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0: t = 0.0
    if t > 1: t = 1.0
    return t * t * (3 - 2 * t)

@njit(**NJ)
def sd_box(x, y, z, bx, by, bz):
    qx = abs(x) - bx; qy = abs(y) - by; qz = abs(z) - bz
    ox = max(qx, 0.0); oy = max(qy, 0.0); oz = max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0)

@njit(**NJ)
def sd_box2(a, b, ha, hb):
    qa = abs(a) - ha; qb = abs(b) - hb
    oa = max(qa, 0.0); ob = max(qb, 0.0)
    return math.sqrt(oa * oa + ob * ob) + min(max(qa, qb), 0.0)

@njit(**NJ)
def quad(a, b):
    oa = max(a, 0.0); ob = max(b, 0.0)
    return math.sqrt(oa * oa + ob * ob) + min(max(a, b), 0.0)

@njit(**NJ)
def sd_cap(x, y, z, ax, ay, az, bx, by, bz, r):
    px = x - ax; py = y - ay; pz = z - az
    ex = bx - ax; ey = by - ay; ez = bz - az
    h = (px * ex + py * ey + pz * ez) / (ex * ex + ey * ey + ez * ez + 1e-12)
    if h < 0: h = 0.0
    if h > 1: h = 1.0
    px -= ex * h; py -= ey * h; pz -= ez * h
    return math.sqrt(px * px + py * py + pz * pz) - r

@njit(**NJ)
def smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25

@njit(**NJ)
def lcg(s):
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return s, ((s >> 8) & 0xffffff) / 16777216.0


# ------------------------------------------------------------------ the stair
@njit(**NJ)
def stair_top(z):
    u = (z - Z0) / PZ; k = math.floor(u); ul = z - Z0 - PZ * k
    if k > 2: return YBOT
    i = int(ul / TR)
    if i > 19: i = 19
    return YA - S * k - RI * (i + 1)

@njit(**NJ)
def sd_stair(x, y, z):
    la = (y - (YA - SLOPE * (z - Z0))) * CSL          # nothing of the stair is above the line of the landing edges
    lb = y - YBOT
    if lb < la: la = lb
    if la > 1.3: return la
    u = (z - Z0) / PZ; k = int(math.floor(u)); ul = z - Z0 - PZ * k
    i = int(ul / TR)
    if i > 19: i = 19
    n0 = 20 * k + i
    w = 0.013 * math.exp(-(x / 1.4) ** 2)             # eight hundred years of feet, in the middle
    d = 1e9
    for n in range(n0 - 3, n0 + 1):
        kk = n // 20; ii = n - 20 * kk
        if kk > 2:
            top = YBOT; ze = 1e6
        else:
            top = YA - S * kk - RI * (ii + 1)
            if ii < 19: ze = Z0 + PZ * kk + TR * (ii + 1)
            else: ze = Z0 + PZ * (kk + 1)
            if kk == 2 and ii == 19: ze = 1e6
        q = quad(z - ze + AR, y - (top - w) + AR) - AR
        if q < d: d = q
    return d


# ------------------------------------------------------------------ the woman
@njit(**NJ)
def sd_fig(lx, ly, lz):
    """feet at the origin, facing +z (away from the lens), her left is -x.  Returns distance, material."""
    h = ly
    # ghaghra
    cz = 0.10 * sstep(0.85, 0.15, h)
    tt = min(max((0.98 - h) / 0.9, 0.0), 1.0)
    rx = 0.155 + 0.095 * tt; rz = 0.115 + 0.10 * tt
    ang = math.atan2(lz - cz, lx)
    fold = 0.011 * math.sin(11.0 * ang + 5.0 * h) * sstep(0.97, 0.55, h)
    e = math.sqrt((lx / rx) ** 2 + ((lz - cz) / rz) ** 2)
    hb = 0.05 - 0.17 * sstep(0.05, 0.30, lz)
    dsk = max(max((e - 1.0) * min(rx, rz) - fold, hb - h), h - 1.0) * 0.8
    d = dsk; m = M_SKIRT
    # odhni: over the head, the shoulders and the back
    rx = 0.18 + 0.03 * sstep(0.85, 1.28, h) - 0.05 * sstep(1.25, 1.40, h); rz = 0.118
    e = math.sqrt((lx / rx) ** 2 + ((lz + 0.012) / rz) ** 2)
    zig = 0.015 * math.sin(lx * 31.0)
    dt = max(max((e - 1.0) * rz, 0.80 + zig - h), h - 1.40)
    ex = lx / 0.105; ey = (h - 1.455) / 0.118; ez = (lz - 0.01) / 0.112
    dh = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.105
    dv = smin(dt, dh, 0.09) * 0.85
    if dv < d: d = dv; m = M_VEIL
    # arms
    da = sd_cap(lx, ly, lz, -0.215, 1.25, 0.0, -0.315, 1.04, 0.05, 0.040)
    da = min(da, sd_cap(lx, ly, lz, 0.215, 1.25, 0.0, 0.25, 1.0, 0.03, 0.040))
    if da < d: d = da; m = M_VEIL
    da = sd_cap(lx, ly, lz, -0.315, 1.04, 0.05, -0.27, 1.0, 0.26, 0.031)
    da = min(da, sd_cap(lx, ly, lz, 0.25, 1.0, 0.03, 0.235, 0.77, 0.11, 0.031))
    da = min(da, math.sqrt((lx - 0.235) ** 2 + (ly - 0.73) ** 2 + (lz - 0.12) ** 2) - 0.038)
    if da < d: d = da; m = M_SKIN
    # feet
    ex = (lx + 0.075) / 0.045; ey = (ly - 0.03) / 0.03; ez = (lz + 0.02) / 0.11
    df = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.03
    ex = (lx - 0.08) / 0.045; ey = (ly + 0.14) / 0.03; ez = (lz - 0.36) / 0.11
    df = min(df, (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.03)
    if df < d: d = df; m = M_SKIN
    # the pot, on her left hip
    px = lx + 0.325; py = ly - 0.985; pz = lz - 0.10
    ex = px / 0.140; ey = py / 0.122; ez = pz / 0.140
    dp = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.122
    rr = math.sqrt(px * px + pz * pz)
    dp = min(dp, max(rr - 0.058, abs(py - 0.13) - 0.04))
    dp = min(dp, math.sqrt((rr - 0.078) ** 2 + (py - 0.172) ** 2) - 0.015)
    if dp < d: d = dp; m = M_BRASS
    return d, m


# ------------------------------------------------------------------ the well
@njit(**NJ)
def zone(z):
    """which storey of plan: each begins with its landing (2.6 m), then its flight (8 m)."""
    kz = math.floor((z - Z0 + LL) / PZ)
    if kz < -1: kz = -1.0
    if kz > 3: kz = 3.0
    return int(kz), z - (Z0 - LL) - PZ * kz

@njit(**NJ)
def sd_bird(lx, ly, lz, yaw):
    c = math.cos(yaw); s = math.sin(yaw)
    u = lx * c + lz * s; w = -lx * s + lz * c
    ex = u / 0.058; ey = (ly - 0.085) / 0.062; ez = w / 0.115
    d = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.058
    d = smin(d, math.sqrt(u * u + (ly - 0.108) ** 2 + (w - 0.06) ** 2) - 0.055, 0.03)
    d = smin(d, math.sqrt(u * u + (ly - 0.178) ** 2 + (w - 0.098) ** 2) - 0.033, 0.025)
    ex = u / 0.036; ey = (ly - 0.072) / 0.014; ez = (w + 0.15) / 0.085
    d = smin(d, (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.014, 0.03)
    return d

@njit(**NJ)
def scene(x, y, z):
    ax = abs(x)
    m = M_STONE
    jf = math.ceil(-y / S)
    if jf < 1: jf = 1.0
    if jf > 7: jf = 7.0
    j = int(jf); sy = y + S * jf                          # storey counted down from the roof; height above its floor
    kz, ul = zone(z)
    jl = min(4 + kz, 7)                                   # the lowest storey here
    inflight = ul > LL and kz <= 2
    nr = math.floor((z - ZG0) / GP + 0.5)
    zrow = ZG0 + GP * nr; zr = z - zrow; azr = abs(zr)    # the nearest row of pillars
    kr, ulr = zone(zrow)
    jlr = min(4 + kr, 7)
    tierow = ulr < LL and kr <= 2                         # a pavilion row: tie beams cross the nave here
    norow = zrow > ZEND - 0.6 or zrow < ZBACK + 0.6
    # ---- side walls, a niche in every bay
    dw = WX - ax
    d = dw
    if dw < 0.6:
        zb = z - ZG0 - GP * (math.floor((z - ZG0) / GP) + 0.5)
        if j <= jl:
            qn = sd_box(ax - WX, sy - 1.45, zb, 0.22, 0.72, 0.42)
            d = max(dw, -qn)
            q = sd_box(ax - (WX + 0.20), sy - 1.40, zb, 0.06, 0.52, 0.27) - 0.03
            q = min(q, sd_box(ax - (WX - 0.035), sy - 2.27, zb, 0.035, 0.06, 0.56))
            q = min(q, sd_box(ax - (WX - 0.025), sy - 2.39, zb, 0.025, 0.05, 0.36))
            q = min(q, sd_box(ax - (WX - 0.04), sy - 0.66, zb, 0.04, 0.05, 0.54))
            if q < d: d = q
    # ---- the well wall at the end
    de = ZEND - z
    if de < 0.8:
        qe = de
        xi = math.floor(x / 2.2 + 0.5)
        if xi < -1: xi = -1.0
        if xi > 1: xi = 1.0
        xn = x - 2.2 * xi
        portal = j >= 6 and xi == 0.0
        if not portal:
            qn = sd_box(xn, sy - 1.45, z - ZEND, 0.55, 0.76, 0.26)
            qe = max(qe, -qn)
            q = sd_box(xn, sy - 1.40, z - (ZEND + 0.25), 0.34, 0.58, 0.09) - 0.03
            q = min(q, sd_box(abs(xn) - 0.64, sy - 1.42, z - (ZEND - 0.04), 0.07, 0.86, 0.04))
            q = min(q, sd_box(xn, sy - 0.60, z - (ZEND - 0.07), 0.77, 0.06, 0.07))
            q = min(q, sd_box(xn, sy - 2.33, z - (ZEND - 0.07), 0.79, 0.08, 0.07))
            q = min(q, sd_box(xn, sy - 2.47, z - (ZEND - 0.04), 0.55, 0.06, 0.04))
            q = min(q, sd_box(xn, sy - 2.58, z - (ZEND - 0.025), 0.30, 0.05, 0.025))
            if q < qe: qe = q
        qp = sd_box(x, y + 20.9, z - (ZEND + 1.2), 0.95, 2.9, 1.4)                      # the way through to the shaft
        qe = max(qe, -qp)
        q = sd_box(ax - 1.13, y + 20.9, z - (ZEND - 0.10), 0.18, 2.9, 0.12)
        q = min(q, sd_box(x, y + 17.78, z - (ZEND - 0.14), 1.55, 0.22, 0.16))
        q = min(q, sd_box(x, y + 17.48, z - (ZEND - 0.10), 1.25, 0.08, 0.12))
        if q < qe: qe = q
        if j >= 2:
            q = sd_box2(z - (ZEND - 0.14), sy - (S - 0.10), 0.14, 0.10)
            xr = x - 0.55 * math.floor(x / 0.55 + 0.5)
            q = min(q, sd_box(xr, sy - (S - 0.30), z - (ZEND - 0.10), 0.07, 0.10, 0.10))
            if j == 7: q = max(q, 1.35 - ax)
            if q < qe: qe = q
        if j <= 6:
            q = sd_box2(z - (ZEND - 0.14), sy + 0.10, 0.14, 0.10)
            if j == 6: q = max(q, 1.35 - ax)
            if q < qe: qe = q
        if qe < d: d = qe
    else:
        bd = de - 0.36
        if bd < d: d = max(bd, 0.03)
    q = z - ZBACK
    if q < d: d = q
    # ---- the roof, its slot, a beam at every row
    q = max(max(-y, y - ROOF), -sd_box(x - SXC, y - 0.25, z - SZC, SXH, 1.0, SZH))
    if q < d: d = q
    if y > -1.0:
        if not norow:
            q = sd_box2(zr, y + 0.225, 0.20, 0.225)
            if q < d: d = q
    else:
        bd = -0.46 - y
        if bd < d: d = max(bd, 0.03)
    # ---- the stair
    q = sd_stair(x, y, z)
    if q < d: d = q
    # ---- where a new lowest storey begins, something may stand just past the line
    dzb = (Z0 - LL + PZ * (kz + 1)) - z
    if kz < 3 and dzb < d: d = max(dzb, 0.02)
    # ---- the tie beams across the nave
    if tierow and not norow:
        if azr < 0.5:
            if ax < PX + 0.1:
                if j < jlr:
                    q = max(sd_box2(zr, sy - (S - 0.225), 0.20, 0.225), ax - PX)
                    if q < d: d = q
                if j + 1 < jlr:
                    q = max(sd_box2(zr, sy + 0.225, 0.20, 0.225), ax - PX)
                    if q < d: d = q
        else:
            bd = azr - 0.22
            if bd < d: d = max(bd, 0.03)
    # ---- the galleries
    if ax > 1.40:
        qx = ax - PX
        t = PIN - ax
        q = 1e9
        if inflight:                                      # the platform beside the flight, level with the landing above it
            q = sd_box(ax - (PIN + 2.0), y - (-S * jl - 20.0), ul - (LL + 4.0), 2.0, 20.0, 4.0)
        if kz >= 0:
            q = min(q, sd_box(ax - (PIN + 2.0), y - (-S * (jl - 1) - 20.0), ul + 4.0, 2.0, 20.0, 4.0))
        if j <= jl:
            q = min(q, sd_box2(qx, sy - (S - 0.225), 0.21, 0.225))                         # beam along
            if j >= 2:
                q = min(q, sd_box2(sy - (S - 0.14), ax - (PIN + 1.0), 0.14, 1.0))          # the floor above
                hc = S - 0.10 - 0.38 * t                                                 # its eave
                q = min(q, max(max((abs(sy - hc) - 0.032) * 0.93, t - 0.42), -t - 0.2))
        if j < jl:
            q = min(q, sd_box2(sy + 0.14, ax - (PIN + 1.0), 0.14, 1.0))                    # the floor under foot
            q = min(q, sd_box2(qx, sy + 0.225, 0.21, 0.225))
            hc = -0.10 - 0.38 * t
            q = min(q, max(max((abs(sy - hc) - 0.032) * 0.93, t - 0.42), -t - 0.2))
            q = min(q, sd_box2(ax - (PIN + 0.07), sy - 0.30, 0.05, 0.30))                  # parapet
            q = min(q, sd_box2(ax - (PIN + 0.07), sy - 0.635, 0.085, 0.035))
        if j <= jlr and not norow:
            rr = math.sqrt(qx * qx + zr * zr)
            q2 = sd_box(qx, sy - 0.10, zr, 0.24, 0.10, 0.24)
            q2 = min(q2, sd_box(qx, sy - 0.25, zr, 0.215, 0.05, 0.215))
            q2 = min(q2, sd_box(qx, sy - 0.68, zr, 0.19, 0.38, 0.19))
            oc = max(sd_box2(qx, zr, 0.19, 0.19), (abs(qx) + abs(zr)) * 0.70711 - 0.205)
            q2 = min(q2, max(oc, abs(sy - 1.45) - 0.40))
            q2 = min(q2, sd_box(qx, sy - 1.87, zr, 0.20, 0.05, 0.20))
            q2 = min(q2, max(rr - 0.172, abs(sy - 2.06) - 0.15))
            cu = (sy - 2.29) / 0.09
            bul = 0.195
            if abs(cu) < 1.0: bul += 0.055 * math.sqrt(1.0 - cu * cu)
            q2 = min(q2, max(rr - bul, abs(sy - 2.29) - 0.09) * 0.8)
            q2 = min(q2, sd_box(qx, sy - 2.44, zr, 0.25, 0.06, 0.25))
            q2 = min(q2, sd_box(qx, sy - 2.72, zr, 0.20, 0.23, 0.20))
            hb = S - 0.45 - sy                                                           # brackets: quarter ellipses under the beams
            e = math.sqrt((azr / 0.75) ** 2 + (hb / 0.45) ** 2)
            q2 = min(q2, max(max((e - 1.0) * 0.45, abs(qx) - 0.15), -hb))
            e = math.sqrt((qx / 0.75) ** 2 + (hb / 0.45) ** 2)
            q2 = min(q2, max(max((e - 1.0) * 0.45, azr - 0.15), max(-hb, -qx)))
            if tierow and j < jlr:
                e = math.sqrt((qx / 0.75) ** 2 + (hb / 0.45) ** 2)
                q2 = min(q2, max(max((e - 1.0) * 0.45, azr - 0.15), max(-hb, qx)))
            else:
                e = math.sqrt((qx / 0.44) ** 2 + (hb / 0.30) ** 2)
                q2 = min(q2, max(max((e - 1.0) * 0.30, azr - 0.13), max(-hb, qx)))
            q2 = min(q2, max(sd_box2(zr, sy - (S - 0.225), 0.20, 0.225), -qx))             # beam across the gallery
            if j < jlr:
                q2 = min(q2, max(sd_box2(zr, sy + 0.225, 0.20, 0.225), -qx))
            q2 = min(q2, sd_box(ax - (WX - 0.08), sy - 1.7, zr, 0.08, 1.7, 0.18))          # wall pilaster
            if q2 < q: q = q2
        if q < d: d = q
    else:
        bd = 1.44 - ax
        if bd < d: d = max(bd, 0.03)
    # ---- time: nothing is flat or sharp any more
    if d < 0.05:
        d += 0.007 * (noise3(x * 5.3, y * 5.3, z * 5.3) - 0.5)
        c = noise3(x * 9.0 + 3.0, y * 9.0, z * 9.0 + 7.0)
        if c > 0.70: d += 0.010 * ((c - 0.70) / 0.30) ** 2
    # ---- the woman
    lx = x - XW; ly = y - YW; lz = z - ZW
    bd = sd_box(lx + 0.05, ly - 0.72, lz - 0.12, 0.50, 0.95, 0.45)
    if bd < 0.03:
        q, mf = sd_fig(lx, ly, lz)
        if q < d: d = q; m = mf
    elif bd < d:
        d = max(bd, 0.03)
    # ---- pigeons on the tie beam of the fifth pavilion
    bd = sd_box(x, y - (BY + 0.12), z - BZ, 1.6, 0.16, 0.25)
    if bd < 0.03:
        q = sd_bird(x - 0.92, y - BY, z - BZ - 0.02, 0.5)
        q = min(q, sd_bird(x - 1.21, y - BY, z - BZ + 0.03, -1.0))
        q = min(q, sd_bird(x + 1.18, y - BY, z - BZ, 2.3))
        if q < d: d = q; m = M_BIRD
    elif bd < d:
        d = max(bd, 0.03)
    return d, m


@njit(**NJ)
def march(ox, oy, oz, dx, dy, dz, tmax):
    """returns t, m.  m = 0: left through the slot to the sky.  m = -1: nothing."""
    t = 0.0
    for s in range(700):
        py = oy + dy * t
        if py > ROOF + 0.05: return t, 0
        d, m = scene(ox + dx * t, py, oz + dz * t)
        if d < 0.00025 * t + 0.0004:
            return t, m
        t += d * 0.85
        if t > tmax: break
    return t, -1

@njit(**NJ)
def occl(ox, oy, oz, dx, dy, dz, maxt):
    t = 0.0
    for s in range(400):
        py = oy + dy * t
        if py > ROOF + 0.05: return False
        d, m = scene(ox + dx * t, py, oz + dz * t)
        if d < 0.0015: return True
        t += d * 0.9
        if t >= maxt: return False
    return False

@njit(**NJ)
def normal(px, py, pz, t):
    e = 0.0003 * t + 0.0012
    nx = scene(px + e, py, pz)[0] - scene(px - e, py, pz)[0]
    ny = scene(px, py + e, pz)[0] - scene(px, py - e, pz)[0]
    nz = scene(px, py, pz + e)[0] - scene(px, py, pz - e)[0]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nl, ny / nl, nz / nl

@njit(**NJ)
def sun_vis(px, py, pz):
    """1 if the point sees the sun through the slot."""
    s = (0.25 - py) / SUNY
    x0 = px + SUNX * s; z0 = pz + SUNZ * s
    if x0 < SX0 - 0.06 or x0 > SX1 + 0.06 or z0 < SZ0 - 0.06 or z0 > SZ1 + 0.06: return 0.0
    if occl(px, py, pz, SUNX, SUNY, SUNZ, (ROOF + 0.1 - py) / SUNY): return 0.0
    return 1.0


# ------------------------------------------------------------------ stone
@njit(**NJ)
def stone_base(x, y, z):
    a = fbm3(x * 0.23 + 3.1, y * 0.31, z * 0.19 + 7.7, 3)
    b = fbm3(x * 0.9 + 11.0, y * 1.1 + 5.0, z * 0.8, 3)
    t = sstep(0.38, 0.62, a)
    r = 0.715 + (0.680 - 0.715) * t; g = 0.565 + (0.480 - 0.565) * t; bl = 0.385 + (0.400 - 0.385) * t   # honey to rose
    gk = min(sstep(0.50, 0.72, b) * 0.60 + 0.45 * sstep(-9.0, -1.0, y), 0.85)                           # and near the top, where the air is
    rs = sstep(-15.0, -22.0, y) * 0.3                                                                  # rose, lower down
    r += (0.66 - r) * rs; g += (0.45 - g) * rs; bl += (0.39 - bl) * rs
    r += (0.56 - r) * gk; g += (0.52 - g) * gk; bl += (0.46 - bl) * gk                                  # grey where it has weathered
    dm = sstep(-14.0, -23.5, y)                                                                         # damp, lower down
    r *= 1 - 0.26 * dm; g *= 1 - 0.23 * dm; bl *= 1 - 0.22 * dm
    return r, g, bl

@njit(**NJ)
def relief(kind, u, v, w):
    """height of the worked surface, metres."""
    er = 0.0045 * (fbm3(u * 5.0, v * 5.0, w, 3) - 0.5) + 0.0016 * (noise3(u * 31.0, v * 31.0, w + 3.0) - 0.5)
    worn = 0.72 + 0.28 * sstep(0.30, 0.62, fbm3(u * 1.7 + 4.0, v * 1.7, w + 9.0, 2))
    h = 0.0
    if kind == 1:                                         # coursed ashlar
        cv = v / 0.34; ci = math.floor(cv); fv = cv - ci
        cu = u / 0.92 + 0.5 * (ci - 2 * math.floor(ci / 2)) + 0.3 * hash3(int(ci), 7, int(w))
        cj = math.floor(cu); fu = cu - cj
        ej = min(min(fv, 1 - fv) * 0.34, min(fu, 1 - fu) * 0.92)
        h = 0.007 * (sstep(0.0, 0.010, ej) - 1.0) + (hash3(int(ci), int(cj), 11) - 0.5) * 0.005
        return er + h
    if kind == 2:                                         # pillar: u across the face from its middle, v above its floor
        if v < 1.04:                                      # square part: a framed panel, a lotus in it
            fr = max(abs(u) / 0.15, abs(v - 0.70) / 0.30)
            h = 0.009 * sstep(1.0, 0.93, fr) - 0.007 * sstep(0.86, 0.80, fr)
            r = math.sqrt(u * u + (v - 0.70) ** 2); th = math.atan2(v - 0.70, u)
            R = 0.105 * (0.72 + 0.28 * abs(math.cos(4.0 * th)))
            h += 0.011 * sstep(R, R - 0.018, r) - 0.005 * sstep(0.058, 0.050, r) + 0.008 * sstep(0.034, 0.024, r)
            for c in (0.44, 0.96):
                r2 = math.sqrt(u * u + (v - c) ** 2 * 2.2)
                h += 0.007 * sstep(0.07, 0.055, r2) * sstep(0.86, 0.80, fr)
        elif v < 1.84:                                    # eight-sided part: beads, then plain, then a chain
            if v < 1.14 or v > 1.74:
                fu = u / 0.045; fu = fu - math.floor(fu) - 0.5
                c = 1.09 if v < 1.4 else 1.79
                h = 0.008 * sstep(0.45, 0.25, math.sqrt(fu * fu + ((v - c) / 0.045) ** 2))
            elif v > 1.36 and v < 1.60:
                fu = u / 0.075; fu = fu - math.floor(fu) - 0.5; fv = (v - 1.48) / 0.11
                h = 0.010 * sstep(0.5, 0.36, abs(fu) + abs(fv) * 0.5) - 0.005 * sstep(0.22, 0.14, abs(fu) + abs(fv) * 0.5)
        elif v > 1.93 and v < 2.19:                       # round part: fluted
            h = 0.005 * math.cos(6.2831853 * u / 0.045)
        return er + h * worn
    if kind == 3:                                         # beam: fillets and a frieze of arches
        h = -0.008 * math.exp(-((v - 0.06) / 0.012) ** 2) - 0.008 * math.exp(-((v - 0.385) / 0.012) ** 2)
        fu = u / 0.20; fu = (fu - math.floor(fu) - 0.5) * 0.20
        b = v - 0.13
        r = math.sqrt(fu * fu + b * b * 0.55)
        if b > 0.0 and v < 0.35:
            h += 0.012 * sstep(0.088, 0.074, r) - 0.008 * sstep(0.050, 0.038, r)
        return er + h * worn
    if kind == 4:                                         # parapet: framed panels, a lozenge in each
        fu = u / 0.42; fu = (fu - math.floor(fu) - 0.5) * 0.42
        fb = v - 0.29
        fr = max(abs(fu) / 0.17, abs(fb) / 0.21)
        h = 0.010 * sstep(1.0, 0.92, fr) - 0.007 * sstep(0.82, 0.74, fr)
        h += 0.013 * sstep(1.0, 0.8, abs(fu) / 0.11 + abs(fb) / 0.15)
        return er + h * worn
    if kind == 5:                                         # ledge: dentils
        fu = u / 0.13; fu = fu - math.floor(fu)
        h = 0.012 * sstep(0.50, 0.58, fu) * sstep(1.0, 0.94, fu)
        return er + h * worn
    if kind == 6:                                         # what is left of a figure
        h = 0.035 * sstep(0.36, 0.66, fbm3(abs(u) * 4.6, v * 4.2, w, 3)) * math.exp(-(u / 0.33) ** 2)
        return er * 1.5 + h
    if kind == 7:                                         # tread
        cu = u / 1.15 + hash3(int(w), 3, 5); fu = cu - math.floor(cu)
        ej = min(fu, 1 - fu) * 1.15
        if w >= 1000.0:
            cv = v / 0.87; fv = cv - math.floor(cv)
            cu = u / 1.15 + 0.5 * (math.floor(cv) - 2 * math.floor(math.floor(cv) / 2)); fu = cu - math.floor(cu)
            ej = min(min(fu, 1 - fu) * 1.15, min(fv, 1 - fv) * 0.87)
        h = 0.005 * (sstep(0.0, 0.010, ej) - 1.0)
        return er * 1.3 + h
    return er


@njit(**NJ)
def material(px, py, pz, nx, ny, nz, m, fp):
    """albedo and the shading normal."""
    if m == M_VEIL:
        f = 0.9 + 0.2 * noise3(px * 40.0, py * 40.0, pz * 40.0)
        ar = 0.70 * f; ag = 0.045 * f; ab = 0.035 * f
        ly = py - YW
        if ly < 0.835 or abs(ly - 1.33) < 0.012: ar = 0.80; ag = 0.56; ab = 0.10                      # its border
        return ar, ag, ab, nx, ny, nz
    if m == M_SKIRT:
        ly = py - YW
        f = 0.9 + 0.2 * noise3(px * 40.0, py * 40.0, pz * 40.0)
        ar = 0.50 * f; ag = 0.035 * f; ab = 0.045 * f
        if ly < 0.13 and ly > 0.02 or (ly < -0.04): ar = 0.78; ag = 0.52; ab = 0.09
        return ar, ag, ab, nx, ny, nz
    if m == M_SKIN:
        ar = 0.34; ag = 0.20; ab = 0.125
        ly = py - YW
        if abs(ly - 0.995) < 0.02 or abs(ly - 0.82) < 0.02: ar = 0.80; ag = 0.76; ab = 0.66          # bangles
        return ar, ag, ab, nx, ny, nz
    if m == M_BRASS:
        return 0.80, 0.57, 0.20, nx, ny, nz
    if m == M_BIRD:
        f = 0.8 + 0.4 * noise3(px * 30.0, py * 30.0, pz * 30.0)
        ar = 0.30 * f; ag = 0.31 * f; ab = 0.35 * f
        if py - BY > 0.135: ar = 0.16; ag = 0.19; ab = 0.20
        return ar, ag, ab, nx, ny, nz
    ax = abs(px)
    jf = math.ceil(-py / S)
    if jf < 1: jf = 1.0
    if jf > 7: jf = 7.0
    sy = py + S * jf; j = int(jf)
    kz, ul = zone(pz)
    jl = min(4 + kz, 7)
    nr = math.floor((pz - ZG0) / GP + 0.5)
    zr = pz - (ZG0 + GP * nr); azr = abs(zr)
    st = stair_top(pz)
    kind = 0; u = 0.0; v = 0.0; w = 1.0; axis = 0
    onstair = py < st + 0.06 and ax < PIN + 0.02
    tux = 1.0; tuz = 0.0
    if ny > 0.7:
        axis = 1; u = px; v = pz
        if onstair:
            kind = 7; w = math.floor((pz - Z0) / TR) + 40.0
            if (pz - Z0) - PZ * math.floor((pz - Z0) / PZ) >= FL - 0.4 or pz > Z0 + 3 * PZ - LL: w = 1000.0 + kz
        elif ax > PIN - 0.45 and ax < PIN: kind = 7; u = pz; v = px; w = 3.0 + j
        else: w = 3.0
    elif ny < -0.7:
        axis = 1; u = px; v = pz; w = 5.0
    else:
        axis = 0
        hl = math.sqrt(nx * nx + nz * nz) + 1e-9
        tux = -nz / hl; tuz = nx / hl
        u = px * tux + pz * tuz; v = py
        w = 7.0 + math.floor(math.atan2(nz, nx) * 2.0 + 0.5)
        if ax > WX - 0.03 or pz > ZEND - 0.03:
            if ax > WX + 0.08 or (pz > ZEND + 0.08 and pz < ZEND + 0.5):
                kind = 6; v = sy; w += j * 3.0
                if ax > WX: u = pz - ZG0 - GP * (math.floor((pz - ZG0) / GP) + 0.5); w += nr
                else: u = px - 2.2 * math.floor(px / 2.2 + 0.5)
            else: kind = 1
        elif sy > 2.95 and j <= jl and (ax > PIN - 0.03 or azr < 0.26):
            kind = 3; v = sy - 2.95
        elif ax > PIN - 0.05 and ax < PX + 0.3 and azr < 0.30 and sy > 0.30 and sy < 2.22 and j <= jl and (sy > 0.69 or azr < 0.20 or j == jl):
            kind = 2; v = sy; w += 20.0 + nr
            qxs = px - PX if px > 0 else px + PX
            u = qxs * tux + zr * tuz
        elif sy < 0.68 and ax > PIN - 0.03 and ax < PIN + 0.2 and j < jl:
            kind = 4; v = sy
        elif ax > PIN - 0.05 and j > jl:
            kind = 1
        elif py < st + 0.3 and ax < PIN:
            kind = 7; w = math.floor((pz - Z0) / TR + 0.5) + 40.0
    # ---- relief
    e = max(0.003, fp * 0.8)
    h0 = relief(kind, u, v, w)
    hu = (relief(kind, u + e, v, w) - h0) / e
    hv = (relief(kind, u, v + e, w) - h0) / e
    fade = sstep(0.05, 0.012, fp)
    hu *= fade; hv *= fade
    bx = nx; by = ny; bz = nz
    if axis == 1: bx -= hu; bz -= hv
    else: bx -= hu * tux; bz -= hu * tuz; by -= hv
    bl_ = math.sqrt(bx * bx + by * by + bz * bz); bx /= bl_; by /= bl_; bz /= bl_
    # ---- colour
    ar, ag, ab = stone_base(px, py, pz)
    tone = 1.0 + 9.0 * min(max(h0, -0.012), 0.012)                       # dirt lies in what is cut
    if kind == 1 or kind == 7:                                         # each block is its own stone
        if kind == 1:
            cv = v / 0.34; ci = math.floor(cv)
            cu = u / 0.92 + 0.5 * (ci - 2 * math.floor(ci / 2)) + 0.3 * hash3(int(ci), 7, int(w))
            bi = int(ci) * 7 + int(math.floor(cu)) * 13
        else:
            bi = int(w) * 5 + int(math.floor(u / 1.15 + hash3(int(w), 3, 5))) * 11
            if w >= 1000.0: bi += int(math.floor(v / 0.87)) * 29 + int(math.floor(u / 1.15 + 0.5 * math.floor(v / 0.87))) * 3
        hb = hash3(bi, 17, int(w))
        tone *= 0.90 + 0.20 * hb
        hc = hash3(bi, 19, int(w)) - 0.5
        ar *= 1 + 0.06 * hc; ab *= 1 - 0.10 * hc
    tone *= 0.90 + 0.20 * fbm3(px * 2.3, py * 2.3, pz * 2.3, 3)
    if fp < 0.012:                                                      # grain of the sandstone
        gi = hash3(int(math.floor(px / 0.004)), int(math.floor(py / 0.004)), int(math.floor(pz / 0.004)))
        tone *= 1.0 + 0.13 * (gi - 0.5) * sstep(0.012, 0.004, fp)
    if abs(ny) < 0.7:
        # what has run down from every ledge
        run = sstep(0.50, 0.74, fbm3(u * 2.6 + 1.0, 0.37 + w, v * 0.22, 3))
        dtop = S - sy
        run *= (0.35 + 0.65 * math.exp(-dtop / 1.3)) * (0.5 + 0.5 * fbm3(u * 9.0, v * 1.3, w, 2))
        ar *= 1 - 0.42 * run; ag *= 1 - 0.40 * run; ab *= 1 - 0.36 * run
        crust = sstep(0.60, 0.80, fbm3(u * 1.1 + 5.0, v * 1.1, w + 2.0, 3))                # black crust, in patches
        ar *= 1 - 0.30 * crust; ag *= 1 - 0.29 * crust; ab *= 1 - 0.26 * crust
    elif ny > 0.7:
        dust = 0.5 + 0.5 * fbm3(px * 3.0, 2.0, pz * 3.0, 2)
        if not onstair:
            ar += (0.56 - ar) * 0.22 * dust; ag += (0.53 - ag) * 0.22 * dust; ab += (0.48 - ab) * 0.22 * dust
            tone *= 0.86
            ci = int(math.floor(px / 0.09)); cj = int(math.floor(pz / 0.09))            # pigeons
            near = math.exp(-((pz - BZ) / 0.5) ** 2) * math.exp(-((py - BY) / 0.3) ** 2)
            if hash3(ci, 91, cj) < 0.018 + 0.20 * near:
                ex = px / 0.09 - ci - 0.3 - 0.4 * hash3(ci, 93, cj); ez = pz / 0.09 - cj - 0.3 - 0.4 * hash3(ci, 95, cj)
                rr = ex * ex + ez * ez * 0.5 + 0.10 * (noise3(px * 90.0, 1.0, pz * 90.0) - 0.5)
                k = sstep(0.07, 0.03, rr) * 0.8
                ar += (0.74 - ar) * k; ag += (0.73 - ag) * k; ab += (0.68 - ab) * k
        else:
            path = math.exp(-(px / 1.3) ** 2)                                              # polished by feet
            tone *= 1.0 + 0.16 * path
            ul2 = pz - Z0 - PZ * math.floor((pz - Z0) / PZ)
            zt = ul2 / TR; fz = zt - math.floor(zt)
            if ul2 < FL and pz < Z0 + 3 * PZ - LL:
                grime = math.exp(-fz * TR / 0.035) * 0.5 * (0.5 + fbm3(px * 5.0, 1.0, pz * 5.0, 2))   # dirt at the back of the tread
                tone *= 1.0 - min(grime, 0.6)
                tone *= 1.0 + 0.10 * math.exp(-(1.0 - fz) * TR / 0.02)                          # the nosing rubbed pale
            # someone has come up from the water not long ago, carrying it
            if pz < 27.6 and py > WY:
                xc = 0.26 + 0.20 * math.sin(pz * 0.31 + 1.0)
                xc += (-0.42 - xc) * sstep(22.5, 26.8, pz)
                fi = math.floor(zt); side = 1.0 if (int(fi) + int(kz)) % 2 == 0 else -1.0
                fx = px - (xc + side * 0.085); fzz = (zt - fi - 0.5) * TR
                e1 = ((fx + side * 0.006) / 0.045) ** 2 + ((fzz + 0.040) / 0.062) ** 2
                e2 = (fx / 0.033) ** 2 + ((fzz - 0.085) / 0.040) ** 2
                e3 = ((fx - side * 0.016) / 0.016) ** 2 + ((fzz - 0.025) / 0.05) ** 2
                wet = max(max(sstep(1.0, 0.7, e1), sstep(1.0, 0.7, e2)), sstep(1.0, 0.6, e3))
                for q in range(5):
                    tx_ = fx + side * (0.036 - 0.018 * q); tz_ = fzz + 0.118 - 0.006 * q * (q - 1.0) * 0.5
                    rq = 0.013 - 0.0012 * q
                    wet = max(wet, sstep(rq * rq, rq * rq * 0.5, tx_ * tx_ + tz_ * tz_))
                wet *= 0.55 + 0.45 * noise3(px * 60.0, 2.0, pz * 60.0)
                ci = int(math.floor(px / 0.05)); cj = int(math.floor(pz / 0.05))
                if abs(px - xc - 0.2) < 0.30 and hash3(ci, 97, cj) < 0.035:                  # and spilled a little
                    ex = px / 0.05 - ci - 0.5; ez = pz / 0.05 - cj - 0.5
                    wet = max(wet, sstep(0.05 + 0.10 * hash3(ci, 99, cj), 0.01, ex * ex + ez * ez))
                wet *= 0.30 + 0.70 * sstep(7.0, 24.0, pz)                                  # the oldest have nearly dried
                ar *= 1 - 0.50 * wet; ag *= 1 - 0.55 * wet; ab *= 1 - 0.56 * wet
    else:
        tone *= 0.86                                                                       # soffits: smoke, bats
    # ---- where the water has stood
    hw = py - WY
    if hw < 5.5:
        wob = 0.05 * (fbm3(px * 0.8, 1.0, pz * 0.8, 2) - 0.5)
        damp = math.exp(-max(hw, 0.0) / 1.5)
        ar *= 1 - 0.30 * damp; ag *= 1 - 0.24 * damp; ab *= 1 - 0.24 * damp
        for q in range(5):
            lv = 0.24 + 0.42 * q + 0.22 * q * q + wob * (1 + q)
            wd = 0.035 + 0.02 * q
            ln = math.exp(-((hw - lv) / wd) ** 2) * (0.45 + 0.55 * fbm3(px * 4.0 + q, py * 9.0, pz * 4.0, 2)) * (1.0 - 0.13 * q)
            ar += (0.66 - ar) * 0.50 * ln; ag += (0.64 - ag) * 0.50 * ln; ab += (0.58 - ab) * 0.50 * ln   # lime, where it stood longest
        al = math.exp(-max(hw, 0.0) / 0.22) * (0.6 + 0.4 * fbm3(px * 7.0, py * 7.0, pz * 7.0, 2))
        if hw < 0.0: al = 0.85
        ar += (0.070 - ar) * al; ag += (0.105 - ag) * al; ab += (0.050 - ab) * al                    # weed at the line
    ar *= tone; ag *= tone; ab *= tone
    return ar, ag, ab, bx, by, bz


# ------------------------------------------------------------------ the lattice of probes
@njit(**NJ)
def vol_lookup(V, VAL, G0, GH, px, py, pz, nx, ny, nz):
    """irradiance on a surface at p facing n, from the probes in front of it."""
    NX = V.shape[0]; NY = V.shape[1]; NZ = V.shape[2]
    sx = 0 if nx > 0 else 1; sy_ = 2 if ny > 0 else 3; sz = 4 if nz > 0 else 5
    wx = nx * nx; wy = ny * ny; wz = nz * nz
    for att in range(3):
        off = 0.24 + 0.24 * att
        gx = (px + nx * off - G0[0]) / GH; gy = (py + ny * off - G0[1]) / GH; gz = (pz + nz * off - G0[2]) / GH
        ix = int(math.floor(gx)); iy = int(math.floor(gy)); iz = int(math.floor(gz))
        ix = min(max(ix, 0), NX - 2); iy = min(max(iy, 0), NY - 2); iz = min(max(iz, 0), NZ - 2)
        fx = min(max(gx - ix, 0.0), 1.0); fy = min(max(gy - iy, 0.0), 1.0); fz = min(max(gz - iz, 0.0), 1.0)
        r = 0.0; g = 0.0; b = 0.0; ws = 0.0
        for a in range(2):
            for bb in range(2):
                for c in range(2):
                    i = ix + a; jj = iy + bb; kk = iz + c
                    if not VAL[i, jj, kk]: continue
                    dd = (G0[0] + GH * i - px) * nx + (G0[1] + GH * jj - py) * ny + (G0[2] + GH * kk - pz) * nz
                    if dd < 0.03: continue
                    wgt = (fx if a == 1 else 1 - fx) * (fy if bb == 1 else 1 - fy) * (fz if c == 1 else 1 - fz) + 0.02
                    r += wgt * (wx * V[i, jj, kk, sx, 0] + wy * V[i, jj, kk, sy_, 0] + wz * V[i, jj, kk, sz, 0])
                    g += wgt * (wx * V[i, jj, kk, sx, 1] + wy * V[i, jj, kk, sy_, 1] + wz * V[i, jj, kk, sz, 1])
                    b += wgt * (wx * V[i, jj, kk, sx, 2] + wy * V[i, jj, kk, sy_, 2] + wz * V[i, jj, kk, sz, 2])
                    ws += wgt
        if ws > 0.05:
            return r / ws, g / ws, b / ws
    return 0.0, 0.0, 0.0

@njit(**NJ)
def vol_mean(VE, VAL, G0, GH, px, py, pz):
    NX = VE.shape[0]; NY = VE.shape[1]; NZ = VE.shape[2]
    gx = (px - G0[0]) / GH; gy = (py - G0[1]) / GH; gz = (pz - G0[2]) / GH
    ix = int(math.floor(gx)); iy = int(math.floor(gy)); iz = int(math.floor(gz))
    ix = min(max(ix, 0), NX - 2); iy = min(max(iy, 0), NY - 2); iz = min(max(iz, 0), NZ - 2)
    fx = min(max(gx - ix, 0.0), 1.0); fy = min(max(gy - iy, 0.0), 1.0); fz = min(max(gz - iz, 0.0), 1.0)
    r = 0.0; g = 0.0; b = 0.0; ws = 0.0
    for a in range(2):
        for bb in range(2):
            for c in range(2):
                i = ix + a; jj = iy + bb; kk = iz + c
                if not VAL[i, jj, kk]: continue
                wgt = (fx if a == 1 else 1 - fx) * (fy if bb == 1 else 1 - fy) * (fz if c == 1 else 1 - fz) + 0.01
                r += wgt * VE[i, jj, kk, 0]; g += wgt * VE[i, jj, kk, 1]; b += wgt * VE[i, jj, kk, 2]; ws += wgt
    if ws > 0.0: return r / ws, g / ws, b / ws
    return 0.0, 0.0, 0.0


@njit(parallel=True, **NJ)
def make_lights(NXs, NZs, NXk, NZk, seed):
    """the sun, followed down through the slot to what it strikes; and the slot itself as a piece of sky."""
    M = NXs * NZs
    LP = np.zeros((M, 3)); LN = np.zeros((M, 3)); LI = np.zeros((M, 3))
    area = (SX1 - SX0) * (SZ1 - SZ0)
    phi = SUNY * area / M
    for a in prange(NZs):
        s = (seed + a * 7919) & 0xffffffff
        for b in range(NXs):
            i = a * NXs + b
            s, r1 = lcg(s); s, r2 = lcg(s)
            ox = SX0 + (b + r1) / NXs * (SX1 - SX0); oz = SZ0 + (a + r2) / NZs * (SZ1 - SZ0); oy = ROOF + 0.04
            t, m = march(ox, oy, oz, -SUNX, -SUNY, -SUNZ, 60.0)
            tw = (WY - oy) / (-SUNY)
            if m <= 0 and tw > t: continue
            if tw < t:
                LP[i, 0] = ox - SUNX * tw; LP[i, 1] = WY + 0.01; LP[i, 2] = oz - SUNZ * tw
                LN[i, 1] = 1.0
                LI[i, 0] = E0R * phi * 0.020 / math.pi; LI[i, 1] = E0G * phi * 0.050 / math.pi; LI[i, 2] = E0B * phi * 0.028 / math.pi
                continue
            px = ox - SUNX * t; py = oy - SUNY * t; pz = oz - SUNZ * t
            gx, gy, gz = normal(px, py, pz, t)
            ar, ag, ab, bx, by, bz = material(px, py, pz, gx, gy, gz, m, 0.05)
            LP[i, 0] = px + gx * 0.02; LP[i, 1] = py + gy * 0.02; LP[i, 2] = pz + gz * 0.02
            LN[i, 0] = gx; LN[i, 1] = gy; LN[i, 2] = gz
            LI[i, 0] = E0R * phi * ar / math.pi; LI[i, 1] = E0G * phi * ag / math.pi; LI[i, 2] = E0B * phi * ab / math.pi
    Mk = NXk * NZk
    KP = np.zeros((Mk, 3))
    s = (seed * 31 + 5) & 0xffffffff
    for a in range(NZk):
        for b in range(NXk):
            s, r1 = lcg(s); s, r2 = lcg(s)
            KP[a * NXk + b, 0] = SX0 + (b + r1) / NXk * (SX1 - SX0); KP[a * NXk + b, 1] = ROOF
            KP[a * NXk + b, 2] = SZ0 + (a + r2) / NZk * (SZ1 - SZ0)
    return LP, LN, LI, KP, area / Mk


@njit(**NJ)
def first_light(px, py, pz, LP, LN, LI, KP, KA, ns, nk, s, out):
    """light arriving at p after one bounce of the sun, and straight from the sky in the slot.
    out[0:6] gets it resolved on the six axis directions (rgb each)."""
    M = LP.shape[0]; Mk = KP.shape[0]
    for q in range(18): out[q] = 0.0
    for a in range(ns):
        s, r = lcg(s)
        i = int((a + r) / ns * M)
        if LI[i, 0] + LI[i, 1] + LI[i, 2] <= 0.0: continue
        lx = LP[i, 0] - px; ly = LP[i, 1] - py; lz = LP[i, 2] - pz
        r2 = lx * lx + ly * ly + lz * lz; rl = math.sqrt(r2); lx /= rl; ly /= rl; lz /= rl
        cl = -(LN[i, 0] * lx + LN[i, 1] * ly + LN[i, 2] * lz)
        if cl <= 0.0: continue
        if occl(px, py, pz, lx, ly, lz, rl - 0.03): continue
        g = cl / max(r2, 0.36) * M / ns
        if lx > 0: k = 0
        else: k = 1
        w = abs(lx) * g
        out[k * 3] += w * LI[i, 0]; out[k * 3 + 1] += w * LI[i, 1]; out[k * 3 + 2] += w * LI[i, 2]
        k = 2 if ly > 0 else 3
        w = abs(ly) * g
        out[k * 3] += w * LI[i, 0]; out[k * 3 + 1] += w * LI[i, 1]; out[k * 3 + 2] += w * LI[i, 2]
        k = 4 if lz > 0 else 5
        w = abs(lz) * g
        out[k * 3] += w * LI[i, 0]; out[k * 3 + 1] += w * LI[i, 1]; out[k * 3 + 2] += w * LI[i, 2]
    for a in range(nk):
        s, r = lcg(s)
        i = int((a + r) / nk * Mk)
        lx = KP[i, 0] - px; ly = KP[i, 1] - py; lz = KP[i, 2] - pz
        if ly <= 0.0: continue
        r2 = lx * lx + ly * ly + lz * lz; rl = math.sqrt(r2); lx /= rl; ly /= rl; lz /= rl
        if occl(px, py, pz, lx, ly, lz, rl - 0.02): continue
        g = ly / max(r2, 0.36) * KA * Mk / nk
        k = 0 if lx > 0 else 1
        w = abs(lx) * g
        out[k * 3] += w * LSR; out[k * 3 + 1] += w * LSG; out[k * 3 + 2] += w * LSB
        w = ly * g
        out[6] += w * LSR; out[7] += w * LSG; out[8] += w * LSB
        k = 4 if lz > 0 else 5
        w = abs(lz) * g
        out[k * 3] += w * LSR; out[k * 3 + 1] += w * LSG; out[k * 3 + 2] += w * LSB
    return s


@njit(parallel=True, **NJ)
def probes_first(NX, NY, NZ, G0, GH, LP, LN, LI, KP, KA, ns, nk, seed):
    VAL = np.zeros((NX, NY, NZ), np.bool_)
    VF = np.zeros((NX, NY, NZ, 6, 3), np.float32)
    for i in prange(NX):
        out = np.zeros(18)
        for j in range(NY):
            for k in range(NZ):
                px = G0[0] + GH * i; py = G0[1] + GH * j; pz = G0[2] + GH * k
                if py < WY + 0.02 or py > -0.02: continue
                if scene(px, py, pz)[0] < 0.05: continue
                VAL[i, j, k] = True
                s = (seed + i * 92821 + j * 68917 + k * 31337) & 0xffffffff
                first_light(px, py, pz, LP, LN, LI, KP, KA, ns, nk, s, out)
                for q in range(6):
                    for c in range(3): VF[i, j, k, q, c] = out[q * 3 + c]
    return VAL, VF


@njit(parallel=True, **NJ)
def probes_hits(VAL, G0, GH, R, seed):
    """from every probe, R rays: where each lands, which way that surface faces, what colour it is."""
    NX, NY, NZ = VAL.shape
    n = 0
    IDX = np.full((NX, NY, NZ), -1, np.int32)
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if VAL[i, j, k]: IDX[i, j, k] = n; n += 1
    HP = np.zeros((n, R, 3), np.float32); HN = np.zeros((n, R, 3), np.float32); HA = np.zeros((n, R, 3), np.float32)
    HD = np.zeros((n, R, 3), np.float32)
    for i in prange(NX):
        for j in range(NY):
            for k in range(NZ):
                p = IDX[i, j, k]
                if p < 0: continue
                px = G0[0] + GH * i; py = G0[1] + GH * j; pz = G0[2] + GH * k
                s = (seed + i * 1237 + j * 7793 + k * 15485) & 0xffffffff
                s, r0 = lcg(s); s, r1 = lcg(s)
                ca = math.cos(6.2831853 * r1); sa = math.sin(6.2831853 * r1)
                for q in range(R):
                    yy = 1.0 - 2.0 * (q + r0) / R
                    rr = math.sqrt(max(1.0 - yy * yy, 0.0)); th = 2.39996323 * q
                    dx0 = rr * math.cos(th); dz0 = rr * math.sin(th)
                    dx = dx0 * ca - dz0 * sa; dz = dx0 * sa + dz0 * ca; dy = yy
                    HD[p, q, 0] = dx; HD[p, q, 1] = dy; HD[p, q, 2] = dz
                    t, m = march(px, py, pz, dx, dy, dz, 70.0)
                    if m <= 0: continue
                    if dy < 0.0:
                        tw = (WY - py) / dy
                        if tw < t:
                            HP[p, q, 0] = px + dx * tw; HP[p, q, 1] = WY; HP[p, q, 2] = pz + dz * tw
                            HN[p, q, 1] = 1.0
                            HA[p, q, 0] = 0.03; HA[p, q, 1] = 0.06; HA[p, q, 2] = 0.035
                            continue
                    hx = px + dx * t; hy = py + dy * t; hz = pz + dz * t
                    gx, gy, gz = normal(hx, hy, hz, t)
                    ar, ag, ab, bx, by, bz = material(hx, hy, hz, gx, gy, gz, m, 0.05)
                    HP[p, q, 0] = hx; HP[p, q, 1] = hy; HP[p, q, 2] = hz
                    HN[p, q, 0] = gx; HN[p, q, 1] = gy; HN[p, q, 2] = gz
                    HA[p, q, 0] = ar; HA[p, q, 1] = ag; HA[p, q, 2] = ab
    return IDX, HP, HN, HA, HD


@njit(parallel=True, **NJ)
def probes_bounce(VT, VAL, IDX, G0, GH, HP, HN, HA, HD):
    """one more bounce: every probe looks at what its rays hit and takes what those surfaces now give off."""
    NX, NY, NZ = VAL.shape
    R = HP.shape[1]
    VM = np.zeros((NX, NY, NZ, 6, 3), np.float32)
    for i in prange(NX):
        for j in range(NY):
            for k in range(NZ):
                p = IDX[i, j, k]
                if p < 0: continue
                for q in range(R):
                    ar = HA[p, q, 0]
                    if ar <= 0.0: continue
                    er, eg, eb = vol_lookup(VT, VAL, G0, GH, HP[p, q, 0], HP[p, q, 1], HP[p, q, 2], HN[p, q, 0], HN[p, q, 1], HN[p, q, 2])
                    lr = ar * er * 4.0 / R; lg = HA[p, q, 1] * eg * 4.0 / R; lb = HA[p, q, 2] * eb * 4.0 / R     # (alb E / pi) (4 pi / R)
                    dx = HD[p, q, 0]; dy = HD[p, q, 1]; dz = HD[p, q, 2]
                    a = 0 if dx > 0 else 1
                    VM[i, j, k, a, 0] += abs(dx) * lr; VM[i, j, k, a, 1] += abs(dx) * lg; VM[i, j, k, a, 2] += abs(dx) * lb
                    a = 2 if dy > 0 else 3
                    VM[i, j, k, a, 0] += abs(dy) * lr; VM[i, j, k, a, 1] += abs(dy) * lg; VM[i, j, k, a, 2] += abs(dy) * lb
                    a = 4 if dz > 0 else 5
                    VM[i, j, k, a, 0] += abs(dz) * lr; VM[i, j, k, a, 1] += abs(dz) * lg; VM[i, j, k, a, 2] += abs(dz) * lb
    return VM


# ------------------------------------------------------------------ air
@njit(**NJ)
def air(ox, oy, oz, dx, dy, dz, t, s, VE, VAL, G0, GH, nb):
    """transmittance to t, and the light the dust sends along the ray: the haze of the well, and the shaft."""
    T = math.exp(-FOG * t)
    lr = 0.0; lg = 0.0; lb = 0.0; hr = 0.0; hg = 0.0; hb = 0.0
    nf = 5
    s, r = lcg(s)
    for i in range(nf):
        ti = (i + r) / nf * t
        er, eg, eb = vol_mean(VE, VAL, G0, GH, ox + dx * ti, oy + dy * ti, oz + dz * ti)
        w = FOG * math.exp(-FOG * ti) * t / nf / math.pi
        lr += w * er; lg += w * eg; lb += w * eb
    # the shaft: a sheared box under the slot
    kx = SUNX / SUNY; kz = SUNZ / SUNY
    X0 = ox + kx * (0.25 - oy); dX = dx - kx * dy
    Z0_ = oz + kz * (0.25 - oy); dZ = dz - kz * dy
    t0 = 0.0; t1 = t
    if abs(dX) > 1e-9:
        a = (SX0 - 0.05 - X0) / dX; b = (SX1 + 0.05 - X0) / dX
        if a > b: a, b = b, a
        t0 = max(t0, a); t1 = min(t1, b)
    elif X0 < SX0 or X0 > SX1: t1 = -1.0
    if abs(dZ) > 1e-9:
        a = (SZ0 - 0.05 - Z0_) / dZ; b = (SZ1 + 0.05 - Z0_) / dZ
        if a > b: a, b = b, a
        t0 = max(t0, a); t1 = min(t1, b)
    elif Z0_ < SZ0 or Z0_ > SZ1: t1 = -1.0
    if t1 > t0:
        s, r = lcg(s)
        acc = 0.0
        for i in range(nb):
            ti = t0 + (i + r) / nb * (t1 - t0)
            px = ox + dx * ti; py = oy + dy * ti; pz = oz + dz * ti
            if py > 0.0:
                acc += math.exp(-FOG * ti); continue
            if sun_vis(px, py, pz) > 0.0:
                dens = 0.75 + 0.5 * fbm3(px * 0.9, py * 0.5, pz * 0.9, 2)                  # dust is not even
                acc += math.exp(-FOG * ti) * dens
        w = BEAM * acc * (t1 - t0) / nb / (4.0 * math.pi)
        hr = w * E0R; hg = w * E0G * 0.93; hb = w * E0B * 0.80
    return T, lr, lg, lb, hr, hg, hb, s


@njit(**NJ)
def plain(ox, oy, oz, dx, dy, dz, VT, VE, VAL, G0, GH, s):
    """what a ray sees, cheaply: used for what the water mirrors."""
    t, m = march(ox, oy, oz, dx, dy, dz, 80.0)
    if m == 0:
        T, lr, lg, lb, hr, hg, hb, s = air(ox, oy, oz, dx, dy, dz, t, s, VE, VAL, G0, GH, 6)
        return LSR * T + lr + hr, LSG * T + lg + hg, LSB * T + lb + hb, s
    if m < 0: return 0.0, 0.0, 0.0, s
    px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
    gx, gy, gz = normal(px, py, pz, t)
    ar, ag, ab, nx, ny, nz = material(px, py, pz, gx, gy, gz, m, t / FOC * 2.0)
    er, eg, eb = vol_lookup(VT, VAL, G0, GH, px, py, pz, nx, ny, nz)
    cs = nx * SUNX + ny * SUNY + nz * SUNZ
    if cs > 0.0 and sun_vis(px + gx * 0.01, py + gy * 0.01, pz + gz * 0.01) > 0.0:
        er += E0R * cs; eg += E0G * cs; eb += E0B * cs
    T, lr, lg, lb, hr, hg, hb, s = air(ox, oy, oz, dx, dy, dz, t, s, VE, VAL, G0, GH, 6)
    return ar * er / math.pi * T + lr + hr, ag * eg / math.pi * T + lg + hg, ab * eb / math.pi * T + lb + hb, s


# ------------------------------------------------------------------ the exposure
@njit(parallel=True, **NJ)
def expose(W, H, F, SS, NS, NK, nee, VT, VM, VE, VAL, G0, GH, LP, LN, LI, KP, KA, seed, OX, OY, WF):
    C = np.zeros((H, W, 3), np.float32)        # everything except the first-bounce light and the sun
    CS = np.zeros((H, W, 3), np.float32)       # what the sun lights directly, kept apart
    CB = np.zeros((H, W, 3), np.float32)       # the shaft in the dust
    EF = np.zeros((H, W, 3), np.float32)       # first-bounce irradiance on the true surface
    DV = np.zeros((H, W, 3), np.float32)       # where it mostly comes from
    AL = np.zeros((H, W, 3), np.float32)       # albedo (through the air)
    NG = np.zeros((H, W, 3), np.float32); NB = np.zeros((H, W, 3), np.float32)
    PP = np.zeros((H, W, 3), np.float32)
    WG = np.zeros((H, W), np.float32)
    M = LP.shape[0]; Mk = KP.shape[0]
    for iy in prange(H):
        for ix in range(W):
            s = (seed + (iy + OY) * 7919 + (ix + OX) * 104729) & 0xffffffff
            s, pr = lcg(s)
            cr = 0.0; cg = 0.0; cb = 0.0; ur = 0.0; ug = 0.0; ub = 0.0; vr = 0.0; vg = 0.0; vb = 0.0
            for sub in range(SS * SS):
                s, j1 = lcg(s); s, j2 = lcg(s)
                fx = ix + OX + (sub % SS + j1) / SS; fy = iy + OY + (sub // SS + j2) / SS
                dx = (fx - WF * 0.5) / F; dy = VTOP - fy / F; dz = 1.0
                dl = math.sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl
                t, m = march(CAMX, CAMY, CAMZ, dx, dy, dz, 90.0)
                water = False
                if dy < 0.0:
                    tw = (WY - CAMY) / dy
                    if tw < t or m < 0: water = True; t = tw
                if m == 0 and not water:
                    T, lr, lg, lb, hr, hg, hb, s = air(CAMX, CAMY, CAMZ, dx, dy, dz, t, s, VE, VAL, G0, GH, 10)
                    cr += 0.115 * T + lr; cg += 0.112 * T + lg; cb += 0.104 * T + lb                # the sky itself: white
                    vr += hr; vg += hg; vb += hb
                    continue
                if m < 0 and not water: continue
                px = CAMX + dx * t; py = CAMY + dy * t; pz = CAMZ + dz * t
                T, fr, fg, fb, hr, hg, hb, s = air(CAMX, CAMY, CAMZ, dx, dy, dz, t, s, VE, VAL, G0, GH, 12)
                cr += fr; cg += fg; cb += fb; vr += hr; vg += hg; vb += hb
                if water:
                    # ---- the surface: nearly still
                    wnx = 0.016 * (noise3(px * 2.3, 0.0, pz * 2.3) - 0.5) + 0.008 * (noise3(px * 7.0, 3.0, pz * 7.0) - 0.5)
                    wnz = 0.016 * (noise3(px * 2.3, 5.0, pz * 2.3) - 0.5) + 0.008 * (noise3(px * 7.0, 8.0, pz * 7.0) - 0.5)
                    wl = math.sqrt(wnx * wnx + 1.0 + wnz * wnz); wnx /= wl; wny = 1.0 / wl; wnz /= wl
                    vn = dx * wnx + dy * wny + dz * wnz
                    rx = dx - 2 * vn * wnx; ry = dy - 2 * vn * wny; rz = dz - 2 * vn * wnz
                    if ry < 0.02: ry = 0.02
                    Fr = 0.02 + 0.98 * (1.0 - min(-vn, 1.0)) ** 5
                    mr, mg, mb, s = plain(px, WY + 0.004, pz, rx, ry, rz, VT, VE, VAL, G0, GH, s)
                    # ---- below it
                    eta = 1.0 / 1.333
                    kk = 1.0 - eta * eta * (1.0 - vn * vn)
                    tx = eta * dx - (eta * vn + math.sqrt(kk)) * wnx; ty = eta * dy - (eta * vn + math.sqrt(kk)) * wny
                    tz = eta * dz - (eta * vn + math.sqrt(kk)) * wnz
                    tb, mb_ = march(px, WY - 0.004, pz, tx, ty, tz, 8.0)
                    if mb_ < 0: tb = 8.0
                    er, eg, eb = vol_lookup(VT, VAL, G0, GH, px, WY + 0.05, pz, 0.0, 1.0, 0.0)       # what falls on the water here
                    br = 0.0; bg = 0.0; bb = 0.0; xr = 0.0; xg = 0.0; xb = 0.0
                    nsw = 5
                    for q in range(nsw):                                                          # the body of the water, lit
                        ti = (q + 0.5) / nsw * tb
                        qx = px + tx * ti; qy = WY + ty * ti; qz = pz + tz * ti
                        dep = WY - qy
                        sv = sun_vis(qx + SUNX * dep / SUNY, WY + 0.02, qz + SUNZ * dep / SUNY)
                        pl = ti + dep
                        w = tb / nsw
                        br += w * 0.020 * (er * 0.3) * math.exp(-SIGR * pl); xr += w * 0.020 * E0R * sv * 0.55 * math.exp(-SIGR * pl)
                        bg += w * 0.060 * (eg * 0.3) * math.exp(-SIGG * pl); xg += w * 0.060 * E0G * sv * 0.55 * math.exp(-SIGG * pl)
                        bb += w * 0.028 * (eb * 0.3) * math.exp(-SIGB * pl); xb += w * 0.028 * E0B * sv * 0.55 * math.exp(-SIGB * pl)
                    if mb_ > 0:                                                                   # the steps under it
                        qx = px + tx * tb; qy = WY - 0.004 + ty * tb; qz = pz + tz * tb
                        gx, gy, gz = normal(qx, qy, qz, t + tb)
                        ar, ag, ab, nx, ny, nz = material(qx, qy, qz, gx, gy, gz, mb_, (t + tb) / F * 2.0)
                        dep = WY - qy
                        cs = max(nx * SUNX + ny * SUNY + nz * SUNZ, 0.0)
                        sv = 0.0
                        if cs > 0.0: sv = sun_vis(qx + SUNX * dep / SUNY, WY + 0.02, qz + SUNZ * dep / SUNY)
                        up = max(ny, 0.0) * 0.8 + 0.2
                        pl = tb + dep
                        br += ar / math.pi * (er * up) * math.exp(-SIGR * pl); xr += ar / math.pi * E0R * sv * cs * 0.9 * math.exp(-SIGR * pl)
                        bg += ag / math.pi * (eg * up) * math.exp(-SIGG * pl); xg += ag / math.pi * E0G * sv * cs * 0.9 * math.exp(-SIGG * pl)
                        bb += ab / math.pi * (eb * up) * math.exp(-SIGB * pl); xb += ab / math.pi * E0B * sv * cs * 0.9 * math.exp(-SIGB * pl)
                    # dust and petals lying on it, gathered at the edges
                    ds = scene(px, WY - 0.01, pz)[0]
                    sc = 0.22 * math.exp(-max(ds, 0.0) / 0.25) * sstep(0.42, 0.70, fbm3(px * 2.7, 5.0, pz * 2.7, 3))
                    sc += 0.35 * math.exp(-max(ds, 0.0) / 0.015)
                    sv = sun_vis(px, WY + 0.02, pz)
                    sr_ = 0.42 * er / math.pi; sg_ = 0.40 * eg / math.pi; sb_ = 0.30 * eb / math.pi
                    ur += T * ((1 - Fr) * xr * (1 - sc) + sc * 0.42 * E0R * sv * SUNY / math.pi)
                    ug += T * ((1 - Fr) * xg * (1 - sc) + sc * 0.40 * E0G * sv * SUNY / math.pi)
                    ub += T * ((1 - Fr) * xb * (1 - sc) + sc * 0.30 * E0B * sv * SUNY / math.pi)
                    lr = Fr * mr + (1 - Fr) * br; lg = Fr * mg + (1 - Fr) * bg; lb = Fr * mb + (1 - Fr) * bb
                    lr = lr * (1 - sc) + sc * sr_; lg = lg * (1 - sc) + sc * sg_; lb = lb * (1 - sc) + sc * sb_
                    cr += lr * T; cg += lg * T; cb += lb * T
                    continue
                gx, gy, gz = normal(px, py, pz, t)
                fp = t / F / max(-(gx * dx + gy * dy + gz * dz), 0.08)
                ar, ag, ab, nx, ny, nz = material(px, py, pz, gx, gy, gz, m, fp)
                qx = px + gx * 0.012; qy = py + gy * 0.012; qz = pz + gz * 0.012
                # ---- the sun
                er = 0.0; eg = 0.0; eb = 0.0
                cs = nx * SUNX + ny * SUNY + nz * SUNZ
                lit = 0.0
                if cs > 0.0 and gx * SUNX + gy * SUNY + gz * SUNZ > 0.0:
                    lit = sun_vis(qx, qy, qz)
                    ur += ar * E0R * cs * lit / math.pi * T; ug += ag * E0G * cs * lit / math.pi * T; ub += ab * E0B * cs * lit / math.pi * T
                # ---- the later bounces, from the probes, less where the surface is shut in
                occ = 0.0; sca = 1.0
                for k in range(4):
                    hh = 0.03 + 0.085 * k
                    occ += max(hh - scene(px + gx * hh, py + gy * hh, pz + gz * hh)[0], 0.0) / hh * sca; sca *= 0.75
                ao = min(max(1.0 - 0.33 * occ, 0.0), 1.0)
                if nee:
                    mr, mg, mb = vol_lookup(VM, VAL, G0, GH, px, py, pz, nx, ny, nz)
                    k_ao = 0.35 + 0.65 * ao
                else:
                    mr, mg, mb = vol_lookup(VT, VAL, G0, GH, px, py, pz, nx, ny, nz)
                    k_ao = 0.25 + 0.75 * ao
                er += mr * k_ao; eg += mg * k_ao; eb += mb * k_ao
                lr = ar * er / math.pi; lg = ag * eg / math.pi; lb = ab * eb / math.pi
                if m == M_BRASS:
                    vn = dx * nx + dy * ny + dz * nz
                    rx = dx - 2 * vn * nx; ry = dy - 2 * vn * ny; rz = dz - 2 * vn * nz
                    sp = max(rx * SUNX + ry * SUNY + rz * SUNZ, 0.0) ** 24
                    svp = sun_vis(qx, qy, qz) if lit == 0.0 else lit
                    ur += 0.9 * sp * svp * E0R * T; ug += 0.7 * sp * svp * E0G * T; ub += 0.3 * sp * svp * E0B * T
                    mr2, mg2, mb2 = vol_lookup(VT, VAL, G0, GH, px, py, pz, rx, ry, rz)
                    lr += 0.25 * mr2 / math.pi * 0.9; lg += 0.25 * mg2 / math.pi * 0.7; lb += 0.25 * mb2 / math.pi * 0.3
                cr += lr * T; cg += lg * T; cb += lb * T
                # ---- the first bounce and the sky, by sampling the small lamps
                if nee:
                    e0 = 0.0; e1 = 0.0; e2 = 0.0; d0 = 0.0; d1 = 0.0; d2 = 0.0
                    for a in range(NS):
                        s, r = lcg(s)
                        i = int((((a + sub * NS) * 7 % (NS * SS * SS)) + r) / (NS * SS * SS) * M)
                        if LI[i, 0] + LI[i, 1] + LI[i, 2] <= 0.0: continue
                        lx = LP[i, 0] - qx; ly = LP[i, 1] - qy; lz = LP[i, 2] - qz
                        r2 = lx * lx + ly * ly + lz * lz; rl = math.sqrt(r2); lx /= rl; ly /= rl; lz /= rl
                        cl = -(LN[i, 0] * lx + LN[i, 1] * ly + LN[i, 2] * lz)
                        cp = gx * lx + gy * ly + gz * lz
                        if cl <= 0.0 or cp <= 0.0: continue
                        if occl(qx, qy, qz, lx, ly, lz, rl - 0.03): continue
                        g = cl / max(r2, 0.16) * M / NS
                        e0 += g * cp * LI[i, 0]; e1 += g * cp * LI[i, 1]; e2 += g * cp * LI[i, 2]
                        gl = g * (LI[i, 0] + LI[i, 1] + LI[i, 2])
                        d0 += gl * lx; d1 += gl * ly; d2 += gl * lz
                    for a in range(NK):
                        s, r = lcg(s)
                        i = int((((a + sub * NK) * 5 % (NK * SS * SS)) + r) / (NK * SS * SS) * Mk)
                        lx = KP[i, 0] - qx; ly = KP[i, 1] - qy; lz = KP[i, 2] - qz
                        if ly <= 0.0: continue
                        r2 = lx * lx + ly * ly + lz * lz; rl = math.sqrt(r2); lx /= rl; ly /= rl; lz /= rl
                        cp = gx * lx + gy * ly + gz * lz
                        if cp <= 0.0: continue
                        if occl(qx, qy, qz, lx, ly, lz, rl - 0.02): continue
                        g = ly / max(r2, 0.16) * KA * Mk / NK
                        e0 += g * cp * LSR; e1 += g * cp * LSG; e2 += g * cp * LSB
                        gl = g * (LSR + LSG + LSB)
                        d0 += gl * lx; d1 += gl * ly; d2 += gl * lz
                    EF[iy, ix, 0] += e0; EF[iy, ix, 1] += e1; EF[iy, ix, 2] += e2
                    DV[iy, ix, 0] += d0; DV[iy, ix, 1] += d1; DV[iy, ix, 2] += d2
                    AL[iy, ix, 0] += ar * T; AL[iy, ix, 1] += ag * T; AL[iy, ix, 2] += ab * T
                    NG[iy, ix, 0] += gx; NG[iy, ix, 1] += gy; NG[iy, ix, 2] += gz
                    NB[iy, ix, 0] += nx; NB[iy, ix, 1] += ny; NB[iy, ix, 2] += nz
                    PP[iy, ix, 0] += px; PP[iy, ix, 1] += py; PP[iy, ix, 2] += pz
                    WG[iy, ix] += 1.0
            n = SS * SS
            C[iy, ix, 0] = cr / n; C[iy, ix, 1] = cg / n; C[iy, ix, 2] = cb / n
            CS[iy, ix, 0] = ur / n; CS[iy, ix, 1] = ug / n; CS[iy, ix, 2] = ub / n
            CB[iy, ix, 0] = vr / n; CB[iy, ix, 1] = vg / n; CB[iy, ix, 2] = vb / n
            w = WG[iy, ix]
            if w > 0.0:
                for c in range(3):
                    EF[iy, ix, c] /= w; DV[iy, ix, c] /= w; AL[iy, ix, c] /= n; NG[iy, ix, c] /= w; NB[iy, ix, c] /= w; PP[iy, ix, c] /= w
    return C, CS, CB, EF, DV, AL, NG, NB, PP, WG


@njit(parallel=True, **NJ)
def atrous(A, B, NG, PP, WG, step, sig):
    """one pass of an edge-keeping blur of A (and B), guided by surface position and facing."""
    H, W = WG.shape
    OA = np.zeros_like(A); OB = np.zeros_like(B)
    kern = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for y in prange(H):
        for x in range(W):
            if WG[y, x] <= 0.0: continue
            nx = NG[y, x, 0]; ny = NG[y, x, 1]; nz = NG[y, x, 2]
            nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9; nx /= nl; ny /= nl; nz /= nl
            ws = 0.0
            a0 = 0.0; a1 = 0.0; a2 = 0.0; b0 = 0.0; b1 = 0.0; b2 = 0.0
            for j in range(-2, 3):
                yy = y + j * step
                if yy < 0 or yy >= H: continue
                for i in range(-2, 3):
                    xx = x + i * step
                    if xx < 0 or xx >= W: continue
                    if WG[yy, xx] <= 0.0: continue
                    mx = NG[yy, xx, 0]; my = NG[yy, xx, 1]; mz = NG[yy, xx, 2]
                    ml = math.sqrt(mx * mx + my * my + mz * mz) + 1e-9
                    dn = max((nx * mx + ny * my + nz * mz) / ml, 0.0)
                    pd = abs((PP[yy, xx, 0] - PP[y, x, 0]) * nx + (PP[yy, xx, 1] - PP[y, x, 1]) * ny + (PP[yy, xx, 2] - PP[y, x, 2]) * nz)
                    w = kern[j + 2] * kern[i + 2] * dn ** 24 * math.exp(-pd / sig) * min(WG[yy, xx], 1.0)
                    ws += w
                    a0 += w * A[yy, xx, 0]; a1 += w * A[yy, xx, 1]; a2 += w * A[yy, xx, 2]
                    b0 += w * B[yy, xx, 0]; b1 += w * B[yy, xx, 1]; b2 += w * B[yy, xx, 2]
            if ws > 0.0:
                OA[y, x, 0] = a0 / ws; OA[y, x, 1] = a1 / ws; OA[y, x, 2] = a2 / ws
                OB[y, x, 0] = b0 / ws; OB[y, x, 1] = b1 / ws; OB[y, x, 2] = b2 / ws
            else:
                for c in range(3): OA[y, x, c] = A[y, x, c]; OB[y, x, c] = B[y, x, c]
    return OA, OB


# ------------------------------------------------------------------ plaster
EXPOSURE = 210.0
ADAPT = 0.20
SUNK = 0.17                     # the eye closes on what the sun strikes: it is printed at a fifth

def develop(lin, scale, rng, exposure):
    """no lens.  The eye's adaptation, a restrained curve, then the tooth of lime plaster."""
    H, W = lin.shape[:2]
    lin = np.maximum(lin, 0.0) * exposure
    L = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    # the eye opens in the dark parts and closes on the bright ones, over a broad neighbourhood
    lg = np.log(np.clip(L, 1e-4, 80.0))
    base = gaussian_filter(lg, 60.0 * scale)
    gain = np.exp(-ADAPT * (base - math.log(0.22)))
    gain = np.clip(gain, 0.25, 5.0)
    lin = lin * gain[..., None]
    L = L * gain
    wp = 6.0
    f = lambda x: x * (1.0 + x / wp ** 2) / (1.0 + x)
    y = 0.70 * lin * (f(L) / np.maximum(L, 1e-9))[..., None] + 0.30 * f(lin)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-9), 1 / 2.4) - 0.055)
    y = y + 0.22 * (y * y * (3 - 2 * y) - y)
    y = 0.030 + 0.955 * y                                                    # matte: no true black, no glare
    # tooth
    g = gaussian_filter(rng.standard_normal((H, W)), 0.6 * max(scale, 0.5)); g /= g.std()
    g2 = gaussian_filter(rng.standard_normal((H, W)), 2.2 * max(scale, 0.5)); g2 /= g2.std()
    lum = y.mean(-1, keepdims=True)
    amp = 0.004 + 0.009 * np.sqrt(np.clip(4 * lum * (1 - lum), 0, 1))
    y = y + amp * (0.85 * g[..., None] + 0.35 * g2[..., None])
    return np.clip(y, 0, 1)


def sign(img, scale, depth):
    """a mason's mark, a triangle over a line, cut lower right, and the depth."""
    H, W = img.shape[:2]
    u = max(scale, 0.2)
    cw, ch = int(330 * u), int(74 * u)
    k = 4
    im = Image.new('L', (cw * k, ch * k), 0); d = ImageDraw.Draw(im)
    lw = max(int(3.2 * u * k), 2)
    tx, ty, ts = 34 * u * k, 12 * u * k, 40 * u * k
    d.line([(tx - ts * 0.5, ty + ts * 0.80), (tx, ty), (tx + ts * 0.5, ty + ts * 0.80), (tx - ts * 0.5, ty + ts * 0.80)], fill=255, width=lw, joint='curve')
    d.line([(tx - ts * 0.62, ty + ts * 1.08), (tx + ts * 0.62, ty + ts * 1.08)], fill=255, width=lw)
    font = ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf', int(38 * u * k))
    d.text((tx + ts * 1.05, ty + ts * 0.10), '%.1f m' % depth, fill=255, font=font)
    im = im.resize((cw, ch), Image.LANCZOS)
    m = np.asarray(im, np.float64) / 255.0
    x0 = W - cw - int(70 * u); y0 = H - ch - int(64 * u)
    reg = img[y0:y0 + ch, x0:x0 + cw]
    lightedge = np.roll(np.roll(m, 1, 0), 1, 1)
    out = reg * (1.0 - 0.50 * m[..., None]) + 0.10 * np.clip(lightedge - m, 0, 1)[..., None]
    img[y0:y0 + ch, x0:x0 + cw] = np.clip(out, 0, 1)
    return img


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    out = sys.argv[2] if len(sys.argv) > 2 else __file__.replace('.py', '.png')
    mode = sys.argv[3] if len(sys.argv) > 3 else ('final' if scale >= 0.99 else 'draft')
    W = int(round(2700 * scale)); H = int(round(3600 * scale)); F = FOC * scale
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    if mode == 'final':
        GH = 0.40; R = 96; pns, pnk = 160, 64; SS, NS, NK, nee = 4, 2, 2, 1; LG = (32, 160, 12, 96)
    elif mode == 'mid':
        GH = 0.45; R = 64; pns, pnk = 96, 48; SS, NS, NK, nee = 2, 5, 3, 1; LG = (24, 128, 10, 80)
    else:
        GH = 0.55; R = 48; pns, pnk = 64, 32; SS, NS, NK, nee = 2, 0, 0, 0; LG = (16, 96, 8, 48)
    LP, LN, LI, KP, KA = make_lights(LG[0], LG[1], LG[2], LG[3], SEED % 100000)
    o = np.argsort(LP[:, 2] + 0.001 * LP[:, 0], kind='stable'); LP = LP[o]; LN = LN[o]; LI = LI[o]
    print('lamps %d (lit %d)  %.0fs' % (LP.shape[0], int((LI.sum(1) > 0).sum()), time.time() - t0), flush=True)
    G0 = np.array([-WX + 0.12, YBOT + 0.1, ZBACK + 0.2])
    NX = int((2 * WX - 0.24) / GH) + 1; NY = int((-YBOT - 0.1) / GH) + 1; NZ = int((ZEND + 2.4 - ZBACK - 0.2) / GH) + 1
    G0[0] = -0.5 * GH * (NX - 1)
    VAL, VF = probes_first(NX, NY, NZ, G0, GH, LP, LN, LI, KP, KA, pns, pnk, SEED % 1000)
    print('probes %dx%dx%d, %d in air  %.0fs' % (NX, NY, NZ, int(VAL.sum()), time.time() - t0), flush=True)
    IDX, HP, HN, HA, HD = probes_hits(VAL, G0, GH, R, SEED % 777)
    VM = np.zeros_like(VF)
    for it in range(5):
        VM = probes_bounce(VF + VM, VAL, IDX, G0, GH, HP, HN, HA, HD)
    del HP, HN, HA, HD
    VT = VF + VM
    VE = np.ascontiguousarray(VT.mean(3))
    print('bounces  %.0fs   first %.5f  later %.5f' % (time.time() - t0, VF[VAL].mean(), VM[VAL].mean()), flush=True)
    OX = OY = 0; WF = W; crop = None
    for a in sys.argv:
        if a.startswith('--crop='):                       # x0,y0,x1,y1 in pixels of the full picture at this scale
            crop = [int(v) for v in a[7:].split(',')]
            OX, OY = crop[0], crop[1]; W = crop[2] - crop[0]; H = crop[3] - crop[1]
    C, CS, CB, EF, DV, AL, NG, NB, PP, WG = expose(W, H, F, SS, NS, NK, nee, VT, VM, VE, VAL, G0, GH, LP, LN, LI, KP, KA, SEED % 9973, OX, OY, WF)
    print('exposed  %.0fs' % (time.time() - t0), flush=True)
    lin = C.astype(np.float64) + SUNK * CS.astype(np.float64)
    cb = CB.astype(np.float64)                                               # dust lights the dust beside it: the shaft has no knife edge
    g1 = np.stack([gaussian_filter(cb[..., c], 9.0 * scale) for c in range(3)], -1)
    g2 = np.stack([gaussian_filter(cb[..., c], 45.0 * scale) for c in range(3)], -1)
    lin += 0.68 * cb + 0.27 * g1 + 0.05 * g2
    if nee:
        A = EF; B = DV
        for step in (1, 2, 4, 8) if scale > 0.6 else (1, 2, 4):
            A, B = atrous(A, B, NG, PP, WG, step, 0.05)
        dn = (NB * B).sum(-1); dg = (NG * B).sum(-1)
        bump = np.clip(dn / np.maximum(dg, 1e-9), 0.0, 2.5)
        bump = np.where(dg > 1e-9, bump, 1.0)
        lin += AL.astype(np.float64) * A * bump[..., None] / math.pi
    if '--keep' in sys.argv: np.save(out.replace('.png', '_lin.npy'), lin.astype(np.float32))
    img = develop(lin, scale, rng, EXPOSURE)
    if crop is None: img = sign(img, scale, ROOF - YBOT)
    Image.fromarray((img * 255 + rng.random(img.shape)).clip(0, 255).astype(np.uint8)).save(out)
    print('saved', out, '%dx%d  %.0fs' % (W, H, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
