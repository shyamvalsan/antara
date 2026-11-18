"""
Giant Forest, 6:50 am, the fire is walking

Ray Avila.

Burn plan.  West slope, 2,050 m, the last week of May, the morning after ignition.  The ground
rises to the east, gently where the lens stands and more steeply beyond.  Overnight humidity
recovery was good and the duff is damp underneath, so the fire backs down the slope towards us:
flame lengths a hand to a knee, rate of spread about a metre in ten minutes.  The lens is at head
height, 1.62 m, looking east.  Eleven monarchs are in the plan, six to eight metres through at the
ground; the nearest, on the left, is twelve metres from the lens and only its flank is in the
picture.  The one with the old scar is twenty-six metres off.  Between them sugar pine and white
fir, a tenth the width.  The sun is thirteen degrees up, ahead and to the left, coming through the
crowns in pieces.  One firefighter, laying a strip with a drip torch below the main fire.

Usage:  python giant_forest_650am_the_fire_is_walking.py [scale=1.0] [out.png] [--crop=x0,y0,x1,y1]
        scale 1.0 is 2700 x 3600.

Built as signed-distance solids in metres and lit by calculation:
  trunks  - each a solid with its own flare, buttresses, furrowed bark and, on the old ones, a fire scar
  ground  - a heightfield of duff with sticks and cones lying on it; black where the fire has been
  sun     - one directional light, broken by the crowns (out of frame) and by every trunk
  smoke   - a density marched along every ray, lit by the sun through a shadow map, so the shafts are true
  fire    - a map of flame heights along the front; the flames are marched as emitters and light the ground
"""
import sys, math, time, warnings
import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaWarning
warnings.simplefilter('ignore', NumbaWarning)
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, maximum_filter

SEED = 19731007
FW, FH = 2700, 3600
FOC = 2250.0                    # pixels per unit tangent at scale 1
VC = 0.70                      # the horizon line, as a fraction of the height from the top
PITCH = 0.0
CAMX, CAMZ, EYE = 0.0, 0.0, 1.62
OLDEST = '2,200 yr'

# the sun: thirteen degrees up, ahead and to the left
_el, _az = math.radians(13.0), math.radians(-37.0)
SUNX, SUNY, SUNZ = math.cos(_el) * math.sin(_az), math.sin(_el), math.cos(_el) * math.cos(_az)
E0R, E0G, E0B = 3.3, 1.95, 0.80                      # low sun, through valley air and our own smoke
SKR, SKG, SKB = 0.150, 0.185, 0.250                  # what comes down between the crowns
HZR, HZG, HZB = 0.30, 0.25, 0.17                     # the lit smoke as a lamp, from the sun's side
# smoke
SIG0 = 0.0080                   # per metre, the morning haze
WINDX, WINDZ = 0.35, -0.10      # drift of the plume per metre of rise

# grid of trees
CS = 2.0
GX0, GZ0 = -70.0, -14.0
GNX, GNZ = 70, 80
KMAX = 6
MARG = 1.0
# fire maps
FM0X, FM0Z, FMS, FMN = -40.0, 2.0, 0.03, 2800        # origin, metres per texel, size
FCK = 16                                             # coarse map factor

M_GROUND, M_TRUNK, M_STICK, M_CONE = 1, 2, 5, 6
M_CHIP = 7
M_SHIRT, M_PANTS, M_BOOT, M_SKIN, M_HELMET, M_PACK, M_TORCH, M_GLOVE, M_SHROUD = 10, 11, 12, 13, 14, 15, 16, 17, 18
NJ = dict(cache=True, fastmath=True, error_model='numpy')

# the firefighter: feet, and which way he faces
FIGX, FIGZ, FIGYAW = 3.1, 19.6, 1.30


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
def smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25

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
def sd_ell(x, y, z, rx, ry, rz):
    k0 = math.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = math.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2 + (z / (rz * rz)) ** 2)
    return k0 * (k0 - 1.0) / (k1 + 1e-9)

@njit(**NJ)
def sd_rbox(x, y, z, bx, by, bz, r):
    qx = abs(x) - bx + r; qy = abs(y) - by + r; qz = abs(z) - bz + r
    ox = max(qx, 0.0); oy = max(qy, 0.0); oz = max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0) - r

@njit(**NJ)
def lcg(s):
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return s, ((s >> 8) & 0xffffff) / 16777216.0


# ------------------------------------------------------------------ the ground
@njit(**NJ)
def slope(x, z):
    """the hill: gentle where the lens stands, steepening to the east"""
    q = (z - 17.0) / 5.0
    sp = q if q > 20.0 else math.log(1.0 + math.exp(q))
    return 0.040 * z + 0.012 * x + 0.105 * 5.0 * sp

@njit(**NJ)
def ground0(x, z):
    """the slope and its swells, without the mounds at the trees"""
    return slope(x, z) + 1.5 * (fbm3(x * 0.045 + 3.1, 0.5, z * 0.045 + 7.7, 3) - 0.5) \
        + 0.22 * (noise3(x * 0.31, 1.5, z * 0.31) - 0.5)


# ------------------------------------------------------------------ a trunk
# T[i] = x, z, R, base y, seed, kind (0 monarch, 1 pine, 2 fir), scar angle, scar half-width (m), scar height, lean x, lean z, height
@njit(**NJ)
def trunk_r(yr, R, kind):
    """the round radius at a height above the base: taper and flare"""
    yy = max(yr, -1.5)
    if kind == 0:
        return R * (1.0 - 0.0040 * yy) * (1.0 + 0.20 * math.exp(-yy / 0.9) + 0.16 * math.exp(-yy / 4.5))
    return R * (1.0 - 0.012 * yy) * (1.0 + 0.30 * math.exp(-yy / 0.45))

@njit(**NJ)
def scar_w(yr, a, T, i):
    """inside the scar: >0, growing to 1 at its middle.  A pointed arch on the bark."""
    hw = T[i, 7]
    if hw <= 0.0: return -1.0
    hs = T[i, 8]
    if yr > hs: return -1.0
    da = a - T[i, 6]
    if da > math.pi: da -= 2 * math.pi
    if da < -math.pi: da += 2 * math.pi
    yy = max(yr, 0.0) / hs
    w = hw * (1.0 - yy ** 1.7) * (0.9 + 0.25 * noise3(yr * 0.9, T[i, 4], 0.5))
    lean = 0.10 * hw * math.sin(yr * 0.8 + T[i, 4])
    s = abs(da * T[i, 2] * 1.25 - lean)
    return (w - s) / max(hw, 0.05)

@njit(**NJ)
def furrows(ca, sa, yr, R, sd):
    """the bark of a monarch: furrows that part and join every metre or two (0 ridge .. 1 furrow), the lesser ones
    on the ridges, and the cross-checks where a ridge has broken.  Each as distance from the line, 0 on it."""
    k = R / 0.30
    w1 = fbm3(ca * k * 0.5 + sd, yr * 0.22 + 3.0, sa * k * 0.5 - sd, 2)
    n = noise3(ca * k + sd * 3.1 + 2.6 * w1, yr * 0.34 + 1.6 * w1, sa * k - sd * 1.7 + 1.3 * w1)
    f1 = abs(2.0 * n - 1.0)
    n2 = noise3(ca * k * 2.9 - sd + 1.5 * w1, yr * 1.25 + 9.0, sa * k * 2.9 + sd * 0.3)
    f2 = abs(2.0 * n2 - 1.0)
    n3 = noise3(ca * k * 1.3 + 5.0, yr * 5.5 + 3.0 * w1, sa * k * 1.3 - 2.0)
    f3 = abs(2.0 * n3 - 1.0)
    return f1, f2, f3

@njit(**NJ)
def sd_trunk(dx, yr, dz, rr, T, i):
    R = T[i, 2]; kind = int(T[i, 5]); sd = T[i, 4]
    a = math.atan2(dz, dx)
    ca = dx / rr; sa = dz / rr
    r0 = trunk_r(yr, R, kind)
    yy = max(yr, -1.5)
    if kind == 0:
        A = 0.10 * math.exp(-yy / 2.0) + 0.018
        b = 0.50 * math.cos(3 * a + sd) + 0.45 * math.cos(4 * a + 2.3 * sd) + 0.40 * math.cos(6 * a + 4.1 * sd) \
            + 0.30 * math.cos(9 * a + 1.7 * sd) + 0.22 * math.cos(14 * a + 3.3 * sd) + 0.12 * math.cos(23 * a + 0.7 * sd)
        r = r0 * (1.0 + A * b)
        f1, f2, f3 = furrows(ca, sa, yr, R, sd)
        fur = 1.0 - sstep(0.0, 0.26, f1)
        r -= 0.115 * fur + (0.034 * (1.0 - sstep(0.0, 0.24, f2)) + 0.016 * (1.0 - sstep(0.0, 0.12, f3))) * (1.0 - fur)
        # the scar: the fire got in here, and the tree has been closing over it since
        sw = scar_w(yr, a, T, i)
        if sw > -0.35:
            cut = sstep(0.0, 0.55, sw)
            lip = math.exp(-(sw / 0.11) ** 2)
            r += 0.10 * lip * (1.0 - cut) + 0.03 * lip
            r -= cut * R * (0.30 + 0.10 * noise3(a * 5.0, yr * 1.1, sd))
    else:
        A = 0.03 * math.exp(-yy / 0.8) + 0.008
        r = r0 * (1.0 + A * (math.cos(4 * a + sd) + 0.6 * math.cos(7 * a + 2.0 * sd)))
        sp = 0.075 if kind == 1 else 0.05
        k = R / sp
        n = noise3(ca * k + sd * 3.1, yr * (0.9 if kind == 1 else 0.55), sa * k - sd * 1.7)
        w = abs(2.0 * n - 1.0)
        r -= (0.022 if kind == 1 else 0.016) * (1.0 - sstep(0.0, 0.32, w))
    d = rr - r
    top = T[i, 11]
    if yr > top - 2.0:
        d = max(d, yr - top)
    return d

# ------------------------------------------------------------------ the firefighter
@njit(**NJ)
def sd_fig(lx, ly, lz):
    """feet at the origin, facing +z, his right is +x.  Mid-stride, head down to the work, torch low in the right hand."""
    d = 9.0; m = M_SHIRT
    lz -= 0.07 * max(ly - 0.95, 0.0)                    # he leans into it a little
    # legs: left forward, right back with the heel coming up
    dl = sd_cap(lx, ly, lz, -0.095, 0.92, 0.0, -0.105, 0.50, 0.13, 0.098)
    dl = smin(dl, sd_cap(lx, ly, lz, -0.105, 0.50, 0.13, -0.11, 0.16, 0.22, 0.082), 0.03)
    dr = sd_cap(lx, ly, lz, 0.095, 0.92, 0.0, 0.105, 0.50, -0.05, 0.098)
    dr = smin(dr, sd_cap(lx, ly, lz, 0.105, 0.50, -0.05, 0.11, 0.20, -0.25, 0.082), 0.03)
    dg = smin(dl, dr, 0.04)
    dg = smin(dg, sd_ell(lx, ly - 0.97, lz, 0.175, 0.125, 0.125), 0.07)
    dg += 0.004 * math.sin(ly * 55.0 + lx * 21.0 + 3.0 * math.sin(lz * 30.0))          # the cloth
    if dg < d: d = dg; m = M_PANTS
    db = sd_rbox(lx + 0.11, ly - 0.075, lz - 0.245, 0.048, 0.075, 0.125, 0.04)
    db = smin(db, sd_cap(lx, ly, lz, -0.11, 0.10, 0.22, -0.11, 0.24, 0.22, 0.062), 0.03)
    bx = lx - 0.11; by = ly - 0.115; bz = lz + 0.235
    cy = by * 0.92 - bz * 0.38; cz = bz * 0.92 + by * 0.38
    db2 = sd_rbox(bx, cy, cz, 0.048, 0.075, 0.125, 0.04)
    db2 = smin(db2, sd_cap(lx, ly, lz, 0.11, 0.16, -0.25, 0.11, 0.28, -0.25, 0.062), 0.03)
    db = min(db, db2)
    if db < d: d = db; m = M_BOOT
    # shirt: chest, belly, shoulders, sleeves
    dt = sd_ell(lx, ly - 1.31, lz, 0.20, 0.215, 0.128)
    dt = smin(dt, sd_ell(lx, ly - 1.12, lz - 0.01, 0.165, 0.15, 0.115), 0.08)
    dt = smin(dt, sd_cap(lx, ly, lz, -0.170, 1.455, 0.0, 0.170, 1.455, 0.0, 0.072), 0.07)
    dt = smin(dt, sd_cap(lx, ly, lz, -0.215, 1.44, 0.0, -0.255, 1.17, -0.03, 0.052), 0.04)
    dt = smin(dt, sd_cap(lx, ly, lz, -0.255, 1.17, -0.03, -0.245, 0.95, 0.07, 0.043), 0.025)
    dt = smin(dt, sd_cap(lx, ly, lz, 0.215, 1.44, 0.0, 0.275, 1.19, 0.06, 0.052), 0.04)
    dt = smin(dt, sd_cap(lx, ly, lz, 0.275, 1.19, 0.06, 0.31, 0.98, 0.19, 0.043), 0.025)
    dt += 0.005 * math.sin(ly * 43.0 + lx * 13.0 + 2.5 * math.sin(lx * 29.0 + lz * 17.0)) * sstep(1.0, 1.2, ly)
    if dt < d: d = dt; m = M_SHIRT
    # gloves
    dh = sd_ell(lx + 0.245, ly - 0.905, lz - 0.085, 0.040, 0.058, 0.046)
    dh = min(dh, sd_ell(lx - 0.315, ly - 0.935, lz - 0.215, 0.040, 0.050, 0.050))
    if dh < d: d = dh; m = M_GLOVE
    # neck and head, chin down
    dn = sd_cap(lx, ly, lz, 0.0, 1.50, 0.0, 0.0, 1.60, 0.02, 0.050)
    dn = smin(dn, sd_ell(lx, ly - 1.655, lz - 0.035, 0.083, 0.100, 0.094), 0.02)
    if dn < d: d = dn; m = M_SKIN
    # the helmet, and the shroud that hangs from it over the neck
    hx = lx; hy = ly - 1.705; hz = lz - 0.03
    he = sd_ell(hx, hy, hz, 0.122, 0.105, 0.142)
    he = max(he, -hy - 0.018)
    br = sd_ell(hx, hy + 0.016, hz - 0.045, 0.118, 0.009, 0.160)
    he = smin(he, br, 0.012)
    he = min(he, sd_rbox(hx, hy - 0.104, hz, 0.008, 0.008, 0.10, 0.006))
    if he < d: d = he; m = M_HELMET
    # the shroud: cloth from the helmet down over the ears and neck
    ds = sd_ell(hx, hy + 0.105, hz + 0.012, 0.100, 0.125, 0.108)
    ds = max(ds, hy + 0.012)
    ds = max(ds, hz - 0.075)
    ds += 0.004 * math.sin(math.atan2(hz, hx) * 9.0 + hy * 30.0)
    if ds < d: d = ds; m = M_SHROUD
    # line gear: the bag low on the back, the straps, a canteen each side
    dp = sd_rbox(lx, ly - 1.085, lz + 0.155, 0.165, 0.105, 0.065, 0.045)
    dp = min(dp, sd_rbox(abs(lx) - 0.105, ly - 1.33, lz + 0.105, 0.019, 0.16, 0.012, 0.008))
    dp = min(dp, sd_rbox(lx, ly - 1.00, lz, 0.185, 0.022, 0.125, 0.015))
    dp = min(dp, sd_cap(lx, ly, lz, -0.205, 1.00, -0.02, -0.205, 1.10, -0.02, 0.048))
    dp = min(dp, sd_cap(lx, ly, lz, 0.205, 1.00, -0.02, 0.205, 1.10, -0.02, 0.048))
    if dp < d: d = dp; m = M_PACK
    # drip torch: a can hanging from the fist, the spout forward and down with a loop in it
    tx = lx - 0.325; ty = ly - 0.725; tz = lz - 0.26
    cy = ty * 0.93 - tz * 0.37; cz = tz * 0.93 + ty * 0.37
    dc = max(math.sqrt(tx * tx + cz * cz) - 0.062, abs(cy) - 0.15) - 0.008
    dc = min(dc, sd_cap(lx, ly, lz, 0.325, 0.87, 0.21, 0.330, 0.80, 0.46, 0.009))
    dc = min(dc, sd_cap(lx, ly, lz, 0.330, 0.80, 0.46, 0.335, 0.47, 0.57, 0.009))
    dc = min(dc, sd_cap(lx, ly, lz, 0.315, 0.93, 0.22, 0.325, 0.87, 0.215, 0.010))
    lq = math.sqrt((ly - 0.83) ** 2 + (lz - 0.36) ** 2) - 0.035
    dc = min(dc, math.sqrt(lq * lq + (lx - 0.33) ** 2) - 0.008)
    if dc < d: d = dc; m = M_TORCH
    return d, m

# ------------------------------------------------------------------ the grove
@njit(**NJ)
def scene(x, y, z, T, GI, lod):
    """distance, material, tree index.  lod 0: everything.  1: no litter (shadows, far)."""
    g = ground0(x, z)
    ti = -1
    d = MARG; m = 0
    ci = int((x - GX0) / CS); cj = int((z - GZ0) / CS)
    dtr = 9.0
    if ci >= 0 and ci < GNX and cj >= 0 and cj < GNZ:
        for q in range(KMAX):
            i = GI[cj, ci, q]
            if i < 0: break
            yr = y - T[i, 3]
            dx = x - T[i, 0] - T[i, 9] * yr; dz = z - T[i, 1] - T[i, 10] * yr
            rr = math.sqrt(dx * dx + dz * dz) + 1e-9
            R = T[i, 2]; kind = int(T[i, 5])
            # the duff banks up against a trunk
            if kind == 0:
                g += (0.10 + 0.50 * fbm3(x * 0.9 + 4.0, 1.0, z * 0.9, 2)) * sstep(R * 1.75 + 1.4, R * 1.05, rr) ** 2
            else:
                g += 0.10 * sstep(R * 1.45 + 1.0, R, rr) ** 2
            rb = trunk_r(yr, R, kind) * (1.14 if kind == 0 else 1.05)
            if rr - rb > 0.8:
                dt = rr - rb - 0.2
            else:
                dt = sd_trunk(dx, yr, dz, rr, T, i) * (0.55 if kind == 0 else 0.7)
            if dt < dtr: dtr = dt; ti = i
    if dtr < d: d = dtr; m = M_TRUNK
    h = y - g
    if h < 0.6:
        g += 0.030 * (noise3(x * 2.1, 0.3, z * 2.1) - 0.5) + 0.012 * (noise3(x * 7.3, 0.9, z * 7.3) - 0.5)
        h = y - g
    dg = h * 0.80
    if m == M_TRUNK:
        dd = smin(dg, d, 0.45)
        if dg < d: m = M_GROUND
        d = dd
    elif dg < d:
        d = dg; m = M_GROUND
    # the firefighter
    fx = x - FIGX; fz = z - FIGZ
    if fx * fx + fz * fz < 2.2 and h < 2.6:
        if fx * fx + fz * fz < 0.80:
            c = math.cos(FIGYAW); s = math.sin(FIGYAW)
            df, mf = sd_fig(fx * c - fz * s, y - ground0(FIGX, FIGZ) - 0.02, fx * s + fz * c)
            if df < d: d = df; m = mf
        else:
            d = min(d, math.sqrt(fx * fx + fz * fz) - 0.80)
    # fallen limbs
    if lod == 0 and h < 0.4 and m != M_TRUNK:
        cs = 3.2
        bi = math.floor(x / cs); bj = math.floor(z / cs)
        for oi in range(-1, 2):
            for oj in range(-1, 2):
                ii = int(bi + oi); jj = int(bj + oj)
                if hash3(ii, jj, 31) > 0.50: continue
                cx = (ii + 0.5 + 0.8 * (hash3(ii, jj, 32) - 0.5)) * cs
                cz = (jj + 0.5 + 0.8 * (hash3(ii, jj, 33) - 0.5)) * cs
                an = hash3(ii, jj, 34) * 6.2832
                ln = 0.5 + 1.3 * hash3(ii, jj, 35)
                rd = 0.012 + 0.045 * hash3(ii, jj, 36) ** 2
                ux = math.cos(an); uz = math.sin(an)
                if (x - cx) ** 2 + (z - cz) ** 2 > (ln * 2.2 + 0.3) ** 2: continue
                x0 = cx - ux * ln; z0 = cz - uz * ln
                bn = (hash3(ii, jj, 37) - 0.5) * 0.9
                x1 = cx + uz * bn * 0.4; z1 = cz - ux * bn * 0.4
                x2 = cx + ux * ln * 0.9 + uz * bn; z2 = cz + uz * ln * 0.9 - ux * bn
                y0 = ground0(x0, z0) + rd * 0.8; y1 = ground0(x1, z1) + rd * 0.8 + 0.03; y2 = ground0(x2, z2) + rd * 0.5
                ds = sd_cap(x, y, z, x0, y0, z0, x1, y1, z1, rd)
                ds = smin(ds, sd_cap(x, y, z, x1, y1, z1, x2, y2, z2, rd * 0.75), 0.01)
                x3 = x1 + (ux * 0.5 - uz * 0.8) * ln * 0.6; z3 = z1 + (uz * 0.5 + ux * 0.8) * ln * 0.6
                ds = min(ds, sd_cap(x, y, z, x1, y1, z1, x3, ground0(x3, z3) + rd * 0.4, z3, rd * 0.5))
                ds += 0.012 * (noise3(x * 14.0, y * 14.0, z * 14.0) - 0.5) + 0.004 * (noise3(x * 60.0, y * 60.0, z * 60.0) - 0.5)
                if ds < d: d = ds; m = M_STICK
    # litter: twigs, the small cones of the big trees, flakes of bark
    if lod == 0 and h < 0.22 and m != M_TRUNK:
        cs = 0.30
        bi = math.floor(x / cs); bj = math.floor(z / cs)
        for oi in range(-1, 2):
            for oj in range(-1, 2):
                ii = int(bi + oi); jj = int(bj + oj)
                r1 = hash3(ii, jj, 11)
                if r1 > 0.27: continue
                cx = (ii + 0.5 + 0.9 * (hash3(ii, jj, 12) - 0.5)) * cs
                cz = (jj + 0.5 + 0.9 * (hash3(ii, jj, 13) - 0.5)) * cs
                an = hash3(ii, jj, 14) * 6.2832
                ux = math.cos(an); uz = math.sin(an)
                if r1 < 0.13:
                    ln = 0.04 + 0.26 * hash3(ii, jj, 15) ** 2.5
                    rd = 0.0025 + 0.008 * hash3(ii, jj, 16) ** 3
                    ya = ground0(cx - ux * ln, cz - uz * ln); yb = ground0(cx + ux * ln, cz + uz * ln)
                    bx = cx + 0.25 * ln * uz; bz = cz - 0.25 * ln * ux
                    ym = 0.5 * (ya + yb) + rd + 0.006
                    ds = sd_cap(x, y, z, cx - ux * ln, ya + rd, cz - uz * ln, bx, ym, bz, rd)
                    ds = min(ds, sd_cap(x, y, z, bx, ym, bz, cx + ux * ln, yb + rd + 0.01 * hash3(ii, jj, 17), cz + uz * ln, rd * 0.8))
                    if ds < d: d = ds; m = M_STICK
                elif r1 < 0.19:
                    px = (x - cx) * ux + (z - cz) * uz; pz = -(x - cx) * uz + (z - cz) * ux
                    sc = 0.75 + 0.5 * hash3(ii, jj, 15)
                    py = y - ground0(cx, cz) - 0.014 * sc
                    dc = sd_ell(px, py, pz, 0.033 * sc, 0.023 * sc, 0.024 * sc)
                    if dc < 0.02:
                        dc += 0.0012 * math.sin(px * 420.0 / sc + 2.0 * math.atan2(py, pz)) * math.sin(4.0 * math.atan2(py, pz))
                    dc *= 0.8
                    if dc < d: d = dc; m = M_CONE
                else:
                    px = (x - cx) * ux + (z - cz) * uz; pz = -(x - cx) * uz + (z - cz) * ux
                    sc = 0.5 + 1.3 * hash3(ii, jj, 15) ** 2
                    tl = (hash3(ii, jj, 16) - 0.5) * 0.5
                    py = y - ground0(cx, cz) - 0.010 - tl * px
                    dc = sd_rbox(px, py, pz, 0.050 * sc, 0.004 + 0.003 * sc, 0.018 * sc * (1.0 + 0.5 * math.sin(px * 40.0 + 6.0 * tl)), 0.003)
                    if dc < d: d = dc; m = M_CHIP
    return d, m, ti


@njit(**NJ)
def march(ox, oy, oz, dx, dy, dz, tmax, T, GI):
    t = 0.02
    for s in range(900):
        d, m, ti = scene(ox + dx * t, oy + dy * t, oz + dz * t, T, GI, 0 if t < 26.0 else 1)
        if d < 0.00022 * t + 0.0003:
            return t, m, ti
        t += d
        if t > tmax: break
    return tmax, 0, -1

@njit(**NJ)
def shadow(px, py, pz, lx, ly, lz, maxt, kk, T, GI):
    """soft shadow towards a light: 1 open, 0 shut"""
    t = 0.03; res = 1.0
    for s in range(260):
        d, m, ti = scene(px + lx * t, py + ly * t, pz + lz * t, T, GI, 1)
        if d < 0.0008: return 0.0
        if d < MARG * 0.95: res = min(res, kk * d / t)
        t += min(max(d, 0.02), 2.5)
        if t >= maxt or py + ly * t > 70.0: break
    return res * res * (3.0 - 2.0 * res)

@njit(**NJ)
def normal(px, py, pz, t, T, GI, lod):
    e = 0.00035 * t + 0.0008
    nx = scene(px + e, py, pz, T, GI, lod)[0] - scene(px - e, py, pz, T, GI, lod)[0]
    ny = scene(px, py + e, pz, T, GI, lod)[0] - scene(px, py - e, pz, T, GI, lod)[0]
    nz = scene(px, py, pz + e, T, GI, lod)[0] - scene(px, py, pz - e, T, GI, lod)[0]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nl, ny / nl, nz / nl

@njit(**NJ)
def ambient_occ(px, py, pz, nx, ny, nz, T, GI):
    a = 0.0; w = 0.0
    hs = (0.04, 0.12, 0.30, 0.70, 1.5)
    for i in range(5):
        h = hs[i]
        d = scene(px + nx * h, py + ny * h, pz + nz * h, T, GI, 1)[0]
        k = 1.0 / (1.0 + i)
        a += k * min(max(d / h, 0.0), 1.0); w += k
    return a / w


# ------------------------------------------------------------------ the crowns, as the sun sees them
@njit(**NJ)
def gobo(px, py, pz, GW):
    """how much of the sun gets through the crowns to this point.  Constant along a ray of sun."""
    # two axes square to the sun
    ul = math.sqrt(SUNX * SUNX + SUNZ * SUNZ)
    ux = SUNZ / ul; uz = -SUNX / ul
    vx = -SUNY * SUNX / ul; vy = ul; vz = -SUNY * SUNZ / ul
    u = px * ux + pz * uz
    v = px * vx + py * vy + pz * vz
    a = fbm3(u * 0.105 + 4.0, v * 0.150 + 1.0, 2.5, 3)
    b = fbm3(u * 0.55 + 9.0, v * 0.60 + 3.0, 7.5, 3)
    c = noise3(u * 2.6, v * 2.6, 1.5)
    g = sstep(0.535, 0.61, a + 0.10 * (b - 0.5)) * sstep(0.30, 0.56, b + 0.22 * (c - 0.5))
    # the gaps we know about
    for k in range(GW.shape[0]):
        du = (u - GW[k, 0]) / GW[k, 2]; dv = (v - GW[k, 1]) / GW[k, 3]
        q = du * du + dv * dv
        if q < 4.0:
            wv = math.exp(-q * q) * sstep(0.18, 0.50, b + 0.25 * (c - 0.5) + 0.25 * math.exp(-q))
            if wv > g: g = wv
    return g


@njit(parallel=True, **NJ)
def make_shadow_map(NU, NV, U0, V0, DU, WS, T, GI):
    """for each line of sun, how far it gets (as distance along the sun axis)"""
    SM = np.empty((NV, NU), np.float32)
    ul = math.sqrt(SUNX * SUNX + SUNZ * SUNZ)
    ux = SUNZ / ul; uz = -SUNX / ul
    vx = -SUNY * SUNX / ul; vy = ul; vz = -SUNY * SUNZ / ul
    for j in prange(NV):
        for i in range(NU):
            u = U0 + (i + 0.5) * DU; v = V0 + (j + 0.5) * DU
            ox = u * ux + v * vx + WS * SUNX; oy = v * vy + WS * SUNY; oz = u * uz + v * vz + WS * SUNZ
            t = 0.0; hit = 400.0
            for s in range(400):
                x = ox - SUNX * t; y = oy - SUNY * t; z = oz - SUNZ * t
                if y > 75.0:
                    t += (y - 75.0) / SUNY + 0.01; continue
                d, m, ti = scene(x, y, z, T, GI, 1)
                if d < 0.004:
                    hit = t; break
                t += d
                if t > 400.0: break
            SM[j, i] = WS - hit
    return SM

@njit(**NJ)
def sun_vol(px, py, pz, SM, U0, V0, DU, sp):
    """the sun at a point in the air: shadow map and crowns.  sp: softening, in texels"""
    ul = math.sqrt(SUNX * SUNX + SUNZ * SUNZ)
    ux = SUNZ / ul; uz = -SUNX / ul
    vx = -SUNY * SUNX / ul; vy = ul; vz = -SUNY * SUNZ / ul
    u = px * ux + pz * uz
    v = px * vx + py * vy + pz * vz
    w = px * SUNX + py * SUNY + pz * SUNZ
    fu = (u - U0) / DU - 0.5; fv = (v - V0) / DU - 0.5
    i = int(math.floor(fu)); j = int(math.floor(fv))
    NV, NU = SM.shape
    if i < 0 or j < 0 or i >= NU - 1 or j >= NV - 1: return 0.0
    a = fu - i; b = fv - j
    s00 = 1.0 if w > SM[j, i] - 0.05 else 0.0
    s10 = 1.0 if w > SM[j, i + 1] - 0.05 else 0.0
    s01 = 1.0 if w > SM[j + 1, i] - 0.05 else 0.0
    s11 = 1.0 if w > SM[j + 1, i + 1] - 0.05 else 0.0
    return (s00 * (1 - a) + s10 * a) * (1 - b) + (s01 * (1 - a) + s11 * a) * b


# ------------------------------------------------------------------ smoke
@njit(**NJ)
def fmap(M, x, z, s):
    fx = (x - FM0X) / s; fz = (z - FM0Z) / s
    i = int(math.floor(fx)); j = int(math.floor(fz))
    if i < 0 or j < 0 or i >= M.shape[1] - 1 or j >= M.shape[0] - 1: return 0.0
    a = fx - i; b = fz - j
    return (M[j, i] * (1 - a) + M[j, i + 1] * a) * (1 - b) + (M[j + 1, i] * (1 - a) + M[j + 1, i + 1] * a) * b

@njit(**NJ)
def smoke(x, y, z, PL):
    """extinction per metre.  Haze, the thin sheets it settles into overnight, and what the fire is giving off now."""
    h = y - slope(x, z)
    n = fbm3(x * 0.050 + 1.0, y * 0.11, z * 0.050 + 5.0, 3)
    s = 0.55 + 1.1 * n
    # sheets
    w = 1.6 * (noise3(x * 0.035, 3.0, z * 0.035) - 0.5)
    a1 = (h - 3.4 - 3.0 * w) / 1.3; a2 = (h - 8.5 - 4.0 * w) / 2.0; a3 = (h - 15.0 - 5.0 * w) / 2.2
    lay = 0.9 * math.exp(-a1 * a1) + 0.8 * math.exp(-a2 * a2) + 0.8 * math.exp(-a3 * a3)
    lay *= sstep(0.25, 0.75, fbm3(x * 0.06 + 20.0, y * 0.1, z * 0.06, 2))
    s += lay
    s += 2.2 * sstep(14.0, 44.0, h)                 # it thickens upward, into the crowns
    # the plume: rises off the flame, leans with the drift, spreads
    if h < 30.0 and h > -0.5:
        hh = max(h, 0.0)
        p = fmap(PL, x - WINDX * hh * (1.0 + 0.02 * hh), z - WINDZ * hh, FMS * FCK)
        if p > 0.0:
            tb = fbm3(x * 0.55, y * 0.35 - 2.0, z * 0.55, 3)
            s += 30.0 * p * sstep(0.25, 0.75, tb) * math.exp(-hh / 9.0) * sstep(0.0, 0.7, hh)
    return SIG0 * s

@njit(**NJ)
def phase(c):
    g = 0.74
    a = (1 - g * g) / (1 + g * g - 2 * g * c) ** 1.5
    g = -0.20
    b = (1 - g * g) / (1 + g * g - 2 * g * c) ** 1.5
    return (0.70 * a + 0.30 * b) / (4 * math.pi)

@njit(**NJ)
def air(ox, oy, oz, dx, dy, dz, tend, jit, NST, SM, U0, V0, DU, PL, GW, out):
    """in-scatter (sun part in out[0:3], sky part in out[3:6]) and transmittance out[6]"""
    tend = min(tend, 190.0)
    ph = phase(dx * SUNX + dy * SUNY + dz * SUNZ) * 4 * math.pi
    Tr = 1.0
    sr = 0.0; ar = 0.0
    # steps lengthen with distance
    kk = 1.035
    t = 0.0
    dt0 = tend * (kk - 1.0) / (kk ** NST - 1.0)
    dt = dt0
    for i in range(NST):
        tm = t + dt * jit
        x = ox + dx * tm; y = oy + dy * tm; z = oz + dz * tm
        sg = smoke(x, y, z, PL)
        e = math.exp(-sg * dt)
        w = Tr * (1.0 - e)
        if w > 1e-5:
            v = sun_vol(x, y, z, SM, U0, V0, DU, 1.0)
            if v > 0.0: v *= gobo(x, y, z, GW)
            sr += w * v
            hgt = sstep(-4.0, 30.0, y - slope(x, z))
            ar += w * (0.55 + 0.9 * hgt)
        Tr *= e
        t += dt; dt *= kk
    out[0] = sr * ph; out[1] = ar; out[2] = Tr


# ------------------------------------------------------------------ flame
@njit(**NJ)
def flames(ox, oy, oz, dx, dy, dz, tend, jit, HF, HC, out):
    """add the light of the flames along a ray.  They are thin: you see through them."""
    out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
    tend = min(tend, 120.0)
    cs = FMS * FCK
    t = 1.0
    stp = cs * 0.8
    while t < tend:
        x = ox + dx * t; z = oz + dz * t
        hc = fmap(HC, x, z, cs)
        if hc > 0.0:
            y = oy + dy * t
            g = ground0(x, z)
            if y - g < hc * 1.9 + 0.3 + abs(dy) * stp and y - g > -0.3 - abs(dy) * stp:
                fs = 0.006 * (1.0 + t / 22.0)
                n = int(stp / fs) + 1
                fs = stp / n
                for k in range(n):
                    tt = t + (k + jit) * fs
                    if tt > tend: break
                    x = ox + dx * tt; y = oy + dy * tt; z = oz + dz * tt
                    h = y - ground0(x, z) - 0.012
                    if h < 0.0 or h > 1.4: continue
                    # a tongue leans with the drift and wanders as it climbs
                    wx = (noise3(x * 6.0, y * 4.0 - 3.0, z * 6.0) - 0.5) * 0.42 * h + 0.14 * h \
                        + (noise3(x * 23.0, y * 11.0, z * 23.0) - 0.5) * 0.10 * h
                    wz = (noise3(x * 6.0 + 31.0, y * 4.0, z * 6.0 + 17.0) - 0.5) * 0.42 * h \
                        + (noise3(x * 23.0 + 9.0, y * 11.0, z * 23.0 + 4.0) - 0.5) * 0.10 * h
                    hf = fmap(HF, x - wx, z - wz, FMS)
                    if hf <= 0.012: continue
                    hf *= 0.45 + 1.15 * noise3(x * 15.0, y * 6.0 - 7.0, z * 15.0)
                    rel = (hf - h) / hf
                    st = noise3(x * 70.0, y * 16.0 - 1.0, z * 70.0)
                    if rel <= 0.0:
                        # the tip, come loose
                        q = 0.30 - abs(rel + 0.38)
                        if q > 0.0 and st > 0.62:
                            e = q * (st - 0.62) * fs * 260.0
                            out[0] += e; out[1] += e * 0.30; out[2] += e * 0.02
                        continue
                    den = rel ** 1.35 * (0.30 + 1.5 * st * st)
                    low = sstep(0.14, 0.0, h)                 # at the root it is thin and blue-clear; the light starts a little up
                    den *= 1.0 - 0.6 * low
                    e = den * fs * 70.0
                    hot = rel
                    out[0] += e * (0.80 + 2.4 * hot)
                    out[1] += e * (0.17 + 0.35 * hot + 1.55 * hot * hot)
                    out[2] += e * (0.008 + 0.62 * hot * hot * hot)
        t += stp

# ------------------------------------------------------------------ surfaces
@njit(**NJ)
def bark_detail(ca, sa, yr, R, sd, fp, kind):
    """the small relief of the bark, 0 (crevice) .. 1 (face), built only from what a pixel this size can hold"""
    if kind == 0:
        h = 0.34 * (1.0 - abs(2.0 * noise3(ca * R * 11.0 + sd, yr * 1.0, sa * R * 11.0) - 1.0)) * sstep(0.06, 0.02, fp)
        h += 0.26 * (1.0 - abs(2.0 * noise3(ca * R * 31.0, yr * 2.4 + sd, sa * R * 31.0) - 1.0)) * sstep(0.022, 0.007, fp)
        h += 0.24 * noise3(ca * R * 85.0, yr * 4.5, sa * R * 85.0) * sstep(0.008, 0.0025, fp)
        h += 0.16 * noise3(ca * R * 250.0, yr * 12.0, sa * R * 250.0) * sstep(0.0028, 0.0009, fp)
        return h
    h = 0.5 * (1.0 - abs(2.0 * noise3(ca * R * 40.0 + sd, yr * 4.0, sa * R * 40.0) - 1.0)) * sstep(0.016, 0.005, fp)
    h += 0.5 * noise3(ca * R * 140.0, yr * 16.0, sa * R * 140.0) * sstep(0.005, 0.0016, fp)
    return h

@njit(**NJ)
def bark_at(x, y, z, T, i, fp):
    yr = y - T[i, 3]
    dx = x - T[i, 0] - T[i, 9] * yr; dz = z - T[i, 1] - T[i, 10] * yr
    rr = math.sqrt(dx * dx + dz * dz) + 1e-9
    return bark_detail(dx / rr, dz / rr, yr, T[i, 2], T[i, 4], fp, int(T[i, 5]))

@njit(**NJ)
def bark_rgb(x, y, z, T, i, fp, out):
    """colour of bark in out[0:3]; how deep in a furrow (0 ridge .. 1 furrow) in out[3]; char in out[4]"""
    yr = y - T[i, 3]
    dx = x - T[i, 0] - T[i, 9] * yr; dz = z - T[i, 1] - T[i, 10] * yr
    rr = math.sqrt(dx * dx + dz * dz) + 1e-9
    R = T[i, 2]; kind = int(T[i, 5]); sd = T[i, 4]
    a = math.atan2(dz, dx); ca = dx / rr; sa = dz / rr
    det = bark_detail(ca, sa, yr, R, sd, fp, kind)
    if kind == 0:
        f1, f2, f3 = furrows(ca, sa, yr, R, sd)
        fur = 1.0 - sstep(0.015, 0.20, f1)
        fur2 = max(1.0 - sstep(0.0, 0.17, f2), 0.8 * (1.0 - sstep(0.0, 0.09, f3)))
        blot = fbm3(ca * R * 1.3 + sd, yr * 0.35, sa * R * 1.3, 3)
        strk = noise3(ca * R * 18.0, yr * 0.5 + sd, sa * R * 18.0)
        # cinnamon on the faces, paler and greyer where it has weathered, dark red-brown in the furrows
        r = 0.40 + 0.16 * blot; g = 0.175 + 0.065 * blot; b = 0.082 + 0.02 * blot
        k = 0.80 + 0.40 * strk
        r *= k; g *= k; b *= k
        wth = sstep(0.42, 0.70, fbm3(ca * R * 3.1, yr * 0.9 + sd, sa * R * 3.1, 3)) * (1.0 - fur)
        r += (0.40 - r) * 0.60 * wth; g += (0.31 - g) * 0.60 * wth; b += (0.24 - b) * 0.60 * wth
        dk = (1.0 - 0.80 * fur) * (1.0 - 0.50 * fur2 * (1.0 - fur))
        dk *= 0.52 + 0.80 * det
        r *= dk; g *= dk * (1.0 - 0.10 * fur); b *= dk * (1.0 - 0.15 * fur)
        # char: black up from the ground on the side the fires come from, and in the scar
        ch = sstep(4.5, 0.3, yr + 3.0 * (fbm3(ca * R * 0.9 + 5.0, yr * 0.25, sa * R * 0.9, 3) - 0.5)
                   - 2.6 * math.cos(a - T[i, 6]))
        ch *= 0.25 + 0.75 * sstep(0.35, 0.60, fbm3(ca * R * 2.2, yr * 0.7, sa * R * 2.2 + sd, 3) + 0.25 * (1 - fur))
        sw = scar_w(yr, a, T, i)
        if sw > 0.0: ch = max(ch, sstep(0.0, 0.16, sw))
        out[3] = fur + 0.5 * fur2 * (1 - fur) + 0.5 * sstep(0.45, 0.15, det); out[4] = ch
        if ch > 0.0:
            # alligator char: black blocks, a dull sheen on their faces
            cr = abs(2.0 * noise3(ca * R * 12.0, yr * 9.0, sa * R * 12.0) - 1.0)
            cc = 0.014 + 0.020 * sstep(0.0, 0.25, cr) + 0.020 * det
            r += (cc * 1.05 - r) * ch; g += (cc - g) * ch; b += (cc * 1.02 - b) * ch
    else:
        sp = 0.075 if kind == 1 else 0.05; k = R / sp
        n = noise3(ca * k + sd * 3.1, yr * (0.9 if kind == 1 else 0.55), sa * k - sd * 1.7)
        fur = 1.0 - sstep(0.0, 0.32, abs(2.0 * n - 1.0))
        blot = fbm3(ca * R * 4.0 + sd, yr * 0.8, sa * R * 4.0, 3)
        if kind == 1:
            r = 0.27 + 0.10 * blot; g = 0.17 + 0.06 * blot; b = 0.12 + 0.04 * blot
        else:
            r = 0.25 + 0.09 * blot; g = 0.235 + 0.085 * blot; b = 0.215 + 0.08 * blot
        dk = (1.0 - 0.6 * fur) * (0.62 + 0.7 * det)
        r *= dk; g *= dk; b *= dk
        ch = sstep(2.6, 0.2, yr + 2.0 * (fbm3(ca * R * 3.0 + 5.0, yr * 0.5, sa * R * 3.0, 2) - 0.5))
        ch *= sstep(0.3, 0.6, fbm3(ca * R * 6.0, yr * 1.5, sa * R * 6.0 + sd, 2))
        cc = 0.018 + 0.03 * det
        r += (cc - r) * ch; g += (cc - g) * ch; b += (cc - b) * ch
        out[3] = fur + 0.4 * sstep(0.45, 0.15, det); out[4] = ch
    out[0] = r; out[1] = g; out[2] = b

@njit(**NJ)
def needles(x, z, fp):
    """dry needles and scale-leaf sprays lying every way: 0..1, where 1 is a needle catching light"""
    if fp > 0.006: return 0.5
    v = 0.0
    for k in range(3):
        an = 0.4 + 1.05 * k + 0.9 * noise3(x * 0.45, 7.0 + k, z * 0.45)
        c = math.cos(an); s = math.sin(an)
        u = x * c + z * s; w = -x * s + z * c
        n = noise3(u * 24.0 + 7.0 * k, 0.3 + k, w * 330.0)
        m = noise3(u * 9.0, 4.0 + k, w * 9.0)
        q = sstep(0.60, 0.80, n) * sstep(0.35, 0.6, m)
        if q > v: v = q
    f = sstep(0.006, 0.002, fp)
    return 0.5 + (v - 0.28) * f

@njit(**NJ)
def duff_h(x, z, fp):
    h = 0.006 * noise3(x * 11.0, 0.2, z * 11.0) * sstep(0.10, 0.03, fp) + 0.004 * noise3(x * 37.0, 0.7, z * 37.0) * sstep(0.03, 0.01, fp)
    h += 0.004 * noise3(x * 110.0, 1.3, z * 110.0) * sstep(0.010, 0.003, fp)
    h += 0.0030 * needles(x, z, fp)
    return h

@njit(**NJ)
def fire_light(px, py, pz, nx, ny, nz, LT, shad, T, GI, out):
    """the flames as lamps.  LT[k] = x, y, z, power"""
    r = 0.0
    for k in range(LT.shape[0]):
        lx = LT[k, 0] - px; ly = LT[k, 1] - py; lz = LT[k, 2] - pz
        d2 = lx * lx + ly * ly + lz * lz
        if d2 > 900.0: continue
        dl = math.sqrt(d2) + 1e-9
        c = (lx * nx + ly * ny + lz * nz) / dl
        if c <= 0.0: continue
        e = LT[k, 3] * c / (d2 + 0.12)
        if e < 2e-5: continue
        if shad and e > 4e-4:
            e *= shadow(px, py, pz, lx / dl, ly / dl, lz / dl, dl - 0.15, 9.0, T, GI)
        r += e
    out[0] = r; out[1] = r * 0.42; out[2] = r * 0.085


@njit(parallel=True, **NJ)
def expose(W, H, F, SS, NST, T, GI, SM, U0, V0, DU, HF, HC, PL, BM, LT, GW, seed, OX, OY, WF, HFULL):
    IMG = np.zeros((H, W, 3), np.float32)       # surfaces and sky-lit smoke
    BMS = np.zeros((H, W, 3), np.float32)       # sun-lit smoke (softened afterwards)
    FLM = np.zeros((H, W, 3), np.float32)       # flame
    cp = math.cos(PITCH); spi = math.sin(PITCH)
    oy0 = ground0(CAMX, CAMZ) + EYE
    for j in prange(H):
        o6 = np.zeros(8); bo = np.zeros(6); fo = np.zeros(3); fl = np.zeros(3)
        s = (seed + j * 7919) & 0xffffffff
        for i in range(W):
            cr = 0.0; cg = 0.0; cb = 0.0
            br = 0.0; fr = 0.0; fg = 0.0; fb = 0.0
            for sy in range(SS):
                for sx in range(SS):
                    s, r1 = lcg(s); s, r2 = lcg(s); s, r3 = lcg(s)
                    u = (i + OX + (sx + r1) / SS - WF * 0.5) / F
                    v = -(j + OY + (sy + r2) / SS - HFULL * VC) / F
                    dx = u; dy = v * cp + spi; dz = cp - v * spi
                    dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                    dx /= dl; dy /= dl; dz /= dl
                    t, m, ti = march(CAMX, oy0, CAMZ, dx, dy, dz, 230.0, T, GI)
                    lr = 0.0; lg = 0.0; lb = 0.0
                    if m > 0:
                        px = CAMX + dx * t; py = oy0 + dy * t; pz = CAMZ + dz * t
                        lod = 0 if t < 26.0 else 1
                        nx, ny, nz = normal(px, py, pz, t, T, GI, lod)
                        ar = 0.2; ag = 0.2; ab = 0.2; cav = 0.0; spec = 0.0
                        fp = t / F
                        if m == M_TRUNK:
                            bark_rgb(px, py, pz, T, ti, fp, bo)
                            ar = bo[0]; ag = bo[1]; ab = bo[2]; cav = min(bo[3], 1.0)
                            spec = 0.05 * bo[4]
                            # rake the fine relief
                            e = max(0.7 * fp, 0.0006)
                            h0 = bark_at(px, py, pz, T, ti, fp)
                            gx = (bark_at(px + e, py, pz, T, ti, fp) - h0) / e
                            gy = (bark_at(px, py + e, pz, T, ti, fp) - h0) / e
                            gz = (bark_at(px, py, pz + e, T, ti, fp) - h0) / e
                            gd = gx * nx + gy * ny + gz * nz
                            kb = 0.020 if T[ti, 5] == 0 else 0.007
                            nx -= kb * (gx - gd * nx); ny -= kb * (gy - gd * ny); nz -= kb * (gz - gd * nz)
                            nl = math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
                        elif m == M_GROUND or m == M_STICK or m == M_CONE or m == M_CHIP:
                            burn = fmap(BM, px, pz, FMS * 4)
                            n1 = fbm3(px * 1.1, 0.5, pz * 1.1, 3); n2 = noise3(px * 17.0, 0.4, pz * 17.0)
                            n3 = noise3(px * 70.0, 0.4, pz * 70.0)
                            if m == M_GROUND:
                                # duff: red-brown scale and tan needle, grey where it is old, darker where it holds the damp
                                ar = 0.150 + 0.085 * n1; ag = 0.092 + 0.050 * n1; ab = 0.056 + 0.030 * n1
                                gy2 = sstep(0.45, 0.75, fbm3(px * 0.37 + 9.0, 2.5, pz * 0.37, 3))
                                ar += (0.22 - ar) * 0.6 * gy2; ag += (0.19 - ag) * 0.6 * gy2; ab += (0.155 - ab) * 0.6 * gy2
                                nd = needles(px, pz, fp)
                                k = (0.62 + 0.42 * n2 + 0.34 * (n3 - 0.5) * sstep(0.012, 0.004, fp)) * (0.55 + 0.9 * nd)
                                ar *= k; ag *= k * (1.0 + 0.12 * (nd - 0.5)); ab *= k
                                e = max(0.7 * fp, 0.0005)
                                h0 = duff_h(px, pz, fp)
                                gx = (duff_h(px + e, pz, fp) - h0) / e; gz = (duff_h(px, pz + e, fp) - h0) / e
                                nx -= gx; nz -= gz
                                nl = math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
                                cav = 0.5 * sstep(0.55, 0.2, n2) * sstep(0.03, 0.01, fp)
                            elif m == M_STICK:
                                k = 0.55 + 0.9 * n3 * n2
                                ar = 0.16 * k; ag = 0.125 * k; ab = 0.095 * k
                            elif m == M_CONE:
                                ar = 0.26 + 0.10 * n3; ag = 0.19 + 0.06 * n3; ab = 0.115 + 0.03 * n3
                            else:
                                ar = 0.22 + 0.10 * n2; ag = 0.115 + 0.05 * n2; ab = 0.062 + 0.02 * n2
                                k = 0.7 + 0.6 * n3
                                ar *= k; ag *= k; ab *= k
                            if burn > 0.0:
                                # ash: black, with white where it burned clean
                                wa = sstep(0.60, 0.82, fbm3(px * 3.3, 0.9, pz * 3.3, 3)) * sstep(0.5, 1.0, burn)
                                c0 = 0.020 + 0.022 * n3 + 0.30 * wa * (0.6 + 0.4 * n2)
                                ar += (c0 * 1.02 - ar) * burn; ag += (c0 - ag) * burn; ab += (c0 * 0.98 - ab) * burn
                        elif m == M_SHIRT:
                            f = 0.86 + 0.20 * noise3(px * 30.0, py * 30.0, pz * 30.0)
                            sm = sstep(0.40, 0.75, fbm3(px * 9.0, py * 9.0, pz * 9.0, 2))      # soot
                            ar = 0.86 * f * (1 - 0.30 * sm); ag = 0.61 * f * (1 - 0.36 * sm); ab = 0.040 * f
                        elif m == M_PANTS:
                            f = 0.8 + 0.4 * noise3(px * 25.0, py * 25.0, pz * 25.0)
                            ar = 0.075 * f; ag = 0.085 * f; ab = 0.055 * f
                        elif m == M_BOOT:
                            ar = 0.035; ag = 0.030; ab = 0.026
                        elif m == M_SKIN:
                            ar = 0.30; ag = 0.20; ab = 0.14
                        elif m == M_HELMET:
                            ar = 0.40; ag = 0.38; ab = 0.34; spec = 0.08
                        elif m == M_PACK:
                            f = 0.7 + 0.6 * noise3(px * 40.0, py * 40.0, pz * 40.0)
                            ar = 0.050 * f; ag = 0.046 * f; ab = 0.042 * f
                        elif m == M_TORCH:
                            ar = 0.30; ag = 0.29; ab = 0.28; spec = 0.25
                        elif m == M_SHROUD:
                            f = 0.8 + 0.3 * noise3(px * 30.0, py * 30.0, pz * 30.0)
                            ar = 0.62 * f; ag = 0.47 * f; ab = 0.10 * f
                        elif m == M_GLOVE:
                            ar = 0.30; ag = 0.22; ab = 0.13
                        ao = ambient_occ(px, py, pz, nx, ny, nz, T, GI)
                        ao *= 1.0 - 0.62 * cav
                        # the sun
                        nl = nx * SUNX + ny * SUNY + nz * SUNZ
                        sr = 0.0
                        if nl > 0.0:
                            gb = gobo(px, py, pz, GW)
                            if gb > 0.002:
                                sr = gb * nl * shadow(px + nx * 0.004, py + ny * 0.004, pz + nz * 0.004, SUNX, SUNY, SUNZ, 150.0, 55.0, T, GI)
                                sr *= 1.0 - 0.5 * cav
                        # the smoke between here and the sun dims it a little
                        hgt = sstep(0.0, 40.0, py - slope(px, pz))
                        skv = 0.5 + 0.5 * ny
                        ha = SUNX * nx + SUNZ * nz
                        hz = (0.5 + 0.5 * ha / 0.976) ** 2 * (0.55 + 0.45 * (1.0 - abs(ny)))
                        gbo = 0.5 - 0.5 * ny
                        er = E0R * sr + ao * (SKR * skv * (0.75 + 0.8 * hgt) + HZR * hz + 0.050 * gbo)
                        eg = E0G * sr + ao * (SKG * skv * (0.75 + 0.8 * hgt) + HZG * hz + 0.036 * gbo)
                        eb = E0B * sr + ao * (SKB * skv * (0.75 + 0.8 * hgt) + HZB * hz + 0.024 * gbo)
                        fire_light(px, py, pz, nx, ny, nz, LT, False, T, GI, fl)
                        er += fl[0] * ao; eg += fl[1] * ao; eb += fl[2] * ao
                        lr = ar * er / math.pi; lg = ag * eg / math.pi; lb = ab * eb / math.pi
                        if spec > 0.0 and sr > 0.0:
                            hx = SUNX - dx; hy = SUNY - dy; hzz = SUNZ - dz
                            hl = math.sqrt(hx * hx + hy * hy + hzz * hzz)
                            sp = max((nx * hx + ny * hy + nz * hzz) / hl, 0.0) ** 24
                            lr += spec * sp * E0R * sr; lg += spec * sp * E0G * sr; lb += spec * sp * E0B * sr
                        # embers in the black
                        if m == M_GROUND:
                            bq = fmap(BM, px, pz, FMS * 4)
                            if bq > 0.3 and bq < 0.985:
                                em = sstep(0.80, 0.90, noise3(px * 41.0, 5.0, pz * 41.0)) * sstep(0.55, 0.75, noise3(px * 2.3, 8.0, pz * 2.3))
                                em *= sstep(0.985, 0.80, bq)
                                lr += 1.6 * em; lg += 0.34 * em; lb += 0.03 * em
                    # the air in front of it
                    air(CAMX, oy0, CAMZ, dx, dy, dz, t, r3, NST, SM, U0, V0, DU, PL, GW, o6)
                    Tr = o6[2]
                    ca = (dx * SUNX + dz * SUNZ) / 0.976
                    wm = sstep(-0.2, 1.0, ca) ** 2 * (1.0 - 0.5 * abs(dy))
                    cr += lr * Tr + o6[1] * (SKR * 0.36 + 0.20 * wm)
                    cg += lg * Tr + o6[1] * (SKG * 0.36 + 0.14 * wm)
                    cb += lb * Tr + o6[1] * (SKB * 0.36 + 0.075 * wm)
                    br += o6[0]
                    if dy < 0.35:
                        flames(CAMX, oy0, CAMZ, dx, dy, dz, t, r3, HF, HC, fo)
                        if fo[0] > 0.0:
                            # seen through the smoke in front
                            tf = math.exp(-SIG0 * 1.2 * min(t, 60.0))
                            fr += fo[0] * tf; fg += fo[1] * tf; fb += fo[2] * tf
            k = 1.0 / (SS * SS)
            IMG[j, i, 0] = cr * k; IMG[j, i, 1] = cg * k; IMG[j, i, 2] = cb * k
            BMS[j, i, 0] = br * k
            FLM[j, i, 0] = fr * k; FLM[j, i, 1] = fg * k; FLM[j, i, 2] = fb * k
    return IMG, BMS, FLM


# ------------------------------------------------------------------ the plan of the grove and of the fire
def plant(rng):
    """T rows: x, z, R, base y, seed, kind, scar angle, scar half-width, scar height, lean x, lean z, height"""
    rows = []
    #            x      z     R    scar: angle (deg, 0 = +x, -90 faces the lens), half-width, height
    monarchs = [(-7.3,  9.2, 3.05,  -60, 0.0, 0.0),
                (6.4,  25.0, 3.45, -122, 1.25, 7.5),
                (-19.0, 41.5, 2.75, -75, 0.75, 4.0),
                (1.5,  58.0, 3.10, -100, 1.0, 6.0),
                (19.0, 46.0, 2.95, -140, 0.0, 0.0),
                (-39.0, 62.0, 3.30, -40, 0.9, 5.0),
                (12.0, 84.0, 2.85, -90, 0.0, 0.0),
                (-7.0, 98.0, 3.00, -90, 0.8, 5.0),
                (33.0, 76.0, 3.10, -90, 0.0, 0.0),
                (-38.0, 92.0, 2.9, -90, 0.0, 0.0),
                (24.0, 112.0, 3.2, -90, 0.0, 0.0)]
    for k, (x, z, R, sa, hw, hs) in enumerate(monarchs):
        rows.append([x, z, R, 0, 3.0 + 7.31 * k, 0, math.radians(sa), hw, hs,
                     0.006 * rng.standard_normal(), 0.006 * rng.standard_normal(), 90.0])
    # two of them were first drawn nearer the middle; the small trees were set out round the old places
    moved = {2: (-17.5, 41.0), 5: (-50.0, 64.0)}
    rows[2][0], rows[2][1] = -16.5, 41.0
    rows[5][0], rows[5][1] = -33.0, 63.0
    n = 0
    while n < 78:
        x = rng.uniform(-55, 55); z = rng.uniform(5, 135)
        if abs(x) > 0.75 * z + 8: continue
        if abs(x) < 2.2 and z < 22: continue                      # keep the walk in clear
        if abs(x - FIGX) < 2.0 and abs(z - FIGZ) < 3.0: continue
        R = rng.choice([0.20, 0.28, 0.36, 0.45, 0.60, 0.80], p=[0.20, 0.25, 0.22, 0.16, 0.11, 0.06])
        if z < 16 and R > 0.4: continue
        ok = True
        for r in rows:
            dd = math.hypot(x - r[0], z - r[1])
            if dd < (r[2] * 2.1 + 1.2 if r[5] == 0 else r[2] + R + 1.4): ok = False; break
        if not ok: continue
        kind = 1 if rng.random() < 0.45 else 2
        rows.append([x, z, R, 0, rng.uniform(0, 50), kind, 0, 0, 0,
                     0.014 * rng.standard_normal(), 0.014 * rng.standard_normal(), 90.0])
        n += 1
    for k, (x, z) in moved.items():
        rows[k][0], rows[k][1] = x, z
    nm = len(monarchs)
    rows = rows[:nm] + [r for r in rows[nm:] if all(math.hypot(r[0] - m[0], r[1] - m[1]) > m[2] * 2.1 + 1.2 for m in rows[:nm])]
    # and we take the sun's side of the walker to be clear of poles, so that the shirt is lit
    ul = math.hypot(SUNX, SUNZ); sx, sz = SUNX / ul, SUNZ / ul
    def clear(r):
        dx, dz = r[0] - FIGX, r[1] - FIGZ
        return dx * sx + dz * sz < 0 or abs(dx * sz - dz * sx) > r[2] + 1.0
    rows = rows[:nm] + [r for r in rows[nm:] if clear(r)]
    T = np.array(rows, np.float64)
    return T

def seat(T):
    """stand each tree on the ground, and index them on a grid"""
    for i in range(T.shape[0]):
        T[i, 3] = ground0(T[i, 0], T[i, 1]) - (0.5 if T[i, 5] == 0 else 0.2)
    GI = -np.ones((GNZ, GNX, KMAX), np.int32)
    for i in range(T.shape[0]):
        rb = T[i, 2] * (1.75 if T[i, 5] == 0 else 1.45) + MARG + 0.55 + 45.0 * math.hypot(T[i, 9], T[i, 10])
        i0 = int((T[i, 0] - rb - GX0) / CS); i1 = int((T[i, 0] + rb - GX0) / CS)
        j0 = int((T[i, 1] - rb - GZ0) / CS); j1 = int((T[i, 1] + rb - GZ0) / CS)
        for cj in range(max(j0, 0), min(j1, GNZ - 1) + 1):
            for ci in range(max(i0, 0), min(i1, GNX - 1) + 1):
                qx = min(max(T[i, 0], GX0 + ci * CS), GX0 + (ci + 1) * CS)
                qz = min(max(T[i, 1], GZ0 + cj * CS), GZ0 + (cj + 1) * CS)
                if math.hypot(qx - T[i, 0], qz - T[i, 1]) > rb: continue
                for q in range(KMAX):
                    if GI[cj, ci, q] < 0:
                        GI[cj, ci, q] = i; break
                else:
                    raise RuntimeError('cell full')
    return GI

def sun_uv(x, y, z):
    ul = math.hypot(SUNX, SUNZ)
    return x * SUNZ / ul - z * SUNX / ul, -x * SUNY * SUNX / ul + y * ul - z * SUNY * SUNZ / ul

# where we want the sun to land: x, z, height above the ground, half-width across, half-height
TARGETS = [(FIGX, FIGZ, 1.0, 1.3, 0.9), (-1.5, 10.5, 0.2, 1.6, 0.45), (3.6, 23.5, 6.0, 1.3, 2.2), (-6.0, 17.0, 0.2, 1.3, 0.4),
           (-0.5, 7.0, 0.1, 0.9, 0.25), (2.6, 9.6, 0.1, 1.1, 0.3), (-2.5, 12.8, 0.1, 0.9, 0.22), (4.4, 13.6, 0.1, 1.2, 0.3), (1.0, 16.0, 0.1, 1.0, 0.25), (-4.0, 5.2, 0.1, 0.8, 0.2), (3.8, 5.6, 0.1, 0.8, 0.2), (-3.0, 33.0, 0.3, 2.0, 0.6), (11.0, 33.0, 0.5, 1.6, 0.8), (2.0, 44.0, 3.0, 2.2, 2.0),
           (-13.0, 30.0, 6.0, 2.0, 2.5), (-2.0, 26.0, 9.0, 1.6, 1.6), (14.0, 60.0, 8.0, 3.0, 3.0), (-4.0, 70.0, 10.0, 3.0, 4.0)]

def open_crowns(T, GI):
    """gaps in the crowns, found where the trunks leave the sun a clear line"""
    rows = []
    for (x, z, h, ru, rv) in TARGETS:
        best = (0.0, x, z)
        for k in range(60):
            a = k * 2.399963; r = 0.42 * math.sqrt(k)
            xx = x + r * math.cos(a); zz = z + r * math.sin(a)
            y = ground0(xx, zz) + h
            o = shadow(xx, y + 0.4, zz, SUNX, SUNY, SUNZ, 170.0, 30.0, T, GI)
            if o > best[0] + 0.02: best = (o, xx, zz)
            if o > 0.97: break
        print('   gap for (%.1f, %.1f): open %.2f at (%.1f, %.1f)' % (x, z, best[0], best[1], best[2]))
        if best[0] > 0.5:
            u, v = sun_uv(best[1], ground0(best[1], best[2]) + h, best[2])
            rows.append([u, v, ru, rv])
    return np.array(rows) if rows else np.zeros((0, 4))

def seg_dist(X, Z, pts):
    d = np.full(X.shape, 1e9, np.float32)
    for (ax, az), (bx, bz) in zip(pts[:-1], pts[1:]):
        ex, ez = bx - ax, bz - az
        h = np.clip(((X - ax) * ex + (Z - az) * ez) / (ex * ex + ez * ez), 0, 1)
        d = np.minimum(d, np.hypot(X - ax - ex * h, Z - az - ez * h))
    return d

def nz(rng, shape, sig):
    a = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sig, mode='wrap')
    return a / (a.std() + 1e-9)

def light_fire(rng, T):
    """the front as it stands at 6:50.  Returns flame-height map, its coarse maximum, plume source, burn mask, lamps."""
    n = FMN
    ax = FM0X + (np.arange(n, dtype=np.float32) + 0.0) * FMS
    X, Z = np.meshgrid(ax, FM0Z + (np.arange(n, dtype=np.float32)) * FMS)
    # the main front: lit yesterday evening along the far side, backing towards the lens all night
    zf = 31.5 - 0.20 * X + 3.2 * np.sin(X * 0.16 + 0.6) + 1.6 * np.sin(X * 0.47 + 2.0)
    warp = 0.9 * nz(rng, (n, n), 45) + 0.35 * nz(rng, (n, n), 12) + 0.10 * nz(rng, (n, n), 3.5)
    phi = (Z - zf) + warp                               # > 0: burned
    # it does not climb the monarchs, and it goes round the wet places
    for i in range(T.shape[0]):
        if T[i, 5] == 0:
            dd = np.hypot(X - T[i, 0], Z - T[i, 1]) - T[i, 2] * 1.5
            phi = np.minimum(phi, dd * 2.5)
    # this morning's strip: laid from the torch, a few minutes old at the far end and seconds old at his feet
    c, s = math.cos(FIGYAW), math.sin(FIGYAW)
    hx, hz = FIGX + 0.385 * c + 0.62 * s, FIGZ - 0.385 * s + 0.62 * c        # where the torch drips
    strip = [(-17.0, 25.0), (-12.5, 23.6), (-9.0, 24.0), (-5.5, 22.4), (-2.6, 21.9), (0.2, 20.6), (hx - 0.3, hz - 0.1)]
    sd = seg_dist(X, Z, strip)
    age = np.clip((hx - X) / (hx + 17.0), 0, 1)                              # 0 at the torch, 1 at the far end
    wid = 0.05 + 0.75 * age ** 1.2
    phi2 = wid - sd + 0.25 * warp * age
    burn = np.clip(np.maximum(phi, phi2 * 1.0) / 0.22, 0, 1).astype(np.float32)
    # flame stands on the edge of the black
    tong = np.clip(0.42 + 0.60 * nz(rng, (n, n), 1.5) + 0.34 * nz(rng, (n, n), 0.75) + 0.30 * nz(rng, (n, n), 5.0), 0, None) ** 1.5
    act = np.clip(0.42 + 0.80 * nz(rng, (n, n), 30) + 0.45 * nz(rng, (n, n), 8), 0, 1.6)
    band1 = np.exp(-((phi - 0.03) / 0.085) ** 2)
    band2 = np.exp(-((phi2 - 0.02) / (0.045 + 0.035 * age)) ** 2)
    HF = 0.34 * band1 * tong * act + (0.26 + 0.20 * age) * band2 * tong * np.clip(act + 0.35, 0, 1.3) * (1.0 - 0.6 * (age > 0.97))
    # a few things still burning in the black: a stump hole, the end of a limb
    for (sx, sz, rad, hh) in [(-6.0, 40.5, 0.28, 0.5), (11.5, 36.0, 0.22, 0.4), (-16.0, 37.0, 0.25, 0.4), (4.0, 46.0, 0.3, 0.45), (15.5, 31.2, 0.2, 0.3)]:
        HF += hh * np.exp(-((X - sx) ** 2 + (Z - sz) ** 2) / rad ** 2) * tong
    # the torch itself drips fire
    HF += 0.16 * np.exp(-((X - hx) ** 2 + (Z - hz) ** 2) / 0.05 ** 2)
    HF = 0.62 * np.tanh(HF / 0.62)
    HF[HF < 0.03] = 0
    HF = HF.astype(np.float32)
    m = n // FCK
    HC = HF.reshape(m, FCK, m, FCK).max(axis=(1, 3))
    HC = maximum_filter(HC, 3).astype(np.float32)
    PL = HF.reshape(m, FCK, m, FCK).mean(axis=(1, 3))
    PL = gaussian_filter(PL, 1.6).astype(np.float32)
    BM = burn.reshape(n // 4, 4, n // 4, 4).mean(axis=(1, 3)).astype(np.float32)
    # lamps: one per metre of fire or so
    q = 35
    P = (HF ** 1.5).reshape(n // q, q, n // q, q).sum(axis=(1, 3)) * FMS * FMS
    jj, ii = np.nonzero(P > 1e-4)
    LT = np.zeros((len(jj), 4))
    for k, (j, i) in enumerate(zip(jj, ii)):
        x = FM0X + (i + 0.5) * q * FMS; z = FM0Z + (j + 0.5) * q * FMS
        LT[k] = (x, ground0(x, z) + 0.28, z, P[j, i] * 520.0)
    return HF, HC, PL, BM, LT


# ------------------------------------------------------------------ the print
EXPOSURE = 2.9
ADAPT = 0.30

def develop(lin, scale, rng, exposure):
    """no lens.  The eye's adaptation, a restrained curve, then a fine tooth."""
    H, W = lin.shape[:2]
    lin = np.maximum(lin, 0.0) * exposure
    L = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    lg = np.log(np.clip(L, 1e-4, 80.0))
    base = gaussian_filter(lg, 70.0 * scale)
    gain = np.clip(np.exp(-ADAPT * (base - math.log(0.20))), 0.3, 4.0)
    lin = lin * gain[..., None]; L = L * gain
    wp = 5.0
    f = lambda x: x * (1.0 + x / wp ** 2) / (1.0 + x)
    y = 0.72 * lin * (f(L) / np.maximum(L, 1e-9))[..., None] + 0.28 * f(lin)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-9), 1 / 2.4) - 0.055)
    y = y + 0.30 * (y * y * (3 - 2 * y) - y)
    y = 0.022 + 0.962 * y
    g = gaussian_filter(rng.standard_normal((H, W)), 0.6 * max(scale, 0.5)); g /= g.std()
    g2 = gaussian_filter(rng.standard_normal((H, W)), 2.2 * max(scale, 0.5)); g2 /= g2.std()
    lum = y.mean(-1, keepdims=True)
    amp = 0.004 + 0.008 * np.sqrt(np.clip(4 * lum * (1 - lum), 0, 1))
    y = y + amp * (0.85 * g[..., None] + 0.35 * g2[..., None])
    return np.clip(y, 0, 1)


def sign(img, scale, text):
    """three rings, like a core seen end-on, lower left, and the age of the oldest tree in the picture."""
    H, W = img.shape[:2]
    u = max(scale, 0.2)
    cw, ch = int(360 * u), int(74 * u)
    k = 4
    im = Image.new('L', (cw * k, ch * k), 0); d = ImageDraw.Draw(im)
    lw = max(int(2.8 * u * k), 2)
    cx, cy = 36 * u * k, 37 * u * k
    for r in (7, 16, 25):
        rr = r * u * k
        d.ellipse([cx - rr, cy - rr * 0.94, cx + rr, cy + rr * 0.94], outline=255, width=lw)
    font = ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf', int(38 * u * k))
    d.text((cx + 44 * u * k, cy - 23 * u * k), text, fill=255, font=font)
    im = im.resize((cw, ch), Image.LANCZOS)
    m = np.asarray(im, np.float64) / 255.0
    x0 = int(70 * u); y0 = H - ch - int(64 * u)
    reg = img[y0:y0 + ch, x0:x0 + cw]
    lum = reg.mean()
    if lum > 0.30:
        out = reg * (1.0 - 0.50 * m[..., None])
    else:
        out = reg + (0.34 - reg) * 0.75 * m[..., None]
    img[y0:y0 + ch, x0:x0 + cw] = np.clip(out, 0, 1)
    return img


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    scale = float(args[0]) if len(args) > 0 else 1.0
    out = args[1] if len(args) > 1 else __file__.replace('.py', '.png')
    W = int(round(FW * scale)); H = int(round(FH * scale)); F = FOC * scale
    WF, HFULL = W, H
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    T = plant(rng)
    GI = seat(T)
    HF, HC, PL, BM, LT = light_fire(rng, T)
    GW = open_crowns(T, GI)
    print('trees %d, lamps %d, gaps %d  %.0fs' % (T.shape[0], LT.shape[0], GW.shape[0], time.time() - t0), flush=True)
    # shadow map: the sun's view of the grove
    DU = 0.07 if scale >= 0.6 else 0.12
    U0, U1, V0, V1, WS = -150.0, 110.0, -45.0, 80.0, 190.0
    NU = int((U1 - U0) / DU); NV = int((V1 - V0) / DU)
    SM = make_shadow_map(NU, NV, U0, V0, DU, WS, T, GI)
    print('shadow map %dx%d  %.0fs' % (NU, NV, time.time() - t0), flush=True)
    OX = OY = 0; crop = None
    for a in sys.argv:
        if a.startswith('--crop='):
            crop = [int(v) for v in a[7:].split(',')]
            OX, OY = crop[0], crop[1]; W = crop[2] - crop[0]; H = crop[3] - crop[1]
    SS = 2 if scale >= 0.6 else 2
    NST = 110 if scale >= 0.6 else 80
    for a in sys.argv:
        if a.startswith('--ss='): SS = int(a[5:])
    IMG, BMS, FLM = expose(W, H, F, SS, NST, T, GI, SM, U0, V0, DU, HF, HC, PL, BM, LT, GW, SEED % 9973, OX, OY, WF, HFULL)
    print('exposed  %.0fs' % (time.time() - t0), flush=True)
    lin = IMG.astype(np.float64)
    # smoke lights the smoke beside it: the shafts have no knife edge
    b = BMS[..., 0].astype(np.float64)
    b = 0.70 * b + 0.22 * gaussian_filter(b, 7.0 * scale) + 0.08 * gaussian_filter(b, 40.0 * scale)
    lin += b[..., None] * np.array([E0R, E0G, E0B]) * 0.42
    # flame, with the little glow the eye puts round it
    fl = FLM.astype(np.float64)
    gl = np.stack([gaussian_filter(fl[..., c], 5.0 * scale) for c in range(3)], -1)
    gl2 = np.stack([gaussian_filter(fl[..., c], 24.0 * scale) for c in range(3)], -1)
    fl = np.stack([gaussian_filter(fl[..., c], 0.55 * scale) for c in range(3)], -1)
    lin += 0.84 * fl + 0.11 * gl + 0.05 * gl2
    if '--keep' in sys.argv: np.save(out.replace('.png', '_lin.npy'), lin.astype(np.float32))
    img = develop(lin, scale, rng, EXPOSURE)
    if crop is None: img = sign(img, scale, OLDEST)
    Image.fromarray((img * 255 + rng.random(img.shape)).clip(0, 255).astype(np.uint8)).save(out)
    print('saved', out, '%dx%d  %.0fs' % (W, H, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
