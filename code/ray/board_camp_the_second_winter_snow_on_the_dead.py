"""
Board Camp, the second winter, snow on the dead

Ray Avila.

Site record.  A grove on a north-west slope at 2,000 m, burned at high severity in September, in a
wind, after a hundred years without fire.  This is the second February after.  We stand on a bench
and look across a draw at the slope the fire ran up.  Nothing in the frame is alive.  The big trees
kept their bark and lost everything else: they stand as black columns, some snapped at forty or
fifty metres, a few limbs left.  The fir and pine that grew in thick while the fires were being put
out are poles now, something over two thousand of them on the slope.  Snow sixty
centimetres, two days old, wind from the left.  Cloud is down on the ridge.  One person, crossing
the draw; their tracks start at the lens.

Usage:  python board_camp_the_second_winter_snow_on_the_dead.py [scale=1.0] [out.png] [--crop=x0,y0,x1,y1]
        scale 1.0 is 4200 x 2800.

Built as signed-distance solids in metres and lit by calculation:
  trunks  - each a solid: flare, furrowed char, a broken top, the stubs of limbs
  snow    - a heightfield with drifts, wells round the trunks, buried logs and one line of tracks
  light   - an overcast sky as a dome, a little brighter to the south, and what the snow throws back up
  cloud   - a density marched along every ray, thickening with height
"""
import sys, math, time, warnings
import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaWarning
warnings.simplefilter('ignore', NumbaWarning)
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

SEED = 20230211
FW, FH = 4200, 2800
FOC = 3500.0                    # pixels per unit tangent at scale 1
VC = 0.60                       # the level line, as a fraction of the height from the top
CAMX, CAMZ, EYE = 0.0, 0.0, 1.62
OLDEST = '2,700 yr'

# the brighter part of the overcast: high, ahead and to the right
_l = math.sqrt(0.45 ** 2 + 0.80 ** 2 + 0.40 ** 2)
BRX, BRY, BRZ = 0.45 / _l, 0.80 / _l, 0.40 / _l
WNX, WNZ = 0.94, 0.34           # the wind blows this way
SIG0 = 0.0011                   # cloud, per metre, at the level of the lens

# grid of trees
CS = 2.5
GX0, GZ0 = -260.0, -12.0
GNX, GNZ = 208, 190
KMAX = 10
MARG = 1.2

M_SNOW, M_TRUNK = 1, 2
M_SHIRT, M_PANTS, M_BOOT, M_SKIN, M_HELMET, M_PACK, M_TORCH, M_GLOVE, M_SHROUD = 10, 11, 12, 13, 14, 15, 16, 17, 18
NJ = dict(cache=True, fastmath=True, error_model='numpy')

# the walker, and the way they came
FIGX, FIGZ, FIGYAW = 5.2, 63.6, 0.12
NPATH = 6
PATHX = (0.9, -0.6, 1.0, 2.8, 4.4, 5.2)
PATHZ = (0.5, 16.0, 31.0, 45.0, 56.0, 63.6)


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


# ------------------------------------------------------------------ the figure
@njit(**NJ)
def sd_fig(lx, ly, lz):
    """feet at the origin, facing +z, his right is +x.  Mid-stride, head down."""
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
    dt = smin(dt, sd_cap(lx, ly, lz, 0.275, 1.19, 0.03, 0.27, 0.96, -0.04, 0.043), 0.025)
    dt += 0.005 * math.sin(ly * 43.0 + lx * 13.0 + 2.5 * math.sin(lx * 29.0 + lz * 17.0)) * sstep(1.0, 1.2, ly)
    if dt < d: d = dt; m = M_SHIRT
    # gloves
    dh = sd_ell(lx + 0.245, ly - 0.905, lz - 0.085, 0.040, 0.058, 0.046)
    dh = min(dh, sd_ell(lx - 0.27, ly - 0.915, lz + 0.05, 0.040, 0.058, 0.046))
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
    return d, m

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


# ------------------------------------------------------------------ the ground under the snow
@njit(**NJ)
def slope(x, z):
    """the bench we stand on, the draw, and the slope the fire ran up"""
    zz = z + 0.22 * x
    q = (zz - 80.0) / 12.0
    sp = q if q > 20.0 else math.log(1.0 + math.exp(q))
    return -0.05 * zz + 0.45 * 12.0 * sp + 0.018 * x

@njit(**NJ)
def tracks(x, z):
    """how far the snow is pressed down here by a boot: 0..1"""
    if z < 0.0 or z > FIGZ + 0.4: return 0.0
    best = 9.0; sb = 0.0; lat = 0.0
    acc = 0.0
    for k in range(NPATH - 1):
        ax = PATHX[k]; az = PATHZ[k]
        ex = PATHX[k + 1] - ax; ez = PATHZ[k + 1] - az
        ln = math.sqrt(ex * ex + ez * ez)
        h = ((x - ax) * ex + (z - az) * ez) / (ln * ln)
        hc = min(max(h, 0.0), 1.0)
        qx = x - ax - ex * hc; qz = z - az - ez * hc
        dd = math.sqrt(qx * qx + qz * qz)
        if dd < best:
            best = dd; sb = acc + hc * ln; lat = (qx * ez - qz * ex) / ln
        acc += ln
    if best > 0.5: return 0.0
    lat += 0.10 * math.sin(sb * 0.31)
    n = math.floor(sb / 0.74 + 0.5)
    side = 1.0 if (int(n) & 1) == 0 else -1.0
    ds = sb - n * 0.74
    dl = lat - side * 0.13
    e = (ds / 0.17) ** 2 + (dl / 0.075) ** 2
    if e > 1.9: return 0.0
    return 1.0 - sstep(0.35, 1.9, e)

@njit(**NJ)
def off_track(x, z):
    """0 on the line the walker took, 1 well away from it: they kept to the easy ground"""
    if z > FIGZ + 12.0 or abs(x) > 22.0: return 1.0
    best = 99.0
    for k in range(NPATH - 1):
        ax = PATHX[k]; az = PATHZ[k]
        ex = PATHX[k + 1] - ax; ez = PATHZ[k + 1] - az
        h = min(max(((x - ax) * ex + (z - az) * ez) / (ex * ex + ez * ez), 0.0), 1.0)
        dd = math.sqrt((x - ax - ex * h) ** 2 + (z - az - ez * h) ** 2)
        if dd < best: best = dd
    return sstep(1.5, 9.0, best)

@njit(**NJ)
def ground0(x, z):
    """the top of the snow, without the wells at the trees"""
    ot = off_track(x, z)
    g = slope(x, z) + 9.0 * (fbm3(x * 0.011 + 3.1, 0.5, z * 0.011 + 7.7, 3) - 0.5) * sstep(20.0, 140.0, z) \
        + 0.8 * (fbm3(x * 0.07, 1.5, z * 0.07, 2) - 0.5) * (0.25 + 0.75 * ot)
    # drifts, laid along the wind
    u = x * WNX + z * WNZ; w = -x * WNZ + z * WNX
    g += 0.16 * (noise3(u * 0.16, 2.5, w * 0.55) - 0.5) + 0.05 * (noise3(u * 0.6, 3.5, w * 1.7) - 0.5)
    # what fell and is under it
    cs = 14.0
    bi = math.floor(x / cs); bj = math.floor(z / cs)
    for oi in range(-1, 2):
        for oj in range(-1, 2):
            ii = int(bi + oi); jj = int(bj + oj)
            r1 = hash3(ii, jj, 51)
            if r1 > 0.42: continue
            cx = (ii + 0.5 + 0.8 * (hash3(ii, jj, 52) - 0.5)) * cs
            cz = (jj + 0.5 + 0.8 * (hash3(ii, jj, 53) - 0.5)) * cs
            an = hash3(ii, jj, 54) * 3.1416
            ln = 3.0 + 9.0 * hash3(ii, jj, 55)
            rd = 0.25 + 0.55 * hash3(ii, jj, 56) ** 2
            ux = math.cos(an); uz = math.sin(an)
            hh = (x - cx) * ux + (z - cz) * uz
            hc = min(max(hh, -ln), ln)
            qx = x - cx - ux * hc; qz = z - cz - uz * hc
            dd = math.sqrt(qx * qx + qz * qz)
            g += rd * 1.1 * sstep(rd * 2.6, 0.0, dd) ** 1.5 * ot
    # the big one, down across the draw
    hh = ((x + 46.0) * 78.0 + (z - 58.0) * 9.0) / (78.0 * 78.0 + 81.0)
    hc = min(max(hh, 0.0), 1.0)
    qx = x + 46.0 - 78.0 * hc; qz = z - 58.0 - 9.0 * hc
    dd = math.sqrt(qx * qx + qz * qz)
    g += (1.9 - 0.9 * hc) * sstep(4.6, 0.0, dd) ** 1.6
    return g

@njit(**NJ)
def ground(x, z):
    g = ground0(x, z)
    t = tracks(x, z)
    if t > 0.0: g -= 0.14 * t
    return g


# ------------------------------------------------------------------ a dead tree
# T[i] = x, z, R, base y, seed, kind (0 monarch, 1 pole), scar angle, scar half-width, scar height, lean x, lean z,
#        height it stands to now, height it had, number of limbs left
@njit(**NJ)
def trunk_r(yr, R, kind, hfull):
    yy = max(yr, -1.5)
    if kind == 0:
        return R * (1.0 + 0.20 * math.exp(-yy / 0.9) + 0.16 * math.exp(-yy / 4.5)) * max(1.0 - 0.80 * (max(yy, 0.0) / hfull) ** 1.2, 0.02)
    return R * (1.0 + 0.30 * math.exp(-yy / 0.45)) * max(1.0 - 0.90 * max(yy, 0.0) / hfull, 0.02)

@njit(**NJ)
def limb(x, y, z, T, i, k, far):
    """one broken limb.  Returns distance."""
    sd = T[i, 4]; kind = int(T[i, 5]); H = T[i, 11]
    i7 = int(sd * 977.0) + 31 * k
    if kind == 0:
        hy = 0.52 * H + 0.40 * H * hash3(i7, k, 1)
        ln = 1.0 + 3.0 * hash3(i7, k, 2) ** 1.5
        rl = 0.10 + 0.16 * hash3(i7, k, 3)
    else:
        hy = 0.30 * H + 0.66 * H * hash3(i7, k, 1)
        ln = 0.25 + 1.3 * hash3(i7, k, 2) ** 2
        rl = 0.018 + 0.035 * hash3(i7, k, 3)
    az = 6.2832 * hash3(i7, k, 4)
    el = -0.65 + 0.75 * hash3(i7, k, 5)
    yb = T[i, 3] + hy
    ax = T[i, 0] + T[i, 9] * hy; az0 = T[i, 1] + T[i, 10] * hy
    ca = math.cos(az); sa = math.sin(az); ce = math.cos(el); se = math.sin(el)
    bx = ax + ca * ce * ln; by = yb + se * ln; bz = az0 + sa * ce * ln
    d = sd_cap(x, y, z, ax, yb, az0, bx, by, bz, rl)
    if kind == 0 and not far and hash3(i7, k, 9) < 0.6:
        # the old limbs turn up at the elbow
        l2 = ln * (0.15 + 0.4 * hash3(i7, k, 6))
        tw = 6.2832 * hash3(i7, k, 7)
        d = smin(d, sd_cap(x, y, z, bx, by, bz, bx + (ca * 0.7 + 0.4 * math.cos(tw)) * l2, by + l2 * (0.2 + 0.6 * hash3(i7, k, 8)),
                           bz + (sa * 0.7 + 0.4 * math.sin(tw)) * l2, rl * 0.55), 0.25)
        d += 0.10 * (noise3(x * 1.1, y * 1.1, z * 1.1) - 0.5) + 0.03 * (noise3(x * 4.0, y * 4.0, z * 4.0) - 0.5)
    return d

@njit(**NJ)
def sd_trunk(dx, yr, dz, rr, T, i, far):
    R = T[i, 2]; kind = int(T[i, 5]); sd = T[i, 4]
    a = math.atan2(dz, dx)
    ca = dx / rr; sa = dz / rr
    r0 = trunk_r(yr, R, kind, T[i, 12])
    yy = max(yr, -1.5)
    H = T[i, 11]
    if kind == 0:
        A = 0.10 * math.exp(-yy / 2.0) + 0.018
        b = 0.50 * math.cos(3 * a + sd) + 0.45 * math.cos(4 * a + 2.3 * sd) + 0.40 * math.cos(6 * a + 4.1 * sd) \
            + 0.30 * math.cos(9 * a + 1.7 * sd) + 0.22 * math.cos(14 * a + 3.3 * sd) + 0.12 * math.cos(23 * a + 0.7 * sd)
        r = r0 * (1.0 + A * b)
        if not far:
            f1, f2, f3 = furrows(ca, sa, yr, R, sd)
            fur = 1.0 - sstep(0.0, 0.26, f1)
            r -= 0.095 * fur + (0.030 * (1.0 - sstep(0.0, 0.24, f2)) + 0.016 * (1.0 - sstep(0.0, 0.12, f3))) * (1.0 - fur)
            # where the fire ate in
            sw = scar_w(yr, a, T, i)
            if sw > -0.35:
                cut = sstep(0.0, 0.55, sw)
                r -= cut * R * (0.34 + 0.12 * noise3(a * 5.0, yr * 1.1, sd))
            # and where slabs of it have come away
            sl = fbm3(ca * R * 0.55 + sd, yr * 0.085, sa * R * 0.55, 3)
            r -= 0.13 * sstep(0.56, 0.62, sl)
        d = rr - r
        # the top went in the fire or in the first winter's wind: a splintered break
        jag = 2.6 * (noise3(ca * 2.2 + sd, 1.0, sa * 2.2) - 0.5) + 1.1 * (noise3(ca * 7.0, sd, sa * 7.0) - 0.5) \
            + 2.0 * (rr / max(r0, 0.05) - 0.5)
        d = max(d, (yr - H - jag) * 0.45)
    else:
        r = r0 * (1.0 + 0.03 * math.cos(4 * a + sd))
        if not far:
            n = noise3(ca * R / 0.06 + sd * 3.1, yr * 0.7, sa * R / 0.06 - sd * 1.7)
            r -= 0.014 * (1.0 - sstep(0.0, 0.32, abs(2.0 * n - 1.0)))
        d = rr - r
        d = max(d, (yr - H - 0.6 * (noise3(ca * 3.0 + sd, 1.0, sa * 3.0) - 0.5) - 1.5 * (rr / max(r0, 0.02) - 0.5)) * 0.5)
    return d


# ------------------------------------------------------------------ the slope
@njit(**NJ)
def scene(x, y, z, T, GI, lod):
    """distance, material, tree index.  lod 0: everything.  1: plain (far, shadows)."""
    g = ground0(x, z)
    ti = -1
    d = MARG; m = 0
    ci = int((x - GX0) / CS); cj = int((z - GZ0) / CS)
    far = lod == 1
    if ci >= 0 and ci < GNX and cj >= 0 and cj < GNZ:
        for q in range(KMAX):
            i = GI[cj, ci, q]
            if i < 0: break
            yr = y - T[i, 3]
            dx = x - T[i, 0] - T[i, 9] * yr; dz = z - T[i, 1] - T[i, 10] * yr
            rr = math.sqrt(dx * dx + dz * dz) + 1e-9
            R = T[i, 2]; kind = int(T[i, 5])
            H = T[i, 11]
            # the snow is scoured into a well round a trunk, and banked on the lee
            if kind == 0:
                wl = sstep(R * 1.5 + 2.2, R * 1.05, rr)
                g += wl * (-0.55 + 0.9 * max((dx * WNX + dz * WNZ) / rr, -0.2))
            else:
                g -= 0.22 * sstep(R + 0.9, R, rr)
            if yr > H + 4.0:
                dt = math.sqrt(max(rr - R, 0.0) ** 2 + (yr - H - 3.0) ** 2)
                if dt < d: d = dt
                continue
            rb = trunk_r(yr, R, kind, T[i, 12]) * (1.14 if kind == 0 else 1.05)
            if rr - rb > 0.8:
                dt = rr - rb - 0.2
            else:
                dt = sd_trunk(dx, yr, dz, rr, T, i, far) * (0.55 if kind == 0 else 0.7)
            nl = int(T[i, 13])
            if nl > 0 and yr > 0.25 * H - 1.5:
                reach = 9.5 if kind == 0 else 1.9
                if rr < reach + 1.3:
                    for k in range(nl):
                        dl = limb(x, y, z, T, i, k, far) * 0.8
                        if dl < dt: dt = dl
            if dt < d: d = dt; ti = i; m = M_TRUNK
    h = y - g
    t3 = 0.0
    if h < 1.0:
        t3 = tracks(x, z)
        if t3 > 0.0: h += 0.14 * t3
        if lod == 0:
            h -= 0.006 * (noise3(x * 9.0, 0.3, z * 9.0) - 0.5)
    dg = h * (0.75 if t3 <= 0.0 else 0.4)
    if dg < d:
        d = dg; m = M_SNOW
    # the walker
    fx = x - FIGX; fz = z - FIGZ
    if fx * fx + fz * fz < 2.2 and h < 2.6:
        if fx * fx + fz * fz < 0.80:
            c = math.cos(FIGYAW); s = math.sin(FIGYAW)
            df, mf = sd_fig(fx * c - fz * s, y - ground0(FIGX, FIGZ) + 0.10, fx * s + fz * c)
            if df < d: d = df; m = mf
        else:
            d = min(d, math.sqrt(fx * fx + fz * fz) - 0.80)
    return d, m, ti


@njit(**NJ)
def march(ox, oy, oz, dx, dy, dz, tmax, T, GI):
    t = 0.02
    for s in range(1400):
        d, m, ti = scene(ox + dx * t, oy + dy * t, oz + dz * t, T, GI, 0 if t < 130.0 else 1)
        if d < 0.00016 * t + 0.0003:
            return t, m, ti
        t += d
        if t > tmax: break
    return tmax, 0, -1

@njit(**NJ)
def shadow(px, py, pz, lx, ly, lz, maxt, kk, T, GI):
    """soft shadow towards a broad light: 1 open, 0 shut"""
    t = 0.05; res = 1.0
    for s in range(90):
        d, m, ti = scene(px + lx * t, py + ly * t, pz + lz * t, T, GI, 1)
        if d < 0.0008: return 0.0
        if d < MARG * 0.95: res = min(res, kk * d / t)
        t += min(max(d, 0.03), 2.5)
        if t >= maxt: break
    return res * res * (3.0 - 2.0 * res)

@njit(**NJ)
def normal(px, py, pz, t, T, GI, lod):
    e = 0.00022 * t + 0.0008
    nx = scene(px + e, py, pz, T, GI, lod)[0] - scene(px - e, py, pz, T, GI, lod)[0]
    ny = scene(px, py + e, pz, T, GI, lod)[0] - scene(px, py - e, pz, T, GI, lod)[0]
    nz = scene(px, py, pz + e, T, GI, lod)[0] - scene(px, py, pz - e, T, GI, lod)[0]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nl, ny / nl, nz / nl

@njit(**NJ)
def ambient_occ(px, py, pz, nx, ny, nz, T, GI):
    a = 0.0; w = 0.0
    hs = (0.05, 0.15, 0.40, 1.0, 2.4)
    for i in range(5):
        h = hs[i]
        d = scene(px + nx * h, py + ny * h, pz + nz * h, T, GI, 1)[0]
        k = 1.0 / (1.0 + 0.7 * i)
        a += k * min(max(d / min(h, MARG), 0.0), 1.0); w += k
    return a / w


# ------------------------------------------------------------------ cloud
@njit(**NJ)
def cloud(x, y, z):
    h = y - slope(x, z)
    n = fbm3(x * 0.012 + 1.0, y * 0.03, z * 0.012 + 5.0, 3)
    s = 0.45 + 1.1 * n
    s *= 1.0 + 7.0 * sstep(4.0, 70.0, y) + 0.8 * sstep(30.0, 4.0, h) * sstep(0.35, 0.7, fbm3(x * 0.02 + 9.0, 0.5, z * 0.02, 2))
    return SIG0 * s

@njit(**NJ)
def air(ox, oy, oz, dx, dy, dz, tend, jit, NST, out):
    """in-scatter weight out[0] and transmittance out[1]"""
    tend = min(tend, 900.0)
    Tr = 1.0; acc = 0.0
    kk = 1.05
    t = 0.0
    dt = tend * (kk - 1.0) / (kk ** NST - 1.0)
    for i in range(NST):
        tm = t + dt * jit
        sg = cloud(ox + dx * tm, oy + dy * tm, oz + dz * tm)
        e = math.exp(-sg * dt)
        acc += Tr * (1.0 - e)
        Tr *= e
        t += dt; dt *= kk
        if Tr < 0.002: break
    out[0] = acc; out[1] = Tr


# ------------------------------------------------------------------ surfaces
@njit(**NJ)
def char_h(ca, sa, yr, R, sd, fp):
    """alligator char: 1 on the face of a block, 0 in a crack, at the sizes a pixel this big can hold"""
    # two families of crack, one running up the trunk and one across it, so the char breaks into blocks
    c1 = min(abs(2.0 * noise3(ca * R * 17.0 + sd, yr * 2.6, sa * R * 17.0) - 1.0),
             abs(2.0 * noise3(ca * R * 3.4 - sd, yr * 13.0, sa * R * 3.4) - 1.0))
    c2 = min(abs(2.0 * noise3(ca * R * 44.0, yr * 8.0 + sd, sa * R * 44.0) - 1.0),
             abs(2.0 * noise3(ca * R * 9.0, yr * 36.0, sa * R * 9.0 + sd) - 1.0))
    c3 = noise3(ca * R * 120.0, yr * 100.0, sa * R * 120.0)
    a = sstep(0.05, 0.016, fp); b = sstep(0.018, 0.006, fp); c = sstep(0.006, 0.002, fp)
    return (1.0 - a * (1.0 - sstep(0.0, 0.14, c1))) * (1.0 - 0.7 * b * (1.0 - sstep(0.0, 0.16, c2))) * (1.0 - 0.35 * c * c3)

@njit(**NJ)
def char_at(x, y, z, T, i, fp):
    yr = y - T[i, 3]
    dx = x - T[i, 0] - T[i, 9] * yr; dz = z - T[i, 1] - T[i, 10] * yr
    rr = math.sqrt(dx * dx + dz * dz) + 1e-9
    return char_h(dx / rr, dz / rr, yr, T[i, 2], T[i, 4], fp)

@njit(**NJ)
def char_rgb(x, y, z, nx, ny, nz, T, i, fp, out):
    """burned bark: out[0:3] colour, out[3] crevice, out[4] how much snow is stuck here"""
    yr = y - T[i, 3]
    dx = x - T[i, 0] - T[i, 9] * yr; dz = z - T[i, 1] - T[i, 10] * yr
    rr = math.sqrt(dx * dx + dz * dz) + 1e-9
    R = T[i, 2]; kind = int(T[i, 5]); sd = T[i, 4]
    ca = dx / rr; sa = dz / rr
    fur = 0.0; fur2 = 0.0
    if kind == 0:
        f1, f2, f3 = furrows(ca, sa, yr, R, sd)
        fur = 1.0 - sstep(0.015, 0.20, f1)
        fur2 = max(1.0 - sstep(0.0, 0.17, f2), 0.8 * (1.0 - sstep(0.0, 0.09, f3)))
        det = char_h(ca, sa, yr, R, sd, fp)
        # char: black blocks with a dull sheen on their faces, blacker in the cracks.  It holds a little of the red
        # under it; where a slab has come away the wood is bare and grey-brown
        v = 0.010 + 0.050 * det + 0.012 * fbm3(ca * R * 1.5, yr * 0.3, sa * R * 1.5 + sd, 2)
        r = v * 1.10; g = v * 0.98; b = v * 0.94
        sl = sstep(0.56, 0.62, fbm3(ca * R * 0.55 + sd, yr * 0.085, sa * R * 0.55, 3))
        rb = sstep(0.50, 0.80, fbm3(ca * R * 2.0 + 3.0, yr * 0.5, sa * R * 2.0, 3)) * sstep(3.0, 14.0, yr) * (1.0 - fur)
        r += (0.18 - r) * 0.55 * rb; g += (0.085 - g) * 0.55 * rb; b += (0.050 - b) * 0.55 * rb
        gv = 0.11 + 0.07 * noise3(ca * R * 30.0, yr * 0.8, sa * R * 30.0)
        r += (gv * 1.12 - r) * sl; g += (gv - g) * sl; b += (gv * 0.86 - b) * sl
        dk = (1.0 - 0.7 * fur) * (1.0 - 0.4 * fur2 * (1.0 - fur))
        r *= dk; g *= dk; b *= dk
    else:
        det = noise3(ca * R * 90.0, yr * 6.0, sa * R * 90.0)
        v = 0.028 + 0.035 * det
        gy = sstep(0.45, 0.7, fbm3(ca * R * 3.0 + sd, yr * 0.25, sa * R * 3.0, 2))     # bark gone, wood weathering silver
        v += (0.12 + 0.08 * det - v) * gy
        r = v * 1.05; g = v; b = v * 0.96
    # snow: driven into the windward side and held in the furrows, sitting on anything that faces up
    wd = -(nx * WNX + nz * WNZ)
    pat = fbm3(ca * R * 2.4 + sd, yr * 0.22, sa * R * 2.4, 3)
    p2 = noise3(ca * R * 9.0, yr * 0.9 + sd, sa * R * 9.0)
    sn = sstep(0.30, 0.62, wd + 0.7 * (pat - 0.5) - 0.25 * sstep(2.0, 40.0, yr))
    sn *= sstep(0.47, 0.53, 0.6 * pat + 0.4 * p2 + 0.22 * fur)
    if kind == 1: sn *= sstep(0.55, 0.75, p2) * 0.8
    sn = max(sn, sstep(0.72, 0.90, ny + 0.2 * (pat - 0.5)))
    out[0] = r; out[1] = g; out[2] = b; out[3] = min(fur + 0.5 * fur2, 1.0); out[4] = sn


@njit(parallel=True, **NJ)
def expose(W, H, F, SS, NST, T, GI, seed, OX, OY, WF, HFULL):
    IMG = np.zeros((H, W, 3), np.float32)
    oy0 = ground0(CAMX, CAMZ) + EYE
    for j in prange(H):
        o2 = np.zeros(2); bo = np.zeros(6)
        s = ((seed + j * 7919) * 2654435761) & 0xffffffff
        s = (s ^ (s >> 15)) & 0xffffffff
        for q in range(4): s, r0 = lcg(s)
        for i in range(W):
            cr = 0.0; cg = 0.0; cb = 0.0
            for sy in range(SS):
                for sx in range(SS):
                    s, r1 = lcg(s); s, r2 = lcg(s); s, r3 = lcg(s)
                    dx = (i + OX + (sx + r1) / SS - WF * 0.5) / F
                    dy = -(j + OY + (sy + r2) / SS - HFULL * VC) / F
                    dz = 1.0
                    dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                    dx /= dl; dy /= dl; dz /= dl
                    t, m, ti = march(CAMX, oy0, CAMZ, dx, dy, dz, 900.0, T, GI)
                    lr = 0.0; lg = 0.0; lb = 0.0
                    if m > 0:
                        px = CAMX + dx * t; py = oy0 + dy * t; pz = CAMZ + dz * t
                        lod = 0 if t < 130.0 else 1
                        nx, ny, nz = normal(px, py, pz, t, T, GI, lod)
                        fp = t / F
                        ar = 0.2; ag = 0.2; ab = 0.2; cav = 0.0; snow = 0.0
                        if m == M_TRUNK:
                            char_rgb(px, py, pz, nx, ny, nz, T, ti, fp, bo)
                            ar = bo[0]; ag = bo[1]; ab = bo[2]; cav = bo[3]; snow = bo[4]
                            if T[ti, 5] == 0 and fp < 0.05:
                                e = max(0.7 * fp, 0.0008)
                                h0 = char_at(px, py, pz, T, ti, fp)
                                gx = (char_at(px + e, py, pz, T, ti, fp) - h0) / e
                                gy = (char_at(px, py + e, pz, T, ti, fp) - h0) / e
                                gz = (char_at(px, py, pz + e, T, ti, fp) - h0) / e
                                gd = gx * nx + gy * ny + gz * nz
                                kb = 0.010 * (1.0 - snow)
                                nx -= kb * (gx - gd * nx); ny -= kb * (gy - gd * ny); nz -= kb * (gz - gd * nz)
                                nl = math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
                        elif m == M_SNOW:
                            snow = 1.0
                            # what the dead trees shed on it: flakes of char, needles long gone brown
                            dirt = 0.0
                            if fp < 0.02:
                                dirt = sstep(0.80, 0.90, noise3(px * 60.0, 0.3, pz * 60.0)) * sstep(0.45, 0.70, fbm3(px * 0.5, 0.7, pz * 0.5, 2)) * sstep(0.02, 0.006, fp)
                            ar = 0.035; ag = 0.032; ab = 0.030
                            snow = 1.0 - 0.85 * dirt
                            e = max(0.7 * fp, 0.002)
                            h0 = noise3(px * 26.0, 0.2, pz * 26.0) * 0.004 + noise3(px * 90.0, 0.6, pz * 90.0) * 0.0012
                            gx = (noise3((px + e) * 26.0, 0.2, pz * 26.0) * 0.004 + noise3((px + e) * 90.0, 0.6, pz * 90.0) * 0.0012 - h0) / e
                            gz = (noise3(px * 26.0, 0.2, (pz + e) * 26.0) * 0.004 + noise3(px * 90.0, 0.6, (pz + e) * 90.0) * 0.0012 - h0) / e
                            kb = sstep(0.02, 0.005, fp)
                            nx -= kb * gx; nz -= kb * gz
                            nl = math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
                        elif m == M_SHIRT:
                            f = 0.86 + 0.20 * noise3(px * 30.0, py * 30.0, pz * 30.0)
                            ar = 0.88 * f; ag = 0.60 * f; ab = 0.030 * f
                        elif m == M_SHROUD:
                            ar = 0.62; ag = 0.47; ab = 0.10
                        elif m == M_PANTS:
                            ar = 0.075; ag = 0.085; ab = 0.055
                        elif m == M_BOOT:
                            ar = 0.035; ag = 0.030; ab = 0.026
                        elif m == M_SKIN:
                            ar = 0.30; ag = 0.20; ab = 0.14
                        elif m == M_HELMET:
                            ar = 0.40; ag = 0.38; ab = 0.34
                        elif m == M_PACK:
                            ar = 0.050; ag = 0.046; ab = 0.042
                        elif m == M_GLOVE:
                            ar = 0.30; ag = 0.22; ab = 0.13
                        if snow > 0.0:
                            ar += (0.90 - ar) * snow; ag += (0.915 - ag) * snow; ab += (0.94 - ab) * snow
                        ao = ambient_occ(px, py, pz, nx, ny, nz, T, GI)
                        ao *= 1.0 - 0.6 * cav * (1.0 - snow)
                        # the overcast: a dome, a little brighter one way, and the snow lighting things from below
                        up = 0.5 + 0.5 * ny
                        nb = nx * BRX + ny * BRY + nz * BRZ
                        br = 0.0
                        if nb > 0.0:
                            br = nb
                        dn = 0.5 - 0.5 * ny
                        er = ao * (0.80 * up + 0.30 * dn + 0.20 * br)
                        eg = ao * (0.83 * up + 0.31 * dn + 0.20 * br)
                        eb = ao * (0.90 * up + 0.335 * dn + 0.19 * br)
                        lr = ar * er; lg = ag * eg; lb = ab * eb
                    air(CAMX, oy0, CAMZ, dx, dy, dz, t, r3, NST, o2)
                    Tr = o2[1]; a = o2[0]
                    # cloud is lit from above and from the snow under it; near the ground it is a shade bluer
                    hz = 0.80 + 0.10 * sstep(-0.1, 0.35, dy)
                    cr += lr * Tr + a * 0.96 * hz
                    cg += lg * Tr + a * 0.985 * hz
                    cb += lb * Tr + a * 1.025 * hz
            k = 1.0 / (SS * SS)
            IMG[j, i, 0] = cr * k; IMG[j, i, 1] = cg * k; IMG[j, i, 2] = cb * k
    return IMG


# ------------------------------------------------------------------ what was standing, and what is left
def near_path(x, z, w):
    for k in range(NPATH - 1):
        ax, az = PATHX[k], PATHZ[k]; ex, ez = PATHX[k + 1] - ax, PATHZ[k + 1] - az
        h = min(max(((x - ax) * ex + (z - az) * ez) / (ex * ex + ez * ez), 0.0), 1.0)
        if math.hypot(x - ax - ex * h, z - az - ez * h) < w: return True
    return False

def plant(rng):
    rows = []
    #            x      z     R    scar angle, half-width, height, stands to, limbs
    monarchs = [(-16.0, 25.0, 3.30, -40, 1.5, 9.0, 61.0, 6),
                (21.0,  45.0, 2.95, -135, 1.2, 7.0, 47.0, 5),
                (-33.0, 92.0, 3.05, -80, 1.3, 8.0, 52.0, 7),
                (13.5, 101.0, 3.20, -100, 1.6, 10.0, 66.0, 7),
                (58.0, 118.0, 2.8, -90, 0.0, 0.0, 31.0, 4),
                (-72.0, 132.0, 3.0, -90, 1.0, 6.0, 44.0, 6),
                (-8.0, 160.0, 2.9, -90, 0.0, 0.0, 39.0, 6),
                (36.0, 176.0, 3.1, -90, 1.2, 7.0, 29.0, 4),
                (88.0, 205.0, 2.9, -90, 0.0, 0.0, 47.0, 6),
                (-52.0, 218.0, 3.2, -90, 1.0, 6.0, 50.0, 6),
                (-118.0, 196.0, 2.8, -90, 0.0, 0.0, 49.0, 5),
                (10.0, 252.0, 3.0, -90, 0.0, 0.0, 41.0, 6),
                (64.0, 290.0, 3.1, -90, 0.0, 0.0, 55.0, 6),
                (-30.0, 318.0, 2.9, -90, 0.0, 0.0, 67.0, 7),
                (132.0, 262.0, 3.0, -90, 0.0, 0.0, 51.0, 5),
                (-96.0, 300.0, 3.1, -90, 0.0, 0.0, 60.0, 6),
                (-150.0, 270.0, 2.9, -90, 0.0, 0.0, 57.0, 6),
                (30.0, 372.0, 3.0, -90, 0.0, 0.0, 63.0, 6)]
    for k, (x, z, R, sa, hw, hs, H, nl) in enumerate(monarchs):
        rows.append([x, z, R, 0, 3.0 + 7.31 * k, 0, math.radians(sa), hw, hs,
                     0.006 * rng.standard_normal(), 0.006 * rng.standard_normal(), H, 86.0, nl])
    occ = {}
    def cells(x, z, rb):
        out = []
        for cj in range(int((z - rb - GZ0) / CS), int((z + rb - GZ0) / CS) + 1):
            for ci in range(int((x - rb - GX0) / CS), int((x + rb - GX0) / CS) + 1):
                out.append((cj, ci))
        return out
    for r in rows:
        for c in cells(r[0], r[1], r[2] * 1.5 + 9.5 + MARG + 0.6): occ[c] = occ.get(c, 0) + 1
    n = 0; tries = 0
    while n < 2300 and tries < 400000:
        tries += 1
        x = rng.uniform(-235, 235); z = rng.uniform(6, 440)
        if abs(x) > 0.72 * z + 14: continue
        if near_path(x, z, 3.2 + 0.02 * z): continue
        if abs(x) < 3.5 and z < 16: continue
        if z < 40 and abs(x) > 0.52 * z: continue       # nothing hard against the edge of the frame
        # they stood in thickets with gaps between
        dens = 0.25 + 0.75 * (gaussian_bump(x, z))
        if rng.random() > dens: continue
        hfull = rng.uniform(14, 44) if rng.random() < 0.8 else rng.uniform(6, 16)
        R = 0.011 * hfull * rng.uniform(0.75, 1.3)
        if z < 30 and R > 0.3: continue
        ok = True
        for r in rows[:len(monarchs)]:
            if math.hypot(x - r[0], z - r[1]) < r[2] * 1.9 + 2.5: ok = False; break
        if not ok: continue
        rb = R * 1.45 + 1.9 + MARG + 0.6 + 44.0 * 0.03
        cl = cells(x, z, rb)
        if any(c[0] < 0 or c[1] < 0 or c[0] >= GNZ or c[1] >= GNX for c in cl): continue
        if any(occ.get(c, 0) >= KMAX - 2 for c in cl): continue
        for c in cl: occ[c] = occ.get(c, 0) + 1
        broke = rng.random()
        H = hfull * (rng.uniform(0.25, 0.8) if broke < 0.45 else rng.uniform(0.85, 0.98))
        rows.append([x, z, R, 0, rng.uniform(0, 50), 1, 0, 0, 0,
                     0.030 * rng.standard_normal(), 0.030 * rng.standard_normal(), H, hfull, int(rng.integers(0, 6))])
        n += 1
    return np.array(rows, np.float64)

_gb = None
def gaussian_bump(x, z):
    global _gb
    if _gb is None:
        r = np.random.default_rng(77)
        _gb = gaussian_filter(r.standard_normal((120, 120)), 3.0, mode='wrap'); _gb = (_gb - _gb.min()) / (_gb.max() - _gb.min())
    return float(_gb[int(z / 4.0) % 120, int((x + 240) / 4.0) % 120]) ** 1.5 * 1.6

def seat(T):
    for i in range(T.shape[0]):
        T[i, 3] = ground0(T[i, 0], T[i, 1]) - (1.0 if T[i, 5] == 0 else 0.5)
    GI = -np.ones((GNZ, GNX, KMAX), np.int32)
    for i in range(T.shape[0]):
        rb = (T[i, 2] * 1.5 + 9.5 if T[i, 5] == 0 else T[i, 2] * 1.45 + 1.9) + MARG + 0.6 + 44.0 * math.hypot(T[i, 9], T[i, 10])
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


# ------------------------------------------------------------------ the print
EXPOSURE = 0.80

def develop(lin, scale, rng, exposure):
    """no lens.  A flat day printed flat: the whites kept under paper white, the blacks kept open, a fine tooth."""
    H, W = lin.shape[:2]
    lin = np.maximum(lin, 0.0) * exposure
    L = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    lg = np.log(np.clip(L, 1e-4, 80.0))
    base = gaussian_filter(lg, 90.0 * scale)
    gain = np.clip(np.exp(-0.18 * (base - math.log(0.45))), 0.5, 2.0)
    lin = lin * gain[..., None]
    wp = 1.6
    y = lin * (1.0 + lin / wp ** 2) / (1.0 + lin) * 1.55
    y = np.clip(y, 0.0, 1.0)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-9), 1 / 2.4) - 0.055)
    y = y + 0.42 * (y * y * (3 - 2 * y) - y)
    # nearly without colour, as the day was; the yellow is left alone
    lum = y.mean(-1, keepdims=True)
    sat = (y.max(-1, keepdims=True) - y.min(-1, keepdims=True))
    keep = np.clip((sat - 0.10) / 0.15, 0, 1)
    y = lum + (y - lum) * (0.80 + 0.20 * keep)
    y = 0.030 + 0.945 * y
    g = gaussian_filter(rng.standard_normal((H, W)), 0.6 * max(scale, 0.5)); g /= g.std()
    g2 = gaussian_filter(rng.standard_normal((H, W)), 2.2 * max(scale, 0.5)); g2 /= g2.std()
    lum = y.mean(-1, keepdims=True)
    amp = 0.004 + 0.007 * np.sqrt(np.clip(4 * lum * (1 - lum), 0, 1))
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
    print('standing dead: %d  %.0fs' % (T.shape[0], time.time() - t0), flush=True)
    OX = OY = 0; crop = None
    SS = 2; NST = 44
    for a in sys.argv:
        if a.startswith('--crop='):
            crop = [int(v) for v in a[7:].split(',')]
            OX, OY = crop[0], crop[1]; W = crop[2] - crop[0]; H = crop[3] - crop[1]
        if a.startswith('--ss='): SS = int(a[5:])
    IMG = expose(W, H, F, SS, NST, T, GI, SEED % 9973, OX, OY, WF, HFULL)
    print('exposed  %.0fs' % (time.time() - t0), flush=True)
    lin = IMG.astype(np.float64)
    if '--keep' in sys.argv: np.save(out.replace('.png', '_lin.npy'), lin.astype(np.float32))
    img = develop(lin, scale, rng, EXPOSURE)
    if crop is None: img = sign(img, scale, OLDEST)
    Image.fromarray((img * 255 + rng.random(img.shape)).clip(0, 255).astype(np.uint8)).save(out)
    print('saved', out, '%dx%d  %.0fs' % (W, H, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
