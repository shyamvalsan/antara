"""
Kund, three thousand five hundred steps, one going down

Ruhi Vaghela.

Survey.  A square kund of the Abhaneri type.  39.0 m square at the rim, thirteen storeys, 30.0 m from
the rim to the floor of the pool.  Each storey is 2.31 m: eight risers of 0.288 m.  Each terrace steps
in 1.25 m.  Against every terrace face stand pairs of opposed flights, 0.72 m wide, seven treads of
0.27 m rising from either hand to a shared landing; the pairs are set half a pair out of step from one
terrace to the next.  Three sides are built so; the fourth, where we stand, is sheer.
The water stands 1.6 m deep over the floor, 9 m square.  The hour is late morning in May; the sun is
from the left and a little behind us, so that it rakes the far wall and leaves the left wall to be lit
by the stone opposite.  One woman, going down, on the ninth storey.

Usage:  python kund_three_thousand_five_hundred_steps_one_going_down.py [scale=1.0] [out.png] [supersampling]
        scale 0.25 is a study; 1.0 is 3600 x 3000.

Built as signed-distance solids in metres and ray-marched (the method of the stair pictures in this
gallery), lit by calculation:
  sun    - a hard directional light with the sun's true half-degree, so far shadows soften a little
  sky    - every lit point sends one ray into its hemisphere; what that ray meets (sky, or stone in sun,
           or stone in shade) is the fill and the bounce, coloured as it should be
  water  - a level plane; mirror above by Fresnel, turbid green below, the lowest steps seen through it
  dust   - the air in the pit marched against the sun, so the lit air stands in front of the shade
  print  - no lens; a restrained curve, then the tooth of lime plaster
"""
import sys, math, time
import numpy as np
from numba import njit, prange
from PIL import Image
from scipy.ndimage import gaussian_filter

SEED = 1985
NJ = dict(cache=True, fastmath=True, error_model='numpy')

# ------------------------------------------------------------------ the survey (metres)
NK = 13                                 # storeys
DEPTH = 30.0
NR = 8                                  # risers in a flight
RI = DEPTH / (NK * NR)                  # 0.2885
H = NR * RI                             # a storey
TR = 0.27                               # tread
NT = NR - 1                             # treads in a flight; the eighth riser lands on the landing
LAND = 0.56                             # shared landing
GAP = 0.06                              # between the feet of neighbouring pairs
P = 2 * NT * TR + LAND + GAP            # one pair
FW = 0.72                               # width of a flight
D = 1.25                                # a terrace steps in this much
POOL = 4.5                              # half the pool
A = (NK - 1) * D + POOL                 # half the rim: 19.5
S = math.sqrt(D * D + H * H)
SS_ = math.sqrt(RI * RI + TR * TR)
YF = -DEPTH
YW = YF + 1.62                          # the water
PLW = A + 3.4                           # the foot of the enclosure wall, back from the rim
PLH = 1.5

# ---- the sun: late morning in May, from the left and behind our left shoulder
_sl = np.array([-0.62, 1.00, -0.42]); _sl /= np.linalg.norm(_sl)
LX, LY, LZ = float(_sl[0]), float(_sl[1]), float(_sl[2])
SUNR, SUNG, SUNB = 3.20, 2.88, 2.36     # sun, as radiance off a white card square to it
SUN_SPREAD = 0.0047                     # its radius, radians

# ---- the lens (a plain pinhole)
CX, CY, CZ = -3.0, 4.6, -A + 0.60                # from the balcony of the pavilion that stands out over the fourth side
TX, TY, TZ = 2.2, -18.7, 12.0
HFOV = 64.0
BW, BH = 3600, 3600

# ---- the water
SIGR, SIGG, SIGB = 3.6, 2.0, 4.2        # extinction per metre: it is green because it is alive

# ---- the woman: ninth storey of the far wall, right-hand flight of the pair, fourth tread down
WK = 8
WCELL = 1
WSC = 0.95                              # she is 1.52 m to the crown of her head
WYAW = math.radians(26.0)               # she comes down turned a little to the wall
WPX = WCELL * P + (WK & 1) * 0.5 * P + LAND * 0.5 + 3.5 * TR
WPY = -WK * H - 4 * RI
WPZ = A - (WK * D + 0.37)
WFX, WFZ = math.cos(WYAW), math.sin(WYAW)
WRX, WRZ = math.sin(WYAW), -math.cos(WYAW)


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
    h0 = hash3(i, j, k); h1 = hash3(i, j + 1, k); h2 = hash3(i, j, k + 1); h3 = hash3(i, j + 1, k + 1)
    a = h0 + u * (hash3(i + 1, j, k) - h0)
    b = h1 + u * (hash3(i + 1, j + 1, k) - h1)
    c = h2 + u * (hash3(i + 1, j, k + 1) - h2)
    d = h3 + u * (hash3(i + 1, j + 1, k + 1) - h3)
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
def quad(a, b):
    """distance to the quarter-plane a <= 0, b <= 0."""
    oa = max(a, 0.0); ob = max(b, 0.0)
    return math.sqrt(oa * oa + ob * ob) + min(max(a, b), 0.0)

@njit(**NJ)
def sd_box(x, y, z, bx, by, bz):
    qx = abs(x) - bx; qy = abs(y) - by; qz = abs(z) - bz
    ox = max(qx, 0.0); oy = max(qy, 0.0); oz = max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0)

@njit(**NJ)
def cone(x, y, z, ax, ay, az, ra, bx, by, bz, rb):
    """a limb: capsule whose radius runs from ra to rb."""
    px = x - ax; py = y - ay; pz = z - az
    ex = bx - ax; ey = by - ay; ez = bz - az
    h = (px * ex + py * ey + pz * ez) / (ex * ex + ey * ey + ez * ez + 1e-12)
    if h < 0: h = 0.0
    if h > 1: h = 1.0
    px -= ex * h; py -= ey * h; pz -= ez * h
    return math.sqrt(px * px + py * py + pz * pz) - (ra + (rb - ra) * h)

@njit(**NJ)
def smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25

@njit(**NJ)
def rnd(s):
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return s, ((s >> 8) & 0xffffff) / 16777216.0


# ------------------------------------------------------------------ one stepped side
@njit(**NJ)
def sag(u, v):
    """eight hundred years of settlement: no course is level any more, by two or three centimetres."""
    return 0.022 * math.sin(u * 0.23 + v * 0.31) + 0.010 * math.sin(u * 0.71 + 1.0 + v * 0.9)

@njit(**NJ)
def wall_at(u, v, y, fine):
    """the stepped side near its surface.  fine: with its wear, its chips and its steps out of true."""
    ARR = 0.018
    if fine:
        n1 = noise3(u * 3.1, y * 3.1, v * 3.1); n2 = noise3(u * 9.0, y * 9.0, v * 9.0)
        ARR = 0.005 + 0.026 * n1 * n1 + 0.035 * max(n2 - 0.58, 0.0)   # the arrises are gone, here more, there less
    k0 = int(math.floor(v / D))
    d = 0.6
    for k in range(k0, k0 + 3):
        kk = min(max(k, 0), NK - 1)
        q = quad(v - kk * D + ARR, y + kk * H + ARR) - ARR
        if q < d: d = q
    for k in range(k0, k0 + 2):
        if k < 0 or k >= NK: continue
        dv = v - (k * D + FW)
        if dv > d: continue
        yr = y + (k + 1) * H - H                          # height over the landing of this pair
        us = u - (k & 1) * 0.5 * P
        cell = math.floor(us / P + 0.5)
        ul = us - P * cell
        a = abs(ul) - LAND * 0.5
        ls = max((a * RI + yr * TR) / SS_, yr)
        if ls > 0.3:
            d2 = ls
        else:
            hs = 0
            if fine:
                yr += 0.017 * math.exp(-((dv + 0.32) / 0.19) ** 2)      # feet have hollowed the middle of every tread
                hs = int(cell) * 4 + (1 if ul > 0.0 else 0) + k * 4096
            j0 = int(math.floor(a / TR)) + 1
            d2 = 1e9
            for j in range(j0 - 1, j0 + 2):
                jj = min(max(j, 0), NT)
                ja = 0.0; jy = 0.0
                if fine:                                                 # no step sits quite where it was set
                    ja = 0.030 * (hash3(hs, jj, 1) - 0.5); jy = 0.022 * (hash3(hs, jj, 2) - 0.5)
                q = quad(a - jj * TR - ja + ARR, yr + jj * RI + jy + ARR) - ARR
                if q < d2: d2 = q
        q = quad(d2 + ARR, dv + ARR) - ARR
        if q < d: d = q
    return d

@njit(**NJ)
def wall(u, v, y):
    """u along the wall, v in from the rim, y up from the rim.  Terraces, and the paired flights on them."""
    la = ((v - FW) * H + y * D) / S                       # height over the line of the landings' outer edges
    b = max(la, y)
    if b > 0.66: return b - 0.035
    y += sag(u, v)
    d = wall_at(u, v, y, False)                           # as it was drawn
    if d > 0.11: return d - 0.055
    return wall_at(u, v, y, True)                         # as it now is


@njit(**NJ)
def woman(x, y, z):
    """returns distance and part: 1 cloth, 2 skin, 3 brass.  Local frame: a forward, b up, c to her right."""
    qx = x - WPX; qz = z - WPZ
    a = (qx * WFX + qz * WFZ) / WSC; b = (y - WPY) / WSC; c = (qx * WRX + qz * WRZ) / WSC
    dr = RI / WSC
    # cloth: the sari from hip to hem, the leading leg in it, bodice, shoulders, the pallu over head and back
    dc = cone(a, b, c, 0.02, 0.93, 0.0, 0.150, 0.05, 0.13, -0.02, 0.200)
    dc = smin(dc, cone(a, b, c, 0.05, 0.88, 0.07, 0.115, 0.215, 0.46 - dr * 0.5, 0.10, 0.095), 0.06)
    dc = smin(dc, cone(a, b, c, 0.215, 0.46 - dr * 0.5, 0.10, 0.095, 0.235, 0.13 - dr, 0.10, 0.125), 0.05)
    dc = smin(dc, cone(a * 1.22, b, c, 0.0, 0.95, 0.0, 0.140, 0.0, 1.29, 0.0, 0.125) * 0.82, 0.05)
    dc = smin(dc, cone(a, b, c, 0.0, 1.335, -0.145, 0.058, 0.0, 1.335, 0.145, 0.058), 0.05)
    hx = a + 0.012; hy = b - 1.505
    dc = smin(dc, math.sqrt(hx * hx + hy * hy + c * c) - 0.113, 0.03)
    ex = (a + 0.105) / 0.050; ey = (b - 1.21) / 0.36; ez = c / 0.165
    k0 = math.sqrt(ex * ex + ey * ey + ez * ez)
    dc = smin(dc, (k0 - 1.0) * 0.05, 0.04)
    d = dc; part = 1
    # skin: face, the arm up to the pot, the arm out for balance, feet
    ds = math.sqrt((a - 0.022) ** 2 + (b - 1.49) ** 2 + c * c) - 0.100
    ds = min(ds, cone(a, b, c, 0.0, 1.335, 0.185, 0.043, 0.07, 1.44, 0.315, 0.036))
    ds = min(ds, cone(a, b, c, 0.07, 1.44, 0.315, 0.036, 0.03, 1.735, 0.175, 0.030))
    ds = min(ds, cone(a, b, c, 0.0, 1.335, -0.185, 0.043, 0.0, 1.08, -0.225, 0.036))
    ds = min(ds, cone(a, b, c, 0.0, 1.08, -0.225, 0.036, 0.11, 0.87, -0.235, 0.030))
    ds = min(ds, cone(a, b, c, -0.06, 0.036, -0.075, 0.036, 0.10, 0.030, -0.075, 0.030))
    ds = min(ds, cone(a, b, c, 0.17, 0.036 - dr, 0.10, 0.036, 0.33, 0.030 - dr, 0.10, 0.030))
    if ds < d: d = ds; part = 2
    # brass: the ghada, its neck and its rolled lip
    ex = a / 0.158; ey = (b - 1.752) / 0.128; ez = c / 0.158
    k0 = math.sqrt(ex * ex + ey * ey + ez * ez)
    db = (k0 - 1.0) * 0.128
    rr = math.sqrt(a * a + c * c)
    db = smin(db, max(rr - 0.058 - 0.3 * abs(b - 1.875) , abs(b - 1.875) - 0.035), 0.02)
    db = min(db, math.sqrt((rr - 0.078) ** 2 + (b - 1.912) ** 2) - 0.014)
    if db < d: d = db; part = 3
    return d * WSC, part


@njit(**NJ)
def scene(x, y, z):
    d = wall(x, A - z, y)                                 # far side
    q = wall(z, A + x, y)                                 # left
    if q < d: d = q
    q = wall(z, A - x, y)                                 # right
    if q < d: d = q
    q = quad(z + A + 0.012, y + 0.012) - 0.012            # the sheer side, where we stand
    if q < d: d = q
    q = y - YF                                            # floor of the pool
    if q < d: d = q
    q = max(PLW - max(abs(x), abs(z)), y - PLH)           # the enclosure wall
    if q < d: d = q
    bd = sd_box(x - WPX - 0.08, y - WPY - 0.78, z - WPZ, 0.50, 1.16, 0.50)
    if bd < 0.03:
        q = woman(x, y, z)[0]
        if q < d: d = q
    elif bd < d:
        d = max(bd, 0.03)
    return d


@njit(**NJ)
def march(ox, oy, oz, dx, dy, dz, tmax, pa):
    t = 0.0
    for s in range(900):
        d = scene(ox + dx * t, oy + dy * t, oz + dz * t)
        if d < pa * t * 0.15 + 0.00015:
            return t
        t += d * 0.82
        if t > tmax: break
    return 1e9

@njit(**NJ)
def normal(px, py, pz, e):
    nx = scene(px + e, py, pz) - scene(px - e, py, pz)
    ny = scene(px, py + e, pz) - scene(px, py - e, pz)
    nz = scene(px, py, pz + e) - scene(px, py, pz - e)
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nl, ny / nl, nz / nl


# ------------------------------------------------------------------ pigeons (ellipsoids, met exactly)
def _pigeons():
    rng = np.random.default_rng(SEED + 7)
    B = []
    def sit(x, y, z, yaw, tone):
        c, s = math.cos(yaw), math.sin(yaw)
        B.append([x, y + 0.075, z, 0.135, 0.074, 0.070, yaw, tone])                    # body
        B.append([x + 0.105 * c, y + 0.158, z + 0.105 * s, 0.036, 0.036, 0.034, yaw, tone * 0.8])   # head
        B.append([x - 0.15 * c, y + 0.07, z - 0.15 * s, 0.085, 0.022, 0.045, yaw, tone * 0.7])      # tail
    # on the far wall: a row on one ledge high up, twos and threes lower
    def far_ledge(k, x):                       # walkway of terrace k
        return x, -(k + 1) * H, A - (k * D + FW + 0.10 + 0.36 * rng.random())
    def far_landing(k, cell):
        return cell * P + (k & 1) * 0.5 * P + rng.uniform(-0.18, 0.18), -k * H, A - (k * D + rng.uniform(0.2, 0.6))
    for i in range(7):
        x, y, z = far_ledge(1, 3.3 + 0.42 * i + 0.25 * rng.random() + (0.9 if i > 3 else 0.0))
        sit(x, y, z, rng.uniform(-3.1, 3.1), rng.uniform(0.7, 1.2))
    for k, cell in ((3, -1), (3, 1), (4, 2), (5, 0), (2, 3), (6, -1), (7, 2), (10, 0)):
        x, y, z = far_landing(k, cell)
        sit(x, y, z, rng.uniform(-3.1, 3.1), rng.uniform(0.7, 1.2))
        if rng.random() < 0.5:
            sit(x + 0.26, y, z + 0.1, rng.uniform(-3.1, 3.1), rng.uniform(0.7, 1.2))
    for i in range(4):
        x, y, z = far_ledge(4, -7.5 + 0.5 * i + 0.3 * rng.random())
        sit(x, y, z, rng.uniform(-3.1, 3.1), rng.uniform(0.7, 1.2))
    # on the left wall, in the shade
    for k, zz in ((2, 6.0), (2, 6.5), (3, 1.0), (5, 3.8)):
        sit(-A + k * D + FW + 0.2, -(k + 1) * H, zz, rng.uniform(-3.1, 3.1), rng.uniform(0.7, 1.2))
    return np.array(B, np.float64)

BIRDS = _pigeons()
NB = BIRDS.shape[0]
BBX0, BBX1 = float(BIRDS[:, 0].min() - 0.5), float(BIRDS[:, 0].max() + 0.5)

@njit(**NJ)
def birds(ox, oy, oz, dx, dy, dz, tmax):
    """nearest pigeon along the ray: t, index (or -1)."""
    best = tmax; bi = -1
    for i in range(NB):
        px = ox - BIRDS[i, 0]; py = oy - BIRDS[i, 1]; pz = oz - BIRDS[i, 2]
        bq = px * dx + py * dy + pz * dz
        cq = px * px + py * py + pz * pz - 0.16
        if bq * bq - cq < 0.0: continue                   # misses a 0.4 m sphere about it
        c = math.cos(BIRDS[i, 6]); s = math.sin(BIRDS[i, 6])
        ia = 1.0 / BIRDS[i, 3]; ib = 1.0 / BIRDS[i, 4]; ic = 1.0 / BIRDS[i, 5]
        qx = (px * c + pz * s) * ia; qy = py * ib; qz = (-px * s + pz * c) * ic
        ex = (dx * c + dz * s) * ia; ey = dy * ib; ez = (-dx * s + dz * c) * ic
        aa = ex * ex + ey * ey + ez * ez; bb = qx * ex + qy * ey + qz * ez; cc = qx * qx + qy * qy + qz * qz - 1.0
        disc = bb * bb - aa * cc
        if disc < 0.0: continue
        t = (-bb - math.sqrt(disc)) / aa
        if t > 1e-4 and t < best: best = t; bi = i
    return best, bi

@njit(**NJ)
def bird_normal(i, x, y, z):
    px = x - BIRDS[i, 0]; py = y - BIRDS[i, 1]; pz = z - BIRDS[i, 2]
    c = math.cos(BIRDS[i, 6]); s = math.sin(BIRDS[i, 6])
    qx = (px * c + pz * s) / BIRDS[i, 3] ** 2; qy = py / BIRDS[i, 4] ** 2; qz = (-px * s + pz * c) / BIRDS[i, 5] ** 2
    nx = qx * c - qz * s; nz = qx * s + qz * c
    nl = math.sqrt(nx * nx + qy * qy + nz * nz) + 1e-12
    return nx / nl, qy / nl, nz / nl


@njit(**NJ)
def in_shadow(px, py, pz, lx, ly, lz):
    """is the sun hidden from this point.  The ray is free once it is over the enclosure wall."""
    tb, bi = birds(px, py, pz, lx, ly, lz, 60.0)
    if bi >= 0: return True
    t = 0.004
    for s in range(400):
        y = py + ly * t
        if y > PLH + 0.02: return False
        d = scene(px + lx * t, y, pz + lz * t)
        if d < 0.0004: return True
        t += d * 0.85
    return False


# ------------------------------------------------------------------ sandstone
@njit(**NJ)
def stone(px, py, pz, nx, ny, nz, fp):
    """albedo and shading normal.  fp is the size of a pixel on the surface, metres."""
    dA = wall(px, A - pz, py); dB = wall(pz, A + px, py); dC = wall(pz, A - px, py)
    w = 0; u = px; v = A - pz; nu = nx; nv = -nz
    if dB < dA and dB <= dC:
        w = 1; u = pz; v = A + px; nu = nz; nv = nx
    elif dC < dA and dC < dB:
        w = 2; u = pz; v = A - px; nu = nz; nv = -nx
    if pz < -A + 0.02:
        w = 3; u = px; v = -1.0; nu = nx; nv = nz
    ys = py + sag(u, v) if w < 3 else py
    horiz = ny > 0.6
    k = min(max(int(math.floor((v + (0.0 if horiz else 0.03)) / D)), 0), NK - 1)
    vr = v - k * D
    yr = ys + (k + 1) * H                                  # height over this terrace's ledge
    us = u - (k & 1) * 0.5 * P
    cell = int(math.floor(us / P + 0.5))
    su = abs(us - P * cell); a = su - LAND * 0.5
    jtop = min(max(int(math.ceil(a / TR - 1e-6)), 0), NR)   # which level of the pair stands at this u (0 landing .. 8 none)
    ftop = H - jtop * RI
    g = min(max(-py / DEPTH, 0.0), 1.0)
    lev = int(math.floor(ys / RI + 0.5))
    crs = int(math.floor(ys / RI + 1e-4))
    fy = ys / RI - math.floor(ys / RI)
    wp = 0.0; en = 9.0; back = 9.0; dtop = 9.0; dbot = 9.0; dj = 9.0
    bid = 0
    if horiz:
        if v < 0.0 or py > 0.5:                                           # the paving of the rim
            row = int(math.floor(v / 0.62))
            cu = u / 0.95 + 0.5 * (row & 1)
            bid = int(hash3(int(math.floor(cu)), row, 7 + w) * 65536)
            dj = min(min(cu - math.floor(cu), 1 - (cu - math.floor(cu))) * 0.95, min(v / 0.62 - row, 1 - (v / 0.62 - row)) * 0.62)
            en = -v; wp = 0.35
        elif vr < FW + 0.02 and yr > 0.1:                                 # a tread, or a landing
            if a <= 0.0:
                bid = int(hash3(cell, lev, 300 + w * 31 + k) * 65536)
                en = FW - vr; back = vr
                wp = math.exp(-((vr - 0.40) / 0.24) ** 2)
            else:
                f = a - (jtop - 1) * TR
                en = min(TR - f, FW - vr); back = min(f, vr * 2.0)
                bid = int(hash3(int(math.floor(us / TR)), lev, w * 31 + k) * 65536)
                wp = math.exp(-((vr - 0.40) / 0.21) ** 2) * (0.75 + 0.25 * math.exp(-((f - 0.15) / 0.1) ** 2))
            dj = 9.0
        else:                                                             # the walk along a ledge
            cu = u / 0.84 + 3.0 * hash3(lev, w, 11)
            bid = int(hash3(int(math.floor(cu)), lev, w + 50) * 65536)
            dj = min(cu - math.floor(cu), 1 - (cu - math.floor(cu))) * 0.84
            en = D - vr if vr < D + 0.05 else 9.0; back = max(vr - FW, 0.0) if (su < LAND * 0.5 + NT * TR) else vr
            wp = 0.75 * math.exp(-((vr - (FW + 0.27)) / 0.2) ** 2)
    elif abs(nv) > 0.6 or w == 3:                                         # a face looking into the tank
        cu = u / 0.74 + 5.3 * hash3(crs, w, 3)
        bid = int(hash3(int(math.floor(cu)), crs, w + 90) * 65536)
        dj = min(min(cu - math.floor(cu), 1 - (cu - math.floor(cu))) * 0.74, min(fy, 1 - fy) * RI)
        if w == 3:
            dtop = -py; dbot = 9.0
        elif py > 0.05:
            dtop = PLH - py; dbot = py
        elif vr > 0.3:
            dtop = ftop - yr; dbot = yr
        else:
            dtop = H - yr; dbot = max(yr - ftop, 0.0)
    else:                                                                 # a riser
        bid = int(hash3(int(math.floor(us / TR + 0.5)), crs, w * 31 + k + 200) * 65536)
        dtop = (1.0 - fy) * RI; dbot = fy * RI
    # ---- the block: honey, some rose, some grey, a few pale
    h1 = hash3(bid, 1, 5); h2 = hash3(bid, 2, 5); h3 = hash3(bid, 3, 5)
    ar = 0.690; ag = 0.490; ab = 0.265
    if h2 > 0.58 and h2 <= 0.78:
        m = min((h2 - 0.58) / 0.06, 1.0) * 0.50; ar += (0.600 - ar) * m; ag += (0.360 - ag) * m; ab += (0.215 - ab) * m
    elif h2 > 0.78 and h2 <= 0.92:
        m = min((h2 - 0.78) / 0.05, 1.0) * 0.7; ar += (0.400 - ar) * m; ag += (0.330 - ag) * m; ab += (0.230 - ab) * m
    elif h2 > 0.92:
        m = 0.7; ar += (0.730 - ar) * m; ag += (0.560 - ag) * m; ab += (0.330 - ab) * m
    tone = (1.0 + 0.20 * (h1 - 0.5)) * (1.06 - 0.20 * g)
    c1 = u; c2 = py; c3 = v
    tone *= 1.0 + 0.30 * (fbm3(c1 * 0.33 + 3.0 * w, c2 * 0.33, c3 * 0.33, 3) - 0.5)       # the bed the stone came from
    tone *= 1.0 + 0.16 * (fbm3(c1 * 3.1, c2 * 3.1, c3 * 3.1 + 7.0 * w, 3) - 0.5)
    ar *= tone; ag *= tone; ab *= tone
    # ---- weather
    if horiz:
        pat = sstep(0.36, 0.66, fbm3(c1 * 1.9 + 5.0, c2 * 1.1, c3 * 1.9, 3))                 # grey crust where no foot goes
        dk = (0.50 * pat + 0.12) * (1.0 - 0.85 * wp)
        dk += 0.30 * sstep(0.60, 0.70, noise3(c1 * 26.0, c2 * 26.0, c3 * 26.0)) * (1.0 - wp) * sstep(0.03, 0.01, fp)   # lichen
        ar += (0.21 - ar) * dk; ag += (0.185 - ag) * dk; ab += (0.15 - ab) * dk
        ar += (0.70 * tone - ar) * 0.34 * wp; ag += (0.50 * tone - ag) * 0.34 * wp; ab += (0.27 * tone - ab) * 0.34 * wp
        bk = 0.45 * math.exp(-max(back, 0.0) / 0.035)                                                # dirt lies at the back of a tread
        ar *= 1 - bk; ag *= 1 - bk; ab *= 1 - bk
        if fp < 0.03:
            chip = sstep(0.52, 0.66, fbm3(c1 * 23.0, c2 * 9.0, c3 * 23.0, 2)) * math.exp(-en / 0.022) * sstep(0.03, 0.012, fp)
            ar += (0.74 - ar) * 0.55 * chip; ag += (0.60 - ag) * 0.55 * chip; ab += (0.40 - ab) * 0.55 * chip
            ci = int(math.floor(c1 / 0.035)); cj = int(math.floor(c3 / 0.035))                 # what pigeons leave
            if hash3(ci, 91 + w, cj + lev * 7) < 0.012 * (1.0 - 0.7 * wp) * (1.0 - g):
                ex = c1 / 0.035 - ci - 0.5; ez = c3 / 0.035 - cj - 0.5
                if ex * ex + ez * ez < 0.03 + 0.15 * hash3(ci, 93, cj):
                    ar += (0.78 - ar) * 0.7; ag += (0.77 - ag) * 0.7; ab += (0.72 - ab) * 0.7
    else:
        dtop = max(dtop, 0.0)
        st = sstep(0.42, 0.72, fbm3(c1 * 4.6 + 9.0 * w + c3 * 4.6, c2 * 0.45, 3.0, 3))
        run = st * math.exp(-dtop / (0.35 + 1.1 * fbm3(c1 * 1.3 + c3 * 1.3, 0.5 * w, 2.0, 2))) * (0.72 + 0.28 * g)
        run += 0.30 * sstep(0.5, 0.8, fbm3(c1 * 0.8 + c3 * 0.8, c2 * 0.8, 4.0 + w, 3))
        run += 0.14 * math.exp(-dtop / 0.03) * (0.4 + 0.6 * noise3(c1 * 7.0 + c3 * 7.0, c2, 1.0))
        run = min(run, 0.85)
        ar += (0.085 - ar) * run; ag += (0.075 - ag) * run; ab += (0.062 - ab) * run
        bs = 0.30 * math.exp(-dbot / 0.045)
        ar *= 1 - bs; ag *= 1 - bs; ab *= 1 - bs
    edge = 1.0 - max(abs(nx), max(abs(ny), abs(nz)))                       # a worn arris shows the stone's own colour
    if edge > 0.02:
        ek = 0.55 * sstep(0.02, 0.22, edge)
        ar += (0.74 * tone - ar) * ek; ag += (0.575 * tone - ag) * ek; ab += (0.36 * tone - ab) * ek
    # ---- deeper, the stone has been under water more of its life
    alg = sstep(0.52, 1.0, g) * (0.25 + 0.75 * fbm3(c1 * 1.1 + c3 * 1.1, c2 * 1.4, 6.0 + w, 3))
    alg = min(alg, 0.8)
    ar += (0.105 - ar) * alg; ag += (0.118 - ag) * alg; ab += (0.070 - ab) * alg
    hw = py - YW
    if hw < 5.5:
        un = c1 + c3
        damp = 0.50 * sstep(1.5, 0.0, hw)
        ar *= 1 - damp; ag *= 1 - damp * 0.94; ab *= 1 - damp
        band = 0.0
        for i in range(6):                                                                 # where it stood longest it left lime
            hb = 0.07 + 0.30 * i + 0.11 * i * i + 0.035 * (noise3(un * 0.6, i * 3.7, 1.0) - 0.5)
            wd = 0.022 + 0.018 * i
            band += (0.66 - 0.085 * i) * math.exp(-((hw - hb) / wd) ** 2) * (0.45 + 0.55 * noise3(un * 5.0, c2 * 5.0, i * 1.7))
        band = min(band, 0.8)
        ar += (0.60 - ar) * band; ag += (0.57 - ag) * band; ab += (0.47 - ab) * band
        sl = sstep(0.10, 0.0, hw) * 0.85
        ar += (0.030 - ar) * sl; ag += (0.050 - ag) * sl; ab += (0.022 - ab) * sl
    # ---- joints, grain
    if fp < 0.03:
        jl = sstep(0.009, 0.002, dj) * sstep(0.03, 0.009, fp)
        ar *= 1 - 0.50 * jl; ag *= 1 - 0.50 * jl; ab *= 1 - 0.50 * jl
    if fp < 0.014:
        gk = sstep(0.014, 0.004, fp)
        gr = 1.0 + 0.16 * gk * (hash3(int(math.floor(px / 0.0045)), int(math.floor(py / 0.0045)), int(math.floor(pz / 0.0045))) - 0.5)
        gr *= 1.0 + 0.14 * gk * (noise3(px * 70.0, py * 70.0, pz * 70.0) - 0.5)
        ar *= gr; ag *= gr; ab *= gr
    # ---- each block sits a hair out of true; the surface has a tooth
    tx = 0.050 * (h3 - 0.5); ty = 0.050 * (hash3(bid, 4, 5) - 0.5)
    e = 0.02
    b0 = fbm3(px * 9.0, py * 9.0, pz * 9.0, 2)
    gx = (fbm3((px + e) * 9.0, py * 9.0, pz * 9.0, 2) - b0) / e
    gy = (fbm3(px * 9.0, (py + e) * 9.0, pz * 9.0, 2) - b0) / e
    gz = (fbm3(px * 9.0, py * 9.0, (pz + e) * 9.0, 2) - b0) / e
    am = 0.010
    if abs(ny) > 0.6:
        nx += tx - am * gx; nz += ty - am * gz
    elif abs(nx) > abs(nz):
        ny += tx - am * gy; nz += ty - am * gz
    else:
        ny += tx - am * gy; nx += ty - am * gx
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    return ar, ag, ab, nx / nl, ny / nl, nz / nl


@njit(**NJ)
def woman_mat(px, py, pz, part, nx, ny, nz):
    qx = px - WPX; qz = pz - WPZ
    a = (qx * WFX + qz * WFZ) / WSC; b = (py - WPY) / WSC; c = (qx * WRX + qz * WRZ) / WSC
    if part == 1:                                                         # deep yellow, hand-dyed, with its folds
        an = math.atan2(c, a)
        fold = math.sin(an * 11.0 + 2.5 * b + 2.0 * math.sin(b * 6.0)) * sstep(1.0, 0.5, b)
        fold += 0.6 * math.sin(b * 23.0 + an * 3.0) * sstep(0.9, 1.2, b) * sstep(1.45, 1.3, b)
        t = 1.0 + 0.10 * fold
        ar = 0.86 * t; ag = 0.50 * t; ab = 0.030 * t
        hem = 0.19 - (RI / WSC) * sstep(0.10, 0.2, a)                    # border at the hem: madder red, a gold line
        if b < hem + 0.02 * math.sin(an * 5.0):
            ar = 0.52; ag = 0.060; ab = 0.035
        if b > 0.95 and b < 1.40 and abs(b - 1.16 - 0.95 * c) < 0.016:    # the border of the pallu, across her back
            ar = 0.52; ag = 0.060; ab = 0.035
        nx += 0.16 * fold * (-math.sin(an) * WFX + math.cos(an) * WRX); nz += 0.16 * fold * (-math.sin(an) * WFZ + math.cos(an) * WRZ)
        nl = math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl
        return ar, ag, ab, nx, ny, nz
    if part == 2:
        if b > 1.25 and b < 1.40 and abs(c) > 0.15:                       # the sleeve of the blouse
            return 0.42, 0.055, 0.035, nx, ny, nz
        if abs(c) > 0.12 and ((b > 0.905 and b < 0.93) or (b > 1.56 and b < 1.59)):   # bangles
            return 0.50, 0.05, 0.04, nx, ny, nz
        return 0.30, 0.165, 0.095, nx, ny, nz
    return 0.30, 0.165, 0.040, nx, ny, nz


@njit(**NJ)
def sky(dx, dy, dz):
    """a white-hot May sky: pale at the horizon, blue only overhead, bright round the sun."""
    mu = max(dy, 0.0)
    k = mu ** 0.5
    r = 0.86 + (0.20 - 0.86) * k; g = 0.90 + (0.36 - 0.90) * k; b = 0.95 + (0.74 - 0.95) * k
    c = max(dx * LX + dy * LY + dz * LZ, 0.0)
    gl = 0.9 * c ** 24 + 0.25 * c ** 4
    s = 0.50
    return s * (r + gl * 1.0), s * (g + gl * 0.93), s * (b + gl * 0.80)

@njit(**NJ)
def ambient(py, ny):
    """what reaches a surface that we do not follow further: the remaining sky and the second bounce."""
    g = min(max(-py / DEPTH, 0.0), 1.0)
    sv = (0.42 - 0.30 * g) * (0.55 + 0.45 * ny)
    bo = 0.17 + 0.10 * (1.0 - g)
    return 0.20 * sv + 0.30 * bo, 0.29 * sv + 0.21 * bo, 0.45 * sv + 0.12 * bo


@njit(**NJ)
def seen(ox, oy, oz, dx, dy, dz, pa, t0, skyk):
    """radiance along a secondary ray: sky, or a surface under sun and ambient."""
    t = march(ox, oy, oz, dx, dy, dz, 140.0, pa)
    if dy < 0.0:
        tw = (YW - oy) / dy
        if tw > 0.0 and tw < t:                                           # it met the water: dark, with a little of the sky in it
            px = ox + dx * tw; pz = oz + dz * tw
            vis = 0.0 if in_shadow(px, YW + 0.002, pz, LX, LY, LZ) else 1.0
            return 0.012 + 0.018 * vis, 0.020 + 0.034 * vis, 0.012 + 0.012 * vis
    if t > 1e8:
        kr, kg, kb = sky(dx, dy, dz)
        return kr * skyk, kg * skyk, kb * skyk
    px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
    e = max(0.0015, 0.3 * pa * (t + t0))
    gx, gy, gz = normal(px, py, pz, e)
    wd, part = woman(px, py, pz)
    if wd < 0.004 + 0.2 * pa * t:
        ar, ag, ab, nx, ny, nz = woman_mat(px, py, pz, part, gx, gy, gz)
    else:
        ar, ag, ab, nx, ny, nz = stone(px, py, pz, gx, gy, gz, (t + t0) * pa * 2.0)
    cs = nx * LX + ny * LY + nz * LZ
    er, eg, eb = ambient(py, gy)
    if cs > 0.0 and (gx * LX + gy * LY + gz * LZ) > 0.0:
        if not in_shadow(px + gx * 0.004, py + gy * 0.004, pz + gz * 0.004, LX, LY, LZ):
            er += SUNR * cs; eg += SUNG * cs; eb += SUNB * cs
    return ar * er, ag * eg, ab * eb


@njit(**NJ)
def basis(nx, ny, nz):
    if abs(ny) < 0.9: tx = -nz; ty = 0.0; tz = nx
    else: tx = 1.0; ty = 0.0; tz = 0.0
    tl = math.sqrt(tx * tx + ty * ty + tz * tz); tx /= tl; ty /= tl; tz /= tl
    bx = ny * tz - nz * ty; by = nz * tx - nx * tz; bz = nx * ty - ny * tx
    return tx, ty, tz, bx, by, bz


# ------------------------------------------------------------------ the exposure
@njit(parallel=True, **NJ)
def expose(W, Hh, SS, cam, seed, X0, Y0, FWd, FHt):
    """returns: what the sun does (and the water), the albedo, the fill and bounce (as radiance), normals, distance."""
    DIR = np.zeros((Hh, W, 3), np.float32); ALB = np.zeros((Hh, W, 3), np.float32)
    IND = np.zeros((Hh, W, 3), np.float32); NRM = np.zeros((Hh, W, 3), np.float32); DEP = np.zeros((Hh, W), np.float32)
    N = SS * SS
    fx_, fy_, fz_, rx_, ry_, rz_, ux_, uy_, uz_, tu = cam
    pa = 2.0 * tu / FWd
    for iy in prange(Hh):
        for ix in range(W):
            st = ((ix + X0) * 7919 + iy * 104729 + seed * 31337) & 0xffffffff
            st, r0 = rnd(st); st, r0 = rnd(st)
            rot = r0 * 6.2831853
            st, r0 = rnd(st)
            sh = int(r0 * N)
            dr = 0.0; dg = 0.0; db = 0.0; qr = 0.0; qg = 0.0; qb = 0.0; ir = 0.0; ig = 0.0; ib = 0.0
            mx = 0.0; my = 0.0; mz = 0.0; dep = 0.0
            for s in range(N):
                sa = s // SS; sb = s % SS
                st, j1 = rnd(st); st, j2 = rnd(st)
                sx = (ix + X0 + (sa + j1) / SS - 0.5 * FWd) * pa
                sy = (0.5 * FHt - iy - Y0 - (sb + j2) / SS) * pa
                dx = fx_ + sx * rx_ + sy * ux_; dy = fy_ + sx * ry_ + sy * uy_; dz = fz_ + sx * rz_ + sy * uz_
                dl = math.sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl
                t = march(CX, CY, CZ, dx, dy, dz, 160.0, pa)
                tb, bi = birds(CX, CY, CZ, dx, dy, dz, min(t, 160.0))
                tw = 1e9
                if dy < 0.0: tw = (YW - CY) / dy
                st, j3 = rnd(st); st, j4 = rnd(st)
                # the sun is a disc
                tx, ty, tz, bx, by, bz = basis(LX, LY, LZ)
                rr = SUN_SPREAD * math.sqrt(j3); th = 6.2831853 * j4
                lx = LX + rr * (math.cos(th) * tx + math.sin(th) * bx); ly = LY + rr * (math.cos(th) * ty + math.sin(th) * by)
                lz = LZ + rr * (math.cos(th) * tz + math.sin(th) * bz)
                ll = math.sqrt(lx * lx + ly * ly + lz * lz); lx /= ll; ly /= ll; lz /= ll
                if tw < t and tw < tb:
                    # ------------------------------------------------ the water
                    px = CX + dx * tw; pz = CZ + dz * tw
                    wx = 0.0016 * (fbm3(px * 1.7, 1.0, pz * 1.7, 2) - 0.5) + 0.0006 * (noise3(px * 9.0, 2.0, pz * 9.0) - 0.5)
                    wz = 0.0016 * (fbm3(px * 1.7, 5.0, pz * 1.7, 2) - 0.5) + 0.0006 * (noise3(px * 9.0, 7.0, pz * 9.0) - 0.5)
                    wl = math.sqrt(1.0 + wx * wx + wz * wz); wx /= wl; wy = 1.0 / wl; wz /= wl
                    c = -(dx * wx + dy * wy + dz * wz)
                    F = 0.035 + 0.965 * (1.0 - c) ** 5
                    F = min(F * 3.0, 1.0)                                  # a skin of dust and pollen makes it more of a mirror
                    mxr = dx + 2 * c * wx; myr = dy + 2 * c * wy; mzr = dz + 2 * c * wz
                    ar, ag, ab = seen(px, YW + 0.002, pz, mxr, myr, mzr, pa, tw, 0.9)   # the one patch of sky it returns
                    # seen through
                    eta = 1.0 / 1.333
                    kk = 1.0 - eta * eta * (1.0 - c * c)
                    gxr = eta * dx + (eta * c - math.sqrt(kk)) * wx; gyr = eta * dy + (eta * c - math.sqrt(kk)) * wy
                    gzr = eta * dz + (eta * c - math.sqrt(kk)) * wz
                    t3 = march(px, YW - 0.002, pz, gxr, gyr, gzr, 5.0, pa)
                    vis = 0.0 if in_shadow(px, YW + 0.002, pz, lx, ly, lz) else 1.0
                    er, eg, eb = ambient(YW, 1.0)
                    er += SUNR * ly * vis; eg += SUNG * ly * vis; eb += SUNB * ly * vis
                    ur = 0.0; ug = 0.0; ub = 0.0
                    if t3 < 1e8:
                        qx = px + gxr * t3; qy = YW - 0.002 + gyr * t3; qz = pz + gzr * t3
                        hx, hy, hz = normal(qx, qy, qz, 0.003)
                        sr_, sg_, sb_, n1, n2, n3 = stone(qx, qy, qz, hx, hy, hz, (tw + t3) * pa * 2.0)
                        ur = sr_ * er * max(n2, 0.25); ug = sg_ * eg * max(n2, 0.25); ub = sb_ * eb * max(n2, 0.25)
                    else:
                        t3 = 5.0
                    a1 = math.exp(-SIGR * t3); a2 = math.exp(-SIGG * t3); a3 = math.exp(-SIGB * t3)
                    ur = ur * a1 + 0.006 * er * (1 - a1); ug = ug * a2 + 0.014 * eg * (1 - a2); ub = ub * a3 + 0.0045 * eb * (1 - a3)
                    lr = F * ar + (1 - F) * ur; lg = F * ag + (1 - F) * ug; lb = F * ab + (1 - F) * ub
                    # what floats: a scum gathered against the stone, and specks
                    ds = scene(px, YW - 0.01, pz)
                    sc = 0.22 * math.exp(-ds / 0.30) * sstep(0.40, 0.72, fbm3(px * 2.1, 5.0, pz * 2.1, 3))
                    sc += 0.40 * math.exp(-ds / 0.02)
                    sc += 0.12 * sstep(0.62, 0.80, fbm3(px * 0.9 + 4.0, 9.0, pz * 0.9, 4))
                    ci = int(math.floor(px / 0.03)); cj = int(math.floor(pz / 0.03))
                    if hash3(ci, 81, cj) < 0.0015: sc += 0.5
                    sc = min(sc, 0.8)
                    lr = lr * (1 - sc) + sc * 0.085 * er; lg = lg * (1 - sc) + sc * 0.115 * eg; lb = lb * (1 - sc) + sc * 0.040 * eb
                    dr += lr; dg += lg; db += lb
                    my += 1.0; dep += tw
                    continue
                if t > 1e8 and bi < 0:
                    ar, ag, ab = sky(dx, dy, dz)
                    dr += ar; dg += ag; db += ab; dep += 160.0; my += 1.0
                    continue
                spr = 0.0; spg = 0.0; spb = 0.0; shin = 0.0
                if bi >= 0 and tb < t:
                    t = tb
                    px = CX + dx * t; py = CY + dy * t; pz = CZ + dz * t
                    gx, gy, gz = bird_normal(bi, px, py, pz)
                    tn = BIRDS[bi, 7] * (0.8 + 0.4 * noise3(px * 30.0, py * 30.0, pz * 30.0))
                    ar = 0.060 * tn; ag = 0.059 * tn; ab = 0.062 * tn
                    nx = gx; ny = gy; nz = gz
                else:
                    px = CX + dx * t; py = CY + dy * t; pz = CZ + dz * t
                    gx, gy, gz = normal(px, py, pz, max(0.0012, 0.35 * pa * t))
                    fp = t * pa / max(-(gx * dx + gy * dy + gz * dz), 0.08)
                    wd, part = woman(px, py, pz)
                    if wd < 0.004 + 0.2 * pa * t:
                        ar, ag, ab, nx, ny, nz = woman_mat(px, py, pz, part, gx, gy, gz)
                        if part == 3: shin = 1.0
                    else:
                        ar, ag, ab, nx, ny, nz = stone(px, py, pz, gx, gy, gz, fp)
                ox = px + gx * 0.004; oy = py + gy * 0.004; oz = pz + gz * 0.004
                cs = nx * lx + ny * ly + nz * lz
                vis = 0.0
                if cs > 0.0 and (gx * lx + gy * ly + gz * lz) > 0.0:
                    if not in_shadow(ox, oy, oz, lx, ly, lz): vis = 1.0
                dr += ar * SUNR * cs * vis; dg += ag * SUNG * cs * vis; db += ab * SUNB * cs * vis
                if shin > 0.0:                                             # brass: the sun in it, and a little of the sky
                    hx = lx - dx; hy = ly - dy; hz = lz - dz
                    hl = math.sqrt(hx * hx + hy * hy + hz * hz)
                    nh = max((nx * hx + ny * hy + nz * hz) / hl, 0.0)
                    sp = 6.0 * nh ** 90 * vis + 0.5 * nh ** 12 * vis
                    vn = dx * nx + dy * ny + dz * nz
                    rx = dx - 2 * vn * nx; ry = dy - 2 * vn * ny; rz = dz - 2 * vn * nz
                    kr, kg, kb = sky(rx, max(ry, 0.0), rz)
                    wgt = 0.35 * sstep(-0.2, 0.3, ry)
                    dr += 0.95 * (sp * SUNR + wgt * kr); dg += 0.72 * (sp * SUNG + wgt * kg); db += 0.34 * (sp * SUNB + wgt * kb)
                # one ray into the hemisphere, cosine-weighted, stratified across the pixel's samples
                q = (s * 7 + sh) % N
                qa = q // SS; qb_ = q % SS
                st, j5 = rnd(st); st, j6 = rnd(st)
                u1 = (qa + j5) / SS; u2 = (qb_ + j6) / SS
                rr = math.sqrt(u1); th = 6.2831853 * u2 + rot
                tx, ty, tz, bx, by, bz = basis(gx, gy, gz)
                zz = math.sqrt(max(1.0 - u1, 0.0))
                hx = rr * (math.cos(th) * tx + math.sin(th) * bx) + zz * gx
                hy = rr * (math.cos(th) * ty + math.sin(th) * by) + zz * gy
                hz = rr * (math.cos(th) * tz + math.sin(th) * bz) + zz * gz
                er, eg, eb = seen(ox, oy, oz, hx, hy, hz, pa * 4.0, t, 1.0)
                er = min(er, 4.0); eg = min(eg, 4.0); eb = min(eb, 4.0)
                ir += er; ig += eg; ib += eb
                qr += ar; qg += ag; qb += ab
                mx += gx; my += gy; mz += gz; dep += t
            k = 1.0 / N
            DIR[iy, ix, 0] = dr * k; DIR[iy, ix, 1] = dg * k; DIR[iy, ix, 2] = db * k
            ALB[iy, ix, 0] = qr * k; ALB[iy, ix, 1] = qg * k; ALB[iy, ix, 2] = qb * k
            IND[iy, ix, 0] = ir * k; IND[iy, ix, 1] = ig * k; IND[iy, ix, 2] = ib * k
            ml = math.sqrt(mx * mx + my * my + mz * mz) + 1e-9
            NRM[iy, ix, 0] = mx / ml; NRM[iy, ix, 1] = my / ml; NRM[iy, ix, 2] = mz / ml
            DEP[iy, ix] = dep * k
    return DIR, ALB, IND, NRM, DEP


@njit(parallel=True, **NJ)
def settle(E, NRM, DEP, step):
    """the fill was measured with few rays; neighbours on the same face at the same distance agree with each other."""
    Hh, W = DEP.shape
    out = np.zeros_like(E)
    kw = (1.0, 4.0, 6.0, 4.0, 1.0)
    for y in prange(Hh):
        for x in range(W):
            z0 = DEP[y, x]; n0 = NRM[y, x, 0]; n1 = NRM[y, x, 1]; n2 = NRM[y, x, 2]
            sr = 0.0; sg = 0.0; sb = 0.0; sw = 0.0
            for j in range(-2, 3):
                yy = y + j * step
                if yy < 0 or yy >= Hh: continue
                for i in range(-2, 3):
                    xx = x + i * step
                    if xx < 0 or xx >= W: continue
                    dn = n0 * NRM[yy, xx, 0] + n1 * NRM[yy, xx, 1] + n2 * NRM[yy, xx, 2]
                    if dn < 0.0: continue
                    wn = dn ** 32
                    wz = math.exp(-abs(DEP[yy, xx] - z0) / (0.004 * z0 + 1e-4))
                    wgt = kw[i + 2] * kw[j + 2] * wn * wz
                    sr += wgt * E[yy, xx, 0]; sg += wgt * E[yy, xx, 1]; sb += wgt * E[yy, xx, 2]; sw += wgt
            if sw > 0.0:
                out[y, x, 0] = sr / sw; out[y, x, 1] = sg / sw; out[y, x, 2] = sb / sw
            else:
                out[y, x, 0] = E[y, x, 0]; out[y, x, 1] = E[y, x, 1]; out[y, x, 2] = E[y, x, 2]
    return out


@njit(parallel=True, **NJ)
def dust(W, Hh, cam, seed, nstep):
    """the air in the pit: how much of each line of sight stands in the sun.  Returns lit path and whole path, metres."""
    LIT = np.zeros((Hh, W), np.float32); LEN = np.zeros((Hh, W), np.float32)
    fx_, fy_, fz_, rx_, ry_, rz_, ux_, uy_, uz_, tu = cam
    pa = 2.0 * tu / W
    for iy in prange(Hh):
        for ix in range(W):
            sx = (ix + 0.5 - 0.5 * W) * pa; sy = (0.5 * Hh - iy - 0.5) * pa
            dx = fx_ + sx * rx_ + sy * ux_; dy = fy_ + sx * ry_ + sy * uy_; dz = fz_ + sx * rz_ + sy * uz_
            dl = math.sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl
            t = march(CX, CY, CZ, dx, dy, dz, 160.0, pa)
            if dy < 0.0: t = min(t, (YW - CY) / dy)
            t = min(t, 160.0)
            j = hash3(ix, iy, seed)
            lit = 0.0
            for s in range(nstep):
                ts = t * (s + j) / nstep
                if not in_shadow(CX + dx * ts, CY + dy * ts, CZ + dz * ts, LX, LY, LZ): lit += 1.0
            LIT[iy, ix] = lit / nstep * t; LEN[iy, ix] = t
    return LIT, LEN


# ------------------------------------------------------------------ the print
def camera(W):
    f = np.array([TX - CX, TY - CY, TZ - CZ]); f /= np.linalg.norm(f)
    r = np.cross([0.0, 1.0, 0.0], f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return (f[0], f[1], f[2], r[0], r[1], r[2], u[0], u[1], u[2], math.tan(math.radians(HFOV) * 0.5))


def develop(lin, scale, rng):
    """no lens.  A restrained curve, then plaster."""
    Hh, W = lin.shape[:2]
    lin = lin * EXPOSURE
    f = lambda x: x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)
    L = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])[..., None]
    y = 0.65 * lin * (f(L) / np.maximum(L, 1e-9)) + 0.35 * f(lin)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-9), 1 / 2.4) - 0.055)
    y = y + 0.22 * (y * y * (3 - 2 * y) - y)
    m = y.mean(-1, keepdims=True); y = np.clip(m + 1.07 * (y - m), 0.0, 1.0)
    # lime plaster: it has no true black and no gloss; a slow unevenness from the float, a fine tooth from the sand
    y = 0.030 + 0.950 * y
    slow = gaussian_filter(rng.standard_normal((Hh, W)), 60.0 * scale); slow /= slow.std()
    mid = gaussian_filter(rng.standard_normal((Hh, W)), 5.0 * scale); mid /= mid.std()
    y = y * (1.0 + 0.012 * slow + 0.006 * mid)[..., None]
    g = gaussian_filter(rng.standard_normal((Hh, W)), 0.6 * max(scale, 0.5)); g /= g.std()
    g2 = gaussian_filter(rng.standard_normal((Hh, W)), 1.3 * max(scale, 0.5)); g2 /= g2.std()
    lum = y.mean(-1, keepdims=True)
    amp = 0.004 + 0.009 * np.sqrt(np.clip(4 * lum * (1 - lum), 0, 1))
    y = y + amp * (0.8 * g + 0.5 * g2)[..., None]
    return np.clip(y, 0, 1)


def mark(img, scale):
    """the mason's mark, a triangle over a line, and the depth, cut lower right."""
    Hh, W = img.shape[:2]
    s = 30.0 * scale                                       # unit
    x0 = W - 9.2 * s; y0 = Hh - 4.3 * s
    strokes = [[(0.0, 1.55), (0.9, 0.0), (1.8, 1.55), (0.0, 1.55)], [(-0.25, 2.1), (2.05, 2.1)]]
    def glyph(pts, ox):
        return [(ox + 0.62 * a, 0.75 + 1.35 * b) for a, b in pts]
    strokes.append(glyph([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 3.0))                      # 3
    strokes.append(glyph([(0.25, 0.5), (1.0, 0.5)], 3.0))
    strokes.append(glyph([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)], 4.05))         # 0
    strokes.append(glyph([(0.0, 1.0), (0.0, 0.45), (0.65, 0.45), (0.65, 1.0)], 5.55))                 # m
    strokes.append(glyph([(0.65, 0.45), (1.3, 0.45), (1.3, 1.0)], 5.55))
    yy, xx = np.mgrid[0:int(4.2 * s), 0:int(9.0 * s)].astype(np.float64)
    dmin = np.full(yy.shape, 1e9)
    for st in strokes:
        for (a0, b0), (a1, b1) in zip(st[:-1], st[1:]):
            ax, ay, bx, by = a0 * s + 0.6 * s, b0 * s + 0.5 * s, a1 * s + 0.6 * s, b1 * s + 0.5 * s
            ex, ey = bx - ax, by - ay
            h = np.clip(((xx - ax) * ex + (yy - ay) * ey) / (ex * ex + ey * ey), 0, 1)
            dmin = np.minimum(dmin, np.hypot(xx - ax - ex * h, yy - ay - ey * h))
    wdt = 0.085 * s
    cut = np.clip((wdt - dmin) / (0.045 * s) + 0.5, 0, 1)
    lip = np.clip((wdt - np.roll(np.roll(dmin, int(max(1, 0.06 * s)), 0), int(max(1, 0.06 * s)), 1)) / (0.045 * s) + 0.5, 0, 1)
    X0, Y0 = int(x0), int(y0)
    reg = img[Y0:Y0 + cut.shape[0], X0:X0 + cut.shape[1]]
    reg = reg + 0.10 * np.clip(lip - cut, 0, 1)[..., None]                    # the lit lip of the cut
    reg = reg * (1.0 - 0.50 * cut[..., None])                                 # the cut, in its own shade
    img[Y0:Y0 + cut.shape[0], X0:X0 + cut.shape[1]] = reg
    return img


def count_steps():
    n = 0
    for k in range(NK):
        half = A - k * D                                   # the side walls' faces at this storey
        m = -40
        while m <= 40:
            c = m * P + (k & 1) * 0.5 * P
            for side in (-1, 1):
                for j in range(1, NR + 1):                 # seven treads and the landing's edge: eight risers
                    uc = c + side * (LAND * 0.5 + (j - 0.5) * TR)
                    if abs(uc) < half: n += 1
            m += 1
    return 3 * n


EXPOSURE = 0.52
DUST = 0.0016                                              # scattering per metre of lit air
DUSTC = (1.00, 0.90, 0.74)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    scale = float(args[0]) if len(args) > 0 else 1.0
    out = args[1] if len(args) > 1 else __file__.replace('.py', '.png')
    W = int(round(BW * scale)); Hh = int(round(BH * scale))
    SS = 3 if scale >= 0.99 else 2
    if len(args) > 2: SS = int(args[2])
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    print('steps on three sides: %d   depth %.1f m   rim %.1f m square' % (count_steps(), DEPTH, 2 * A), flush=True)
    cam = camera(W)
    crop = [a for a in sys.argv[1:] if a.startswith('--crop=')]            # a detail, for study: x0,y0,w,h in pixels at this scale
    X0 = Y0 = 0; FWd, FHt = W, Hh
    if crop:
        X0, Y0, W, Hh = [int(v) for v in crop[0][7:].split(',')]
    DIR, ALB, IND, NRM, DEP = expose(W, Hh, SS, cam, SEED % 1000, X0, Y0, FWd, FHt)
    print('exposed %.0fs' % (time.time() - t0), flush=True)
    st = max(int(round(scale * 2)), 1)
    E = settle(IND, NRM, DEP, 1)
    E = settle(E, NRM, DEP, max(st, 2))
    if scale >= 0.99: E = settle(E, NRM, DEP, 4)
    lin = DIR.astype(np.float64) + ALB.astype(np.float64) * E
    # dust
    w2 = max(FWd // 6, 100); h2 = int(round(w2 * FHt / FWd))
    LIT, LEN = dust(w2, h2, cam, 5, 40)
    LIT = np.asarray(Image.fromarray(gaussian_filter(LIT, 1.0)).resize((FWd, FHt), Image.BICUBIC), np.float64)[Y0:Y0 + Hh, X0:X0 + W]
    LEN = np.asarray(Image.fromarray(LEN).resize((FWd, FHt), Image.BICUBIC), np.float64)[Y0:Y0 + Hh, X0:X0 + W]
    trn = np.exp(-DUST * 0.8 * LEN)[..., None]
    sc = (DUST * LIT)[..., None] * np.array(DUSTC) * np.array([SUNR, SUNG, SUNB]) * 0.30
    sc = sc + (DUST * (LEN - LIT))[..., None] * np.array([0.10, 0.12, 0.15])
    lin = lin * trn + sc
    print('settled, dust %.0fs' % (time.time() - t0), flush=True)
    if '--keep' in sys.argv: np.save(out.replace('.png', '_lin.npy'), lin.astype(np.float32))
    img = develop(lin, scale, rng)
    if not crop: img = mark(img, scale)
    Image.fromarray((img * 255 + rng.random(img.shape)).clip(0, 255).astype(np.uint8)).save(out)
    print('saved', out, '%dx%d' % (W, Hh), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
