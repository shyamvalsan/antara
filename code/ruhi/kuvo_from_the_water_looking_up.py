"""
Kuvo, from the water, looking up

Ruhi Vaghela.

Survey.  The well shaft at the end of a stepwell: round, 8.40 m across the wall faces, the ground
27.60 m above the water and 29.80 m above the bed.  The wall is ringed every 3.40 m by a string course
(a cornice on ninety-six corbels, a roll, a band of forty-eight lozenges); between the rings sixteen
bays, a pilaster between each, a niche in every bay the stair does not cross and a lotus panel in every
bay it does.  A stair of single stones built into the wall, 1.00 m wide, winds up the shaft: one turn
in 6.80 m, forty-four risers of 0.155 m to the turn and a landing 1.50 m deep on brackets before the
doorway to each second storey of the stepwell.  Three turns, 132 risers, from the water to the top
doorway; the last seven metres of wall have no stair.  At the top the wall is corbelled in, five
courses, to a mouth 6.40 m across with a parapet, and a pair of stone beams for the pulleys lies
across it.
The lens stands on its tripod in the water, 1.80 m off the axis, 0.35 m above the surface, pointing at
the zenith; no tilt, so every ring stays a circle.  The sun is 14 degrees off the zenith, fifty-six
minutes after noon in the middle of May: its light comes down the mouth as one slanted shaft, lies on
the east wall as a long ellipse, pointed at the top, and narrows again to reach the water.

Usage:  python kuvo_from_the_water_looking_up.py [scale=1.0] [out.png] [draft|mid|final]
        scale 1.0 is 3000 x 3600; 0.5 or 0.25 are studies.
        --crop=x0,y0,x1,y1 renders only that part of the picture (pixels at the given scale).
Needs:  numpy, scipy, numba, pillow.  No source images.  The same picture every time.

Built as signed-distance solids in metres (the carving is cut into the solid, not painted on) and lit
by calculation:
  sun     - one hard directional light through the mouth
  sky     - the mouth as a pale area light
  bounce  - what the sun lights becomes many small lamps, each with its shadow (first bounce);
            the later bounces are gathered into a lattice of irradiance probes
  dust    - marched through the shaft of sun, with the shadows of the pulley beams in it; the damp air lies low
  print   - no lens; the eye's adaptation, a restrained curve, then the tooth of lime plaster
"""
import sys, math, time, warnings
import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaWarning
warnings.simplefilter('ignore', NumbaWarning)
from PIL import Image
from scipy.ndimage import gaussian_filter

SEED = 19850512
PI = math.pi
TAU = 2.0 * math.pi

# ------------------------------------------------------------------ the survey (metres)
R = 4.20                        # the wall face, from the axis
R0 = 3.20                       # the mouth
S = 3.40                        # ring to ring
P = 6.80                        # one turn of the stair
NT = 48                         # stones to the turn, landing included
NLAND = 5                       # of which the landing
RI = P / (NT - NLAND + 1)       # riser
DPHI = TAU / NT
WS, WL = 1.00, 1.50             # stair and landing, out from the wall
TH = 0.22                       # a tread stone
WY = -27.60                     # water
YBED = -29.80                   # the bed of the well: the depth that is signed
YL0 = WY + 0.15                 # the lowest landing
YR0 = YL0 - 0.25                # ring levels: YR0 + S j
NLAST = 3 * NT + NLAND - 1      # the last stone of the stair
YHEAD = -2.60                   # the corbelled head begins
HC, HP = 0.44, 0.20             # its courses: height, projection
YTOP = 0.80                     # top of the parapet
NB = 16                         # bays
BPHI = TAU / NB
BW = R * BPHI
PB = 3                          # the bay of the doorways
PHIP = PB * BPHI - 2.5 * DPHI   # where each landing begins
PHIC = PB * BPHI                # its middle
D96, D48, D12, D8 = TAU / 96, TAU / 48, TAU / 12, TAU / 8
# the pulley beams: a pair, chords of the mouth
_bb = math.radians(118.0)
BNX, BNZ, BOFF, BY = math.cos(_bb), math.sin(_bb), 1.25, 0.12
# the sun
ZEN, SAZ = math.radians(14.0), math.radians(9.0)
SUNX, SUNY, SUNZ = math.sin(ZEN) * math.cos(SAZ), math.cos(ZEN), math.sin(ZEN) * math.sin(SAZ)   # towards it
E0R, E0G, E0B = 1.00, 0.89, 0.70                      # on a surface square to it
LSR, LSG, LSB = 0.080, 0.094, 0.112                   # radiance of the May sky: white, a little blue
SUNA, SUNB = 0.32, 0.27                               # the eye closes on what the sun strikes (a shoulder)
# the lens
CAMX, CAMY, CAMZ = 1.80, WY + 0.35, 0.0
FOC = 1400.0                                          # pixels per unit tangent at scale 1
CXF, CYF = 0.50, 0.385                                # where the zenith falls in the picture
_rot = math.radians(-17.0)
CROT, SROT = math.cos(_rot), math.sin(_rot)
# the woman: on the stair, second turn
NW = NT + 18
_iw = NW - NT
PHIW = PHIP + (_iw + 0.5) * DPHI
YW = YL0 + P + RI * (_iw - (NLAND - 1))
XW, ZW = (R - 0.78) * math.cos(PHIW), (R - 0.78) * math.sin(PHIW)
RWX, RWZ = math.cos(PHIW), math.sin(PHIW)
# air
FOG = 0.0035                    # scattering per metre
BEAM = 0.0060                   # the same dust, seen against the sun

M_STONE, M_VEIL, M_SKIRT, M_SKIN, M_BRASS, M_BIRD, M_LEAF = 1, 2, 3, 4, 5, 6, 7
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

@njit(**NJ)
def wrap(a):
    return a - TAU * math.floor(a / TAU + 0.5)


# ------------------------------------------------------------------ the stair
@njit(**NJ)
def stair_y(tau, k):
    """the top of the stair, turn k, at tau stones round from the start of the landing."""
    return YL0 + P * k + RI * max(0.0, tau - (NLAND - 0.5))

@njit(**NJ)
def ring_free(tau, yr):
    """1 where a ring is clear of the stair; 0 where someone must walk past it."""
    f = 0.0
    for k in range(4):
        if k == 3 and tau > NLAND + 0.3: continue
        ys = stair_y(tau, k)
        g = sstep(ys - 0.62, ys - 0.47, yr) * sstep(ys + 2.25, ys + 2.10, yr)
        if g > f: f = g
    return 1.0 - f

def _bays():
    """what each bay holds, ring by ring: 0 nothing, 1 a niche, 2 a lotus panel; and which niches still hold a figure."""
    kind = np.zeros((NB, 8), np.int64); fig = np.zeros((NB, 8), np.int64)
    rs = np.random.default_rng(SEED)
    for j in range(7):
        yr = YR0 + S * j
        for b in range(NB):
            tc = ((b * BPHI - PHIP) % TAU) / DPHI
            k = 1
            for kk in range(4):
                if kk == 3 and tc > NLAND: continue
                ys = YL0 + P * kk + RI * max(0.0, tc - (NLAND - 0.5))
                if ys + 0.55 > yr + 0.9 and ys - 0.65 < yr + 2.6: k = 2
                if b == PB and abs(yr - (YL0 + P * kk - 0.25)) < 0.01: k = 0
            kind[b, j] = k
            fig[b, j] = 1 if rs.random() < 0.72 else 0
    return kind, fig
KIND, FIG = _bays()

def _plants():
    """saplings rooted in the joints: stems and leaves, set out once."""
    spec = [(203.0, -21.15, 1.25), (150.0, -17.75, 0.95), (262.0, -14.35, 1.0), (118.0, -21.2, 0.9),
            (20.0, -17.7, 0.9), (310.0, -10.95, 0.8), (176.0, -10.9, 0.85), (238.0, -7.5, 0.8), (330.0, -21.1, 1.0)]
    rs = np.random.default_rng(SEED + 7)
    n = len(spec); NS_, NL_ = 3, 11
    stems = np.zeros((n, NS_, 6)); leaves = np.zeros((n, NS_ * NL_, 10)); bound = np.zeros((n, 4))
    for p, (ph, y, sz) in enumerate(spec):
        ph = math.radians(ph)
        root = np.array([R * math.cos(ph), y, R * math.sin(ph)])
        inw = np.array([-math.cos(ph), 0.0, -math.sin(ph)]); tan = np.array([-math.sin(ph), 0.0, math.cos(ph)])
        up = np.array([0.0, 1.0, 0.0])
        pts = [root]
        for s in range(NS_):
            end = root + sz * (inw * (0.22 + 0.25 * rs.random()) + up * (0.05 + 0.40 * rs.random()) + tan * (rs.random() - 0.5) * 0.7)
            stems[p, s, :3] = root - inw * 0.03; stems[p, s, 3:] = end
            for l in range(NL_):
                f = 0.30 + 0.78 * (l + rs.random() * 0.6) / NL_
                c = root + (end - root) * f + sz * 0.22 * (rs.random(3) - 0.5)
                nn = np.array([(rs.random() - 0.5) * 2.2, 1.0, (rs.random() - 0.5) * 2.2]); nn /= np.linalg.norm(nn)
                leaves[p, s * NL_ + l, :3] = c; leaves[p, s * NL_ + l, 3:6] = nn
                leaves[p, s * NL_ + l, 6] = sz * (0.016 + 0.030 * rs.random() ** 2)
                ax = np.cross(nn, rs.random(3) - 0.5 + np.array([0.0, -0.8, 0.0])); ax /= np.linalg.norm(ax)
                leaves[p, s * NL_ + l, 7:10] = ax
                pts.append(c)
            pts.append(end)
        pts = np.array(pts); c = pts.mean(0)
        bound[p, :3] = c; bound[p, 3] = np.sqrt(((pts - c) ** 2).sum(1)).max() + 0.10 * sz
    return stems, leaves, bound
STEMS, LEAVES, PBOUND = _plants()

# the lines the water has left: height above today's water, width, strength
FLOOD = np.array([[5.15, 0.085, 0.80], [6.35, 0.06, 0.50], [7.9, 0.05, 0.38], [10.2, 0.05, 0.28], [3.9, 0.06, 0.35], [2.6, 0.05, 0.30]])
# pigeons in the air: x, y, z, heading, how far the wings are lifted, how far it is banked, how far the wings are spread
FLY = np.array([[0.30, -1.0, 0.75, 0.6, 0.35, 0.5, 1.0], [-1.15, -2.4, -0.55, 1.0, -0.20, -0.7, 0.72], [1.05, -0.5, -1.05, 0.3, 0.55, 0.2, 0.9],
                [-2.00, -8.5, 1.00, 2.4, 0.30, 0.9, 1.0], [1.50, -6.0, -2.00, -1.2, -0.25, -0.4, 0.8], [-2.6, -13.0, -1.3, 4.0, 0.45, 0.6, 0.6],
                [-0.5, -4.6, 1.6, 1.5, 0.1, -1.0, 0.85], [0.75, -1.6, 0.55, 0.75, 0.5, 0.3, 0.55]])
# and on the ledges: angle round the wall (degrees), ring, heading
SIT = np.array([[196.0, 2.0, 0.4], [199.5, 2.0, 2.6], [141.0, 3.0, -0.7], [248.0, 2.0, 1.9], [226.0, 4.0, 0.2]])


@njit(**NJ)
def rosette(u, v, h, r0):
    """a lotus cut in relief: u, v on the face, h out from it."""
    r = math.sqrt(u * u + v * v)
    d = max(r - r0, h - 0.030)
    d = min(d, max(abs(r - 0.93 * r0) - 0.07 * r0, h - 0.055))                 # its rim
    th = math.atan2(v, u)
    ths = th - D12 * math.floor(th / D12 + 0.5)
    pr = r * math.cos(ths) - 0.56 * r0; pt = r * math.sin(ths)
    pe = (math.sqrt((pr / (0.30 * r0)) ** 2 + (pt / (0.125 * r0)) ** 2) - 1.0) * 0.125 * r0
    d = min(d, max(pe + 0.9 * max(h - 0.035, 0.0) ** 2 / 0.03, h - 0.068))
    ths = th + 0.5 * D8; ths = ths - D8 * math.floor(ths / D8 + 0.5)
    pr = r * math.cos(ths) - 0.25 * r0; pt = r * math.sin(ths)
    pe = (math.sqrt((pr / (0.15 * r0)) ** 2 + (pt / (0.085 * r0)) ** 2) - 1.0) * 0.085 * r0
    d = min(d, max(pe, h - 0.085))
    d = min(d, math.sqrt(r * r + (h - 0.055) ** 2) - 0.13 * r0)
    return d


@njit(**NJ)
def sd_stairs(rho, phi, y, q, tau):
    i0 = int(math.floor(tau))
    if i0 > NT - 1: i0 = NT - 1
    k = int(math.floor((y + 0.12 - YL0 - RI * max(0, i0 - (NLAND - 1))) / P + 0.5))
    n0 = NT * k + i0
    d = 1e9
    for n in range(n0 - 1, n0 + 2):
        if n < 0 or n > NLAST: continue
        kk = n // NT; ii = n - NT * kk
        top = YL0 + P * kk + RI * max(0, ii - (NLAND - 1))
        land = ii < NLAND
        dph = wrap(phi - (PHIP + (ii + 0.5) * DPHI))
        a = rho * dph; aw = R * dph
        w = WL if land else WS
        th = 0.25 if land else TH
        qa = abs(a) - (0.5 * DPHI * rho + 0.03)
        qr = (R - w) - rho
        qy = abs(y - (top - 0.5 * th)) - 0.5 * th
        ox = max(qa + 0.012, 0.0); oy = max(qr + 0.012, 0.0); oz = max(qy + 0.012, 0.0)
        t = math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qa, max(qr, qy)) + 0.012, 0.0) - 0.012
        if t < d: d = t
        hb = (top - th) - y                                   # below the soffit
        if land:
            if ii == 0 or ii == 2 or ii == 4:                 # three brackets carry the landing
                for c in range(3):                            # each of three stones, the upper oversailing the lower
                    pj = 1.28 - 0.40 * c; h0 = 0.27 * c
                    e = math.sqrt(max(q - (pj - 0.27), 0.0) ** 2 + (hb - h0) ** 2) - 0.27
                    t = max(max(max(e, h0 - hb), hb - h0 - 0.27), abs(aw) - (0.12 - 0.012 * c))
                    if t < d: d = t
            else:                                             # a lotus under each span between them
                t = max(rosette(a, rho - (R - 0.80), hb, 0.27), -hb - 0.1)
                if t < d: d = t
            # the parapet, its coping, a lozenge cut in every stone of its face
            rp = rho - (R - WL + 0.07)
            t = sd_box2(rp, y - (top + 0.25), 0.07, 0.25)
            dl = (abs(a) / 0.13 + abs(y - (top + 0.25)) / 0.15 - 1.0) * 0.098
            t = max(t, -max(dl, rho - (R - WL + 0.03)))
            t = min(t, math.sqrt(rp * rp + (y - (top + 0.52)) ** 2) - 0.095)
            t = max(t, qa - 0.03)
            if t < d: d = t
        else:                                                 # a corbel under every tread
            e = math.sqrt((q / 0.40) ** 2 + (hb / 0.30) ** 2)
            t = max(max((e - 1.0) * 0.30, abs(aw) - 0.075), -hb)
            if t < d: d = t
            dl = (abs(a) / 0.15 + abs(y - (top - 0.5 * th)) / 0.075 - 1.0) * 0.067        # and a lozenge on its end
            t = max(d, -max(dl, rho - (R - w + 0.025)))
            if t > d and qa < 0.0: d = t
    return d


# ------------------------------------------------------------------ the woman
@njit(**NJ)
def sd_fig(lx, ly, lz):
    """feet at the origin, facing +z (up the stair, away from us), her left is -x.  Returns distance, material."""
    h = ly
    # ghaghra
    cz = 0.08 * sstep(0.85, 0.15, h)
    tt = min(max((0.98 - h) / 0.9, 0.0), 1.0)
    rx = 0.155 + 0.095 * tt; rz = 0.115 + 0.10 * tt
    ang = math.atan2(lz - cz, lx)
    fold = 0.011 * math.sin(11.0 * ang + 5.0 * h) * sstep(0.97, 0.55, h)
    e = math.sqrt((lx / rx) ** 2 + ((lz - cz) / rz) ** 2)
    hb = 0.05 + 0.13 * sstep(0.05, 0.30, lz)
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
    # arms: the left raised to steady the pot, the right hanging
    da = sd_cap(lx, ly, lz, -0.215, 1.25, 0.0, -0.325, 1.40, 0.05, 0.040)
    da = min(da, sd_cap(lx, ly, lz, 0.215, 1.25, 0.0, 0.25, 1.0, 0.03, 0.040))
    if da < d: d = da; m = M_VEIL
    da = sd_cap(lx, ly, lz, -0.325, 1.40, 0.05, -0.20, 1.64, 0.04, 0.031)
    da = min(da, math.sqrt((lx + 0.185) ** 2 + (ly - 1.68) ** 2 + (lz - 0.04) ** 2) - 0.040)
    da = min(da, sd_cap(lx, ly, lz, 0.25, 1.0, 0.03, 0.235, 0.77, 0.11, 0.031))
    da = min(da, math.sqrt((lx - 0.235) ** 2 + (ly - 0.73) ** 2 + (lz - 0.12) ** 2) - 0.038)
    if da < d: d = da; m = M_SKIN
    # feet: one on this tread, one on the next
    ex = (lx + 0.075) / 0.045; ey = (ly - 0.03) / 0.03; ez = (lz + 0.02) / 0.11
    df = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.03
    ex = (lx - 0.08) / 0.045; ey = (ly - RI - 0.03) / 0.03; ez = (lz - 0.40) / 0.11
    df = min(df, (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.03)
    if df < d: d = df; m = M_SKIN
    # the pot, on her head
    px = lx; py = ly - 1.715; pz = lz - 0.01
    ex = px / 0.150; ey = py / 0.128; ez = pz / 0.150
    dp = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.128
    rr = math.sqrt(px * px + pz * pz)
    dp = min(dp, max(rr - 0.060, abs(py - 0.135) - 0.04))
    dp = min(dp, math.sqrt((rr - 0.082) ** 2 + (py - 0.178) ** 2) - 0.016)
    if dp < d: d = dp; m = M_BRASS
    return d, m


# ------------------------------------------------------------------ pigeons, saplings
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
def sd_flyer(lx, ly, lz, yaw, lift, roll, span):
    c = math.cos(yaw); s = math.sin(yaw)
    u = lx * c + lz * s; w = -lx * s + lz * c
    c = math.cos(roll); s = math.sin(roll)
    u, ly = u * c + ly * s, -u * s + ly * c
    ex = u / 0.050; ey = ly / 0.045; ez = w / 0.135
    d = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.045
    d = smin(d, math.sqrt(u * u + (ly - 0.02) ** 2 + (w - 0.14) ** 2) - 0.032, 0.03)
    au = abs(u)
    hh = ly - lift * au - 0.01
    sw = 0.25 + 1.1 * (1.0 - span)
    e = math.sqrt(((au - 0.17 * span) / (0.165 * span)) ** 2 + ((w + 0.01 + sw * au) / (0.085 - 0.10 * max(au - 0.17 * span, 0.0))) ** 2)
    d = min(d, quad((e - 1.0) * 0.06, abs(hh) - 0.007) * 0.8)
    e = math.sqrt((u / (0.045 + 0.25 * max(-w - 0.10, 0.0))) ** 2 + ((w + 0.19) / 0.085) ** 2)
    d = min(d, quad((e - 1.0) * 0.04, abs(ly) - 0.006) * 0.8)
    return d

@njit(**NJ)
def sd_plant(p, x, y, z):
    d = 1e9
    for s in range(STEMS.shape[1]):
        t = sd_cap(x, y, z, STEMS[p, s, 0], STEMS[p, s, 1], STEMS[p, s, 2], STEMS[p, s, 3], STEMS[p, s, 4], STEMS[p, s, 5], 0.007)
        if t < d: d = t
    for l in range(LEAVES.shape[1]):
        px = x - LEAVES[p, l, 0]; py = y - LEAVES[p, l, 1]; pz = z - LEAVES[p, l, 2]
        h = px * LEAVES[p, l, 3] + py * LEAVES[p, l, 4] + pz * LEAVES[p, l, 5]
        al = px * LEAVES[p, l, 7] + py * LEAVES[p, l, 8] + pz * LEAVES[p, l, 9]
        r2 = max(px * px + py * py + pz * pz - h * h - al * al, 0.0)
        al = al / 1.7
        wd = LEAVES[p, l, 6] * (1.0 - 0.45 * al / LEAVES[p, l, 6])          # broad at the stalk, drawn to a point
        t = quad(math.sqrt(al * al + r2 * (LEAVES[p, l, 6] / max(wd, 0.004)) ** 2) - LEAVES[p, l, 6], abs(h) - 0.003) * 0.6
        if t < d: d = t
    return d


# ------------------------------------------------------------------ the wall
@njit(**NJ)
def wall(rho, phi, y, q, tau):
    """the wall face and everything cut in it or standing out of it; q is the distance in from the face."""
    # coursed ashlar: the joints are open a little
    ci = math.floor(y / 0.34); fv = y / 0.34 - ci
    cu = R * phi / 0.92 + 0.5 * (ci - 2.0 * math.floor(ci * 0.5)); fu = cu - math.floor(cu)
    ej = min(min(fv, 1.0 - fv) * 0.34, min(fu, 1.0 - fu) * 0.92)
    d = q + 0.009 * (1.0 - sstep(0.0, 0.016, ej))
    # ---- the head: five courses corbelled in to the mouth, a tooth under each
    if y > YHEAD - 0.5:
        for c in range(5):
            yc = YHEAD + HC * c
            t = quad(q - HP * (c + 1), yc - y) - 0.0
            if t < d: d = t
            a = rho * (phi - D96 * math.floor(phi / D96 + 0.5))
            t = sd_box(a, yc - y - 0.05, q - HP * c - 0.135, 0.045, 0.05, 0.095)
            if t < d: d = t
        if y > YHEAD: return d
    # ---- the nearest ring: bed mould and corbels below, cornice, roll and fillet above
    jr = math.floor((y - YR0) / S + 0.5)
    if jr < 0.0: jr = 0.0
    if jr > 7.0: jr = 7.0
    yr = YR0 + S * jr; v = y - yr
    if v > -0.55 and v < 0.65:
        fr = ring_free(tau, yr)
        pc = 0.05 + 0.21 * fr
        t = sd_box2(q - (pc - 0.5), v - 0.09, 0.475, 0.065) - 0.025
        if t < d: d = t
        t = math.sqrt((q - 0.025) ** 2 + (v - 0.265) ** 2) - 0.082
        if t < d: d = t
        t = sd_box2(q + 0.46, v - 0.27, 0.5, 0.09)
        if t < d: d = t
        hb = -v
        t = sd_box2(q - (0.045 + 0.03 * fr - 0.5), hb - 0.13, 0.5, 0.13)
        if t < d: d = t
        if fr > 0.5:
            a = R * (phi - D96 * math.floor(phi / D96 + 0.5))
            e = math.sqrt((q / 0.215) ** 2 + (hb / 0.215) ** 2)
            t = max(max((e - 1.0) * 0.215, abs(a) - 0.062), -hb)
            if t < d: d = t
    # ---- the storey above the ring below
    jf = math.floor((y - YR0) / S)
    if jf < 0.0: jf = 0.0
    if jf > 7.0: jf = 7.0
    j = int(jf); sy = y - (YR0 + S * jf)
    # the band of lozenges, sunk, a bead between each
    if sy > 0.25 and sy < 1.0:
        a = R * (phi - D48 * math.floor(phi / D48 + 0.5)); v = sy - 0.63
        dl = (abs(a) / 0.215 + abs(v) / 0.165 - 1.0) * 0.131
        d2 = max(abs(v) - 0.20, -dl)
        d = max(d, min(-d2, q + 0.04))
        dli = (abs(a) / 0.115 + abs(v) / 0.088 - 1.0) * 0.070
        d = max(d, min(-dli, q + 0.028))
        ab = abs(a) - 0.5 * D48 * R
        t = math.sqrt(ab * ab + v * v + (q + 0.045) ** 2) - 0.055
        if t < d: d = t
    if j <= 6:
        # the frieze of pendants under the bed mould
        if sy > 2.7 and sy < 3.3:
            a = R * (phi - D96 * math.floor(phi / D96 + 0.5))
            pd = min(sd_box2(a, sy - 3.07, 0.082, 0.09), math.sqrt(a * a + (sy - 2.98) ** 2) - 0.082)
            t = max(pd, q - 0.04)
            if t < d: d = t
        if sy > 0.75 and sy < 2.95:
            bf = math.floor(phi / BPHI + 0.5)
            a = R * (phi - BPHI * bf)
            b = int(bf) % NB
            if b < 0: b += NB
            kind = KIND[b, j]
            # the pilaster between bays
            ap = 0.5 * BW - abs(a)
            if ap < 0.45:
                rr = math.sqrt(ap * ap + q * q); th = math.atan2(q, ap)
                t = sd_box(ap, sy - 1.01, q + 0.33, 0.17, 0.06, 0.5)                               # plinth
                cu = (sy - 1.16) / 0.09                                                          # the pot it stands in
                rb = 0.125
                if abs(cu) < 1.0: rb += 0.040 * math.sqrt(1.0 - cu * cu)
                t = min(t, max(rr - rb, abs(sy - 1.16) - 0.09) * 0.85)
                t = min(t, max(max(sd_box2(ap, q, 0.105, 0.105), (abs(ap) + abs(q)) * 0.7071 - 0.122), abs(sy - 1.44) - 0.20))   # eight-sided
                t = min(t, math.sqrt((rr - 0.112) ** 2 + (sy - 1.67) ** 2) - 0.034)             # a band
                t = min(t, max(rr - 0.098 - 0.007 * math.cos(12.0 * th), abs(sy - 1.98) - 0.29) * 0.85)                        # fluted
                t = min(t, math.sqrt((rr - 0.100) ** 2 + (sy - 2.285) ** 2) - 0.026)
                cu = (sy - 2.385) / 0.075                                                        # the ribbed cushion
                rb = 0.108
                if abs(cu) < 1.0: rb += (0.060 + 0.011 * math.cos(14.0 * th)) * math.sqrt(1.0 - cu * cu)
                t = min(t, max(rr - rb, abs(sy - 2.385) - 0.075) * 0.75)
                t = min(t, sd_box(ap, sy - 2.505, q + 0.32, 0.19, 0.045, 0.5))                   # abacus
                t = min(t, max(sd_box2(ap, sy - 2.635, 0.21, 0.025) - 0.06, q - 0.12))           # the bracket's two arms
                t = min(t, sd_box(ap, sy - 2.71, q + 0.41, 0.11, 0.05, 0.5))
                if t < d: d = t
            if kind == 1 and abs(a) < 0.80:
                d2 = min(sd_box2(a, sy - 1.565, 0.42, 0.485), (math.sqrt((a / 0.42) ** 2 + ((sy - 2.05) / 0.27) ** 2) - 1.0) * 0.27)
                t = max(d2 - 0.10, q - 0.045)                                       # its frame
                t = min(t, max(abs(d2 - 0.10) - 0.022, q - 0.065))                  # and a bead round the frame
                t = min(t, sd_box(a, sy - 1.03, q + 0.36, 0.58, 0.05, 0.5))         # sill
                hc = 2.52 - 0.30 * q                                                # hood
                t = min(t, max(max((abs(sy - hc) - 0.03) * 0.95, q - 0.30), abs(a) - 0.62))
                t = min(t, sd_box(a, sy - 2.585, q + 0.45, 0.50, 0.045, 0.5))       # the stepped crown
                t = min(t, sd_box(a, sy - 2.675, q + 0.42, 0.33, 0.045, 0.5))
                t = min(t, sd_box(a, sy - 2.765, q + 0.39, 0.16, 0.045, 0.5))
                if t < d: d = t
                d = max(d, min(-d2, q + 0.45))
                if q < 0.1 and d2 < 0.1:
                    if FIG[b, j] == 1:                                               # what is left of a figure
                        hs = hash3(b, j, 5) - 0.5
                        ac = a - 0.05 * hs * (sy - 1.2)
                        ex = ac / 0.17; ey = (sy - 1.58) / 0.36; ez = (q + 0.33) / 0.13
                        t = (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.13
                        ex = ac / 0.26; ey = (sy - 1.80) / 0.12; ez = (q + 0.34) / 0.10
                        t = smin(t, (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.10, 0.08)
                        ex = ac / 0.23; ey = (sy - 1.30) / 0.17; ez = (q + 0.31) / 0.14
                        t = smin(t, (math.sqrt(ex * ex + ey * ey + ez * ez) - 1.0) * 0.14, 0.08)
                        t = smin(t, math.sqrt((ac - 0.03 * hs) ** 2 + (sy - 2.03) ** 2 + (q + 0.31) ** 2) - 0.115, 0.06)
                        t = smin(t, math.sqrt(ac * ac + (sy - 2.18) ** 2 + (q + 0.33) ** 2) - 0.07, 0.05)
                        t = min(t, sd_box(a, sy - 1.13, q + 0.30, 0.27, 0.05, 0.15))
                        t += 0.035 * (fbm3(a * 7.0 + b * 3.1, sy * 7.0 + j * 1.7, q * 7.0, 2) - 0.5)
                        t *= 0.8
                        if t < d: d = t
                    else:
                        t = sd_box(a, sy - 1.13, q + 0.30, 0.22, 0.05, 0.15)         # an empty seat
                        if t < d: d = t
            elif kind == 2 and abs(a) < 0.72:
                d2 = sd_box2(a, sy - 1.75, 0.50, 0.64)
                t = max(abs(d2 - 0.045) - 0.03, q - 0.03)
                if t < d: d = t
                d = max(d, min(-d2, q + 0.05))
                t = rosette(a, sy - 1.75, q + 0.05, 0.42)
                if t < d: d = t
                for sg in (-1.0, 1.0):                                                # a half lotus above and below
                    vv = (sy - 1.75) * sg - 0.64
                    if vv > -0.2:
                        t = max(max(math.sqrt(a * a + vv * vv) - 0.15, q + 0.012), vv)
                        if t < d: d = t
    # ---- the doorways from the stepwell, one to every landing
    aw = R * wrap(phi - PHIC)
    if abs(aw) < 1.6:
        for kk in range(4):
            yl = YL0 + P * kk
            if y > yl - 0.6 and y < yl + 3.0:
                d2 = sd_box2(aw, y - (yl + 1.05), 0.62, 1.05)
                t = max(max(d2 - 0.16, q - 0.07), yl - y)
                t = min(t, max(max(abs(d2 - 0.16) - 0.03, q - 0.10), yl - y))
                hc = yl + 2.42 - 0.28 * q
                t = min(t, max(max((abs(y - hc) - 0.035) * 0.95, q - 0.36), abs(aw) - 0.98))
                if t < d: d = t
                d = max(d, -sd_box(aw, y - (yl + 1.05), q + 0.85, 0.62, 1.05, 1.05))
    return d


@njit(**NJ)
def scene_full(x, y, z, prox, rough):
    """distance and material.  prox: far things may answer with the ball that holds them (for marching);
    rough: with the wear of the surface."""
    rho = math.sqrt(x * x + z * z)
    q = R - rho
    m = M_STONE
    if q > 2.25 and y < YHEAD - 0.8:
        d = q - 1.72
    else:
        phi = math.atan2(z, x)
        tau = (phi - PHIP) / DPHI
        tau -= NT * math.floor(tau / NT)
        if q > 0.80 and y < YHEAD - 0.8: d = q - 0.34
        else: d = wall(rho, phi, y, q, tau)
        if y < YL0 + 3 * P + 1.5:
            t = sd_stairs(rho, phi, y, q, tau)
            if t < d: d = t
        # ---- time: nothing is flat or sharp any more
        if rough and d < 0.05:
            d += 0.007 * (noise3(x * 1.9, y * 1.9, z * 1.9) - 0.5) + 0.006 * (noise3(x * 5.3, y * 5.3, z * 5.3) - 0.5)
            d += 0.0030 * (noise3(x * 19.0, y * 19.0, z * 19.0) - 0.5) + 0.0012 * (noise3(x * 61.0, y * 61.0, z * 61.0) - 0.5)
            c = noise3(x * 9.0 + 3.0, y * 9.0, z * 9.0 + 7.0)
            if c > 0.66: d += 0.014 * ((c - 0.66) / 0.34) ** 2
            c = noise3(x * 4.7 + 9.0, y * 4.7 + 1.0, z * 4.7)                        # where a face has spalled, or a corner gone
            if c > 0.68: d += 0.026 * ((c - 0.68) / 0.32) ** 2
            c = noise3(x * 31.0 + 1.0, y * 31.0, z * 31.0 + 2.0)
            if c > 0.74: d += 0.005 * ((c - 0.74) / 0.26) ** 2
        # ---- pigeons on the ledges
        for i in range(SIT.shape[0] if q < 1.0 else 0):
            ph = SIT[i, 0] * 0.017453293
            bx = (R - 0.13) * math.cos(ph); bz = (R - 0.13) * math.sin(ph); by = YR0 + S * SIT[i, 1] + 0.155
            bd = math.sqrt((x - bx) ** 2 + (y - by - 0.1) ** 2 + (z - bz) ** 2) - 0.26
            if bd < 0.03:
                t = sd_bird(x - bx, y - by, z - bz, SIT[i, 2])
                if t < d: d = t; m = M_BIRD
            elif prox and bd < d: d = bd
        # ---- saplings
        for p in range(PBOUND.shape[0] if q < 1.6 else 0):
            bd = math.sqrt((x - PBOUND[p, 0]) ** 2 + (y - PBOUND[p, 1]) ** 2 + (z - PBOUND[p, 2]) ** 2) - PBOUND[p, 3]
            if bd < 0.03:
                t = sd_plant(p, x, y, z)
                if t < d: d = t; m = M_LEAF
            elif prox and bd < d: d = bd
    # ---- the bed
    t = y - YBED
    if t < d: d = t
    # ---- the pulley beams
    if y > -1.5:
        t = sd_box2(abs(x * BNX + z * BNZ - BOFF) - 0.52, y - BY, 0.14, 0.19) - 0.01
        if t < d: d = t; m = M_STONE
    # ---- pigeons in the air
    if y > -14.5:
        for i in range(FLY.shape[0]):
            lx = x - FLY[i, 0]; ly = y - FLY[i, 1]; lz = z - FLY[i, 2]
            bd = math.sqrt(lx * lx + ly * ly + lz * lz) - 0.42
            if bd < 0.03:
                t = sd_flyer(lx, ly, lz, FLY[i, 3], FLY[i, 4], FLY[i, 5], FLY[i, 6])
                if t < d: d = t; m = M_BIRD
            elif prox and bd < d: d = bd
    # ---- the woman
    lx = x - XW; ly = y - YW; lz = z - ZW
    bd = math.sqrt(lx * lx + (ly - 0.95) ** 2 + lz * lz) - 1.12
    if bd < 0.03:
        ux = lx * RWX + lz * RWZ; uz = -lx * RWZ + lz * RWX
        t, mf = sd_fig(ux, ly, uz)
        if t < d: d = t; m = mf
    elif prox and bd < d:
        d = bd
    return d, m


@njit(**NJ)
def scene(x, y, z):
    return scene_full(x, y, z, True, True)

@njit(**NJ)
def scene_x(x, y, z):
    """exact near a surface: for the normal and for how shut in a point is."""
    return scene_full(x, y, z, False, True)


@njit(**NJ)
def march(ox, oy, oz, dx, dy, dz, tmax):
    """returns t, m.  m = 0: left through the mouth to the sky.  m = -1: nothing."""
    t = 0.0
    for s in range(900):
        py = oy + dy * t
        if py > YTOP + 0.05: return t, 0
        d, m = scene(ox + dx * t, py, oz + dz * t)
        if d < 0.00035 * t + 0.0004:
            return t, m
        t += d * 0.9
        if t > tmax: break
    return t, -1

@njit(**NJ)
def occl(ox, oy, oz, dx, dy, dz, maxt):
    t = 0.0
    for s in range(500):
        py = oy + dy * t
        if py > YTOP + 0.05: return False
        d, m = scene(ox + dx * t, py, oz + dz * t)
        if d < 0.0015: return True
        t += d * 0.9
        if t >= maxt: return False
    return False

@njit(**NJ)
def normal(px, py, pz, t):
    e = 0.0003 * t + 0.0012
    a = scene_x(px + e, py - e, pz - e)[0]; b = scene_x(px - e, py - e, pz + e)[0]
    c = scene_x(px - e, py + e, pz - e)[0]; d = scene_x(px + e, py + e, pz + e)[0]
    nx = a - b - c + d; ny = -a - b + c + d; nz = -a + b - c + d
    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / nl, ny / nl, nz / nl

@njit(**NJ)
def sun_vis(px, py, pz):
    """1 if the point sees the sun through the mouth.  The shadow is cast by the stone as cut, without its wear,
    so that a wall the sun only grazes is not eaten by its own grain."""
    s = (YTOP - py) / SUNY
    x0 = px + SUNX * s; z0 = pz + SUNZ * s
    if x0 * x0 + z0 * z0 > (R0 + 0.03) ** 2: return 0.0
    t = 0.0
    for i in range(500):
        y = py + SUNY * t
        if y > YTOP + 0.05: return 1.0
        d, m = scene_full(px + SUNX * t, y, pz + SUNZ * t, True, False)
        if d < 0.0015: return 0.0
        t += d * 0.9
    return 1.0


@njit(parallel=True, **NJ)
def sun_map(N):
    """for the dust: how far down each ray of the sun gets before stone stops it."""
    SM = np.full((N, N), YTOP + 1.0, np.float32)
    for i in prange(N):
        for j in range(N):
            ox = (i + 0.5) / N * 2.0 * R0 - R0; oz = (j + 0.5) / N * 2.0 * R0 - R0
            if ox * ox + oz * oz > R0 * R0: continue
            t, m = march(ox, YTOP + 0.04, oz, -SUNX, -SUNY, -SUNZ, 60.0)
            yh = YTOP + 0.04 - SUNY * t
            if m <= 0 or yh < WY: yh = WY
            SM[i, j] = yh
    return SM


# ------------------------------------------------------------------ stone
@njit(**NJ)
def stone_base(x, y, z):
    a = fbm3(x * 0.23 + 3.1, y * 0.31, z * 0.19 + 7.7, 3)
    b = fbm3(x * 0.9 + 11.0, y * 1.1 + 5.0, z * 0.8, 3)
    t = sstep(0.38, 0.62, a)
    r = 0.715 + (0.680 - 0.715) * t; g = 0.565 + (0.480 - 0.565) * t; bl = 0.385 + (0.400 - 0.385) * t   # honey to rose
    gk = min(sstep(0.50, 0.72, b) * 0.55 + 0.50 * sstep(-10.0, -1.0, y), 0.85)                          # grey near the top, where the air is
    rs = sstep(-15.0, -25.0, y) * 0.3                                                                  # rose, lower down
    r += (0.66 - r) * rs; g += (0.45 - g) * rs; bl += (0.39 - bl) * rs
    r += (0.56 - r) * gk; g += (0.52 - g) * gk; bl += (0.46 - bl) * gk
    dm = sstep(-17.0, -27.5, y)                                                                         # damp, lower down
    r *= 1 - 0.24 * dm; g *= 1 - 0.21 * dm; bl *= 1 - 0.20 * dm
    return r, g, bl


@njit(**NJ)
def material(px, py, pz, nx, ny, nz, m, fp):
    """albedo."""
    if m == M_VEIL:
        f = 0.9 + 0.2 * noise3(px * 40.0, py * 40.0, pz * 40.0)
        ar = 0.72 * f; ag = 0.045 * f; ab = 0.035 * f
        ly = py - YW
        if ly < 0.835 or abs(ly - 1.33) < 0.012: ar = 0.80; ag = 0.56; ab = 0.10                      # its border
        return ar, ag, ab
    if m == M_SKIRT:
        ly = py - YW
        f = 0.9 + 0.2 * noise3(px * 40.0, py * 40.0, pz * 40.0)
        ar = 0.54 * f; ag = 0.035 * f; ab = 0.045 * f
        if ly < 0.30 and ly > 0.20: ar = 0.78; ag = 0.52; ab = 0.09
        return ar, ag, ab
    if m == M_SKIN:
        ar = 0.34; ag = 0.20; ab = 0.125
        ly = py - YW
        if abs(ly - 1.60) < 0.02 or abs(ly - 0.82) < 0.02: ar = 0.80; ag = 0.76; ab = 0.66          # bangles
        return ar, ag, ab
    if m == M_BRASS:
        return 0.80, 0.57, 0.20
    if m == M_BIRD:
        f = 0.8 + 0.4 * noise3(px * 30.0, py * 30.0, pz * 30.0)
        return 0.10 * f, 0.105 * f, 0.125 * f
    if m == M_LEAF:
        f = 0.75 + 0.5 * noise3(px * 23.0, py * 23.0, pz * 23.0)
        return 0.105 * f, 0.185 * f, 0.045 * f
    rho = math.sqrt(px * px + pz * pz); q = R - rho
    phi = math.atan2(pz, px)
    u = R * phi
    jf = math.floor((py - YR0) / S)
    sy = py - (YR0 + S * jf)
    ar, ag, ab = stone_base(px, py, pz)
    tone = 1.0
    # each block is its own stone
    if q < 0.34 or py > YHEAD:
        ci = math.floor(py / 0.34)
        cu = u / 0.92 + 0.5 * (ci - 2.0 * math.floor(ci * 0.5))
        bi = int(ci) * 7 + int(math.floor(cu)) * 13
    else:
        tau = (phi - PHIP) / DPHI
        bi = int(math.floor(tau)) * 17 + int(math.floor(py / P + 0.5)) * 5 + 1000
        if fp < 0.02:                                                   # the stair stones still show the point chisel
            wv = rho + 0.02 * noise3(px * 6.0, py * 6.0, pz * 6.0) + 0.3 * hash3(bi, 5, 7)
            tone *= 1.0 + 0.075 * math.sin(wv * 190.0) * sstep(0.02, 0.008, fp) * (0.4 + 0.6 * noise3(px * 2.0, py * 2.0, pz * 2.0))
    hb = hash3(bi, 17, 3)
    tone *= 0.84 + 0.32 * hb
    hc = hash3(bi, 19, 3) - 0.5
    ar *= 1 + 0.14 * hc; ab *= 1 - 0.30 * hc
    tone *= 0.72 + 0.56 * fbm3(px * 2.3, py * 2.3, pz * 2.3, 4)
    tone *= 0.74 + 0.52 * fbm3(px * 7.0 + 3.0, py * 7.0, pz * 7.0, 5)
    tone *= 0.89 + 0.22 * noise3(px * 3.0 + 4.0 * noise3(px * 1.3, py * 1.3, pz * 1.3), py * 55.0, pz * 3.0) * noise3(px * 0.9, py * 2.0, pz * 0.9) * 2.0      # the bedding of the sandstone
    if fp < 0.03:
        sp = noise3(px * 83.0, py * 83.0, pz * 83.0)                    # pitting
        tone *= 1.0 - 0.16 * sstep(0.74, 0.92, sp) * sstep(0.03, 0.010, fp)
    if fp < 0.012:                                                      # grain
        gi = hash3(int(math.floor(px / 0.004)), int(math.floor(py / 0.004)), int(math.floor(pz / 0.004)))
        tone *= 1.0 + 0.13 * (gi - 0.5) * sstep(0.012, 0.004, fp)
    hw = py - WY
    if abs(ny) < 0.7:
        if q < 0.34 and (q > -0.02 or q < -0.43):                       # dirt in the joints
            ci = math.floor(py / 0.34); fv = py / 0.34 - ci
            cu = u / 0.92 + 0.5 * (ci - 2.0 * math.floor(ci * 0.5)); fu = cu - math.floor(cu)
            ej = min(fu, 1.0 - fu) * 0.92
            if q < 0.06: ej = min(ej, min(fv, 1.0 - fv) * 0.34)
            tone *= 1.0 - 0.42 * sstep(0.020, 0.004, ej + 0.006 * (noise3(px * 9.0, py * 9.0, pz * 9.0) - 0.5))
        # what has run down from every ledge
        run = sstep(0.42, 0.70, fbm3(u * 3.4 + 1.0, 0.37 + jf, py * 0.20, 3))
        dtop = S - sy
        if py > YR0 + 7 * S: dtop = max(YHEAD - py, 0.0)
        run *= (0.30 + 0.70 * math.exp(-dtop / 1.2)) * (0.45 + 0.55 * fbm3(u * 11.0, py * 1.1, jf, 2))
        ar *= 1 - 0.58 * run; ag *= 1 - 0.56 * run; ab *= 1 - 0.50 * run
        crust = sstep(0.56, 0.80, fbm3(px * 1.1 + 5.0, py * 1.1, pz * 1.1 + 2.0, 3))                # black crust, in patches
        ar *= 1 - 0.46 * crust; ag *= 1 - 0.45 * crust; ab *= 1 - 0.40 * crust
        # pigeons: white, down from where they sit
        if q < 0.30 and dtop < 1.6 and dtop > 0.20:
            ci = int(math.floor(u / 0.21))
            hh = hash3(ci, int(jf), 71)
            if hh < 0.10:
                fx = u / 0.21 - ci - 0.5 + 0.25 * (noise3(u * 3.0, py * 4.0, 1.0) - 0.5)
                ln = 0.5 + 9.0 * hh
                k = sstep(0.30, 0.05, abs(fx)) * sstep(ln, 0.2 * ln, dtop - 0.2) * (0.45 + 0.55 * noise3(u * 31.0, py * 9.0, 4.0))
                ar += (0.74 - ar) * 0.7 * k; ag += (0.73 - ag) * 0.7 * k; ab += (0.68 - ab) * 0.7 * k
    elif ny < -0.7:
        tone *= 0.86                                                                       # soffits: smoke, bats, lichen
        li = sstep(0.52, 0.74, fbm3(px * 1.9 + 9.0, py * 1.9, pz * 1.9, 3)) * 0.40
        ar += (0.50 - ar) * li; ag += (0.52 - ag) * li; ab += (0.44 - ab) * li
        sm = sstep(0.50, 0.78, fbm3(px * 0.9 + 3.0, py * 0.9, pz * 0.9 + 8.0, 3))
        tone *= 1.0 - 0.35 * sm
    else:
        dust = 0.5 + 0.5 * fbm3(px * 3.0, 2.0, pz * 3.0, 2)
        ar += (0.56 - ar) * 0.22 * dust; ag += (0.53 - ag) * 0.22 * dust; ab += (0.48 - ab) * 0.22 * dust
    # ---- lichen where the air is drier: grey in rounds, a little orange
    lk = sstep(8.0, 12.0, hw) * (0.6 + 0.4 * sstep(-12.0, -3.0, py))
    if lk > 0.0:
        li = sstep(0.60, 0.655, fbm3(px * 3.1 + 7.0, py * 3.1, pz * 3.1 + 1.0, 3)) * (0.55 + 0.45 * noise3(px * 40.0, py * 40.0, pz * 40.0)) * 0.50 * lk
        ar += (0.60 - ar) * li; ag += (0.60 - ag) * li; ab += (0.51 - ab) * li
        li = sstep(0.69, 0.73, fbm3(px * 4.3 + 1.0, py * 4.3 + 9.0, pz * 4.3, 3)) * 0.45 * lk
        ar += (0.62 - ar) * li; ag += (0.40 - ag) * li; ab += (0.15 - ab) * li
    # ---- moss in the damp
    if hw < 14.0:
        ms = sstep(0.54, 0.72, fbm3(px * 1.4 + 2.0, py * 0.9, pz * 1.4, 3)) * math.exp(-max(hw, 0.0) / 4.5) * 0.60
        ar += (0.105 - ar) * ms; ag += (0.140 - ag) * ms; ab += (0.060 - ab) * ms
    # ---- where the water has stood: dark and green where it stands most of the year, lime where it stood longest
    if hw < 13.0:
        wob = 0.10 * (fbm3(px * 0.8, 1.0, pz * 0.8, 2) - 0.5) + 0.03 * (noise3(px * 5.0, 2.0, pz * 5.0) - 0.5)
        sub = sstep(5.35, 4.95, hw + wob)
        fl = 0.55 + 0.45 * fbm3(px * 3.0, py * 3.0, pz * 3.0, 2)
        ar *= 1 - 0.40 * sub * fl; ag *= 1 - 0.33 * sub * fl; ab *= 1 - 0.38 * sub * fl
        damp = math.exp(-max(hw - 5.2, 0.0) / 2.6)
        ar *= 1 - 0.14 * damp; ag *= 1 - 0.12 * damp; ab *= 1 - 0.13 * damp
        for k in range(FLOOD.shape[0]):
            lv = FLOOD[k, 0] + wob * (0.6 + 0.15 * k)
            ln = math.exp(-((hw - lv) / FLOOD[k, 1]) ** 2)
            ln += 0.5 * sstep(0.0, -4.0 * FLOOD[k, 1], hw - lv) * sstep(-0.9, -0.2, hw - lv) * sstep(0.35, 0.7, fbm3(u * 6.0, py * 0.5, 3.0 + k, 2))   # and what ran down from the line
            ln *= (0.40 + 0.60 * fbm3(px * 4.0 + k, py * 9.0, pz * 4.0, 2)) * FLOOD[k, 2]
            ar += (0.70 - ar) * ln; ag += (0.68 - ag) * ln; ab += (0.61 - ab) * ln
        al = math.exp(-max(hw, 0.0) / 0.22) * (0.6 + 0.4 * fbm3(px * 7.0, py * 7.0, pz * 7.0, 2))
        if hw < 0.0: al = 0.85
        ar += (0.070 - ar) * al; ag += (0.105 - ag) * al; ab += (0.050 - ab) * al                    # weed at the line
    ar *= tone; ag *= tone; ab *= tone
    return ar, ag, ab


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
    # shut in: take the nearest probes there are, whichever way they lie
    gx = (px - G0[0]) / GH; gy = (py - G0[1]) / GH; gz = (pz - G0[2]) / GH
    ix = int(math.floor(gx + 0.5)); iy = int(math.floor(gy + 0.5)); iz = int(math.floor(gz + 0.5))
    r = 0.0; g = 0.0; b = 0.0; ws = 0.0
    for a in range(-2, 3):
        for bb in range(-2, 3):
            for c in range(-2, 3):
                i = ix + a; jj = iy + bb; kk = iz + c
                if i < 0 or jj < 0 or kk < 0 or i >= NX or jj >= NY or kk >= NZ: continue
                if not VAL[i, jj, kk]: continue
                wgt = 1.0 / (0.05 + (gx - i) ** 2 + (gy - jj) ** 2 + (gz - kk) ** 2) ** 2
                r += wgt * (wx * V[i, jj, kk, sx, 0] + wy * V[i, jj, kk, sy_, 0] + wz * V[i, jj, kk, sz, 0])
                g += wgt * (wx * V[i, jj, kk, sx, 1] + wy * V[i, jj, kk, sy_, 1] + wz * V[i, jj, kk, sz, 1])
                b += wgt * (wx * V[i, jj, kk, sx, 2] + wy * V[i, jj, kk, sy_, 2] + wz * V[i, jj, kk, sz, 2])
                ws += wgt
    if ws > 0.0: return 0.6 * r / ws, 0.6 * g / ws, 0.6 * b / ws
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
def make_lights(NA, NK_, seed):
    """the sun, followed down through the mouth to what it strikes; and the mouth itself as a piece of sky."""
    M = NA * NA
    LP = np.zeros((M, 3)); LN = np.zeros((M, 3)); LI = np.zeros((M, 3))
    area = PI * R0 * R0
    phi = SUNY * area / M
    for a in prange(NA):
        s = (seed + a * 7919) & 0xffffffff
        for b in range(NA):
            i = a * NA + b
            s, r1 = lcg(s); s, r2 = lcg(s)
            rr = R0 * math.sqrt((a + r1) / NA); th = TAU * (b + r2) / NA
            ox = rr * math.cos(th); oz = rr * math.sin(th); oy = YTOP + 0.04
            t, m = march(ox, oy, oz, -SUNX, -SUNY, -SUNZ, 60.0)
            tw = (WY - oy) / (-SUNY)
            if m <= 0 and tw > t: continue
            if tw < t:
                LP[i, 0] = ox - SUNX * tw; LP[i, 1] = WY + 0.01; LP[i, 2] = oz - SUNZ * tw
                LN[i, 1] = 1.0
                LI[i, 0] = E0R * phi * 0.020 / PI; LI[i, 1] = E0G * phi * 0.050 / PI; LI[i, 2] = E0B * phi * 0.028 / PI
                continue
            px = ox - SUNX * t; py = oy - SUNY * t; pz = oz - SUNZ * t
            gx, gy, gz = normal(px, py, pz, t)
            ar, ag, ab = material(px, py, pz, gx, gy, gz, m, 0.05)
            LP[i, 0] = px + gx * 0.02; LP[i, 1] = py + gy * 0.02; LP[i, 2] = pz + gz * 0.02
            LN[i, 0] = gx; LN[i, 1] = gy; LN[i, 2] = gz
            LI[i, 0] = E0R * phi * ar / PI; LI[i, 1] = E0G * phi * ag / PI; LI[i, 2] = E0B * phi * ab / PI
    Mk = NK_ * NK_
    KP = np.zeros((Mk, 3))
    s = (seed * 31 + 5) & 0xffffffff
    for a in range(NK_):
        for b in range(NK_):
            s, r1 = lcg(s); s, r2 = lcg(s)
            rr = R0 * math.sqrt((a + r1) / NK_); th = TAU * (b + r2) / NK_
            KP[a * NK_ + b, 0] = rr * math.cos(th); KP[a * NK_ + b, 1] = YTOP; KP[a * NK_ + b, 2] = rr * math.sin(th)
    return LP, LN, LI, KP, area / Mk


@njit(**NJ)
def first_light(px, py, pz, LP, LN, LI, KP, KA, ns, nk, s, out):
    """light arriving at p after one bounce of the sun, and straight from the sky in the mouth.
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
        k = 0 if lx > 0 else 1
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
    for j in prange(NY):
        out = np.zeros(18)
        for i in range(NX):
            for k in range(NZ):
                px = G0[0] + GH * i; py = G0[1] + GH * j; pz = G0[2] + GH * k
                if py < WY + 0.02 or py > YTOP - 0.02: continue
                if scene(px, py, pz)[0] < 0.05: continue
                VAL[i, j, k] = True
                s = (seed + i * 92821 + j * 68917 + k * 31337) & 0xffffffff
                first_light(px, py, pz, LP, LN, LI, KP, KA, ns, nk, s, out)
                for q in range(6):
                    for c in range(3): VF[i, j, k, q, c] = out[q * 3 + c]
    return VAL, VF


@njit(parallel=True, **NJ)
def probes_hits(VAL, G0, GH, RN, seed):
    """from every probe, RN rays: where each lands, which way that surface faces, what colour it is."""
    NX, NY, NZ = VAL.shape
    n = 0
    IDX = np.full((NX, NY, NZ), -1, np.int32)
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if VAL[i, j, k]: IDX[i, j, k] = n; n += 1
    HP = np.zeros((n, RN, 3), np.float32); HN = np.zeros((n, RN, 3), np.float32); HA = np.zeros((n, RN, 3), np.float32)
    HD = np.zeros((n, RN, 3), np.float32)
    for j in prange(NY):
        for i in range(NX):
            for k in range(NZ):
                p = IDX[i, j, k]
                if p < 0: continue
                px = G0[0] + GH * i; py = G0[1] + GH * j; pz = G0[2] + GH * k
                s = (seed + i * 1237 + j * 7793 + k * 15485) & 0xffffffff
                s, r0 = lcg(s); s, r1 = lcg(s)
                ca = math.cos(TAU * r1); sa = math.sin(TAU * r1)
                for q in range(RN):
                    yy = 1.0 - 2.0 * (q + r0) / RN
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
                    ar, ag, ab = material(hx, hy, hz, gx, gy, gz, m, 0.05)
                    HP[p, q, 0] = hx; HP[p, q, 1] = hy; HP[p, q, 2] = hz
                    HN[p, q, 0] = gx; HN[p, q, 1] = gy; HN[p, q, 2] = gz
                    HA[p, q, 0] = ar; HA[p, q, 1] = ag; HA[p, q, 2] = ab
    return IDX, HP, HN, HA, HD


@njit(parallel=True, **NJ)
def probes_bounce(VT, VAL, IDX, G0, GH, HP, HN, HA, HD):
    """one more bounce: every probe looks at what its rays hit and takes what those surfaces now give off."""
    NX, NY, NZ = VAL.shape
    RN = HP.shape[1]
    VM = np.zeros((NX, NY, NZ, 6, 3), np.float32)
    for j in prange(NY):
        for i in range(NX):
            for k in range(NZ):
                p = IDX[i, j, k]
                if p < 0: continue
                for q in range(RN):
                    ar = HA[p, q, 0]
                    if ar <= 0.0: continue
                    er, eg, eb = vol_lookup(VT, VAL, G0, GH, HP[p, q, 0], HP[p, q, 1], HP[p, q, 2], HN[p, q, 0], HN[p, q, 1], HN[p, q, 2])
                    lr = ar * er * 4.0 / RN; lg = HA[p, q, 1] * eg * 4.0 / RN; lb = HA[p, q, 2] * eb * 4.0 / RN     # (alb E / pi) (4 pi / RN)
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
def air(ox, oy, oz, dx, dy, dz, t, s, VE, VAL, G0, GH, SM, nb):
    """transmittance to t, and the light the dust sends along the ray: the haze of the well, and the shaft."""
    T = math.exp(-FOG * t)
    lr = 0.0; lg = 0.0; lb = 0.0; hr = 0.0; hg = 0.0; hb = 0.0
    nf = 5
    s, r = lcg(s)
    for i in range(nf):
        ti = (i + r) / nf * t
        er, eg, eb = vol_mean(VE, VAL, G0, GH, ox + dx * ti, oy + dy * ti, oz + dz * ti)
        w = FOG * math.exp(-FOG * ti) * t / nf / PI
        lr += w * er; lg += w * eg; lb += w * eb
    # the shaft: a leaning cylinder under the mouth
    kx = SUNX / SUNY; kz = SUNZ / SUNY
    X0 = ox + kx * (YTOP - oy); dX = dx - kx * dy
    Z0 = oz + kz * (YTOP - oy); dZ = dz - kz * dy
    A = dX * dX + dZ * dZ; B = X0 * dX + Z0 * dZ; C = X0 * X0 + Z0 * Z0 - R0 * R0
    disc = B * B - A * C
    NM = SM.shape[0]
    if disc > 0.0 and A > 1e-12:
        sq = math.sqrt(disc)
        t0 = max((-B - sq) / A, 0.0); t1 = min((-B + sq) / A, t)
        if t1 > t0:
            s, r = lcg(s)
            acc = 0.0
            for i in range(nb):
                ti = t0 + (i + r) / nb * (t1 - t0)
                px = ox + dx * ti; py = oy + dy * ti; pz = oz + dz * ti
                if py > YTOP:
                    acc += math.exp(-FOG * ti); continue
                ii = int((px + kx * (YTOP - py) + R0) / (2.0 * R0) * NM); jj = int((pz + kz * (YTOP - py) + R0) / (2.0 * R0) * NM)
                if ii < 0 or jj < 0 or ii >= NM or jj >= NM: continue
                if py > SM[ii, jj] + 0.03:
                    dens = (0.70 + 0.60 * fbm3(px * 0.9, py * 0.45, pz * 0.9, 2)) * (0.30 + 1.0 * math.exp(-(py - WY) / 7.0))     # dust is not even, and the damp air lies low
                    acc += math.exp(-FOG * ti) * dens
            cg = dx * SUNX + dy * SUNY + dz * SUNZ                                           # it shines most looked at against the sun
            ph = 0.70 + 0.30 * 0.75 / (1.25 - cg) ** 1.5
            w = BEAM * acc * (t1 - t0) / nb / (4.0 * PI) * ph
            hr = w * E0R; hg = w * E0G * 0.93; hb = w * E0B * 0.80
    return T, lr, lg, lb, hr, hg, hb, s


# ------------------------------------------------------------------ the exposure
@njit(parallel=True, **NJ)
def expose(W, H, F, SS, NS, NK, nee, VT, VM, VE, VAL, G0, GH, SM, LP, LN, LI, KP, KA, seed, OX, OY, WF, HF):
    C = np.zeros((H, W, 3), np.float32)        # everything except the first-bounce light and the sun
    CS = np.zeros((H, W, 3), np.float32)       # what the sun lights directly, kept apart
    CB = np.zeros((H, W, 3), np.float32)       # the shaft in the dust
    EF = np.zeros((H, W, 3), np.float32)       # first-bounce irradiance on the true surface
    AL = np.zeros((H, W, 3), np.float32)       # albedo (through the air)
    NG = np.zeros((H, W, 3), np.float32)
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
                u = (fx - WF * CXF) / F; v = (fy - HF * CYF) / F
                dx = -(u * SROT + v * CROT); dz = u * CROT - v * SROT; dy = 1.0
                dl = math.sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl
                t, m = march(CAMX, CAMY, CAMZ, dx, dy, dz, 80.0)
                if m < 0: continue
                T, fr, fg, fb, hr, hg, hb, s = air(CAMX, CAMY, CAMZ, dx, dy, dz, t, s, VE, VAL, G0, GH, SM, 24)
                cr += fr; cg += fg; cb += fb; vr += hr; vg += hg; vb += hb
                if m == 0:
                    cr += 0.060 * T; cg += 0.064 * T; cb += 0.069 * T                         # the sky itself: white, a little blue
                    continue
                px = CAMX + dx * t; py = CAMY + dy * t; pz = CAMZ + dz * t
                gx, gy, gz = normal(px, py, pz, t)
                if m == M_LEAF and gx * dx + gy * dy + gz * dz > 0.0:
                    gx = -gx; gy = -gy; gz = -gz
                fp = t / F / max(-(gx * dx + gy * dy + gz * dz), 0.08)
                ar, ag, ab = material(px, py, pz, gx, gy, gz, m, fp)
                nx = gx; ny = gy; nz = gz
                qx = px + gx * 0.012; qy = py + gy * 0.012; qz = pz + gz * 0.012
                # ---- the sun
                er = 0.0; eg = 0.0; eb = 0.0
                cs = nx * SUNX + ny * SUNY + nz * SUNZ
                lit = 0.0
                if m == M_LEAF:                                                               # a leaf lets some through
                    sg = 1.0 if cs > 0.0 else -1.0
                    lit = sun_vis(px + gx * 0.012 * sg, py + gy * 0.012 * sg, pz + gz * 0.012 * sg)
                    cs = cs if cs > 0.0 else -0.7 * cs
                    cs = SUNA * cs / (1.0 + cs / SUNB)
                    ur += ar * E0R * cs * lit / PI * T; ug += ag * E0G * cs * lit / PI * T * 1.15; ub += ab * E0B * cs * lit / PI * T
                elif cs > 0.0:
                    lit = sun_vis(px + gx * 0.028, py + gy * 0.028, pz + gz * 0.028)
                    cs = SUNA * cs / (1.0 + cs / SUNB)
                    ur += ar * E0R * cs * lit / PI * T; ug += ag * E0G * cs * lit / PI * T; ub += ab * E0B * cs * lit / PI * T
                # ---- the later bounces, from the probes, less where the surface is shut in
                occ = 0.0; sca = 1.0
                for k in range(4):
                    hh = 0.03 + 0.085 * k
                    occ += max(hh - scene_x(px + gx * hh, py + gy * hh, pz + gz * hh)[0], 0.0) / hh * sca; sca *= 0.75
                ao = min(max(1.0 - 0.33 * occ, 0.0), 1.0)
                if nee:
                    mr, mg, mb = vol_lookup(VM, VAL, G0, GH, px, py, pz, nx, ny, nz)
                    k_ao = 0.35 + 0.65 * ao
                else:
                    mr, mg, mb = vol_lookup(VT, VAL, G0, GH, px, py, pz, nx, ny, nz)
                    k_ao = 0.25 + 0.75 * ao
                er += mr * k_ao; eg += mg * k_ao; eb += mb * k_ao
                lr = ar * er / PI; lg = ag * eg / PI; lb = ab * eb / PI
                if m == M_BRASS:
                    vn = dx * nx + dy * ny + dz * nz
                    rx = dx - 2 * vn * nx; ry = dy - 2 * vn * ny; rz = dz - 2 * vn * nz
                    sp = max(rx * SUNX + ry * SUNY + rz * SUNZ, 0.0) ** 24
                    svp = sun_vis(qx, qy, qz) if lit == 0.0 else lit
                    ur += 0.9 * sp * svp * E0R * T; ug += 0.7 * sp * svp * E0G * T; ub += 0.3 * sp * svp * E0B * T
                    mr2, mg2, mb2 = vol_lookup(VT, VAL, G0, GH, px, py, pz, rx, ry, rz)
                    lr += 0.25 * mr2 / PI * 0.9; lg += 0.25 * mg2 / PI * 0.7; lb += 0.25 * mb2 / PI * 0.3
                cr += lr * T; cg += lg * T; cb += lb * T
                # ---- the first bounce and the sky, by sampling the small lamps
                if nee:
                    e0 = 0.0; e1 = 0.0; e2 = 0.0
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
                    EF[iy, ix, 0] += e0; EF[iy, ix, 1] += e1; EF[iy, ix, 2] += e2
                    AL[iy, ix, 0] += ar * T; AL[iy, ix, 1] += ag * T; AL[iy, ix, 2] += ab * T
                    NG[iy, ix, 0] += gx; NG[iy, ix, 1] += gy; NG[iy, ix, 2] += gz
                    PP[iy, ix, 0] += px; PP[iy, ix, 1] += py; PP[iy, ix, 2] += pz
                    WG[iy, ix] += 1.0
            n = SS * SS
            C[iy, ix, 0] = cr / n; C[iy, ix, 1] = cg / n; C[iy, ix, 2] = cb / n
            CS[iy, ix, 0] = ur / n; CS[iy, ix, 1] = ug / n; CS[iy, ix, 2] = ub / n
            CB[iy, ix, 0] = vr / n; CB[iy, ix, 1] = vg / n; CB[iy, ix, 2] = vb / n
            w = WG[iy, ix]
            if w > 0.0:
                for c in range(3):
                    EF[iy, ix, c] /= w; AL[iy, ix, c] /= n; NG[iy, ix, c] /= w; PP[iy, ix, c] /= w
    return C, CS, CB, EF, AL, NG, PP, WG


@njit(parallel=True, **NJ)
def atrous(A, NG, PP, WG, step, sig):
    """one pass of an edge-keeping blur of A, guided by surface position and facing."""
    H, W = WG.shape
    OA = np.zeros_like(A)
    kern = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for y in prange(H):
        for x in range(W):
            if WG[y, x] <= 0.0: continue
            nx = NG[y, x, 0]; ny = NG[y, x, 1]; nz = NG[y, x, 2]
            nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9; nx /= nl; ny /= nl; nz /= nl
            ws = 0.0
            a0 = 0.0; a1 = 0.0; a2 = 0.0
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
            if ws > 0.0:
                OA[y, x, 0] = a0 / ws; OA[y, x, 1] = a1 / ws; OA[y, x, 2] = a2 / ws
            else:
                for c in range(3): OA[y, x, c] = A[y, x, c]
    return OA


# ------------------------------------------------------------------ plaster
EXPOSURE = 125.0
ADAPT = 0.20

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


GLYPH = {'0': [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]], '1': [[(0.5, 0), (0.5, 1)]],
         '2': [[(0, 0), (1, 0), (1, .5), (0, .5), (0, 1), (1, 1)]], '3': [[(0, 0), (1, 0), (1, 1), (0, 1)], [(.25, .5), (1, .5)]],
         '4': [[(0, 0), (0, .5), (1, .5)], [(1, 0), (1, 1)]], '5': [[(1, 0), (0, 0), (0, .5), (1, .5), (1, 1), (0, 1)]],
         '6': [[(1, 0), (0, 0), (0, 1), (1, 1), (1, .5), (0, .5)]], '7': [[(0, 0), (1, 0), (1, 1)]],
         '8': [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], [(0, .5), (1, .5)]], '9': [[(1, .5), (0, .5), (0, 0), (1, 0), (1, 1), (0, 1)]],
         '.': [[(0.5, 0.93), (0.5, 1.0)]], ' ': [],
         'm': [[(0, 1), (0, .45), (.85, .45), (.85, 1)], [(.85, .45), (1.7, .45), (1.7, 1)]]}

def sign(img, scale, depth):
    """the mason's mark, a triangle over a line, and the depth, cut lower right."""
    Hh, W = img.shape[:2]
    s = 30.0 * scale
    strokes = [[(0.0, 1.55), (0.9, 0.0), (1.8, 1.55), (0.0, 1.55)], [(-0.25, 2.1), (2.05, 2.1)]]
    ox = 3.0
    for ch in '%.1f m' % depth:
        for st in GLYPH[ch]:
            strokes.append([(ox + 0.62 * a, 0.75 + 1.35 * b) for a, b in st])
        ox += 1.05 if ch not in '. ' else 0.62
    wd = ox + 1.6
    x0 = W - (wd + 1.6) * s; y0 = Hh - 4.6 * s
    yy, xx = np.mgrid[0:int(3.6 * s), 0:int(wd * s)].astype(np.float64)
    dmin = np.full(yy.shape, 1e9)
    for st in strokes:
        for (a0, b0), (a1, b1) in zip(st[:-1], st[1:]):
            ax, ay, bx, by = a0 * s + 0.6 * s, b0 * s + 0.5 * s, a1 * s + 0.6 * s, b1 * s + 0.5 * s
            ex, ey = bx - ax, by - ay
            h = np.clip(((xx - ax) * ex + (yy - ay) * ey) / (ex * ex + ey * ey), 0, 1)
            dmin = np.minimum(dmin, np.hypot(xx - ax - ex * h, yy - ay - ey * h))
    wdt = 0.085 * s
    cut = np.clip((wdt - dmin) / (0.045 * s) + 0.5, 0, 1)
    sh = int(max(1, 0.06 * s))
    lip = np.clip((wdt - np.roll(np.roll(dmin, sh, 0), sh, 1)) / (0.045 * s) + 0.5, 0, 1)
    X0, Y0 = int(x0), int(y0)
    reg = img[Y0:Y0 + cut.shape[0], X0:X0 + cut.shape[1]]
    reg = reg + 0.10 * np.clip(lip - cut, 0, 1)[..., None]                    # the lit lip of the cut
    reg = reg * (1.0 - 0.50 * cut[..., None])                                 # the cut, in its own shade
    img[Y0:Y0 + cut.shape[0], X0:X0 + cut.shape[1]] = np.clip(reg, 0, 1)
    return img


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    scale = float(args[0]) if len(args) > 0 else 1.0
    out = args[1] if len(args) > 1 else __file__.replace('.py', '.png')
    mode = args[2] if len(args) > 2 else ('final' if scale >= 0.99 else 'draft')
    W = int(round(3000 * scale)); H = int(round(3600 * scale)); F = FOC * scale
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    if mode == 'final':
        GH = 0.40; RN = 96; pns, pnk = 160, 64; SS, NS, NK, nee = 2, 3, 2, 1; LG = (72, 24)
    elif mode == 'mid':
        GH = 0.45; RN = 64; pns, pnk = 96, 48; SS, NS, NK, nee = 2, 5, 3, 1; LG = (56, 20)
    else:
        GH = 0.55; RN = 48; pns, pnk = 64, 32; SS, NS, NK, nee = 2, 0, 0, 0; LG = (40, 16)
    LP, LN, LI, KP, KA = make_lights(LG[0], LG[1], SEED % 100000)
    o = np.argsort(LP[:, 1] + 0.001 * LP[:, 0], kind='stable'); LP = LP[o]; LN = LN[o]; LI = LI[o]
    print('lamps %d (lit %d)  %.0fs' % (LP.shape[0], int((LI.sum(1) > 0).sum()), time.time() - t0), flush=True)
    NX = int((2 * R - 0.2) / GH) + 1; NY = int((YTOP - WY - 0.2) / GH) + 1
    G0 = np.array([-0.5 * GH * (NX - 1), WY + 0.12, -0.5 * GH * (NX - 1)])
    VAL, VF = probes_first(NX, NY, NX, G0, GH, LP, LN, LI, KP, KA, pns, pnk, SEED % 1000)
    print('probes %dx%dx%d, %d in air  %.0fs' % (NX, NY, NX, int(VAL.sum()), time.time() - t0), flush=True)
    IDX, HP, HN, HA, HD = probes_hits(VAL, G0, GH, RN, SEED % 777)
    VM = np.zeros_like(VF)
    for it in range(6):
        VM = probes_bounce(VF + VM, VAL, IDX, G0, GH, HP, HN, HA, HD)
    del HP, HN, HA, HD
    VT = VF + VM
    VE = np.ascontiguousarray(VT.mean(3))
    print('bounces  %.0fs   first %.5f  later %.5f' % (time.time() - t0, VF[VAL].mean(), VM[VAL].mean()), flush=True)
    SM = sun_map(768)
    OX = OY = 0; WF = W; HF = H; crop = None
    for a in sys.argv:
        if a.startswith('--crop='):
            crop = [int(v) for v in a[7:].split(',')]
            OX, OY = crop[0], crop[1]; W = crop[2] - crop[0]; H = crop[3] - crop[1]
    C, CS, CB, EF, AL, NG, PP, WG = expose(W, H, F, SS, NS, NK, nee, VT, VM, VE, VAL, G0, GH, SM, LP, LN, LI, KP, KA, SEED % 9973, OX, OY, WF, HF)
    print('exposed  %.0fs' % (time.time() - t0), flush=True)
    lin = C.astype(np.float64) + CS.astype(np.float64)
    cb = CB.astype(np.float64)                                               # dust lights the dust beside it: the shaft has no knife edge
    g1 = np.stack([gaussian_filter(cb[..., c], 9.0 * scale) for c in range(3)], -1)
    g2 = np.stack([gaussian_filter(cb[..., c], 45.0 * scale) for c in range(3)], -1)
    lin += 0.68 * cb + 0.27 * g1 + 0.05 * g2
    if nee:
        A = EF
        for step in (1, 2, 4, 8) if scale > 0.6 else (1, 2, 4):
            A = atrous(A, NG, PP, WG, step, 0.05)
        lin += AL.astype(np.float64) * A / PI
    img = develop(lin, scale, rng, EXPOSURE)
    if crop is None: img = sign(img, scale, -YBED)
    Image.fromarray((img * 255 + rng.random(img.shape)).clip(0, 255).astype(np.uint8)).save(out)
    print('saved', out, '%dx%d  %.0fs' % (W, H, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
