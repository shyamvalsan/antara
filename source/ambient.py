#!/usr/bin/env python3
"""Room music for the gallery: one parameterised ambient renderer, seven loops.

Pure synthesis (numpy/scipy). Everything is rendered circularly (notes wrap
around the loop end, reverb is a circular FFT convolution, filters run on a
3x tiled signal, static partials are snapped to whole cycles per loop) so the
end flows seamlessly into the start. Writes web/music/<slug>.mp3, 96 s each.

Tonal scheme (so any 3 s crossfade between rooms is consonant)
  * one tuning reference: A4 = 440 Hz, equal temperament, no per-room detune
    offsets (any 'off-centre' effect is symmetric beating around true pitch);
  * one pitch collection: D major / B minor = D E F# G A B C#, every sustained
    pitch in every room is drawn from it, with D / A (and B, G) as roots;
  * every room carries D and/or A in its drone, so the fifth D-A is the shared
    ground; chord roots only move among D, B, G, A (all diatonic neighbours).

Run: .venv/bin/python ambient.py [slug ...]      (default: all seven)
"""
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
DUR = 96.0                       # seconds; every period below divides it
N = int(SR * DUR)
TARGET_LUFS = -17.0
PEAK_CEIL_DB = -6.0
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "web" / "music"
T = np.arange(N) / SR


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def snap(f):
    """Nearest frequency with a whole number of cycles per loop."""
    return max(1, round(f * DUR)) / DUR


def pan2(s, pan):
    return np.stack([s * np.sqrt(0.5 * (1 - pan)), s * np.sqrt(0.5 * (1 + pan))], 1)


def add_wrap(buf, sig, start):
    """Add sig into circular buffer buf (n,) or (n,2) starting at sample start."""
    start %= len(buf)
    L = len(sig)
    first = min(L, len(buf) - start)
    buf[start:start + first] += sig[:first]
    rest = L - first
    while rest > 0:
        k = min(rest, len(buf))
        buf[:k] += sig[first:first + k]
        first += k
        rest -= k


def circ_filter(x, sos):
    """Filtering without edge transients: tile, filter, cut the middle copy."""
    n = len(x)
    t = np.concatenate([x, x, x], axis=0)
    return signal.sosfilt(sos, t, axis=0)[n:2 * n]


def lp(x, fc, order=2):
    return circ_filter(x, signal.butter(order, fc, "low", fs=SR, output="sos"))


def hp(x, fc, order=2):
    return circ_filter(x, signal.butter(order, fc, "high", fs=SR, output="sos"))


def smooth_noise(n, cutoff_hz, rng):
    """Slow random modulation in roughly [-1, 1], circularly periodic."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1 / (1 + (f / cutoff_hz) ** 4)
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / (np.abs(y).max() + 1e-12)


def env(n, a, r):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    e[n - nr:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    return e


def lfo(period, phase=0.0):
    """Sine with an integer number of cycles per loop."""
    assert abs(DUR / period - round(DUR / period)) < 1e-9, period
    return np.sin(2 * np.pi * T / period + phase)


# ---------------------------------------------------------------- pads
def pad_voice(freq, n, pan, rng, hmax=8, rolloff=1.35, fmax=3200.0,
              det=(0.0006, 0.0016), lfo_rate=(0.02, 0.09), lfo_depth=0.35,
              periodic=False):
    """Additive pad voice: detuned partials with slow per-partial amplitude LFOs.
    periodic=True (n must be N) snaps every frequency to whole loop cycles."""
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    for h in range(1, hmax + 1):
        f = freq * h
        if f > fmax:
            break
        amp = 1.0 / h ** rolloff
        for d in (-1, 1):
            fd = f * (1 + d * rng.uniform(*det))
            if fd > fmax:
                continue
            lf = rng.uniform(*lfo_rate)
            if periodic:
                fd, lf = snap(fd), snap(lf)
            ph = rng.uniform(0, 2 * np.pi)
            lf_s = 1 - lfo_depth + lfo_depth * np.sin(2 * np.pi * lf * t + rng.uniform(0, 6.28))
            s = np.sin(2 * np.pi * fd * t + ph) * amp * lf_s
            p = np.clip(pan + rng.uniform(-0.08, 0.08), -1, 1)
            L += s * np.sqrt(0.5 * (1 - p))
            R += s * np.sqrt(0.5 * (1 + p))
    return L, R


def build_pads(rng, spec):
    """Crossfaded chord centres. spec: segments=[dict(drone=[(midi,gain)], chord=[midi])],
    xf (s), chord_gain, chord_pan, vp (pad_voice kwargs), breath [(period, depth, phase)]."""
    segs = spec["segments"]
    ns = len(segs)
    SEG = N // ns
    xf = int(spec["xf"] * SR)
    vp = spec.get("vp", {})
    out = np.zeros((N, 2))
    for si, seg in enumerate(segs):
        start = si * SEG - xf // 2
        n = SEG + xf
        e = env(n, spec["xf"], spec["xf"])
        for m, g in seg["drone"]:
            L, R = pad_voice(hz(m), n, rng.uniform(-0.15, 0.15), rng, **vp)
            add_wrap(out, np.stack([L, R], 1) * (e * g)[:, None], start)
        for m in seg["chord"]:
            pan = rng.uniform(*spec.get("chord_pan", (-0.3, 0.3)))
            L, R = pad_voice(hz(m), n, pan, rng, **vp)
            g = spec["chord_gain"] * rng.uniform(0.7, 1.0)
            add_wrap(out, np.stack([L, R], 1) * (e * g)[:, None], start)
    br = np.ones(N)
    for per, depth, ph in spec.get("breath", []):
        br += depth * lfo(per, ph)
    out *= br[:, None]
    return out


def static_pad(rng, notes, **vp):
    """Whole-loop periodic voices: notes = [(midi, gain, pan)]."""
    out = np.zeros((N, 2))
    for m, g, pan in notes:
        L, R = pad_voice(hz(m), N, pan, rng, periodic=True, **vp)
        out += np.stack([L, R], 1) * g
    return out


# ---------------------------------------------------------------- note events
BELL = [(1.0, 1.0, 3.2), (2.0, 0.42, 1.88), (3.003, 0.16, 1.28), (4.01, 0.07, 0.94), (5.4, 0.035, 0.71)]


def voice(freq, dur, rng, partials=BELL, attack=0.035, release=0.8, twin=(0.35, 1.0025, 2.4),
          glide=None, jitter=0.0008, fmax=3000.0):
    """One decaying tone. partials = [(ratio, amp, tau_seconds)];
    glide=(depth, tau): starts depth*freq sharp and settles; twin=(amp, ratio, tau)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for r, a, tau in partials:
        f = freq * r * (1 + rng.uniform(-jitter, jitter))
        if f > fmax:
            continue
        if glide:
            gd, gt = glide
            phase = 2 * np.pi * (f * t + f * gd * gt * (1 - np.exp(-t / gt)))
        else:
            phase = 2 * np.pi * f * t
        s += a * np.sin(phase + rng.uniform(0, 6.28)) * np.exp(-t / tau)
    if twin:
        ta, tr, tt = twin
        s += ta * np.sin(2 * np.pi * freq * tr * t) * np.exp(-t / tt)
    na = max(2, int(attack * SR))
    s[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    nr = int(release * SR)
    s[n - nr:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr)
    return s


def put(layer, s, t, pan):
    add_wrap(layer, pan2(s, pan), int(t * SR))


def echo(x, taps, damp=2500):
    """Circular ping-pong echo: taps=[(delay_s, gain)], alternate channels."""
    out = x.copy()
    for k, (d, g) in enumerate(taps):
        y = np.roll(x, int(d * SR), axis=0)
        if k % 2 == 0:
            y = y[:, ::-1]
        out += g * lp(y, damp)
    return out


def bright_mod(x, lo_fc, hi_fc, period, phase, order=2):
    """Slowly opening/closing lowpass: crossfade two filtered copies by a loop-periodic LFO."""
    g = (0.5 + 0.5 * lfo(period, phase))[:, None]
    lo, hi = lp(x, lo_fc, order), lp(x, hi_fc, order)
    return lo + (hi - lo) * g


# ---------------------------------------------------------------- reverb
DEFAULT_BANDS = [(60, 400, 1.1), (400, 1500, 1.0), (1500, 4000, 0.55), (4000, 9000, 0.25)]


def make_ir(rng, rt60=5.0, length=6.5, bands=DEFAULT_BANDS):
    n = int(length * SR)
    ir = np.zeros((n, 2))
    t = np.arange(n) / SR
    for c in range(2):
        noise = rng.standard_normal(n)
        y = np.zeros(n)
        for lo, hi, mult in bands:
            sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
            y += signal.sosfilt(sos, noise) * 10 ** (-3 * t / (rt60 * mult))
        y *= np.minimum(1, t / 0.04)
        ir[:, c] = y
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    return ir


def circ_reverb(x, ir):
    n = len(x)
    out = np.zeros_like(x)
    X = [np.fft.rfft(x[:, ch]) for ch in range(2)]
    for c in range(2):
        H = np.fft.rfft(ir[:, c], n)
        for ch in range(2):
            out[:, c] += np.fft.irfft(X[ch] * H, n) * 0.7
    return out


# ================================================================ rooms
# midi: D2 38  A2 45  B2 47  D3 50  E3 52  F#3 54  G3 55  A3 57  B3 59  C#4 61
#       D4 62  E4 64  F#4 66  G4 67  A4 69  B4 71  C#5 73  D5 74  E5 76  F#5 78
#       A5 81  B5 83  D6 86  E6 88  F#6 90

def room_nave(rng):
    """The long gallery at dusk: the original piece's settings, 4 centres x 24 s."""
    pads = build_pads(rng, dict(
        xf=8.0, chord_gain=0.30,
        breath=[(DUR / 4, 0.12, 0.7), (DUR / 9, 0.06, 2.0)],
        segments=[
            dict(drone=[(38, .55), (45, .28)], chord=[57, 61, 64, 69, 73]),   # D maj9
            dict(drone=[(43, .55), (50, .28)], chord=[59, 62, 66, 69, 74]),   # G maj9 (Lydian)
            dict(drone=[(45, .55), (52, .28)], chord=[57, 61, 64, 67, 71]),   # A 9sus-ish
            dict(drone=[(47, .55), (54, .28)], chord=[59, 62, 66, 69, 73]),   # B m9
        ]))
    penta = [62, 64, 66, 69, 71, 74, 76, 78, 81]
    chords = [{57, 61, 64, 69, 73}, {59, 62, 66, 69, 74}, {57, 61, 64, 67, 71}, {59, 62, 66, 69, 73}]
    seg_s = DUR / 4
    bells = np.zeros((N, 2))
    t, last = 4.0 + rng.uniform(0, 3), None
    while t < DUR - 4:
        seg = min(int(t / seg_s), 3)
        pcs = {m % 12 for m in chords[seg]}
        pool = [m for m in penta if m % 12 in pcs] or penta
        if rng.random() < 0.3:
            pool = penta
        m = int(rng.choice(pool))
        if m == last:
            m = int(rng.choice(penta))
        last = m
        put(bells, voice(hz(m), 9.0, rng) * rng.uniform(0.35, 0.8), t, rng.uniform(-0.7, 0.7))
        if rng.random() < 0.25:
            m2 = int(rng.choice([p for p in penta if p < m] or penta))
            put(bells, voice(hz(m2), 9.0, rng) * rng.uniform(0.25, 0.5), t + rng.uniform(1.5, 3.0), rng.uniform(-0.7, 0.7) * -0.6)
        t += rng.uniform(7.0, 14.0)
    return dict(
        layers=[dict(x=pads, dry=.55, wet=.55, lp_dry=1800, lp_wet=1800),
                dict(x=bells, dry=.35, wet=1.1, lp_dry=2800)],
        rev=dict(rt60=5.0, length=6.5),
        master=dict(width=0.6, wobble=0.15, hp=50, lp=6500, dip=-4.0, lev=0.25))


def room_stanzin(rng):
    """Night sky at 4,500 m: thin, cold, glassy; long silences; far-off wind."""
    # distant wind: a deep open fifth plus band-limited noise that comes and goes
    wind = static_pad(rng, [(38, 0.5, -0.3), (45, 0.25, 0.3), (50, 0.10, 0.0)],
                      hmax=4, rolloff=1.6, fmax=900.0, lfo_rate=(0.02, 0.06))
    nz = circ_filter(np.stack([rng.standard_normal(N), rng.standard_normal(N)], 1),
                     signal.butter(2, [110, 380], "band", fs=SR, output="sos"))
    nz *= (0.55 + 0.45 * smooth_noise(N, 0.07, rng))[:, None] * 0.55
    wind = wind + nz * 0.16
    # thin high pad of open (third-less) chords, slow glassy tremolo
    thin = build_pads(rng, dict(
        xf=9.0, chord_gain=0.16, chord_pan=(-0.8, 0.8),
        vp=dict(hmax=3, rolloff=1.8, fmax=2800.0, det=(0.0008, 0.0022), lfo_rate=(0.03, 0.14), lfo_depth=0.6),
        segments=[
            dict(drone=[], chord=[69, 76, 83]),        # A E B
            dict(drone=[], chord=[74, 81, 76]),        # D A E
            dict(drone=[], chord=[78, 83, 90, 76]),    # F# B F# E
            dict(drone=[], chord=[69, 76, 74, 81]),    # A E D A
        ]))
    # shimmer: high sines that swell out of silence at different slow periods
    shim = np.zeros((N, 2))
    for m, per, ph, pan in [(81, 32, 0.4, -0.9), (88, 48, 2.2, 0.85), (83, 24, 4.0, 0.6),
                            (90, 96, 1.0, -0.6), (86, 32, 3.1, 0.95), (93, 48, 5.2, -0.95)]:
        f = hz(m)
        g = (0.5 + 0.5 * lfo(per, ph)) ** 5
        for off in (-0.15, 0.15):
            s = np.sin(2 * np.pi * snap(f + off) * T + rng.uniform(0, 6.28)) * g
            shim += pan2(s, pan) * 0.5
    # sparse glass tones, wide
    glass = np.zeros((N, 2))
    pool = [74, 76, 78, 81, 83, 86, 88, 90]
    P = [(1.0, 1.0, 4.5), (2.0, 0.28, 3.2), (3.0, 0.10, 2.2), (4.0, 0.04, 1.4)]
    last = None
    for i in range(6):
        t = 7 + i * 15.5 + rng.uniform(-3, 3)
        m = int(rng.choice([p for p in pool if p != last]))
        last = m
        pan = rng.choice([-1, 1]) * rng.uniform(0.6, 0.95)
        put(glass, voice(hz(m), 15.0, rng, partials=P, attack=0.12, release=1.5,
                         twin=(0.5, 1.0018, 4.0)) * rng.uniform(0.5, 0.8), t, pan)
    glass = echo(glass, [(1.9, 0.35), (3.4, 0.2)], damp=3000)
    return dict(
        layers=[dict(x=wind, dry=.9, wet=.4, lp_dry=1200, lp_wet=1200),
                dict(x=thin, dry=.35, wet=.9, lp_dry=3000, lp_wet=3000),
                dict(x=shim, dry=.10, wet=.55),
                dict(x=glass, dry=.30, wet=1.0, lp_dry=3500)],
        rev=dict(rt60=9.0, length=11.0, bands=[(60, 400, 0.8), (400, 1500, 1.0), (1500, 4000, 0.9), (4000, 9000, 0.6)]),
        master=dict(width=1.55, wobble=0.1, hp=45, lp=5200, dip=-3.0, lev=0.3))


def room_gosundeok(rng):
    """Underwater on one breath: muffled, a tide-slow swell, tones that bloom and sink."""
    pads = build_pads(rng, dict(
        xf=10.0, chord_gain=0.28,
        vp=dict(hmax=6, rolloff=1.9, fmax=1500.0, lfo_rate=(0.02, 0.07)),
        segments=[
            dict(drone=[(38, .55), (45, .30)], chord=[50, 57, 64, 66]),   # D add9
            dict(drone=[(38, .55), (45, .30)], chord=[47, 54, 62, 66]),   # B m
            dict(drone=[(43, .55), (50, .30)], chord=[43, 50, 59, 62]),   # G
            dict(drone=[(45, .55), (40, .25)], chord=[45, 52, 57, 61]),   # A (open)
        ]))
    tide = 1 + 0.30 * lfo(32, 0.9)                       # 3 swells per loop
    pads *= tide[:, None]
    # held-breath pulse: a soft low fifth that swells every 6 s (16 per loop), never a beat
    pw = ((0.5 + 0.5 * lfo(6, 1.2)) ** 1.6)
    pulse = static_pad(rng, [(50, 0.5, -0.15), (57, 0.32, 0.15), (62, 0.14, 0.0)],
                       hmax=3, rolloff=1.5, fmax=700.0, lfo_rate=(0.02, 0.05)) * pw[:, None]
    # blooming tones
    bloom = np.zeros((N, 2))
    pool = [50, 52, 54, 57, 59, 62, 64, 66]
    P = [(1.0, 1.0, 4.0), (2.0, 0.22, 2.6), (3.0, 0.05, 1.6)]
    t, last = 3.0, None
    while t < DUR - 2:
        m = int(rng.choice([p for p in pool if p != last]))
        last = m
        put(bloom, voice(hz(m), 16.0, rng, partials=P, attack=2.6, release=3.0,
                         twin=(0.4, 1.002, 4.0), fmax=1400.0) * rng.uniform(0.45, 0.8), t, rng.uniform(-0.6, 0.6))
        t += rng.uniform(6.0, 11.0)
    return dict(
        layers=[dict(x=pads, dry=.8, wet=.6, lp_dry=900, lp_wet=900),
                dict(x=pulse, dry=.45, wet=.3, lp_dry=700, lp_wet=700),
                dict(x=bloom, dry=.35, wet=.9, lp_dry=900, lp_wet=900)],
        rev=dict(rt60=4.5, length=6.0, bands=[(60, 400, 1.0), (400, 1500, 0.5), (1500, 4000, 0.2), (4000, 9000, 0.05)]),
        bright=dict(lo=330, hi=850, period=32, phase=0.9, order=3),
        master=dict(width=0.7, wobble=0.1, hp=40, lp=1200, dip=0.0, lev=0.15))


def room_ruhi(rng):
    """Deep stone and still water: a cistern's long reverb, a low bell, drops with echo."""
    drone = static_pad(rng, [(38, 0.55, -0.2), (45, 0.30, 0.2), (50, 0.22, 0.0), (57, 0.10, 0.25), (54, 0.07, -0.25)],
                       hmax=6, rolloff=1.25, fmax=1800.0, lfo_rate=(0.02, 0.06))
    drone *= (1 + 0.1 * lfo(48, 0.6))[:, None]
    # single low bell tones, tails that run the whole 24 s
    bells = np.zeros((N, 2))
    P = [(1.0, 1.0, 9.0), (2.0, 0.5, 6.0), (3.0, 0.18, 4.0), (4.01, 0.08, 3.0), (5.0, 0.03, 2.0)]
    for i, m in enumerate([50, 57, 54, 47, 50]):
        t = 6 + i * 19 + rng.uniform(-3, 3)
        put(bells, voice(hz(m), 26.0, rng, partials=P, attack=0.06, release=3.0, twin=(0.3, 1.002, 7.0),
                         fmax=1500.0) * rng.uniform(0.7, 1.0), t, rng.uniform(-0.4, 0.4))
    # water drops: soft pings with a falling edge and an echo
    pings = np.zeros((N, 2))
    pool = [74, 76, 78, 81, 83, 86]
    PP = [(1.0, 1.0, 0.16), (2.0, 0.10, 0.08)]
    t, last = 2.5, None
    while t < DUR - 1:
        m = int(rng.choice([p for p in pool if p != last]))
        last = m
        s = voice(hz(m), 1.4, rng, partials=PP, attack=0.004, release=0.3, twin=None,
                  glide=(0.035, 0.03), fmax=2500.0) * rng.uniform(0.3, 0.7)
        put(pings, s, t, rng.uniform(-0.8, 0.8))
        t += rng.uniform(4.0, 10.0)
    pings = echo(pings, [(0.41, 0.5), (0.83, 0.3), (1.37, 0.18), (2.1, 0.1)], damp=2200)
    return dict(
        layers=[dict(x=drone, dry=.55, wet=.9, lp_dry=1500, lp_wet=1500),
                dict(x=bells, dry=.3, wet=1.2, lp_dry=1800, lp_wet=1800),
                dict(x=pings, dry=.35, wet=1.0, lp_dry=2800)],
        rev=dict(rt60=13.0, length=15.0, bands=[(50, 400, 1.3), (400, 1500, 1.0), (1500, 4000, 0.4), (4000, 9000, 0.12)]),
        master=dict(width=1.0, wobble=0.1, hp=38, lp=4500, dip=-3.0, lev=0.25))


def room_halina(rng):
    """Empty waiting room at night: a calm chord with one sus tone that never resolves; room hum."""
    # fluorescent-ish hum: D2 and its harmonics, faint
    hum = np.zeros((N, 2))
    f0 = hz(38)
    for h, a in zip(range(1, 6), [1.0, 0.5, 0.3, 0.15, 0.07]):
        s = np.sin(2 * np.pi * snap(f0 * h) * T + rng.uniform(0, 6.28)) * a * (1 + 0.12 * lfo(24, rng.uniform(0, 6.28)))
        hum += pan2(s, rng.uniform(-0.3, 0.3))
    room = circ_filter(np.stack([rng.standard_normal(N), rng.standard_normal(N)], 1),
                       signal.butter(2, [140, 700], "band", fs=SR, output="sos"))
    room *= (0.6 + 0.4 * smooth_noise(N, 0.05, rng))[:, None]
    # calm chord: D A F#, open
    chord = static_pad(rng, [(50, 0.5, -0.2), (57, 0.4, 0.2), (66, 0.22, -0.1), (69, 0.12, 0.3)],
                       hmax=5, rolloff=1.5, fmax=1800.0, lfo_rate=(0.02, 0.05))
    # the wrong note: G (sus4) as a pair beating at 0.24 Hz around true pitch, always there, never resolving
    gx = np.zeros((N, 2))
    gswell = 0.62 + 0.38 * lfo(48, 1.3)
    for off in (-0.12, 0.12):
        for h, a in [(1, 1.0), (2, 0.18)]:
            s = np.sin(2 * np.pi * snap(hz(67) * h + off * h) * T + rng.uniform(0, 6.28)) * a * gswell
            gx += pan2(s, 0.3 if off < 0 else 0.1) * 0.5
    # a few quiet tones; G keeps coming back
    tones = np.zeros((N, 2))
    P = [(1.0, 1.0, 2.8), (2.0, 0.2, 1.6)]
    for i, m in enumerate([67, 62, 67, 69, 67]):
        t = 9 + i * 18 + rng.uniform(-3, 3)
        put(tones, voice(hz(m), 10.0, rng, partials=P, attack=0.5, release=1.5, twin=None, fmax=1600.0) * rng.uniform(0.5, 0.8),
            t, rng.uniform(-0.5, 0.5))
    return dict(
        layers=[dict(x=hum, dry=.40, wet=.15, lp_dry=700, lp_wet=700),
                dict(x=room, dry=.06, wet=.04),
                dict(x=chord, dry=.55, wet=.5, lp_dry=1600, lp_wet=1600),
                dict(x=gx, dry=.45, wet=.5, lp_dry=1500, lp_wet=1500),
                dict(x=tones, dry=.30, wet=.8, lp_dry=1800)],
        rev=dict(rt60=1.7, length=2.6, bands=[(100, 500, 1.0), (500, 1500, 0.8), (1500, 4000, 0.4), (4000, 9000, 0.2)]),
        master=dict(width=0.45, wobble=0.05, hp=45, lp=3500, dip=-3.0, lev=0.2))


def room_tobi(rng):
    """A city junction remembered: phasing tone patterns, a rising line, warm and bright."""
    pads = build_pads(rng, dict(
        xf=8.0, chord_gain=0.24,
        vp=dict(hmax=8, rolloff=1.15, fmax=3200.0),
        breath=[(DUR / 4, 0.10, 0.2)],
        segments=[
            dict(drone=[(38, .5), (45, .3)], chord=[57, 62, 66, 69, 73]),      # D maj7
            dict(drone=[(47, .5), (54, .3)], chord=[59, 62, 66, 71, 74]),      # B m
            dict(drone=[(43, .5), (50, .3)], chord=[55, 59, 62, 67, 71]),      # G
            dict(drone=[(45, .5), (52, .3)], chord=[57, 61, 64, 69, 71]),      # A
        ]))
    # slow rising line: one long swell per chord (C# - D - B - E)
    line = np.zeros((N, 2))
    LP_ = [(1.0, 1.0, 12.0), (2.0, 0.30, 9.0), (3.0, 0.12, 6.0)]
    for i, m in enumerate([73, 74, 71, 76]):
        put(line, voice(hz(m), 32.0, rng, partials=LP_, attack=8.0, release=8.0, twin=(0.4, 1.0015, 12.0)) * 0.7,
            i * 24 - 4.0, [-0.3, 0.2, -0.1, 0.35][i])
    # phasing patterns: (period s, notes, voice level, pan period, level)
    PT = [(1.0, 1.0, 2.4), (2.0, 0.45, 1.6), (3.0, 0.26, 1.1), (4.0, 0.13, 0.8)]
    pat = np.zeros((N, 2))
    for per, notes, lvl, ppan, off in [(12.0, [66, 69, 74, 71], 0.55, 36.0, 0.0),
                                       (9.6, [78, 74, 76], 0.38, 48.0, 1.3),
                                       (19.2, [62, 64, 69, 66, 71], 0.45, 32.0, 2.1),
                                       (32.0, [57, 59, 62, 64, 66, 69, 71], 0.50, 96.0, 0.7)]:
        k = len(notes)
        for rep in range(int(round(DUR / per))):
            for j, m in enumerate(notes):
                t = off + rep * per + j * per / k
                pan = 0.65 * np.sin(2 * np.pi * t / ppan + off)
                put(pat, voice(hz(m), 9.5, rng, partials=PT, attack=0.35, release=1.5,
                               twin=(0.25, 1.002, 2.0), fmax=3800.0) * lvl, t % DUR, pan)
    return dict(
        layers=[dict(x=pads, dry=.6, wet=.5, lp_dry=2500, lp_wet=2500),
                dict(x=line, dry=.35, wet=.7, lp_dry=3000),
                dict(x=pat, dry=.55, wet=.9, lp_dry=5000)],
        rev=dict(rt60=4.5, length=6.0, bands=[(60, 400, 1.0), (400, 1500, 1.0), (1500, 4000, 0.65), (4000, 9000, 0.3)]),
        bright=dict(lo=3200, hi=7500, period=48, phase=0.5, order=2),
        master=dict(width=0.95, wobble=0.12, hp=45, lp=7500, dip=-2.0, lev=0.2))


def room_ilanji(rng):
    """Lamp-lighting hour in a grove: warm, low, intimate; breathing drone, few soft plucks."""
    drone = static_pad(rng, [(38, 0.55, -0.15), (45, 0.28, 0.15), (50, 0.26, 0.0), (57, 0.10, 0.2)],
                       hmax=6, rolloff=1.3, fmax=2000.0, lfo_rate=(0.02, 0.06))
    drone *= (1 + 0.30 * lfo(12, 0.4) + 0.08 * lfo(48, 2.0))[:, None]       # breathing
    pads = build_pads(rng, dict(
        xf=12.0, chord_gain=0.22,
        vp=dict(hmax=5, rolloff=1.6, fmax=1800.0),
        segments=[
            dict(drone=[], chord=[47, 54, 62, 66]),          # B minor
            dict(drone=[], chord=[43, 50, 59, 62]),          # G
        ]))
    pads *= (1 + 0.25 * lfo(12, 0.4))[:, None]
    # lamp-like plucks: soft attack, rich low harmonics, long decay that fades the upper partials first
    PL = [(h, 1.0 / h ** 1.1, 7.5 / (1 + 0.9 * (h - 1))) for h in range(1, 7)]
    plucks = np.zeros((N, 2))
    pool = [50, 54, 57, 59, 62, 64, 66, 69]
    t, last, i = 5.0, None, 0
    while t < DUR - 6:
        m = int(rng.choice([p for p in pool if p != last]))
        last = m
        pan = rng.uniform(-0.5, 0.5)
        put(plucks, voice(hz(m), 13.0, rng, partials=PL, attack=0.012, release=2.0, twin=None, fmax=2400.0) * rng.uniform(0.5, 0.9), t, pan)
        if rng.random() < 0.3:
            m2 = int(rng.choice([p for p in pool if p < m] or pool))
            put(plucks, voice(hz(m2), 13.0, rng, partials=PL, attack=0.012, release=2.0, twin=None, fmax=2400.0) * rng.uniform(0.35, 0.6),
                t + rng.uniform(1.2, 2.5), -pan * 0.5)
        t += rng.uniform(7.0, 13.0)
        i += 1
    return dict(
        layers=[dict(x=drone, dry=.85, wet=.35, lp_dry=1300, lp_wet=1300),
                dict(x=pads, dry=.5, wet=.5, lp_dry=1100, lp_wet=1100),
                dict(x=plucks, dry=.55, wet=.8, lp_dry=1900, lp_wet=2200)],
        rev=dict(rt60=3.4, length=5.0, bands=[(60, 400, 1.1), (400, 1500, 1.0), (1500, 4000, 0.4), (4000, 9000, 0.12)]),
        bright=dict(lo=450, hi=1200, period=12, phase=0.4, order=2),
        master=dict(width=0.55, wobble=0.08, hp=40, lp=2400, dip=-2.0, lev=0.12))


ROOMS = {   # slug -> (builder, seed)
    "nave": (room_nave, 20261112),
    "stanzin-dolkar": (room_stanzin, 20261201),
    "go-sun-deok": (room_gosundeok, 20261202),
    "ruhi-vaghela": (room_ruhi, 20261203),
    "halina-mroz-quispe": (room_halina, 20261204),
    "tobi-adeyemi": (room_tobi, 20261205),
    "ilanji": (room_ilanji, 20261206),
}


# ---------------------------------------------------------------- mix + master
def render(slug):
    builder, seed = ROOMS[slug]
    rng = np.random.default_rng(seed)
    room = builder(rng)
    ir = make_ir(rng, **room["rev"])
    dry = np.zeros((N, 2))
    wet_in = np.zeros((N, 2))
    for ly in room["layers"]:
        x = ly["x"]
        dry += (lp(x, ly["lp_dry"]) if ly.get("lp_dry") else x) * ly["dry"]
        wet_in += (lp(x, ly["lp_wet"]) if ly.get("lp_wet") else x) * ly["wet"]
    mix = dry + circ_reverb(wet_in, ir)
    b = room.get("bright")
    if b:
        mix = bright_mod(mix, b["lo"], b["hi"], b["period"], b["phase"], b.get("order", 2))
    m = room["master"]
    mid = (mix[:, 0] + mix[:, 1]) / 2
    side = (mix[:, 0] - mix[:, 1]) / 2
    side *= m["width"] + m["wobble"] * smooth_noise(N, 0.05, rng)
    mix = np.stack([mid + side, mid - side], 1)
    mix = hp(mix, m["hp"])
    mix = lp(mix, m["lp"])
    if m["dip"]:   # peaking cut around 2.2 kHz (RBJ biquad) to leave room for speech
        f0, Q = 2200.0, 0.8
        A = 10 ** (m["dip"] / 40)
        w0 = 2 * np.pi * f0 / SR
        al = np.sin(w0) / (2 * Q)
        bb = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
        aa = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
        mix = circ_filter(mix, signal.tf2sos(bb / aa[0], aa / aa[0]))
    mix -= mix.mean(0)
    # very slow circular leveller (~12 s window)
    pw = (mix ** 2).mean(1)
    sp = np.fft.rfft(pw) / (1 + (np.fft.rfftfreq(N, 1 / SR) / 0.04) ** 2)
    env_p = np.maximum(np.fft.irfft(sp, N), 1e-12)
    mix *= ((env_p / env_p.mean()) ** -m["lev"])[:, None]
    # loudness-match: same integrated LUFS for every room; peak ceiling with a soft knee if needed
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    gain = 10 ** ((TARGET_LUFS - meter.integrated_loudness(mix)) / 20)
    ceil = 10 ** (PEAK_CEIL_DB / 20)
    mix *= gain
    if np.abs(mix).max() > ceil:
        mix = ceil * np.tanh(mix / ceil * 0.9) / np.tanh(0.9)
        g2 = 10 ** ((TARGET_LUFS - meter.integrated_loudness(mix)) / 20)
        mix *= g2
        print(f"  [{slug}] soft-limited, trim {20 * np.log10(g2):.2f} dB")
    print(f"[{slug}] gain {20 * np.log10(gain):.2f} dB, peak {20 * np.log10(np.abs(mix).max()):.2f} dBFS", flush=True)
    return mix


def write(slug, mix):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTDIR / f"_{slug}.wav"
    out = OUTDIR / f"{slug}.mp3"
    sf.write(tmp, mix.astype(np.float32), SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ar", "44100", "-ac", "2",
                    "-codec:a", "libmp3lame", "-b:a", "96k", "-id3v2_version", "0", "-write_xing", "1", str(out)], check=True)
    tmp.unlink()
    print("wrote", out, out.stat().st_size, "bytes", flush=True)


def job(slug):
    write(slug, render(slug))


def main():
    slugs = sys.argv[1:] or list(ROOMS)
    with Pool(min(3, len(slugs))) as p:
        p.map(job, slugs)


if __name__ == "__main__":
    main()
