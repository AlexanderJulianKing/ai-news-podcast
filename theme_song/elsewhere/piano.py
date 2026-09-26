"""GarageBand's Steinway, rendered offline on the Mac from its EXS map (Apple's sampler can't load this instrument).

Same decoding as theme_song/off_the_wire/build_steinway_bank.py, but renders any note list directly.
"""
import struct
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf

EXS = "/Library/Application Support/Logic/Sampler Instruments/01 Acoustic Pianos/Steinway Grand Piano 2.exs"
CAF = "/Library/Application Support/Logic/EXS Factory Samples/01 Acoustic Pianos/Steinway Piano_consolidated.caf"
SRC_SR, SR = 44100, 48000
SIDE = 0.6          # stereo width kept
RELEASE = 0.16      # damper, seconds


def _load_map():
    d = open(EXS, 'rb').read()
    o, zones, groups = 0, [], []
    while o + 84 <= len(d):
        sig, size = struct.unpack_from('<II', d, o)
        typ = (sig & 0x0F000000) >> 24
        b = d[o + 84:o + 84 + size]
        if typ == 1:
            zones.append(dict(root=b[1], fine=struct.unpack('b', b[2:3])[0], vol=struct.unpack('b', b[4:5])[0], klo=b[6], khi=b[7],
                              start=struct.unpack_from('<I', b, 12)[0], end=struct.unpack_from('<I', b, 16)[0],
                              group=struct.unpack_from('<i', b, 88)[0]))
        if typ == 2:
            groups.append(dict(vol=struct.unpack('b', b[0:1])[0], vlo=b[5], vhi=b[6], cc=b[85], cclo=b[86], cchi=b[87], klo=b[88], khi=b[89]))
        o += 84 + size
    return zones, groups


ZONES, GROUPS = _load_map()
LAYER_VEL = (30, 50, 75, 110)     # representative velocity of each of the four pedal-up layers


def _zone(key, vel):
    for z in ZONES:
        g = GROUPS[z['group']]
        if g['vlo'] <= vel <= g['vhi'] and g['klo'] <= key <= g['khi'] and z['klo'] <= key <= z['khi'] and not (g['cc'] == 64 and g['cclo'] > 0):
            return z, g
    raise KeyError((key, vel))


@lru_cache(maxsize=None)
def note_sample(key, layer, max_s=6.0):
    """Stereo float32 recording of `key` at velocity layer `layer`, pitch-exact, 48 kHz."""
    z, g = _zone(key, LAYER_VEL[layer])
    with sf.SoundFile(CAF) as f:
        f.seek(z['start'])
        n = min(z['end'] - z['start'], int(max_s * SRC_SR * 1.2))
        y = f.read(n, dtype='float32', always_2d=True).T
    ratio = 2 ** ((key - z['root'] + z['fine'] / 100.0) / 12.0)
    y = np.stack([librosa.resample(ch, orig_sr=SRC_SR, target_sr=SR / ratio, res_type='soxr_hq') for ch in y])
    y *= 10 ** ((g['vol'] + z['vol']) / 20.0)
    mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2 * SIDE
    return np.stack([mid + side, mid - side])[:, :int(max_s * SR)]


def layer_for(vel):
    return 0 if vel < 40 else 1 if vel < 60 else 2 if vel < 90 else 3


def render(notes, total_s):
    """notes: [(start_s, dur_s, midi, velocity)] -> (2, n) float64. Damped at note end (RELEASE)."""
    out = np.zeros((2, int(total_s * SR) + SR))
    for s, d, m, v in notes:
        y = note_sample(int(m), layer_for(v))
        n = min(y.shape[1], int((d + RELEASE) * SR))
        seg = y[:, :n].astype(np.float64) * (v / 127.0) ** 1.2
        r = min(n, int(RELEASE * SR))
        if r > 0 and n == int((d + RELEASE) * SR):
            seg[:, n - r:] *= np.linspace(1, 0, r) ** 2
        i = int(max(0.0, s) * SR)
        j = min(out.shape[1], i + n)
        out[:, i:j] += seg[:, :j - i]
    return out
