"""GarageBand's 'Shaker Heavy', rendered offline from its EXS map (Apple's sampler can't load it).

The instrument holds real strokes in 12 groups: accent / forward / backward, each recorded at 60, 90, 120 and
150 bpm, with velocity layers. Notes here name the stroke: 60 = accent, 61 = forward, 62 = backward.
"""
import struct
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf

EXS = "/Library/Application Support/Logic/Sampler Instruments/03 Drums & Percussion/05 Percussion/Single Instruments/Shaker Heavy.exs"
CAF = "/Library/Application Support/Logic/EXS Factory Samples/03 Drums & Percussion/05 Percussion/Shaker Heavy_consolidated.caf"
SR = 48000
STROKE = {60: 'acc', 61: 'fwd', 62: 'bwd'}
TEMPO = 90


def _load():
    d = open(EXS, 'rb').read()
    o, zones, groups = 0, [], []
    while o + 84 <= len(d):
        sig, size = struct.unpack_from('<II', d, o)
        typ = (sig & 0x0F000000) >> 24
        name = d[o + 20:o + 84].split(b'\0')[0].decode('latin1')
        b = d[o + 84:o + 84 + size]
        if typ == 1:
            zones.append(dict(vol=struct.unpack('b', b[4:5])[0], vlo=b[9], vhi=b[10], start=struct.unpack_from('<I', b, 12)[0],
                              end=struct.unpack_from('<I', b, 16)[0], group=struct.unpack_from('<i', b, 88)[0]))
        if typ == 2:
            groups.append(dict(name=name, vol=struct.unpack('b', b[0:1])[0]))
        o += 84 + size
    return zones, groups


ZONES, GROUPS = _load()
SRC_SR = sf.info(CAF).samplerate


@lru_cache(maxsize=None)
def stroke(kind, vel):
    gname = f'ShakerHeavy:{kind}{TEMPO}'
    gi = next(i for i, g in enumerate(GROUPS) if g['name'] == gname)
    cands = [z for z in ZONES if z['group'] == gi]
    z = next((z for z in cands if z['vlo'] <= vel <= z['vhi']), max(cands, key=lambda z: z['vhi']) if vel > 64 else min(cands, key=lambda z: z['vlo']))
    with sf.SoundFile(CAF) as f:
        f.seek(z['start'])
        y = f.read(z['end'] - z['start'], dtype='float32', always_2d=True).T
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    y = np.stack([librosa.resample(ch, orig_sr=SRC_SR, target_sr=SR, res_type='soxr_hq') for ch in y])
    return y * 10 ** ((GROUPS[gi]['vol'] + z['vol']) / 20.0)


def render(notes, total_s):
    """notes: [(start_s, dur_s, stroke_note, velocity)] -> (2, n)."""
    out = np.zeros((2, int(total_s * SR) + SR))
    for s, d, m, v in notes:
        y = stroke(STROKE.get(int(m), 'fwd'), int(np.clip(v, 1, 127)))
        i = int(max(0.0, s) * SR)
        j = min(out.shape[1], i + y.shape[1])
        out[:, i:j] += y[:, :j - i]
    return out
