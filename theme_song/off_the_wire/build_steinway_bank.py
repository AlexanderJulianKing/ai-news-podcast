"""Build the sample bank for the 'Off the Wire' intro (run on a Mac with GarageBand installed).

Reads GarageBand's Steinway Grand Piano map (Steinway Grand Piano 2.exs) and its packed recordings, and renders
every pitch the intro can play (off_the_wire_score.all_pitches) at three velocity layers, pitch-shifted to the
exact key and resampled to 48 kHz, stereo narrowed a little for mono listeners. Writes steinway_bank_48k.npz next
to this file. The bank is Apple sample content: copy it to the Pi (scp) and never commit it.

    python3 theme_song/off_the_wire/build_steinway_bank.py
    scp theme_song/off_the_wire/steinway_bank_48k.npz alex@raspberrypi...:/home/alex/ai-news-podcast/theme_song/off_the_wire/

Needs numpy, soundfile and librosa (Mac only; the Pi only reads the result).
"""
import os
import struct
import sys

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from newscaster.audio import off_the_wire_score as score  # noqa: E402

EXS = "/Library/Application Support/Logic/Sampler Instruments/01 Acoustic Pianos/Steinway Grand Piano 2.exs"
CAF = "/Library/Application Support/Logic/EXS Factory Samples/01 Acoustic Pianos/Steinway Piano_consolidated.caf"
SRC_SR, SR = 44100, 48000
LAYER_VELOCITY = {1: 50, 2: 75, 3: 110}      # one representative velocity per layer (layer 0, below 40, is not needed)
SHORT_S, LONG_S = 1.2, 4.5
SIDE = 0.6                                   # stereo width kept (1.0 = as recorded)


def load_map():
    """Zones and groups of the EXS instrument. Chunks: 84-byte header + body; type (sig >> 24) & 0xF: 1 zone, 2 group."""
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
        if typ == 2:   # velocity range at 5-6; selected by controller 64 (sustain) range at 86-87; key range at 88-89
            groups.append(dict(vol=struct.unpack('b', b[0:1])[0], vlo=b[5], vhi=b[6], cc=b[85], cclo=b[86], cchi=b[87], klo=b[88], khi=b[89]))
        o += 84 + size
    return zones, groups


def zone_for(zones, groups, key, vel):
    for z in zones:
        g = groups[z['group']]
        if g['vlo'] <= vel <= g['vhi'] and g['klo'] <= key <= g['khi'] and z['klo'] <= key <= z['khi'] and not (g['cc'] == 64 and g['cclo'] > 0):
            return z, g
    raise KeyError(f"no pedal-up zone for key {key} velocity {vel}")


def main():
    zones, groups = load_map()
    audio = sf.SoundFile(CAF)
    out, longs = {}, set(score.long_pitches())
    for key in score.all_pitches():
        for layer, vel in LAYER_VELOCITY.items():
            z, g = zone_for(zones, groups, key, vel)
            audio.seek(z['start'])
            y = audio.read(z['end'] - z['start'], dtype='float32', always_2d=True).T
            ratio = 2 ** ((key - z['root'] + z['fine'] / 100.0) / 12.0)
            y = np.stack([librosa.resample(ch, orig_sr=SRC_SR, target_sr=SR / ratio, res_type='soxr_hq') for ch in y])
            y *= 10 ** ((g['vol'] + z['vol']) / 20.0)
            mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2 * SIDE
            y = np.stack([mid + side, mid - side])
            out[f'p{key}_v{layer}'] = y[:, :int(SHORT_S * SR)]
            if key in longs:
                out[f'p{key}_v{layer}_long'] = y[:, :int(LONG_S * SR)]
    peak = max(float(np.abs(v).max()) for v in out.values())
    scale = peak / 32767.0
    arrays = {k: np.round(v / scale).astype('<i2') for k, v in out.items()}
    path = os.path.join(HERE, 'steinway_bank_48k.npz')
    np.savez_compressed(path, scale=np.array(scale), **arrays)
    print(f"{len(arrays)} recordings ({len(score.all_pitches())} pitches x {len(LAYER_VELOCITY)} layers + long versions) -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
