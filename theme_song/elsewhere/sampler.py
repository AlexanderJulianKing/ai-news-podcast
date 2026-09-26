"""Render parts through Apple's sampler with GarageBand/Logic EXS instruments (Mac only).

render_part.swift is compiled on first use (swiftc). Notes are (start_s, dur_s, midi, velocity) at sounding pitch.
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from newscaster.audio.loudness import integrated_lufs  # noqa: E402

SR = 48000
LIB = '/Library/Application Support/Logic/Sampler Instruments'
INSTRUMENTS = {
    'synth': '04 Keyboards/Keyboard Collection/Liquid Synth Keys.exs',       # the pulse
    'carnival': '04 Keyboards/Keyboard Collection/New Carnival Synth.exs',   # low notes and pad
}
# Patches that sound an octave below the MIDI note (none of the ones used here; kept for the upright/electric basses)
TRANSPOSE = {'upright': 12, 'fingerbass': 12}
BINARY = os.path.join(HERE, 'render_part')


def _binary():
    if not os.path.exists(BINARY):
        subprocess.run(['swiftc', '-O', os.path.join(HERE, 'render_part.swift'), '-o', BINARY], check=True)
    return BINARY


def render_part(name, instrument, notes, total_s, workdir):
    """Render notes through the instrument; returns (2, n) float64 at 48 kHz."""
    os.makedirs(workdir, exist_ok=True)
    tr = TRANSPOSE.get(instrument, 0)
    # a sampler matches note-offs by pitch: end each note just before the next strike of the same pitch
    notes = sorted(notes, key=lambda n: n[0])
    nxt, clamped = {}, []
    for s, d, m, v in reversed(notes):
        if m in nxt:
            d = min(d, nxt[m] - s - 0.005)
        nxt[m] = s
        clamped.append((s, max(d, 0.01), m, v))
    ev = []
    for s, d, m, v in clamped[::-1]:
        ev.append(dict(t=max(0.0, s), k='on', n=int(m) + tr, v=int(np.clip(v, 1, 127))))
        ev.append(dict(t=max(0.0, s + d), k='off', n=int(m) + tr, v=0))
    ev.sort(key=lambda e: (e['t'], 0 if e['k'] == 'off' else 1))
    jpath, wpath = os.path.join(workdir, f'{name}.json'), os.path.join(workdir, f'{name}.wav')
    json.dump(ev, open(jpath, 'w'))
    r = subprocess.run([_binary(), jpath, os.path.join(LIB, INSTRUMENTS[instrument]), wpath, f'{total_s:.3f}'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{name}: {r.stdout} {r.stderr}')
    y, _ = sf.read(wpath, dtype='float64', always_2d=True)
    return y.T


def humanize(notes, rng, t_ms=7, v=4):
    return [(s + rng.normal(0, t_ms / 1000.0), d, m, vel + rng.integers(-v, v + 1)) for s, d, m, vel in notes]


def lufs(x):
    m = x.mean(0) if x.ndim == 2 else x
    return integrated_lufs(m, SR) if np.any(m) else -np.inf
