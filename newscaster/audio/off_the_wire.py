"""Build the day's intro with 'Off the Wire' (see off_the_wire_score.py for the music).

The host's two TTS lines are placed right after a one-second logo; the piano grooves under the reading at one
steady level about 9 dB below the voice (with the 0.8-4 kHz speech band dipped a further 4 dB), answers in the
host's sentence breaks, and finishes about 4 seconds after the last word. The music adapts to the reading's
length and changes every day (seeded by the date).

numpy + pydub only, so it runs on the Pi. The piano is GarageBand's Steinway Grand Piano, pre-rendered per pitch
and velocity layer into OFF_THE_WIRE_BANK_PATH by theme_song/off_the_wire/build_steinway_bank.py on a Mac.
That bank is Apple sample content: it is copied to the Pi directly and never committed (see .gitignore).
Any failure raises; pipeline.generate_audio then falls back to the classic intro.
"""

import os

import numpy as np
from pydub import AudioSegment, silence

import newscaster.config as _config
from newscaster.audio import off_the_wire_score as score
from newscaster.audio.loudness import integrated_lufs, read_wav_mono
from newscaster.logging import print_and_write

SR = 48000
LAYERS = ((0, 39), (40, 59), (60, 89), (90, 127))    # the Steinway's four velocity layers
TARGET_LOGO_LUFS = -21.0                               # the logo, alone: a touch above the voice (voices are -23)
TARGET_UNDER_LUFS = -31.5                              # the groove under the reading
TARGET_BUTTON_LUFS = -22.5                             # the button, after the last word
DUCK_DB = -10.0                                        # steady duck across the whole reading (no swells in pauses)
DIP_DB = -4.0                                          # extra dip of the speech band while the host reads
DIP_BAND = (800.0, 4000.0)
RELEASE = 0.14                                         # damper release, seconds
MIN_BREAK_MS = 500                                     # sentence breaks (commas run 0.35-0.48 s in these voices)


class Bank:
    """Pre-rendered Steinway notes: one stereo recording per (pitch, velocity layer), plus long versions for the
    notes that must ring. Stored as int16 at 48 kHz with a common scale factor."""

    def __init__(self, path):
        data = np.load(path)
        self.scale = float(data['scale'])
        self.short, self.long = {}, {}
        for key in data.files:
            if key == 'scale':
                continue
            name, _, kind = key.partition('_long')
            pitch, layer = (int(x) for x in name[1:].split('_v'))
            (self.long if key.endswith('_long') else self.short)[(pitch, layer)] = data[key]

    def get(self, pitch, velocity, want_long=False):
        layer = next(i for i, (lo, hi) in enumerate(LAYERS) if lo <= velocity <= hi)
        table = self.long if want_long and any(p == pitch for p, _ in self.long) else self.short
        layers = sorted(l for (p, l) in table if p == pitch)
        if not layers:
            raise KeyError(f"no sample for pitch {pitch}")
        use = min(layers, key=lambda l: abs(l - layer))
        return table[(pitch, use)].astype(np.float32) * self.scale


def _sentence_breaks(path, t_offset):
    """Pauses of MIN_BREAK_MS or more inside a TTS line, on the intro timeline."""
    seg = AudioSegment.from_file(path)
    gaps = silence.detect_silence(seg, min_silence_len=MIN_BREAK_MS, silence_thresh=seg.max_dBFS - 38, seek_step=10)
    return [(t_offset + s / 1000.0, t_offset + e / 1000.0) for s, e in gaps if s > 50 and e < len(seg) - 50]


def _fft_convolve(x, h):
    n = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)[:len(x)]


def _reverb(bed, wet=0.13, seconds=1.9, seed=5):
    rng = np.random.default_rng(seed)
    L = int(seconds * SR)
    env = np.exp(-6.9 * np.arange(L) / L)
    out = bed.copy()
    for c in range(2):
        ir = rng.standard_normal(L) * env
        ir[:int(0.018 * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum())
        out[c] += wet * _fft_convolve(bed[c], ir)
    return out


def render(notes, bank, total_s):
    """Overlap-add the notes (start_s, end_s, pitch, velocity) into a stereo bed, with a light room."""
    out = np.zeros((2, int(total_s * SR) + SR), dtype=np.float64)
    for s, e, m, v in notes:
        want_long = (e - s) > 1.0
        y = bank.get(m, v, want_long)
        n = min(y.shape[1], int((e - s + RELEASE) * SR))
        seg = y[:, :n].astype(np.float64) * (v / 127.0) ** 1.45
        r = min(n, int(RELEASE * SR))
        if n == int((e - s + RELEASE) * SR) and r > 0:
            seg[:, -r:] *= np.linspace(1, 0, r) ** 2
        a = int(s * SR)
        if a >= out.shape[1]:
            continue
        seg = seg[:, :out.shape[1] - a]
        out[:, a:a + seg.shape[1]] += seg
    out = _reverb(out[:, :int(total_s * SR)])
    return out / (np.abs(out).max() + 1e-12) * 0.89


def _moving_average(x, k):
    c = np.cumsum(np.concatenate([[0.0], x]))
    lo = np.clip(np.arange(len(x)) - k // 2, 0, len(x))
    hi = np.clip(np.arange(len(x)) + k // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def _band_dipped(bed):
    """The bed with the speech band (DIP_BAND) lowered by DIP_DB, via FFT (numpy only)."""
    out = np.empty_like(bed)
    for c in range(2):
        n = 1 << int(np.ceil(np.log2(bed.shape[1])))
        X = np.fft.rfft(bed[c], n)
        f = np.fft.rfftfreq(n, 1.0 / SR)
        lo, hi = DIP_BAND
        # smooth half-octave edges around the band
        w = np.clip(np.minimum(np.log2(np.maximum(f, 1.0) / lo) * 2 + 1, np.log2(hi / np.maximum(f, 1.0)) * 2 + 1), 0, 1)
        X *= 10 ** (DIP_DB * w / 20.0)
        out[c] = np.fft.irfft(X, n)[:bed.shape[1]]
    return out


def _gain_to(signal_mono, t0, t1, target, default_db):
    a, b = int(max(t0, 0) * SR), int(t1 * SR)
    lufs = integrated_lufs(signal_mono[a:b], SR) if b - a > int(0.5 * SR) else None
    return 10 ** ((target - lufs) / 20.0) if lufs is not None else 10 ** (default_db / 20.0)


def mix(bed, voice, s1, v_end, end):
    """Voice (mono) over the bed (stereo) with the agreed levels. Returns stereo float."""
    N = bed.shape[1]
    t = np.arange(N) / SR
    span = _moving_average(((t >= s1 - 0.25) & (t <= v_end + 0.5)).astype(float), int(0.25 * SR))
    bed = bed * (1 - span) + _band_dipped(bed) * span
    mono = bed.mean(0)
    g_open = _gain_to(mono, 0, s1 - 0.05, TARGET_LOGO_LUFS, -8.0)
    g_under = _gain_to(mono * 10 ** (DUCK_DB / 20.0), s1 + 1, v_end - 1, TARGET_UNDER_LUFS, -18.0) * 10 ** (DUCK_DB / 20.0)
    g_end = _gain_to(mono, v_end + 0.2, end, TARGET_BUTTON_LUFS, -8.0)
    gain = np.where(t < s1 - 0.1, g_open, np.where(t < v_end, g_under, g_end))
    gain = _moving_average(gain, int(0.12 * SR))
    out = bed * gain + voice[:N]
    peak = np.abs(out).max()
    return out / (peak / 0.95) if peak > 0.95 else out


def _to_48k(x, sr):
    if sr == SR:
        return x
    t = np.arange(int(len(x) * SR / sr)) * sr / SR
    return np.interp(t, np.arange(len(x)), x)


def make_intro(formatted_date2, out_path=None, voice_dir='segment_audio', bank_path=None):
    """Write <voice_dir>/<date>_intro.mp3 (or out_path) from <date>_intro1.wav and _intro2.wav. Returns a summary."""
    f1 = os.path.join(voice_dir, f'{formatted_date2}_intro1.wav')
    f2 = os.path.join(voice_dir, f'{formatted_date2}_intro2.wav')
    out_path = out_path or os.path.join(voice_dir, f'{formatted_date2}_intro.mp3')
    bank = Bank(bank_path or getattr(_config, 'OFF_THE_WIRE_BANK_PATH', 'theme_song/off_the_wire/steinway_bank_48k.npz'))
    v1, sr1 = read_wav_mono(f1)
    v2, sr2 = read_wav_mono(f2)
    if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
        raise ValueError("intro voice lines missing or unreadable")
    v1, v2 = _to_48k(v1, sr1), _to_48k(v2, sr2)
    d1, d2 = len(v1) / SR, len(v2) / SR
    n, s1, pause, v_end = score.plan(d1, d2)
    t2 = s1 + d1 + pause
    breaks = _sentence_breaks(f1, s1) + [(s1 + d1, t2)] + _sentence_breaks(f2, t2)
    seed = int(''.join(ch for ch in formatted_date2 if ch.isdigit()) or 0)
    notes, fills, bars = score.arrange(n, s1, v_end, breaks, seed)
    end = score.end_time(n)
    bed = render(notes, bank, end + 0.3)
    N = int(end * SR)
    fade = np.ones(bed.shape[1])
    a = int((end - 0.9) * SR)
    fade[a:N] = np.linspace(1, 0, N - a) ** 1.5
    fade[N:] = 0
    bed = (bed * fade)[:, :N]
    voice = np.zeros(N)
    i = int(s1 * SR); voice[i:i + len(v1)] += v1[:N - i]
    i = int(t2 * SR); voice[i:i + len(v2)] += v2[:N - i]
    out = mix(bed, voice, s1, v_end, end)
    pcm = (np.clip(out, -1, 1) * 32767).astype('<i2').T.copy()
    AudioSegment(pcm.tobytes(), frame_rate=SR, sample_width=2, channels=2).export(out_path, format='mp3')
    summary = dict(bars=n, host_start=round(s1, 2), host_end=round(v_end, 2), music_end=round(end, 2), fills=fills,
                   breaks=len(breaks), path=out_path)
    print_and_write(f"Off the Wire intro: {n} groove bars, host {s1:.1f}-{v_end:.1f}s, music ends {end - v_end:.1f}s after the "
                    f"last word, {len(fills)} fills in {len(breaks)} breaks -> {out_path}")
    return summary
