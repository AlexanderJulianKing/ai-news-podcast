"""Score the day's roundup with 'Elsewhere', fitted to that day's reading.

The music is pre-rendered on a Mac by theme_song/elsewhere/build_elsewhere.py into ROUNDUP_BED_ASSETS: a one-bar
shaker head, about 6.4 minutes of body (everything but the shaker, and the shaker on its own track), and the
ending, reverb included, as int16 at 48 kHz. The notes are the same every day; only the length and timing change.
The assets hold Apple sample content: they are copied to the Pi directly and never committed (see .gitignore).

The output, <date>_overview_scored.wav, runs:
  head    one bar of shaker alone, already at its level in the bed, so the shaker never changes level
  body    the roundup read over the bed; the other instruments ease in over ENTRY_S as the voice starts; the bed
          sits UNDER_LUFS under the voice (voices are -23), held steady, with the speech band dipped DIP_DB
  ending  starts BREATH_AIM after the last word (the voice may start up to a quarter bar late to land it on a
          half-bar); the level climbs gradually into it (never faster than MAX_RISE_DB_PER_S) and peaks at OUT_PEAK
  quiet   GAP_AFTER_MUSIC of silence once the music has rung out
These are Alex's rules from 2026-09-25/26. numpy + pydub only, so it runs on the Pi. Any failure raises;
pipeline.make_roundup then falls back to the plain roundup.
"""

import os
import wave

import numpy as np
from pydub import AudioSegment

import newscaster.config as _config
from newscaster.audio.loudness import integrated_lufs, k_weight
from newscaster.logging import print_and_write

SR = 48000
UNDER_LUFS = -39.5          # the bed under the voice
OUT_LUFS = -26.0            # the level the rise climbs towards ...
OUT_PEAK = -26.3            # ... then set exactly: the ending's loudest 400 ms (it averages about -27)
BREATH_AIM = 0.6            # last word -> ending downbeat
BREATH = (0.45, 1.1)        # allowed range for that breath
XFADE_S = 0.4               # body -> ending
ENTRY_S = 1.2               # the bed (all but the shaker) eases in over this long ...
ENTRY_DB = 8.0              # ... from this far below its level
GAP_AFTER_MUSIC = 1.0       # silence after the music rings out
MAX_RISE_DB_PER_S = 13.0    # the fastest the level may climb after the last word
DIP_DB = -4.0               # speech band dip under the voice
DIP_BAND = (800.0, 4000.0)
RIDE_MAX_DB = 6.0           # how far the steadying ride may push the bed either way


class Assets:
    """The pre-rendered music, as float32 slices on demand."""

    def __init__(self, path):
        data = np.load(path)
        self._data = data
        self.scale = float(data['scale'])
        self.bpm = float(data['bpm'])
        self.head_bars = int(data['head_bars'])
        self.bar = 4 * 60.0 / self.bpm
        self.body_frames = data['body'].shape[1]

    def get(self, name, frames=None):
        arr = self._data[name]
        if frames is not None:
            arr = arr[:, :frames]
        return arr.astype(np.float32) * self.scale


def _read_wav(path):
    """16-bit WAV -> mono float32 at 48 kHz (numpy.frombuffer: fast and light for minutes of audio)."""
    with wave.open(path, 'rb') as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f'{path}: expected 16-bit audio, got {8 * width}-bit')
    y = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    if ch > 1:
        y = y[:len(y) // ch * ch].reshape(-1, ch).mean(1)
    if sr != SR:
        t = np.arange(int(len(y) * SR / sr)) * (sr / SR)
        y = np.interp(t, np.arange(len(y)), y).astype(np.float32)
    return y


def _speech_bounds(y, floor_db=-40.0):
    """First and last moment (s) the 10 ms level is within floor_db of the loudest 10 ms."""
    hop = int(0.01 * SR)
    n = len(y) // hop
    r = np.sqrt(np.mean(y[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)) + 1e-12
    idx = np.nonzero(20 * np.log10(r / r.max()) > floor_db)[0]
    if not len(idx):
        raise ValueError('roundup audio is silent')
    return idx[0] * hop / SR, (idx[-1] + 1) * hop / SR


def plan(first_word, last_word, bar, head_s):
    """Where things go. Times relative to the output's start. Returns dict(delta, voice_at, fw, lw, downbeat, breath).

    The voice starts `delta` (0 .. a quarter bar) after the head, chosen so the ending's downbeat, on a half-bar of
    the body's grid, lands as close to BREATH_AIM after the last word as the range allows."""
    body_start = head_s
    length = last_word - first_word
    best = None
    for delta in np.arange(0.0, bar / 4, 0.01):
        lw = body_start + delta + length
        k = int(np.ceil((lw + BREATH[0] - body_start) / (bar / 2)))
        breath = body_start + k * bar / 2 - lw
        if breath <= BREATH[1] + 1e-9 and (best is None or abs(breath - BREATH_AIM) < abs(best[2] - BREATH_AIM)):
            best = (float(delta), k, float(breath))
    delta, k, breath = best
    fw = body_start + delta
    return dict(delta=delta, voice_at=fw - first_word, fw=fw, lw=fw + length, downbeat=body_start + k * bar / 2,
                breath=breath, body_start=body_start)


def _smooth_step(t, t0, t1):
    """0 before t0, 1 after t1, a steady climb between with briefly rounded ends."""
    x = np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1)
    e = 0.12
    return np.where(x < e, x * x / (2 * e * (1 - e)),
                    np.where(x > 1 - e, 1 - (1 - x) ** 2 / (2 * e * (1 - e)), (x - e / 2) / (1 - e))).astype(np.float32)


def _short_term_db(mono, win_s, hop_s):
    """K-weighted loudness (dB) in win_s windows every hop_s. Returns (times, dB)."""
    y = k_weight(np.asarray(mono, dtype=np.float64), SR)
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    hop, win = int(hop_s * SR), int(win_s * SR)
    idx = np.arange(0, len(y), hop)
    lo = np.clip(idx - win // 2, 0, len(y))
    hi = np.clip(idx + win // 2, 0, len(y))
    ms = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return idx / SR, 10 * np.log10(ms + 1e-12) - 0.691


def _ride(mono, t0, t1, n):
    """Per-sample gain (dB) that holds the 3 s loudness of `mono` steady between t0 and t1 (edge values held)."""
    t, db = _short_term_db(mono, 3.0, 0.1)
    sel = (t >= t0) & (t <= t1)
    if not sel.any():
        return np.zeros(n, dtype=np.float32)
    g = np.clip(np.median(db[sel]) - db, -RIDE_MAX_DB, RIDE_MAX_DB)
    idx = np.nonzero(sel)[0]
    g[:idx[0]] = g[idx[0]]
    g[idx[-1] + 1:] = g[idx[-1]]
    k = 20
    g = np.convolve(np.pad(g, (k, k), mode='edge'), np.ones(k) / k, mode='same')[k:-k]
    return np.interp(np.arange(n) / SR, t, g).astype(np.float32)


def _limit_rise(x, t0):
    """Per-sample gain (dB, <= 0) keeping the 400 ms loudness of x from climbing faster than MAX_RISE_DB_PER_S
    after t0. It only turns attacks down; it never boosts."""
    i0 = max(0, int((t0 - 1.0) * SR))
    seg = x[:, i0:].mean(0)
    t, m = _short_term_db(seg, 0.4, 0.01)
    t = t + i0 / SR
    e = m.copy()
    start = int(np.searchsorted(t, t0))
    step = MAX_RISE_DB_PER_S * 0.01
    for i in range(start + 1, len(m)):
        e[i] = min(m[i], e[i - 1] + step)
    g = e - m
    g[:start] = 0.0
    g = np.convolve(np.pad(g, (5, 5), mode='edge'), np.ones(5) / 5, mode='same')[5:-5]
    out = np.zeros(x.shape[1], dtype=np.float32)
    out[i0:] = np.interp(np.arange(i0, x.shape[1]) / SR, t, np.minimum(g, 0.0))
    return out


def _band_dip(x, taps=4096, block=1 << 16):
    """The speech band (DIP_BAND) lowered by DIP_DB: a linear-phase FIR applied block by block (low memory)."""
    f = np.fft.rfftfreq(taps, 1.0 / SR)
    lo, hi = DIP_BAND
    w = np.clip(np.minimum(np.log2(np.maximum(f, 1.0) / lo) * 2 + 1, np.log2(hi / np.maximum(f, 1.0)) * 2 + 1), 0, 1)
    h = np.fft.irfft(10 ** (DIP_DB * w / 20.0), taps)
    h = np.roll(h, taps // 2) * np.hanning(taps)
    delay = taps // 2
    n_fft = block + taps
    H = np.fft.rfft(h, n_fft)
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        acc = np.zeros(x.shape[1] + taps, dtype=np.float64)
        for s in range(0, x.shape[1], block):
            chunk = x[c, s:s + block].astype(np.float64)
            y = np.fft.irfft(np.fft.rfft(chunk, n_fft) * H, n_fft)[:len(chunk) + taps - 1]
            acc[s:s + len(y)] += y
        out[c] = acc[delay:delay + x.shape[1]]
    return out


def _place(n, x, at):
    out = np.zeros((2, n), dtype=np.float32)
    i = int(round(at * SR))
    j = min(n, i + x.shape[1])
    if j > i:
        out[:, i:j] = x[:, :j - i]
    return out


def _write(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype('<i2').T.copy()
    AudioSegment(pcm.tobytes(), frame_rate=SR, sample_width=2, channels=2).export(path, format='wav')


def _lufs(x):
    v = integrated_lufs(np.asarray(x.mean(0) if x.ndim == 2 else x, dtype=np.float64), SR)
    if v is None:
        raise ValueError('could not measure loudness (silent section)')
    return v


def make_roundup(formatted_date2, voice_dir='segment_audio', assets_path=None, out_path=None, music_path=None):
    """Write <voice_dir>/<date>_overview_scored.wav from <date>_overview.wav. Returns a summary dict.

    music_path: also write the music alone (for checking levels)."""
    in_path = os.path.join(voice_dir, f'{formatted_date2}_overview.wav')
    out_path = out_path or os.path.join(voice_dir, f'{formatted_date2}_overview_scored.wav')
    A = Assets(assets_path or getattr(_config, 'ROUNDUP_BED_ASSETS', 'theme_song/elsewhere/elsewhere_assets.npz'))
    voice = _read_wav(in_path)
    fw0, lw0 = _speech_bounds(voice)
    bar, head_s = A.bar, A.head_bars * A.bar
    p = plan(fw0, lw0, bar, head_s)
    FW, LW, D, body_start = p['fw'], p['lw'], p['downbeat'], p['body_start']
    body_frames = int((D + XFADE_S - body_start) * SR) + 1
    if body_frames > A.body_frames:
        raise ValueError(f'roundup ({lw0 - fw0:.0f} s) is longer than the bed ({A.body_frames / SR:.0f} s)')

    ending = A.get('ending')
    n = int((D + ending.shape[1] / SR) * SR) + 1
    t = np.arange(n, dtype=np.float32) / SR

    # the bed: everything but the shaker, and the shaker, both dipped in the speech band while she reads
    rest = _place(n, A.get('body', body_frames), body_start)
    shk = _place(n, A.get('body_shaker', body_frames), body_start)
    rise = _smooth_step(t, LW, D + bar)                       # the climb peaks on the ending's final chord
    lw_i = int(LW * SR)
    rest_clean_tail = rest[:, lw_i:].copy()
    rest = _band_dip(rest)
    shk = _band_dip(shk)
    rest[:, lw_i:] = rest[:, lw_i:] * (1 - rise[lw_i:]) + rest_clean_tail * rise[lw_i:]
    del rest_clean_tail
    g = _ride((rest + shk).mean(0), FW + 1.0, LW - 0.5, n)
    gain = 10 ** (g / 20)
    rest *= gain
    shk *= gain
    norm = UNDER_LUFS - _lufs((rest + shk)[:, int(FW * SR):lw_i])
    k_norm = np.float32(10 ** (norm / 20))
    cut = np.clip(1 - (t - D) / XFADE_S, 0, 1).astype(np.float32)
    rest *= k_norm * cut * (10 ** (-ENTRY_DB * (1 - _smooth_step(t, body_start, body_start + ENTRY_S)) / 20)).astype(np.float32)
    shk *= k_norm * cut
    shk_join = shk[:, int(body_start * SR):int((body_start + 2 * bar) * SR)].copy()
    bed = rest
    bed += shk
    del rest, shk

    # the head: the shaker alone, at exactly the level the bed gives it (the same gains at the join)
    head = _place(n, A.get('head'), 0.0) * k_norm * np.float32(10 ** (g[int(body_start * SR)] / 20))
    head_lufs = _lufs(head[:, :int(body_start * SR)])
    shaker_lufs = _lufs(shk_join)

    # the ending: level with the bed at its downbeat, then carried up by the rise
    end = _place(n, ending, D)
    end *= np.float32(10 ** ((UNDER_LUFS - _lufs(end[:, int(D * SR):int((D + bar / 2) * SR)])) / 20))
    music = bed
    music += end
    del end
    music *= (10 ** (rise * (OUT_LUFS - UNDER_LUFS) / 20)).astype(np.float32)
    mt, mdb = _short_term_db(music[:, int(D * SR):].mean(0), 0.4, 0.02)
    peak = mdb[mt <= 2 * bar].max()
    music *= (np.float32(10 ** ((OUT_PEAK - peak) / 20)) ** rise).astype(np.float32)
    for _ in range(4):                                         # a sharp attack takes a few passes to settle
        music *= (10 ** (_limit_rise(music, LW - 0.3) / 20)).astype(np.float32)
    music += head

    # the music has rung out when its 400 ms level is 40 dB under the ending's loudest; then GAP_AFTER_MUSIC
    et, edb = _short_term_db(music[:, int(D * SR):].mean(0), 0.4, 0.05)
    ref = edb[et <= 5].max()
    music_end = D + et[edb > ref - 40].max()
    total = int((music_end + GAP_AFTER_MUSIC) * SR)
    out = np.zeros((2, max(total, n)), dtype=np.float32)
    out[:, :min(n, out.shape[1])] = music[:, :min(n, out.shape[1])]
    out = out[:, :total]
    i = int(round(p['voice_at'] * SR))
    j = min(total, i + len(voice))
    out[:, max(i, 0):j] += voice[max(0, -i):j - i]
    peak = float(np.abs(out).max())
    if peak > 10 ** (-1 / 20):
        out *= np.float32(10 ** (-1 / 20) / peak)

    _write(out_path, out)
    if music_path:
        _write(music_path, music[:, :total])
    summary = dict(voice_start=round(FW, 2), voice_end=round(LW, 2), downbeat=round(D, 2), breath=round(p['breath'], 2),
                   delta=round(p['delta'], 2), music_end=round(music_end, 2), length=round(total / SR, 2),
                   bed_bars=round((D - body_start) / bar, 1), head_lufs=round(head_lufs, 1), shaker_lufs=round(shaker_lufs, 1),
                   path=out_path)
    print_and_write(f"Elsewhere roundup bed: voice {FW:.1f}-{LW:.1f}s over {summary['bed_bars']} bars, ending "
                    f"{p['breath']:.2f}s after the last word, music ends {music_end - LW:.1f}s after it -> {out_path}")
    return summary
