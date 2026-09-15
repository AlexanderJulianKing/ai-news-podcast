"""ITU-R BS.1770-4 loudness measurement and per-file loudness normalization.

Google's Chirp3-HD voices ship at noticeably different levels. Measured over
the episodes sitting in segment_audio/, Chloe (Chirp3-HD-Leda) runs about
2.5 LU below Grace (Chirp3-HD-Aoede) inside the same interview segment, where
the two alternate line by line. Nothing upstream equalises them, so each
synthesised line is measured and pulled to a common target here.

numpy only: the Pi's venv has no scipy, so the K-weighting biquads are applied
by overlap-save FFT filtering instead of scipy.signal.lfilter. The two agree to
well under a tenth of a LU; tests/test_loudness.py checks that against scipy
wherever scipy happens to be installed.
"""

import struct
import wave

import numpy as np

# BS.1770-4 block structure.
_BLOCK_SEC = 0.400          # gating block length
_OVERLAP = 0.75             # blocks overlap by 75%, i.e. one every 100 ms
_ABSOLUTE_GATE = -70.0      # LKFS; blocks below this never count
_RELATIVE_GATE = -10.0      # LU below the ungated mean

# Overlap-save parameters. _TAIL is how much of each biquad's impulse response
# we assume is non-negligible; the 38 Hz high-pass has a ~4 ms time constant, so
# 8192 samples (170 ms at 48 kHz) is far past its decay.
_N_FFT = 65536
_TAIL = 8192


def _shelf_coeffs(sr):
    """BS.1770 stage 1: high-frequency shelving filter (head modelling)."""
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    denom = 1.0 + k / q + k * k
    b = np.array([
        (vh + vb * k / q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / q + k * k) / denom,
    ])
    a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ])
    return b, a


def _highpass_coeffs(sr):
    """BS.1770 stage 2: RLB high-pass filter."""
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    denom = 1.0 + k / q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ])
    return b, a


def _biquad_response(b, a, n_fft):
    """Frequency response of a biquad sampled at n_fft rfft bins."""
    w = 2.0 * np.pi * np.fft.rfftfreq(n_fft)
    z = np.exp(-1j * w)
    z2 = z * z
    num = b[0] + b[1] * z + b[2] * z2
    den = a[0] + a[1] * z + a[2] * z2
    return num / den


def _filter_biquad(x, b, a):
    """Apply a biquad by overlap-save FFT filtering. Returns len(x) samples.

    Equivalent to a direct time-domain recursion to within the energy of the
    impulse response past _TAIL samples, which for these two filters is far
    below the measurement's resolution.
    """
    n = len(x)
    if n == 0:
        return x
    n_fft = _N_FFT
    tail = _TAIL
    if n + tail < n_fft:
        # Short signal: one zero-padded transform covers it outright.
        n_fft = int(2 ** np.ceil(np.log2(n + tail)))
        tail = min(tail, n_fft - n)
    hop = n_fft - tail
    resp = _biquad_response(b, a, n_fft)
    padded = np.concatenate([np.zeros(tail), x])
    out = np.empty(n)
    pos = 0
    while pos < n:
        chunk = padded[pos:pos + n_fft]
        if len(chunk) < n_fft:
            chunk = np.concatenate([chunk, np.zeros(n_fft - len(chunk))])
        filtered = np.fft.irfft(np.fft.rfft(chunk) * resp, n_fft)
        take = min(hop, n - pos)
        out[pos:pos + take] = filtered[tail:tail + take]
        pos += hop
    return out


def k_weight(x, sr):
    """Apply the BS.1770 K-weighting filter chain."""
    b1, a1 = _shelf_coeffs(sr)
    b2, a2 = _highpass_coeffs(sr)
    return _filter_biquad(_filter_biquad(x, b1, a1), b2, a2)


def integrated_lufs(x, sr):
    """Gated integrated loudness of a mono signal, or None if unmeasurable.

    Returns None when the signal is shorter than one 400 ms gating block or
    when every block falls below the absolute gate (silence).
    """
    block = int(round(_BLOCK_SEC * sr))
    hop = max(1, int(round(block * (1.0 - _OVERLAP))))
    if block <= 0 or len(x) < block:
        return None

    y = k_weight(np.asarray(x, dtype=np.float64), sr)

    starts = np.arange(0, len(y) - block + 1, hop)
    power = np.empty(len(starts))
    for i, s in enumerate(starts):
        seg = y[s:s + block]
        power[i] = float(np.dot(seg, seg)) / block

    with np.errstate(divide='ignore'):
        loud = -0.691 + 10.0 * np.log10(power + 1e-20)

    above_absolute = loud > _ABSOLUTE_GATE
    if not above_absolute.any():
        return None

    ungated_mean = -0.691 + 10.0 * np.log10(np.mean(power[above_absolute]))
    gated = above_absolute & (loud > ungated_mean + _RELATIVE_GATE)
    if not gated.any():
        return None

    return float(-0.691 + 10.0 * np.log10(np.mean(power[gated])))


def read_wav_mono(filename):
    """Read a 16-bit WAV as float samples in [-1, 1). Returns (samples, sr).

    Returns (None, sr) for sample widths this module does not handle, matching
    check_clipping's behaviour of quietly declining rather than raising.
    """
    with wave.open(filename, 'r') as w:
        n_frames = w.getnframes()
        sr = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(n_frames)
    if sampwidth != 2:
        return None, sr
    n_samples = len(frames) // 2
    if n_samples == 0:
        return np.zeros(0), sr
    samples = np.array(struct.unpack('<' + 'h' * n_samples, frames), dtype=np.float64) / 32768.0
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples, sr


def measure_wav(filename):
    """Return (integrated_lufs, peak_dbfs) for a WAV file.

    Either value may be None when it cannot be measured (unsupported sample
    width, too short to gate, or digital silence).
    """
    samples, _sr = read_wav_mono(filename)
    if samples is None:
        return None, None
    if len(samples) == 0:
        return None, None
    peak = float(np.max(np.abs(samples)))
    peak_db = 20.0 * np.log10(peak) if peak > 0 else None
    return integrated_lufs(samples, _sr), peak_db


def plan_gain(loudness, peak_db, target_lufs, peak_ceiling_db, max_gain_db):
    """Work out the gain to apply, in dB, honouring the peak and gain limits.

    Split out from normalize_loudness so the decision is testable without
    touching audio files.
    """
    if loudness is None:
        return None
    gain = target_lufs - loudness
    gain = max(-max_gain_db, min(max_gain_db, gain))
    if peak_db is not None:
        gain = min(gain, peak_ceiling_db - peak_db)
    return gain
