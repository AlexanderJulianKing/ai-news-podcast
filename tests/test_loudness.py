import importlib
import struct
import sys
import types
import wave

import numpy as np
import pytest

from newscaster.audio import loudness as L

SR = 48000


def write_wav(path, samples, sr=SR, channels=1, sampwidth=2):
    data = np.clip(np.asarray(samples), -1.0, 0.999969)
    ints = np.round(data * 32768.0).astype('<i2')
    with wave.open(str(path), 'w') as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(ints.tobytes())


def sine(seconds=5.0, dbfs=-23.0, freq=1000.0, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return 10 ** (dbfs / 20.0) * np.sin(2 * np.pi * freq * t)


# --- measurement correctness -------------------------------------------------

def test_matches_ebu_tech_3341_reference_tone():
    """EBU Tech 3341 case 1: a -23 dBFS 1 kHz sine reads -23.0 LUFS in stereo.

    We measure mono, so the two equal channels contribute +10*log10(2).
    """
    mono = L.integrated_lufs(sine(), SR)
    assert mono is not None
    assert abs((mono + 10 * np.log10(2)) - (-23.0)) < 0.1


def test_measurement_is_exactly_linear_in_gain():
    """The property normalization depends on: G dB of gain moves LUFS by G."""
    base_samples = sine()
    base = L.integrated_lufs(base_samples, SR)
    for gain in (-11.0, -6.0, -0.5, 3.0, 11.0):
        shifted = L.integrated_lufs(base_samples * 10 ** (gain / 20.0), SR)
        assert abs((shifted - base) - gain) < 0.001


def test_fft_filter_matches_direct_recursion():
    """The overlap-save K-weighting must agree with a plain IIR recursion.

    scipy is not installed on the Pi, so the module filters by FFT; this checks
    that choice against scipy.signal.lfilter wherever scipy is available.
    """
    lfilter = pytest.importorskip('scipy.signal').lfilter
    rng = np.random.default_rng(1234)
    # Noise plus tones, long enough to span several overlap-save blocks.
    x = rng.normal(0, 0.1, SR * 3) + 0.2 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)
    b1, a1 = L._shelf_coeffs(SR)
    b2, a2 = L._highpass_coeffs(SR)
    reference = lfilter(b2, a2, lfilter(b1, a1, x))
    assert np.max(np.abs(L.k_weight(x, SR) - reference)) < 1e-9


def test_relative_gate_ignores_quiet_passages():
    """A long quiet tail must not drag the measurement down."""
    loud = sine(seconds=5.0, dbfs=-20.0)
    quiet = sine(seconds=20.0, dbfs=-60.0)
    assert abs(L.integrated_lufs(np.concatenate([loud, quiet]), SR)
               - L.integrated_lufs(loud, SR)) < 0.2


def test_unmeasurable_signals_return_none():
    assert L.integrated_lufs(np.zeros(SR), SR) is None          # digital silence
    assert L.integrated_lufs(sine(seconds=0.2), SR) is None      # under one block
    assert L.integrated_lufs(np.zeros(0), SR) is None


# --- file reading ------------------------------------------------------------

def test_reads_mono_wav_and_reports_peak(tmp_path):
    path = tmp_path / 'tone.wav'
    write_wav(path, sine(dbfs=-6.0))
    loudness, peak_db = L.measure_wav(str(path))
    assert loudness is not None
    assert abs(peak_db - (-6.0)) < 0.1


def test_stereo_is_downmixed(tmp_path):
    path = tmp_path / 'stereo.wav'
    mono = sine()
    write_wav(path, np.repeat(mono, 2), channels=2)
    samples, sr = L.read_wav_mono(str(path))
    assert sr == SR
    assert len(samples) == len(mono)
    assert np.max(np.abs(samples - mono)) < 1e-4


def test_unsupported_sample_width_declines(tmp_path):
    path = tmp_path / 'wide.wav'
    with wave.open(str(path), 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(4)
        w.setframerate(SR)
        w.writeframes(struct.pack('<' + 'i' * SR, *([0] * SR)))
    assert L.measure_wav(str(path)) == (None, None)


# --- gain planning -----------------------------------------------------------

def test_plan_gain_hits_the_target_when_nothing_binds():
    assert L.plan_gain(-25.5, -12.0, -23.0, -1.0, 12.0) == pytest.approx(2.5)


def test_plan_gain_respects_the_peak_ceiling():
    # Wants +2.5 dB but only 1.0 dB of headroom remains.
    assert L.plan_gain(-25.5, -2.0, -23.0, -1.0, 12.0) == pytest.approx(1.0)


def test_plan_gain_pulls_hot_peaks_down_to_the_ceiling():
    assert L.plan_gain(-23.0, 0.0, -23.0, -1.0, 12.0) == pytest.approx(-1.0)


def test_plan_gain_refuses_to_amplify_a_degenerate_render():
    # A near-silent file would need +37 dB; the clamp stops it at the limit.
    assert L.plan_gain(-60.0, -40.0, -23.0, -1.0, 12.0) == pytest.approx(12.0)


def test_plan_gain_clamps_extreme_attenuation_too():
    assert L.plan_gain(-2.0, -30.0, -23.0, -1.0, 12.0) == pytest.approx(-12.0)


def test_plan_gain_without_a_measurement_is_none():
    assert L.plan_gain(None, -6.0, -23.0, -1.0, 12.0) is None


# --- normalize_loudness, end to end -----------------------------------------

@pytest.fixture
def tts(monkeypatch):
    """Import newscaster.audio.tts with the Google TTS client stubbed out."""
    fake = types.ModuleType('google.cloud.texttospeech')
    google_cloud = importlib.import_module('google.cloud')
    monkeypatch.setattr(google_cloud, 'texttospeech', fake, raising=False)
    monkeypatch.setitem(sys.modules, 'google.cloud.texttospeech', fake)
    from newscaster.audio import tts as module
    return module


def test_normalize_loudness_brings_a_quiet_file_to_target(tmp_path, tts):
    path = tmp_path / 'quiet.wav'
    write_wav(path, sine(dbfs=-30.0))   # -33 LUFS: needs +10 dB, inside the clamp
    before, _ = L.measure_wav(str(path))

    gain = tts.normalize_loudness(str(path))

    after, peak_db = L.measure_wav(str(path))
    assert gain == pytest.approx(-23.0 - before, abs=0.05)
    assert abs(after - (-23.0)) < 0.1
    assert peak_db <= -1.0 + 0.05


def test_normalize_loudness_brings_a_loud_file_to_target(tmp_path, tts):
    path = tmp_path / 'loud.wav'
    write_wav(path, sine(dbfs=-13.0))
    tts.normalize_loudness(str(path))
    after, peak_db = L.measure_wav(str(path))
    assert abs(after - (-23.0)) < 0.1
    assert peak_db <= -1.0 + 0.05


def test_normalize_loudness_equalises_two_voices(tmp_path, tts):
    """The actual bug: two renders 2.5 dB apart must end up matched."""
    grace = tmp_path / 'grace.wav'
    chloe = tmp_path / 'chloe.wav'
    write_wav(grace, sine(dbfs=-22.0, freq=300.0))
    write_wav(chloe, sine(dbfs=-24.5, freq=300.0))
    assert abs(L.measure_wav(str(grace))[0] - L.measure_wav(str(chloe))[0]) > 2.0

    tts.normalize_loudness(str(grace))
    tts.normalize_loudness(str(chloe))

    assert abs(L.measure_wav(str(grace))[0] - L.measure_wav(str(chloe))[0]) < 0.1


def test_normalize_loudness_skips_negligible_adjustments(tmp_path, tts):
    path = tmp_path / 'ontarget.wav'
    write_wav(path, sine(dbfs=-20.0))      # measures ~-23.0 LUFS already
    assert abs(L.measure_wav(str(path))[0] - (-23.0)) < 0.1
    before = path.read_bytes()
    assert tts.normalize_loudness(str(path)) is None
    assert path.read_bytes() == before     # file untouched, no re-encode


def test_normalize_loudness_stops_at_the_max_gain_clamp(tmp_path, tts):
    """A file 13 dB below target is lifted by the 12 dB limit, not all the way."""
    path = tmp_path / 'verydark.wav'
    write_wav(path, sine(dbfs=-33.0))
    assert tts.normalize_loudness(str(path)) == pytest.approx(12.0)
    assert L.measure_wav(str(path))[0] == pytest.approx(-24.0, abs=0.1)


def test_kill_switch_leaves_audio_alone(tmp_path, tts, monkeypatch):
    from newscaster import config as cfg
    monkeypatch.setattr(cfg, 'LOUDNESS_NORMALIZE_ENABLED', False)
    path = tmp_path / 'quiet.wav'
    write_wav(path, sine(dbfs=-30.0))
    before = path.read_bytes()
    assert tts.normalize_loudness(str(path)) is None
    assert path.read_bytes() == before


def test_silence_is_left_alone_rather_than_amplified(tmp_path, tts):
    path = tmp_path / 'silence.wav'
    write_wav(path, np.zeros(SR * 2))
    before = path.read_bytes()
    assert tts.normalize_loudness(str(path)) is None
    assert path.read_bytes() == before


def test_measurement_failure_leaves_audio_alone(tmp_path, tts):
    path = tmp_path / 'missing.wav'
    assert tts.normalize_loudness(str(path)) is None
    assert not path.exists()
