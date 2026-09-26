"""The roundup's music bed ('Elsewhere'): timing, levels, fallbacks. Synthetic assets and voice; no network."""

import os
import wave

import numpy as np
import pytest

from newscaster.audio import roundup_bed as rb

SR = rb.SR
BPM = 94
BAR = 4 * 60 / BPM


def _write_wav(path, y, channels=1):
    pcm = (np.clip(y, -1, 1) * 32767).astype('<i2')
    if channels == 2:
        pcm = pcm.T.copy()
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _read_stereo(path):
    with wave.open(str(path), 'rb') as w:
        ch, n, sr = w.getnchannels(), w.getnframes(), w.getframerate()
        y = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float32) / 32768
    return y.reshape(-1, ch).T, sr


def _assets(path, body_bars=40, seed=0):
    """Stand-in music: a shaker-like noise pulse on sixteenths, a chord that changes every bar, a decaying ending."""
    rng = np.random.default_rng(seed)

    def shaker(bars):
        n = int(bars * BAR * SR)
        out = np.zeros(n)
        for k in range(int(bars * 16)):
            i = int(k * BAR / 16 * SR)
            L = min(int(0.05 * SR), n - i)
            out[i:i + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (0.01 * SR)) * (0.5 if k % 4 else 1.0)
        return np.stack([out, out]) * 0.05

    def chords(bars):
        n = int(bars * BAR * SR)
        t = np.arange(n) / SR
        out = np.zeros(n)
        for b in range(bars):
            i, j = int(b * BAR * SR), int((b + 1) * BAR * SR)
            for f in (110 * 2 ** ((b % 5) / 12), 220 * 2 ** ((b % 3) / 12), 330):
                out[i:j] += np.sin(2 * np.pi * f * t[i:j]) * 0.1
        return np.stack([out, out * 0.9])

    head = shaker(1) * 1.0
    body_shaker = shaker(body_bars)
    body = chords(body_bars)
    t = np.arange(int(4 * BAR * SR)) / SR
    end = np.sin(2 * np.pi * 220 * t) * np.exp(-t / 2) * 0.3
    ending = np.stack([end, end])
    arrays = dict(head=head, body=body, body_shaker=body_shaker, ending=ending)
    peak = max(np.abs(a).max() for a in arrays.values())
    scale = peak / 32767
    np.savez_compressed(path, scale=np.array(scale), bpm=np.array(BPM), head_bars=np.array(1), body_bars=np.array(body_bars),
                        **{k: np.round(a / scale).astype('<i2') for k, a in arrays.items()})
    return path


def _voice(seconds, seed=1):
    """Speech-like: bursts of band-limited noise with short pauses, 0.4 s lead-in and 0.5 s tail of silence."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int((seconds + 0.9) * SR))
    t = 0.4
    while t < seconds + 0.4 - 0.3:
        L = min(rng.uniform(0.8, 2.5), seconds + 0.4 - t)
        n = int(L * SR)
        burst = np.convolve(rng.standard_normal(n), np.ones(12) / 12, mode='same')
        i = int(t * SR)
        y[i:i + n] += burst * 0.25
        t += L + rng.uniform(0.2, 0.6)
    return y


@pytest.fixture
def day(tmp_path):
    assets = _assets(tmp_path / 'assets.npz')
    _write_wav(tmp_path / '2026_01_02_overview.wav', _voice(45))
    return tmp_path, str(assets)


def test_plan_lands_the_ending_on_a_half_bar_with_a_breath():
    for length in (20.0, 45.3, 125.0, 243.1, 357.0):
        p = rb.plan(0.4, 0.4 + length, BAR, BAR)
        assert 0 <= p['delta'] < BAR / 4
        assert rb.BREATH[0] - 1e-9 <= p['breath'] <= rb.BREATH[1] + 1e-9
        halves = (p['downbeat'] - p['body_start']) / (BAR / 2)
        assert abs(halves - round(halves)) < 1e-6
        assert abs(p['downbeat'] - p['lw'] - p['breath']) < 1e-6


def test_make_roundup_levels_timing_and_shaker_join(day):
    d, assets = day
    s = rb.make_roundup('2026_01_02', voice_dir=str(d), assets_path=assets, music_path=str(d / 'music.wav'))
    out, sr = _read_stereo(s['path'])
    assert sr == SR and out.shape[0] == 2
    music, _ = _read_stereo(d / 'music.wav')
    FW, LW, D = s['voice_start'], s['voice_end'], s['downbeat']
    assert rb.BREATH[0] <= s['breath'] <= rb.BREATH[1]
    # the bed under the voice at its level
    bed = rb._lufs(music[:, int(FW * SR):int(LW * SR)])
    assert abs(bed - rb.UNDER_LUFS) < 1.0
    # the shaker keeps one level from the head into the bed
    assert abs(s['head_lufs'] - s['shaker_lufs']) < 1.0
    # after the last word the level climbs gradually, and it gets well above the bed
    t, m = rb._short_term_db(music.mean(0), 0.4, 0.1)
    x = m[(t >= LW - 0.3) & (t <= D + BAR + 0.5)]
    assert max(x[i + 3] - x[i] for i in range(len(x) - 3)) < 6.0
    assert m[(t > D) & (t < D + 2 * BAR)].max() > bed + 8
    # it ends with about GAP_AFTER_MUSIC of quiet
    assert abs(s['length'] - s['music_end'] - rb.GAP_AFTER_MUSIC) < 0.05
    # the whole reading is in the output
    assert out.shape[1] / SR > LW


def test_roundup_longer_than_the_bed_raises(tmp_path):
    assets = _assets(tmp_path / 'assets.npz', body_bars=10)
    _write_wav(tmp_path / '2026_01_03_overview.wav', _voice(60))
    with pytest.raises(ValueError, match='longer than the bed'):
        rb.make_roundup('2026_01_03', voice_dir=str(tmp_path), assets_path=str(assets))


def test_band_dip_lowers_the_speech_band_only_and_keeps_timing():
    t = np.arange(SR * 2) / SR
    for f, want in ((2000.0, rb.DIP_DB), (150.0, 0.0)):
        x = np.stack([np.sin(2 * np.pi * f * t)] * 2).astype(np.float32)
        y = rb._band_dip(x)
        mid = slice(SR // 2, 3 * SR // 2)
        got = 20 * np.log10(np.sqrt(np.mean(y[0, mid] ** 2)) / np.sqrt(np.mean(x[0, mid] ** 2)))
        assert abs(got - want) < 0.5, (f, got)
    # timing: broadband noise has one clear alignment (a pure tone repeats every period)
    rng = np.random.default_rng(0)
    x = np.stack([rng.standard_normal(SR * 2)] * 2).astype(np.float32)
    y = rb._band_dip(x)
    mid = slice(SR // 2, 3 * SR // 2)
    lag = np.argmax(np.correlate(y[0, mid], x[0, SR // 2 - 50:3 * SR // 2 + 50], 'valid')) - 50
    assert lag == 0


def test_limit_rise_never_boosts():
    x = np.zeros((2, SR * 4), dtype=np.float32)
    x[:, SR:] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 3) / SR)
    g = rb._limit_rise(x, 0.5)
    assert g.max() <= 1e-6
    assert g.min() < -3.0           # the sudden entry was turned down


def test_assembly_uses_the_scored_roundup(tmp_path, monkeypatch):
    from pydub import AudioSegment
    from newscaster.audio import assembly
    monkeypatch.chdir(tmp_path)
    os.makedirs('segment_audio')
    os.makedirs('output_audio')
    tone = AudioSegment.silent(duration=1000, frame_rate=SR)
    tone.export('segment_audio/2026_01_04_intro.mp3', format='mp3')
    tone.export('segment_audio/2026_01_04_segment_0.mp3', format='mp3')
    AudioSegment.silent(duration=3000, frame_rate=SR).export('segment_audio/2026_01_04_overview.wav', format='wav')
    AudioSegment.silent(duration=6000, frame_rate=SR).export('segment_audio/2026_01_04_overview_scored.wav', format='wav')
    tone.export('segment_audio/2026_01_04_outro.wav', format='wav')
    assembly.assemble_podcast('2026_01_04')
    with_scored = len(AudioSegment.from_mp3('output_audio/2026_01_04_HQ.mp3'))
    os.remove('segment_audio/2026_01_04_overview_scored.wav')
    assembly.assemble_podcast('2026_01_04')
    plain = len(AudioSegment.from_mp3('output_audio/2026_01_04_HQ.mp3'))
    # scored: intro 1 + gap 2 + segment 1 + gap 2 + scored 6 (its own quiet included, no extra gap) + outro 1 = 13 s
    # plain:  intro 1 + gap 2 + segment 1 + gap 2 + overview 3 + gap 2 + outro 1 = 12 s
    assert abs(with_scored - 13000) < 150
    assert abs(plain - 12000) < 150


def test_pipeline_falls_back_to_the_plain_roundup(tmp_path, monkeypatch):
    from newscaster import pipeline
    monkeypatch.chdir(tmp_path)
    os.makedirs('segment_audio')
    stale = 'segment_audio/2026_01_05_overview_scored.wav'
    open(stale, 'w').close()
    monkeypatch.setattr(pipeline._config, 'ROUNDUP_BED_ENABLED', True, raising=False)

    def boom(*a, **k):
        raise RuntimeError('no assets')

    monkeypatch.setattr(rb, 'make_roundup', boom)
    pipeline.make_roundup('2026_01_05')          # must not raise
    assert not os.path.exists(stale)             # and must not leave an old scored file for assembly to pick up
