"""'Off the Wire' intro: timing, variety, sample coverage, fill placement, an end-to-end build, and the fallback."""
import shutil
import wave

import numpy as np
import pytest

from newscaster.audio import off_the_wire_score as score

SR = 48000


@pytest.mark.parametrize("d1", [4.0, 8.1, 8.5, 12.0])
@pytest.mark.parametrize("d2", [10.0, 34.7, 39.8, 61.3, 95.0])
def test_breath_lands_just_after_the_last_word(d1, d2):
    n, s1, pause, v_end = score.plan(d1, d2)
    breath = score.T0 + n * score.BAR + 3 * score.BEAT
    assert 0.45 <= breath - v_end <= 0.95
    assert score.T0 + 0.45 <= s1 <= score.T0 + 1.85      # the host starts within ~2.5 s, right after the logo
    assert 0.4 <= pause <= 1.0 and n >= 1
    assert score.end_time(n) - v_end < 5.0                # the music is done within 5 s of the last word


def test_groove_is_the_same_for_a_date_and_different_across_dates():
    a, b = score.groove(20, 20260925), score.groove(20, 20260925)
    assert [x['tags'] for x in a] == [x['tags'] for x in b]
    days = [score.groove(20, 20260900 + d) for d in range(30)]
    same = np.mean([np.mean([x['tags'] == y['tags'] for x, y in zip(days[d], days[d + 1])]) for d in range(29)])
    assert same < 0.25


def test_no_bar_rhythm_repeats_back_to_back_and_hands_can_play_it():
    for seed in range(20260901, 20260931):
        bars = score.groove(26, seed)
        assert all(bars[i]['tags'][:2] != bars[i - 1]['tags'][:2] for i in range(1, len(bars)))
        for bar in bars:
            for (_, _, ms, *_) in bar['RH']:
                assert max(ms) - min(ms) <= 12
            for (_, _, ms, *_) in bar['LH']:
                assert 28 <= min(ms) and max(ms) <= 60


def test_every_note_played_is_in_the_sample_bank():
    have = set(score.all_pitches())
    for seed in range(20260901, 20260921):
        n, s1, _, v_end = score.plan(8.0, 40.0)
        breaks = [(s1 + t, s1 + t + 0.7) for t in np.arange(3.0, v_end - s1 - 2, 4.1)]
        notes, _, _ = score.arrange(n, s1, v_end, breaks, seed)
        assert {m for _, _, m, _ in notes} <= have


def test_fills_only_answer_inside_sentence_breaks():
    n, s1, _, v_end = score.plan(8.0, 40.0)
    breaks = [(s1 + 6.0, s1 + 6.8), (s1 + 15.0, s1 + 15.6), (s1 + 27.3, s1 + 28.1)]
    _, fills, _ = score.arrange(n, s1, v_end, breaks, 20260925)
    assert fills
    for (t, _, _) in fills:
        assert any(a0 + 0.05 <= t <= a1 + 0.05 for a0, a1 in breaks)


def _tone_bursts(path, seconds, gaps):
    t = np.arange(int(seconds * SR)) / SR
    x = 0.1 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    for a, b in gaps:
        x[int(a * SR):int(b * SR)] = 0
    with wave.open(str(path), 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype('<i2').tobytes())


def _fake_bank(path):
    arrays = {}
    for p in score.all_pitches():
        f = 440 * 2 ** ((p - 69) / 12)
        for layer in (1, 2, 3):
            for suffix, sec in (('', 1.2), ('_long', 4.5)):
                if suffix and p not in score.long_pitches():
                    continue
                t = np.arange(int(sec * SR)) / SR
                y = np.sin(2 * np.pi * f * t) * np.exp(-3 * t)
                arrays[f'p{p}_v{layer}{suffix}'] = np.round(np.stack([y, y]) * 20000).astype('<i2')
    np.savez(path, scale=np.array(1 / 32767.0), **arrays)


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="mp3 export needs ffmpeg")
def test_builds_an_intro_end_to_end(tmp_path):
    from pydub import AudioSegment
    from newscaster.audio.off_the_wire import make_intro
    _tone_bursts(tmp_path / '2026_09_25_intro1.wav', 8.0, [(2.5, 3.1), (5.4, 6.0)])
    _tone_bursts(tmp_path / '2026_09_25_intro2.wav', 30.0, [(6.0, 6.8), (13.0, 13.7), (21.0, 21.9)])
    _fake_bank(tmp_path / 'bank.npz')
    out = tmp_path / 'intro.mp3'
    s = make_intro('2026_09_25', out_path=str(out), voice_dir=str(tmp_path), bank_path=str(tmp_path / 'bank.npz'))
    seg = AudioSegment.from_file(out)
    assert seg.channels == 2 and seg.frame_rate == SR
    assert abs(len(seg) / 1000 - s['music_end']) < 0.2
    assert s['host_start'] < 2.5 and s['music_end'] - s['host_end'] < 5.0
    assert len(s['fills']) >= 2


def test_pipeline_falls_back_to_the_classic_intro(monkeypatch):
    import newscaster.config as config
    from newscaster import pipeline
    from newscaster.audio import intro_music, off_the_wire
    calls = []
    monkeypatch.setattr(intro_music, 'fun_intromaker', lambda d: calls.append(('classic', d)))

    def broken(d):
        raise FileNotFoundError('no sample bank')
    monkeypatch.setattr(off_the_wire, 'make_intro', broken)
    monkeypatch.setattr(config, 'OFF_THE_WIRE_INTRO_ENABLED', True, raising=False)
    pipeline.make_intro('2026_09_25')
    assert calls == [('classic', '2026_09_25')]

    calls.clear()
    monkeypatch.setattr(off_the_wire, 'make_intro', lambda d: calls.append(('wire', d)))
    pipeline.make_intro('2026_09_25')
    assert calls == [('wire', '2026_09_25')]

    calls.clear()
    monkeypatch.setattr(config, 'OFF_THE_WIRE_INTRO_ENABLED', False, raising=False)
    pipeline.make_intro('2026_09_25')
    assert calls == [('classic', '2026_09_25')]
