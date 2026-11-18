import os
import importlib
import sys
import types


class _FakeAudio:
    def __iadd__(self, other):
        return self

    def export(self, path, format=None):
        with open(path, "wb") as handle:
            handle.write(b"fake audio")


class _FakeAudioSegment:
    @staticmethod
    def from_mp3(path):
        return _FakeAudio()

    @staticmethod
    def silent(duration=0):
        return _FakeAudio()


def test_text2speech_speaks_bracketed_quote_clarification(tmp_path, monkeypatch):
    fake_texttospeech = types.ModuleType("google.cloud.texttospeech")
    google_cloud = importlib.import_module("google.cloud")
    monkeypatch.setattr(google_cloud, "texttospeech", fake_texttospeech, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud.texttospeech", fake_texttospeech)

    from newscaster.audio import tts

    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts")
    os.makedirs("segment_audio")
    os.makedirs("logs")

    date2 = "2026_07_08"
    with open(f"output_scripts/{date2}_segment_0.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Grace: What evidence points to that motive?\n"
            "Chloe: Robinson allegedly said he had enough of [Kirk's] hatred.\n"
            "Chloe: [pause]\n"
        )
    for suffix in ("intro1", "intro2", "outro", "overview"):
        with open(f"output_scripts/{date2}_{suffix}.txt", "w", encoding="utf-8") as handle:
            handle.write("Grace: short")

    calls = []

    def fake_google_speak(name, text, filename):
        calls.append((name, text, filename))
        with open(filename, "wb") as handle:
            handle.write(b"fake wav")

    monkeypatch.setattr(tts, "google_speak", fake_google_speak)
    monkeypatch.setattr(tts, "AudioSegment", _FakeAudioSegment)

    import newscaster.audio.overview as overview
    monkeypatch.setattr(overview, "overview_audio_maker", lambda date_str, output_path: None)

    tts.text2speech(date2, ["Grace", "Chloe"])

    spoken_texts = [text for _, text, _ in calls]
    assert "Robinson allegedly said he had enough of Kirk's hatred." in spoken_texts
    assert all("[pause]" not in text for text in spoken_texts)


def _write_wav(path, x, sr=48000):
    import wave
    import numpy as np
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def _read_wav(path):
    import wave
    import numpy as np
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768


def _voice(seconds, sr=48000, level=0.06):
    """A voiced, speech-like signal: harmonics of 140 Hz with a syllable-rate swell."""
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    return level * tone * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) / 2


def test_a_clipped_static_burst_is_silenced(tmp_path, monkeypatch):
    """2026-11-17 pm: Chirp put 0.3 s of full-scale static before Ethan's line on all three renders,
    and turning the whole line down 3 dB left it on air."""
    import numpy as np
    sys.modules.setdefault("google.cloud.texttospeech", types.ModuleType("google.cloud.texttospeech"))
    from newscaster.audio import tts
    sr = 48000
    x = _voice(6)
    rng = np.random.default_rng(0)
    x[int(0.7 * sr):int(1.0 * sr)] = np.clip(rng.normal(0, 0.8, int(0.3 * sr)), -1, 1)
    path = tmp_path / "line.wav"
    _write_wav(path, x)

    assert tts.silence_burst(str(path)) is not None
    y = _read_wav(path)
    assert np.abs(y[int(0.7 * sr):int(1.0 * sr)]).max() < 0.01          # the static is gone
    assert np.allclose(y[int(1.2 * sr):], x[int(1.2 * sr):], atol=1e-3)  # the speech is untouched
    assert tts.check_clipping(str(path))[0] is False


def test_a_loud_voiced_passage_is_not_silenced(tmp_path):
    """Clipping on a shouted word is speech, not static: leave it for the gain fallback."""
    import numpy as np
    sys.modules.setdefault("google.cloud.texttospeech", types.ModuleType("google.cloud.texttospeech"))
    from newscaster.audio import tts
    sr = 48000
    x = _voice(6)
    x[2 * sr:2 * sr + 14400] = np.clip(_voice(0.3, level=1.6), -1, 1)
    path = tmp_path / "line.wav"
    _write_wav(path, x)

    assert tts.silence_burst(str(path)) is None
    assert np.allclose(_read_wav(path), np.clip(x, -1, 1), atol=1e-3)   # file unchanged
    assert tts.check_clipping(str(path))[0] is True
