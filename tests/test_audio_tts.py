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
    assert " Robinson allegedly said he had enough of Kirk's hatred." in spoken_texts
    assert all("[pause]" not in text for text in spoken_texts)
