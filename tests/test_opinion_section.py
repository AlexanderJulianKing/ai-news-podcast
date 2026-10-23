"""Sunday opinion section (2026-10-23).

On Sunday mornings one opinion piece, read by its own voice (Gemini TTS, Schedar),
replaces the two main stories; the roundup follows. A once-in-years story keeps the
normal show. Alex's specs: it takes a side, any topic, between one and two main
stories long, and it must not sound AI-written (see the banned list in opinion.py).
"""
import json
import sys
import types
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

try:                                       # broken on the Mac (protobuf mismatch), fine on the Pi
    from google.cloud import texttospeech  # noqa: F401
except Exception:
    import google.cloud
    _stub = types.ModuleType("google.cloud.texttospeech")
    google.cloud.texttospeech = _stub
    sys.modules["google.cloud.texttospeech"] = _stub

from newscaster import edition, opinion, pipeline, review
from newscaster.audio import gemini_tts, tts
from newscaster.scrapers.topic_finder import TopicFinderResult

SUN, SAT = date(2026, 10, 25), date(2026, 10, 24)


def test_opinion_runs_on_sunday_mornings_only(monkeypatch):
    monkeypatch.setattr(opinion._config, "OPINION_SUNDAY_ENABLED", True)
    assert opinion.is_opinion_day(SUN, edition.MORNING)
    assert not opinion.is_opinion_day(SAT, edition.MORNING)
    assert not opinion.is_opinion_day(SUN, edition.EVENING)
    monkeypatch.setattr(opinion._config, "OPINION_SUNDAY_ENABLED", False)
    assert not opinion.is_opinion_day(SUN, edition.MORNING)


def test_a_once_in_years_story_keeps_the_normal_show():
    with patch.object(opinion, "get_llm_response", return_value="OVERRIDE: NO\nREASON: an ordinary news day"):
        assert opinion.breaking_override("Senate passes farm bill", "brief", ["a", "b"]) is False
    with patch.object(opinion, "get_llm_response", return_value="OVERRIDE: YES\nREASON: a head of state was killed"):
        assert opinion.breaking_override("President shot", "brief", ["a"]) is True
    # If the check itself fails, don't risk skipping a huge story: run the normal show.
    with patch.object(opinion, "get_llm_response", side_effect=RuntimeError("down")):
        assert opinion.breaking_override("x", "brief", []) is True


def test_the_main_picks_still_air_in_the_roundup():
    out = opinion.roundup_with_mains(["Big story", "Everyday story"], ["Side one", "Side two"])
    assert out == ["Big story", "Everyday story", "Side one", "Side two"]


def test_gather_completes_with_no_main_stories_on_an_opinion_day(tmp_path, monkeypatch):
    # Without its marker, a gather with zero stories would retry forever (see write_scripts).
    monkeypatch.chdir(tmp_path)
    pipeline._ensure_output_dirs()
    result = TopicFinderResult(topics=[], overview="here are some other headlines: ...",
                               follow_up_prompt_text="", challenging_follow_up_prompt_text="", opinion=True)
    with patch.object(pipeline, "topic_finder", return_value=result), \
         patch.object(pipeline, "_gather_one_topic", side_effect=AssertionError("no main-story research")), \
         patch.object(pipeline, "index_day", return_value=0):
        pipeline.gather_news("October 25, 2026", "2026_10_25")
    assert (tmp_path / "segment_summaries/2026_10_25_GATHER_COMPLETE.flag").exists()
    assert (tmp_path / "output_scripts/2026_10_25_overview.txt").read_text().startswith("here are")


def _fake_llm(prompt, system_prompt=None, mode=None, **_):
    if system_prompt == opinion.PITCH_PROMPT:
        return json.dumps({"candidates": [{"topic": "Antibiotics", "claim": "Pay for readiness.",
                                           "why_new": "x", "style": "off the news cycle"}],
                           "chosen": 0, "questions": ["How many US deaths from resistant infections?"]})
    if system_prompt == opinion.GAPS_PROMPT:
        return json.dumps({"claim": "Pay for readiness, not doses.", "questions": ["What does PASTEUR pay?"]})
    if system_prompt == opinion.WRITER_PROMPT:
        body = ("If you work in an office, there's a red cabinet down the hall. " * 40 + "\n\n"
                + "Here is my argument. We pay by the dose. " * 30 + "\n\n"
                + "The CDC counts more than thirty-five thousand deaths a year. "
                + "Hospitals keep the newest drugs locked away. " * 40)
        return f"TITLE: Pay for antibiotics like a fire department\nTEASER: Why we should pay drugmakers to wait.\nSCRIPT:\n{body}"
    if system_prompt == opinion.REVISE_PROMPT:
        return prompt.split("SCRIPT:\n", 1)[1].replace("Here is my argument. ", "")
    if system_prompt == opinion.SOURCING_PROMPT:
        assert "1. FLAG: thirty-five thousand" in prompt and "more than 35,000 people die" in prompt
        assert "2. FLAG" not in prompt
        return json.dumps({"verdicts": [{"flag": 1, "verdict": "fix",
                                         "find": "The CDC counts more than thirty-five thousand deaths a year",
                                         "replace": "The CDC counts tens of thousands of deaths a year"}]})
    raise AssertionError(f"unexpected prompt: {system_prompt[:60]}")


def _fake_research(question, formatted_date):
    return {"status": "success", "answer": 'FINDINGS:\n- "more than 35,000 people die" [https://www.cdc.gov/x]',
            "sources": [{"title": "CDC page", "url": "https://www.cdc.gov/x", "excerpt": "more than 35,000 people die"}],
            "metadata": {}}


def test_the_writer_leaves_a_script_the_voices_and_the_fact_check_can_use(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline._ensure_output_dirs()
    checked = {}

    def fake_faithfulness(script, evidence):
        checked["evidence"] = evidence
        return ["FLAG: thirty-five thousand deaths a year — not in the quote's wording"] if "thirty-five" in script else []
    # (the check runs twice and the identical flags are merged into one)
    with patch.object(opinion, "get_llm_response", side_effect=_fake_llm), \
         patch.object(opinion, "research_with_tools", side_effect=_fake_research), \
         patch.object(opinion, "faithfulness_flags", side_effect=fake_faithfulness):
        opinion.write_opinion("2026_10_25", "October 25, 2026")
    lines = [l for l in (tmp_path / "output_scripts/2026_10_25_segment_0.txt").read_text().split("\n") if l.strip()]
    assert lines[0].startswith("Grace:") and "opinion" in lines[0].lower()
    assert "doesn't necessarily reflect the views of Alex" in lines[0]    # Alex's disclaimer, read by Grace
    assert len(lines) > 2 and all(l.startswith("Opinion: ") for l in lines[1:])
    assert not opinion.banned_phrases("\n".join(lines[1:]))        # the "Here is my argument" tic was rewritten
    assert "thirty-five" not in "\n".join(lines)                    # a claim the self-check flagged was fixed
    assert "more than 35,000 people die" in checked["evidence"]      # ...checked against the writer's own quotes
    corpus = review.build_segment_corpus("2026_10_25", 0)          # the fact check sees every source, labeled
    assert "SOURCE: CDC page — https://www.cdc.gov/x" in corpus and "35,000 people die" in corpus
    assert (tmp_path / "episode_titles/2026_10_25.txt").read_text() == "Opinion: Pay for antibiotics like a fire department"
    history = json.loads((tmp_path / opinion._config.OPINION_HISTORY_PATH).read_text())
    assert history[-1]["key"] == "2026_10_25" and "readiness" in history[-1]["claim"]


def test_a_line_with_colons_is_spoken_whole_by_its_voice():
    voices = ["Grace", "Ethan", "Opinion"]
    assert tts.split_speaker("Opinion: the act has a name only Congress could love: PASTEUR.", voices, "Grace") == \
        ("Opinion", "the act has a name only Congress could love: PASTEUR.")
    assert tts.split_speaker("Note: this line has no speaker.", voices, "Ethan") == ("Ethan", "Note: this line has no speaker.")


def test_gemini_audio_is_read_as_a_wav_file_not_as_raw_samples():
    # 2026-10-23: Gemini 3.8 TTS returns a whole WAV file: a 36-byte header, the samples, then a 6 KB
    # "C2PA" content-credentials block. Read as raw samples, the header aired as a click and the
    # C2PA block as a ~125 ms burst of noise at the end of every paragraph; Alex heard both.
    import io, struct, wave as wavemod
    samples = (np.sin(np.arange(2400) / 10) * 8000).astype("<i2")
    buf = io.BytesIO()
    with wavemod.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(samples.tobytes())
    c2pa = b"C2PA" + struct.pack("<I", 6064) + bytes(range(256)) * 23 + bytes(176)
    data = bytearray(buf.getvalue() + c2pa)
    data[4:8] = struct.pack("<I", len(data) - 8)
    x, rate = gemini_tts.decode(bytes(data), "audio/wav")
    assert rate == 24000 and len(x) == len(samples)
    np.testing.assert_allclose(x, samples / 32768)
    raw, rate = gemini_tts.decode(samples.tobytes(), "audio/L16;codec=pcm;rate=24000")
    assert rate == 24000 and len(raw) == len(samples)


def test_opinion_voice_falls_back_to_chirp_schedar_when_gemini_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asked = {}

    class Client:
        def synthesize_speech(self, input, voice, audio_config):
            asked["voice"] = voice.name
            raise RuntimeError("stop here")       # the voice choice is all this test needs

    fake = SimpleNamespace(
        TextToSpeechClient=lambda: Client(), SynthesisInput=lambda text: text,
        VoiceSelectionParams=lambda language_code, name: SimpleNamespace(name=name),
        AudioConfig=lambda **k: k, AudioEncoding=SimpleNamespace(LINEAR16="LINEAR16"))
    monkeypatch.setattr(tts, "texttospeech", fake)
    class Stop(Exception):
        pass

    def stop(seconds):
        raise Stop                                # ends google_speak's retry loop
    monkeypatch.setattr(tts.time, "sleep", stop)
    with patch.object(gemini_tts, "speak", side_effect=RuntimeError("Gemini down")):
        with pytest.raises(Stop):
            tts.google_speak("Opinion", "Hello there.", str(tmp_path / "x.wav"))
    assert asked["voice"] == "en-US-Chirp3-HD-Schedar"


def test_the_fact_check_sees_the_relevant_evidence_when_the_sources_run_long():
    # 2026-10-23 rehearsal: the opinion evidence was 188,000 characters and the faithfulness pass
    # kept only the first 160,000, so it flagged a CalMatters figure that was in the cut-off part.
    filler = "\n\n".join(f"SOURCE: Page {i} — https://example.com/{i}\n" + "Weather and sports notes. " * 140
                         for i in range(60))
    corpus = filler + "\n\nSOURCE: CalMatters — https://calmatters.org/x\nA 7% second mortgage could add $1,000 a month."
    assert len(corpus) > 160000
    seen = {}
    with patch.object(review, "_ask_for_flags", side_effect=lambda label, prompt, *a: seen.setdefault("p", prompt) and []):
        review.faithfulness_flags("CalMatters says a second mortgage could add a thousand dollars a month.", corpus)
    assert "could add $1,000 a month" in seen["p"] and len(seen["p"]) < 170000


def test_a_fix_that_would_leave_a_sentence_fragment_is_skipped():
    script = "The caps are highest where permits are slowest. Prices rose."
    reply = json.dumps({"verdicts": [{"flag": 1, "verdict": "fix", "find": "where permits are", "replace": "in big counties"}]})
    with patch.object(opinion, "faithfulness_flags", return_value=["FLAG: permits — unsupported"]), \
         patch.object(opinion, "get_llm_response", return_value=reply):
        out, audit = opinion.self_check(script, "evidence")
    assert out == script and len(audit["skipped"]) == 1
