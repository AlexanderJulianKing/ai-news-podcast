"""The super-summary call must not pass a collapsed LLM response downstream.

Regression cover for 2026-08-24, when the standard model emitted the word "our"
for 16,384 tokens, that text became the story summary, and the script writer's
"I don't see a news summary in your message" reply aired as the lead story.
"""
import pytest

from newscaster.pipeline import _degenerate_reason, _super_summary_with_retry
from newscaster.llm import LLMError

GOOD = " ".join(
    "The Department of Homeland Security said Border Patrol agents arrested "
    "Luis Aviles in Key West on Saturday while his son served aboard the USS "
    "Abraham Lincoln, according to reporting from PBS NewsHour and the "
    "Associated Press." for _ in range(6)
) + " " + " ".join(f"Additional distinct detail number {i} follows here." for i in range(40))


def test_the_real_failure_is_caught():
    assert _degenerate_reason("our " * 16384) is not None


def test_empty_and_whitespace_are_caught():
    assert _degenerate_reason("") == "empty response"
    assert _degenerate_reason("   \n  ") == "empty response"
    assert _degenerate_reason(None) == "empty response"


def test_short_meta_stub_is_caught():
    stub = ("I'm ready to help, but I don't see a news summary in your message. "
            "Please send the actual story summary you want rewritten.")
    assert "too short" in _degenerate_reason(stub)


def test_consecutive_repetition_is_caught():
    text = " ".join(["alpha"] * 40) + " " + GOOD
    assert _degenerate_reason(text) is not None


def test_realistic_summary_passes():
    assert _degenerate_reason(GOOD) is None


def test_retry_escalates_past_a_collapsed_model(monkeypatch):
    calls = []

    def fake(prompt, system_prompt=None, mode=None):
        calls.append(mode)
        return "our " * 4000 if mode == "standard" else GOOD

    monkeypatch.setattr("newscaster.pipeline.get_llm_response", fake)
    out = _super_summary_with_retry("prompt", "system", 0)
    assert out == GOOD
    assert calls == ["standard", "advanced"]


def test_first_attempt_is_kept_when_healthy(monkeypatch):
    calls = []

    def fake(prompt, system_prompt=None, mode=None):
        calls.append(mode)
        return GOOD

    monkeypatch.setattr("newscaster.pipeline.get_llm_response", fake)
    assert _super_summary_with_retry("prompt", "system", 0) == GOOD
    assert calls == ["standard"]


def test_raises_when_every_model_collapses(monkeypatch):
    monkeypatch.setattr("newscaster.pipeline.get_llm_response",
                        lambda *a, **k: "our " * 4000)
    with pytest.raises(LLMError):
        _super_summary_with_retry("prompt", "system", 1)
