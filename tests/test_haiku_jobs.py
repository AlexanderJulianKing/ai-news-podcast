"""Claude Haiku 5.5 takes article summaries and the background-fact check (2026-10-10).

Both judges (Opus subagents and Codex) preferred Haiku on these two jobs; see
benchmarks/haiku_vs_luna/. Luna answers if Haiku fails.
"""
from newscaster import config as cfg
from newscaster import review
from newscaster.llm import router
from newscaster.scrapers import topic_finder


def test_haiku_mode_is_haiku_low_with_luna_as_backup():
    spec = router._select_primary("haiku", False, False)
    assert spec["provider"] == "anthropic" and spec["model"] == "claude-haiku-5-5" and spec["effort"] == "low"
    assert spec["fallback"]["model"] == cfg.STANDARD_MODEL


def test_a_haiku_failure_is_answered_by_luna(monkeypatch):
    from newscaster.llm.errors import LLMRetriesExhaustedError
    used = []

    def fake_retry(spec, user, system, **kw):
        used.append(spec["model"])
        if spec["provider"] == "anthropic":
            raise LLMRetriesExhaustedError("haiku down", provider="anthropic", model=spec["model"])
        return "luna answer"
    monkeypatch.setattr(router, "_call_with_retry", fake_retry)
    assert router.get_llm_response("p", "s", mode="haiku") == "luna answer"
    assert used == ["claude-haiku-5-5", cfg.STANDARD_MODEL]


def test_summaries_and_fact_check_ask_for_haiku(monkeypatch):
    modes = []
    monkeypatch.setattr(topic_finder, "get_llm_response", lambda *a, **k: modes.append(k.get("mode")) or "summary")
    topic_finder.summarize_text("Some article text.", "article")
    monkeypatch.setattr(review, "get_llm_response", lambda *a, **k: modes.append(k.get("mode")) or "NONE")
    review.stable_fact_flags("Grace: The Eiffel Tower is in Paris.")
    assert modes == ["haiku", "haiku"]
