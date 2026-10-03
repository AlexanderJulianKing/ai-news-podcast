"""Cost settings from 2026-10-02: scores-only triage, and GPT-6.1 Sol for every Opus job except the scripts."""
from unittest.mock import MagicMock

from newscaster import config as cfg
from newscaster import prompts
from newscaster.llm import claude as claude_mod
from newscaster.llm import router
from newscaster.scrapers.topic_finder import _parse_tier1_scores


def test_triage_asks_for_scores_without_justifications():
    for prompt in (prompts.TIER1_TRIAGE_PROMPT, prompts.TIER1_CALIFORNIA_TRIAGE_PROMPT):
        assert "one-sentence justification" not in prompt
        assert "single dash" in prompt
    parsed = _parse_tier1_scores("SCORE: 8 | HEADLINE: Fed cuts rates | REASON: -")
    assert parsed == [{"score": 8, "headline": "Fed cuts rates", "reason": "-"}]


def test_editorial_mode_is_sol_high_with_opus_as_backup():
    spec = router._select_primary("editorial", False, False)
    assert spec["provider"] == "openrouter" and spec["model"] == "openai/gpt-6.1-sol" and spec["reasoning"] == "high"
    assert spec["fallback"]["provider"] == "anthropic" and spec["fallback"]["model"] == cfg.HEAVY_MODEL
    assert router._select_primary("heavy", False, False)["provider"] == "anthropic"   # segment scripts stay on Opus


def test_editorial_failure_falls_back_to_opus_not_sol(monkeypatch):
    from newscaster.llm.errors import LLMRetriesExhaustedError
    used = []

    def fake_retry(spec, user, system, **kw):
        used.append(spec["provider"])
        if spec["provider"] == "openrouter":
            raise LLMRetriesExhaustedError("sol down", provider="openrouter", model=spec["model"])
        return "opus answer"
    monkeypatch.setattr(router, "_call_with_retry", fake_retry)
    assert router.get_llm_response("p", "s", mode="editorial") == "opus answer"
    assert used == ["openrouter", "anthropic"]


def test_only_the_segment_scripts_still_call_opus():
    import pathlib, re
    heavy = [str(f) for f in pathlib.Path("newscaster").rglob("*.py")
             if re.search(r"mode=['\"]heavy['\"]", f.read_text())]
    assert heavy == [str(pathlib.Path("newscaster/script/segments.py"))]


def test_claude_sends_the_requested_effort(monkeypatch):
    sent = {}

    def fake_create(client, **kwargs):
        sent.update(kwargs)
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text="ok")]
        msg.usage = MagicMock(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return msg
    monkeypatch.setattr(claude_mod, "_create_message", fake_create)
    router._dispatch({"provider": "anthropic", "model": "claude-opus-5-5", "effort": "medium"}, "p", "s")
    assert sent["extra_body"] == {"output_config": {"effort": "medium"}}
    claude_mod.claude("p", "claude-opus-5-5", "s")
    assert sent["extra_body"] == {"output_config": {"effort": "high"}}
