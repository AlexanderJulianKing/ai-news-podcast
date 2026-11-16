"""Cost settings from 2026-10-02: scores-only triage and medium-effort Opus for routine calls."""
from unittest.mock import MagicMock

from newscaster import config as cfg
from newscaster import prompts
from newscaster.llm import claude as claude_mod
from newscaster.llm import router


def test_triage_asks_for_numbers_and_scores_only():
    for prompt in (prompts.TIER1_TRIAGE_PROMPT, prompts.TIER1_CALIFORNIA_TRIAGE_PROMPT):
        assert "one-sentence justification" not in prompt
        assert "N: X" in prompt and "Do not repeat the headline" in prompt


def test_routine_mode_is_opus_at_medium_effort():
    spec = router._select_primary("routine", False, False)
    assert spec["provider"] == "anthropic" and spec["model"] == cfg.HEAVY_MODEL
    assert spec["effort"] == "medium"
    assert "effort" not in router._select_primary("heavy", False, False)   # scripts and picks stay high


def test_claude_sends_the_requested_effort(monkeypatch):
    sent = {}

    def fake_create(client, **kwargs):
        sent.update(kwargs)
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text="ok")]
        msg.usage = MagicMock(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return msg
    monkeypatch.setattr(claude_mod, "_create_message", fake_create)
    router._dispatch(router._select_primary("routine", False, False), "p", "s")
    assert sent["extra_body"] == {"output_config": {"effort": "medium"}}
    claude_mod.claude("p", "claude-opus-5-5", "s")
    assert sent["extra_body"] == {"output_config": {"effort": "high"}}
