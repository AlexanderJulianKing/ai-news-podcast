"""Prompt caching for the research controller (2026-10-02).

The controller re-sends the same story evidence on each of its 8 rounds. Caching
only works if that part of the request is byte-identical from round to round and
comes first, and only Claude may see the cache marker.
"""
from unittest.mock import MagicMock, patch

from newscaster import research_agent as ra
from newscaster.llm import claude as claude_mod
from newscaster.llm import router
from newscaster.llm.claude import CACHE_BREAK


def _state(summary_prompt, iterations, followups):
    return {"topic": "Sheriff seized ballots", "formatted_date": "October 2, 2026",
            "summary_prompt": summary_prompt, "base_evidence": "ARTICLE 1 text. SEED findings.",
            "iterations": iterations, "max_iterations": 8, "min_iterations": 2,
            "articles": [], "followups": followups, "memory_note": "past coverage note",
            "adversary_decision": {}}


def test_controller_payload_keeps_a_stable_prefix_as_evidence_grows():
    first = ra._controller_payload(_state("ARTICLE 1 text. SEED findings.", 0, []))
    later = ra._controller_payload(_state("ARTICLE 1 text. SEED findings.\nQ: bills? A: AB 282 ...", 3,
                                          [{"iteration": 1, "question": "bills?", "answer": "FINDINGS: AB 282"}]))
    assert CACHE_BREAK in first and CACHE_BREAK in later
    stable_first, stable_later = first.split(CACHE_BREAK)[0], later.split(CACHE_BREAK)[0]
    assert stable_first == stable_later
    assert "ARTICLE 1 text" in stable_first and "past coverage note" in stable_first
    assert "AB 282" in later.split(CACHE_BREAK)[1]          # new evidence still reaches the controller
    assert "ITERATIONS_COMPLETED: 3" in later.split(CACHE_BREAK)[1]


def test_claude_caches_the_text_before_the_marker_for_an_hour(monkeypatch):
    sent = {}

    def fake_create(client, **kwargs):
        sent.update(kwargs)
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text="ok")]
        msg.usage = MagicMock(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return msg
    monkeypatch.setattr(claude_mod, "_create_message", fake_create)
    claude_mod.claude(f"STABLE PART{CACHE_BREAK}CHANGING PART", "claude-opus-5-5", "system")
    blocks = sent["messages"][0]["content"]
    assert blocks[0]["text"] == "STABLE PART" and blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[1]["text"] == "CHANGING PART" and "cache_control" not in blocks[1]


def test_other_models_never_see_the_marker():
    seen = {}
    with patch.object(router, "get_openrouter_response", side_effect=lambda *a, **k: seen.setdefault("prompt", a[0] if a else k.get("user_prompt")) or "ok"):
        router._dispatch({"provider": "openrouter", "model": "openai/gpt-6-sol", "name": "Sol"},
                         f"STABLE{CACHE_BREAK}CHANGING", "system")
    assert CACHE_BREAK not in seen["prompt"] and "STABLE" in seen["prompt"] and "CHANGING" in seen["prompt"]
