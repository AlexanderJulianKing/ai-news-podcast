"""Tests for the LangGraph selected-story research agent."""
import json
from unittest.mock import patch

import pytest

import newscaster.config as cfg
import newscaster.research_agent as agent
from newscaster.rag.store import Hit
from newscaster.source_hunter import SourceHunterResult


@pytest.fixture(autouse=True)
def _disable_adversary_by_default(monkeypatch):
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_ADVERSARY_ENABLED", False)


def test_parse_controller_done():
    decision = agent.parse_controller_decision(
        '{"status":"done","reason":"enough","confidence":"high"}'
    )
    assert decision == {"status": "done", "reason": "enough", "confidence": "high"}


def test_parse_controller_grounded_search():
    decision = agent.parse_controller_decision(json.dumps({
        "status": "continue",
        "action": "grounded_search",
        "question": "What is the scale?",
        "question_type": "scale_check",
        "reason": "scale is missing",
    }))
    assert decision["action"] == "grounded_search"
    assert decision["question_type"] == "scale_check"
    assert decision["question"] == "What is the scale?"


def test_parse_controller_article_search():
    decision = agent.parse_controller_decision(json.dumps({
        "status": "continue",
        "action": "article_search",
        "query": "dam failure investigation local report",
        "reason": "need another source",
    }))
    assert decision["action"] == "article_search"
    assert decision["query"] == "dam failure investigation local report"


def test_parse_controller_repair_path():
    repaired = '{"status":"done","reason":"fixed","confidence":"medium"}'
    with patch("newscaster.research_agent.get_llm_response", return_value=repaired) as mock_llm:
        decision = agent.parse_controller_decision("not json", allow_repair=True)
    assert decision["status"] == "done"
    mock_llm.assert_called_once()


def test_memory_no_hits_returns_empty(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_RESEARCH_MEMORY_ENABLED", True)
    with patch("newscaster.research_agent.retrieve_prior_research", return_value=[]), \
         patch("newscaster.research_agent.get_llm_response") as mock_llm:
        note = agent.build_research_memory_note("topic", "March 9, 2026", "2026_03_09", "seed")
    assert note == ""
    mock_llm.assert_not_called()


def test_memory_hits_build_note(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_RESEARCH_MEMORY_ENABLED", True)
    hit = Hit(
        chunk_id="c0", date="2026_03_01", chunk_type="article",
        outlet="NPR", headline="Old dam story", url="https://npr.org/dam",
        text="prior dam context", similarity=0.91,
    )
    with patch("newscaster.research_agent.retrieve_prior_research", return_value=[hit]) as mock_ret, \
         patch("newscaster.research_agent.get_llm_response", return_value="memory note") as mock_llm:
        note = agent.build_research_memory_note("dam", "March 9, 2026", "2026_03_09", "seed")
    assert note == "memory note"
    assert "dam" in mock_ret.call_args[0][0]
    assert "prior dam context" in mock_llm.call_args[0][0]
    assert mock_llm.call_args.kwargs["mode"] == "heavy"


def test_memory_retrieval_failure_is_nonfatal(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_RESEARCH_MEMORY_ENABLED", True)
    with patch("newscaster.research_agent.retrieve_prior_research", side_effect=RuntimeError("boom")):
        assert agent.build_research_memory_note("dam", "March 9, 2026", "2026_03_09", "seed") == ""


def test_adaptive_loop_min_iterations_forces_one_grounded_search(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MIN_ITERATIONS", 2)
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MAX_ITERATIONS", 5)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.research_agent.build_research_memory_note", return_value=""), \
         patch("newscaster.source_hunter.answer_with_source_hunter", return_value=SourceHunterResult(
             answer="controlled answer",
             sources=[{"url": "https://example.com"}],
             status="success",
         )), \
         patch("newscaster.research_agent.get_llm_response") as mock_llm:
        controller_modes = []
        def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
            controller_modes.append(mode)
            return '{"status":"done","reason":"enough","confidence":"high"}'
        mock_llm.side_effect = fake_llm
        followups = []
        result = agent.run_adaptive_research(
            "dam", 0, "March 9, 2026", "2026_03_09", "seed", 1,
            articles=[], followups=followups,
        )
    assert len(result.followups) == 1
    assert result.followups[0]["question_type"] == "freshness_check"
    assert result.followups[0]["action"] == "source_hunter"
    assert "controlled answer" in result.summary_prompt
    assert controller_modes == ["heavy", "heavy"]


def test_adversary_runs_after_controller_says_done_then_returns_to_controller(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_ADVERSARY_ENABLED", True)
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MIN_ITERATIONS", 0)
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MAX_ITERATIONS", 5)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)

    call_order = []

    def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
        call_order.append(mode)
        if mode == "adversary":
            return json.dumps({
                "question": "What evidence would challenge the central claim?",
                "question_type": "counterevidence_check",
                "reason": "test the premise",
            })
        assert mode == "heavy"
        if len(call_order) == 1:
            assert "adversarial answer" not in prompt
        else:
            assert "SECOND-PERSPECTIVE ADVERSARIAL QUESTION" in prompt
            assert "adversarial answer" in prompt
        return '{"status":"done","reason":"enough","confidence":"high"}'

    with patch("newscaster.research_agent.build_research_memory_note", return_value=""), \
         patch("newscaster.source_hunter.answer_with_source_hunter", return_value=SourceHunterResult(
             answer="adversarial answer",
             sources=[{"url": "https://example.com/adversary"}],
             status="success",
         )) as mock_source_hunter, \
         patch("newscaster.research_agent.get_llm_response", side_effect=fake_llm):
        result = agent.run_adaptive_research(
            "dam", 0, "March 9, 2026", "2026_03_09", "seed", 1,
            articles=[], followups=[],
        )

    mock_source_hunter.assert_called_once()
    assert call_order == ["heavy", "adversary", "heavy"]
    assert len(result.followups) == 1
    assert result.followups[0]["asker"] == "GPT-5.5 Adversary"
    assert result.followups[0]["adversary_guided"] is True
    assert result.followups[0]["question_type"] == "counterevidence_check"
    assert "adversarial answer" in result.summary_prompt


def test_adaptive_loop_hard_stops_at_max_iterations(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MIN_ITERATIONS", 0)
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MAX_ITERATIONS", 5)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.research_agent.build_research_memory_note", return_value=""), \
         patch("newscaster.source_hunter.answer_with_source_hunter", return_value=SourceHunterResult(
             answer="answer",
             sources=[{"url": "https://example.com"}],
             status="success",
         )), \
         patch("newscaster.research_agent.get_llm_response") as mock_llm:
        def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
            return json.dumps({
                "status": "continue",
                "action": "grounded_search",
                "question": "next?",
                "question_type": "source_check",
                "reason": "keep checking",
            })
        mock_llm.side_effect = fake_llm
        result = agent.run_adaptive_research(
            "dam", 0, "March 9, 2026", "2026_03_09", "seed", 1,
            articles=[], followups=[],
        )
    assert len(result.followups) == 5
    assert all(f["iteration"] in {1, 2, 3, 4, 5} for f in result.followups)


def test_adaptive_article_search_calls_result_piper(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MIN_ITERATIONS", 0)
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MAX_ITERATIONS", 5)
    decisions = iter([
        json.dumps({
            "status": "continue", "action": "article_search",
            "query": "dam investigation", "reason": "need source",
        }),
        '{"status":"done","reason":"enough","confidence":"medium"}',
    ])

    def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
        return next(decisions)

    def fake_piper(summary_prompt, counter, topic, result, slot, date, articles=None):
        articles.append({
            "chunk_id": "2026_03_09_seg0_art1",
            "url": result["url"],
            "outlet": "NPR",
            "original_headline": result["headline"],
            "summary": "article summary",
        })
        return summary_prompt + "\narticle summary", counter + 1

    with patch("newscaster.research_agent.build_research_memory_note", return_value=""), \
         patch("newscaster.research_agent.get_llm_response", side_effect=fake_llm), \
         patch("newscaster.research_agent.search_web", return_value=[
             {"headline": "Dam", "url": "https://npr.org/dam", "snippet": "s"}
         ]) as mock_search, \
         patch("newscaster.research_agent.result_piper", side_effect=fake_piper) as mock_piper:
        result = agent.run_adaptive_research(
            "dam", 0, "March 9, 2026", "2026_03_09", "seed", 1,
            articles=[], followups=[],
        )
    mock_search.assert_called_once_with("dam investigation", 9)
    mock_piper.assert_called_once()
    assert len(result.articles) == 1
    assert "article summary" in result.summary_prompt


def test_adaptive_repeated_tool_failures_stop(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MIN_ITERATIONS", 0)
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_MAX_ITERATIONS", 5)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)

    def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
        return json.dumps({
            "status": "continue",
            "action": "grounded_search",
            "question": "verify?",
            "question_type": "source_check",
            "reason": "need verification",
        })

    with patch("newscaster.research_agent.build_research_memory_note", return_value=""), \
         patch("newscaster.source_hunter.answer_with_source_hunter", side_effect=RuntimeError("source hunter down")), \
         patch("newscaster.research_agent.get_llm_response", side_effect=fake_llm):
        result = agent.run_adaptive_research(
            "dam", 0, "March 9, 2026", "2026_03_09", "seed", 1,
            articles=[], followups=[],
        )
    assert result.followups == []
    assert result.done_reason.startswith("grounded search failed")


def test_grounded_search_records_no_evidence_instead_of_failing(monkeypatch):
    # Part B: a partial / no-evidence source-hunter result must NOT end the loop as a
    # failure. Its findings + gaps are recorded so the controller (Opus) can target the
    # gap next turn or move on. Only a thrown exception is a real failure.
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    state = {
        "last_decision": {"question": "What rate?", "question_type": "source_check"},
        "topic": "fed", "formatted_date": "June 18, 2026",
        "iterations": 1, "followups": [], "summary_prompt": "seed",
    }
    with patch("newscaster.source_hunter.answer_with_source_hunter", return_value=SourceHunterResult(
        answer="No accepted source evidence was found for this question.",
        sources=[], status="no_evidence",
    )):
        out = agent._grounded_search_node(state)

    assert out.get("consecutive_failures") == 0       # no-evidence is not a failure
    assert "done_reason" not in out                    # the loop is not terminated
    assert len(out["followups"]) == 1
    assert out["followups"][0]["source_hunter_status"] == "no_evidence"
