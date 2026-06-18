"""Pipeline integration points for adaptive selected-story research."""
from unittest.mock import patch

import newscaster.config as cfg
import newscaster.pipeline as pipeline
from newscaster.research_agent import AdaptiveResearchResult
from newscaster.source_hunter import SourceHunterResult


def _fake_result_piper(summary_prompt, counter, topic, result, slot, date, articles=None):
    if articles is not None:
        articles.append({
            "chunk_id": f"{date}_seg{slot}_art{counter}",
            "url": result["url"],
            "outlet": "NPR",
            "original_headline": result["headline"],
            "summary": "article summary",
        })
    return summary_prompt + "\narticle summary", counter + 1


def test_gather_one_topic_uses_adaptive_research(monkeypatch):
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_ENABLED", True)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    articles = []
    followups = []
    captured = {}

    def fake_adaptive(topic, topic_index, formatted_date, formatted_date2,
                      summary_prompt, counter, articles=None, followups=None):
        followups.append({
            "asker": "Research Agent",
            "question": "q",
            "answer": "a",
            "challenging": False,
            "iteration": 1,
        })
        captured["adaptive_prompt"] = summary_prompt
        return AdaptiveResearchResult(
            summary_prompt="adapted prompt",
            successful_summary_counter=counter,
            articles=list(articles),
            followups=list(followups),
            done_reason="done",
        )

    def fake_llm(prompt, system_prompt="You are an intelligent assistant.",
                 mode="light", grounding=False, url_context=False):
        if grounding:
            return "grounded context"
        captured["synthesis_prompt"] = prompt
        return "super summary"

    with patch("newscaster.pipeline.time.sleep"), \
         patch("newscaster.pipeline.search_web", return_value=[
             {"headline": "Dam", "url": "https://npr.org/dam", "snippet": "s"}
         ]), \
         patch("newscaster.pipeline.result_piper", side_effect=_fake_result_piper), \
         patch("newscaster.pipeline.answer_with_source_hunter", return_value=SourceHunterResult(
             answer="controlled seed context",
             sources=[{"url": "https://npr.org/dam"}],
             status="success",
         )), \
         patch("newscaster.pipeline.run_adaptive_research", side_effect=fake_adaptive) as mock_adaptive, \
         patch("newscaster.pipeline.get_llm_response", side_effect=fake_llm), \
         patch("newscaster.pipeline.call_with_default", return_value="no"):
        out = pipeline._gather_one_topic(
            "dam", 0, "March 9, 2026", "2026_03_09",
            "follow up", "challenging", articles=articles, followups=followups,
        )

    assert out == "super summary"
    mock_adaptive.assert_called_once()
    assert "controlled seed context" in captured["adaptive_prompt"]
    assert captured["synthesis_prompt"] == "adapted prompt"
    assert followups[0]["asker"] == "Research Agent"


def test_gather_one_topic_falls_back_to_fixed_rounds(monkeypatch):
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_ENABLED", True)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    captured = {}

    def fake_llm(prompt, system_prompt="You are an intelligent assistant.",
                 mode="light", grounding=False, url_context=False):
        if grounding:
            return "grounded context"
        captured["synthesis_prompt"] = prompt
        return "super summary"

    with patch("newscaster.pipeline.time.sleep"), \
         patch("newscaster.pipeline.search_web", return_value=[
             {"headline": "Dam", "url": "https://npr.org/dam", "snippet": "s"}
         ]), \
         patch("newscaster.pipeline.result_piper", side_effect=_fake_result_piper), \
         patch("newscaster.pipeline.answer_with_source_hunter", return_value=SourceHunterResult(
             answer="controlled seed context",
             sources=[{"url": "https://npr.org/dam"}],
             status="success",
         )), \
         patch("newscaster.pipeline.run_adaptive_research", side_effect=RuntimeError("graph boom")), \
         patch("newscaster.pipeline._run_follow_up_rounds", return_value="fixed prompt") as mock_fixed, \
         patch("newscaster.pipeline.get_llm_response", side_effect=fake_llm), \
         patch("newscaster.pipeline.call_with_default", return_value="no"):
        out = pipeline._gather_one_topic(
            "dam", 0, "March 9, 2026", "2026_03_09",
            "follow up", "challenging", articles=[], followups=[],
        )

    assert out == "super summary"
    mock_fixed.assert_called_once()
    assert captured["synthesis_prompt"] == "fixed prompt"


def test_gather_one_topic_skips_agent_when_source_hunter_disabled(monkeypatch):
    # A5: the research agent's only research tool is the source hunter. With the hunter
    # disabled the agent must NOT run (it would otherwise collapse to early termination);
    # the pipeline falls back to the fixed follow-up rounds instead.
    monkeypatch.setattr(cfg, "AGENTIC_RESEARCH_ENABLED", True)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", False)

    def fake_llm(prompt, system_prompt="You are an intelligent assistant.",
                 mode="light", grounding=False, url_context=False):
        return "super summary"

    with patch("newscaster.pipeline.time.sleep"), \
         patch("newscaster.pipeline.search_web", return_value=[
             {"headline": "Dam", "url": "https://npr.org/dam", "snippet": "s"}
         ]), \
         patch("newscaster.pipeline.result_piper", side_effect=_fake_result_piper), \
         patch("newscaster.pipeline.answer_with_source_hunter", return_value=SourceHunterResult(
             answer="No evidence", status="no_evidence",
         )), \
         patch("newscaster.pipeline.run_adaptive_research") as mock_adaptive, \
         patch("newscaster.pipeline._run_follow_up_rounds", return_value="fixed prompt") as mock_fixed, \
         patch("newscaster.pipeline.get_llm_response", side_effect=fake_llm), \
         patch("newscaster.pipeline.call_with_default", return_value="no"):
        out = pipeline._gather_one_topic(
            "dam", 0, "March 9, 2026", "2026_03_09",
            "follow up", "challenging", articles=[], followups=[],
        )

    assert out == "super summary"
    mock_adaptive.assert_not_called()
    mock_fixed.assert_called_once()
