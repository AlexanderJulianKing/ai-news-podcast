from unittest.mock import patch

import newscaster.config as cfg
from newscaster.source_hunter import SourceHunterResult, answer_with_source_hunter
from newscaster.scrapers import topic_finder


def test_source_hunter_success_uses_controlled_evidence(monkeypatch):
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_MAX_ITERATIONS", 1)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_CANDIDATE_LIMIT", 3)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_NEARBY_SOURCE_DEPTH", 0)
    raw_evidence = {
        "sources": [{
            "ok": True,
            "url": "https://example.com/report",
            "final_url": "https://example.com/report",
            "title": "Official report",
            "content_type": "text/html",
            "char_count": 1200,
            "excerpt": "The official report says the program starts June 1.",
        }]
    }
    validated = {"sources": [raw_evidence["sources"][0] | {"validation": {"score": 8}}], "rejected_sources": []}

    with patch("newscaster.source_hunter.search_web", return_value=[
        {"headline": "Official report", "url": "https://example.com/report", "snippet": "s"}
    ]), \
         patch("newscaster.source_hunter.fetch_discovered_evidence", return_value=raw_evidence) as mock_fetch, \
         patch("newscaster.source_hunter.filter_validated_evidence", return_value=validated), \
         patch("newscaster.source_hunter.get_llm_response", return_value="controlled answer") as mock_llm:
        result = answer_with_source_hunter("When does the program start?", topic="program", mode="standard")

    assert result.status == "success"
    assert result.answer == "controlled answer"
    assert result.sources[0]["url"] == "https://example.com/report"
    mock_fetch.assert_called_once()
    assert mock_llm.call_args.kwargs["grounding"] is False


def test_source_hunter_no_evidence_does_not_synthesize(monkeypatch):
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_MAX_ITERATIONS", 1)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_NEARBY_SOURCE_DEPTH", 0)
    with patch("newscaster.source_hunter.search_web", return_value=[
        {"headline": "Wrong page", "url": "https://example.com/webinar", "snippet": "s"}
    ]), \
         patch("newscaster.source_hunter.fetch_discovered_evidence", return_value={"sources": [{"ok": True}]}), \
         patch("newscaster.source_hunter.filter_validated_evidence", return_value={
             "sources": [],
             "rejected_sources": [{"url": "https://example.com/webinar", "validation": {"reason": "wrong page"}}],
         }), \
         patch("newscaster.source_hunter._generate_evidence_contract", return_value={}), \
         patch("newscaster.source_hunter.get_llm_response") as mock_llm:
        result = answer_with_source_hunter("What is the count?", topic="count")

    assert result.status == "no_evidence"
    assert result.sources == []
    assert result.rejected_sources[0]["url"] == "https://example.com/webinar"
    mock_llm.assert_not_called()


def test_tier2_brief_uses_source_hunter(monkeypatch):
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.scrapers.topic_finder.answer_with_source_hunter", return_value=SourceHunterResult(
        answer="source hunter brief",
        sources=[{"url": "https://example.com/story"}],
        status="success",
    )) as mock_hunter, \
         patch("newscaster.scrapers.topic_finder.get_llm_response") as mock_llm:
        brief = topic_finder._research_headline_brief("headline", "June 18, 2026")

    assert brief == "source hunter brief"
    mock_hunter.assert_called_once()
    mock_llm.assert_not_called()


def test_tier2_brief_uses_advanced_when_standard_has_no_evidence(monkeypatch):
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.scrapers.topic_finder.answer_with_source_hunter", side_effect=[
        SourceHunterResult(answer="No evidence", status="no_evidence"),
        SourceHunterResult(answer="advanced brief", sources=[{"url": "https://example.com"}], status="success"),
    ]) as mock_hunter, \
         patch("newscaster.scrapers.topic_finder.get_llm_response") as mock_llm:
        brief = topic_finder._research_headline_brief("headline", "June 18, 2026")

    assert brief == "advanced brief"
    assert mock_hunter.call_count == 2
    assert mock_hunter.call_args_list[0].kwargs["mode"] == "standard"
    assert mock_hunter.call_args_list[1].kwargs["mode"] == "advanced"
    mock_llm.assert_not_called()


def test_tier2_brief_marks_unverified_when_source_hunter_fails(monkeypatch):
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.scrapers.topic_finder.answer_with_source_hunter", return_value=SourceHunterResult(
        answer="No evidence",
        status="no_evidence",
    )), \
         patch("newscaster.scrapers.topic_finder.get_llm_response") as mock_llm:
        brief = topic_finder._research_headline_brief("headline", "June 18, 2026")

    assert brief.startswith("UNVERIFIED:")
    mock_llm.assert_not_called()


def test_summarize_headline_does_not_rerun_apparatus_when_unverifiable(monkeypatch):
    """An unverifiable headline must not trigger a second full source-hunter pass.

    The helper already escalates standard -> advanced internally, so a no-evidence
    result should mark UNVERIFIED after exactly those two invocations rather than
    rerunning the entire apparatus on the near-identical retry prompt.
    """
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.scrapers.topic_finder.answer_with_source_hunter", return_value=SourceHunterResult(
        answer="No evidence",
        status="no_evidence",
    )) as mock_hunter, \
         patch("newscaster.scrapers.topic_finder.get_llm_response") as mock_llm:
        result = topic_finder.summarize_headline_with_grounding("Unverifiable headline")

    assert result.startswith("UNVERIFIED:")
    assert mock_hunter.call_count == 2
    assert mock_hunter.call_args_list[0].kwargs["mode"] == "standard"
    assert mock_hunter.call_args_list[1].kwargs["mode"] == "advanced"
    mock_llm.assert_not_called()


def test_summarize_headline_retries_when_answer_denies_story(monkeypatch):
    """A successful-but-denying answer should still trigger the retry prompt.

    The retry path exists to push past false-negative denials, so the
    no-evidence short-circuit must not collapse it when the source hunter
    actually returned evidence.
    """
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.scrapers.topic_finder.answer_with_source_hunter", side_effect=[
        SourceHunterResult(
            answer="There is no indication that this happened.",
            sources=[{"url": "https://example.com/a"}],
            status="success",
        ),
        SourceHunterResult(
            answer="Reuters confirms the program launched today. Sources: Reuters - wire report.",
            sources=[{"url": "https://example.com/b"}],
            status="success",
        ),
    ]) as mock_hunter:
        result = topic_finder.summarize_headline_with_grounding("A real story")

    assert result.startswith("Reuters confirms")
    assert mock_hunter.call_count == 2


def test_source_hunter_downgrades_non_answer_to_no_evidence(monkeypatch):
    """A validated source whose synthesis can't actually answer must not return 'success'."""
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_MAX_ITERATIONS", 1)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_NEARBY_SOURCE_DEPTH", 0)
    raw_evidence = {"sources": [{
        "ok": True, "url": "https://example.com/report", "final_url": "https://example.com/report",
        "title": "Report", "content_type": "text/html", "char_count": 100, "excerpt": "unrelated text",
    }]}
    validated = {"sources": [raw_evidence["sources"][0] | {"validation": {"score": 5}}], "rejected_sources": []}

    with patch("newscaster.source_hunter._generate_evidence_contract", return_value={}), \
         patch("newscaster.source_hunter.search_web", return_value=[
             {"headline": "Report", "url": "https://example.com/report", "snippet": "s"}
         ]), \
         patch("newscaster.source_hunter.fetch_discovered_evidence", return_value=raw_evidence), \
         patch("newscaster.source_hunter.filter_validated_evidence", return_value=validated), \
         patch("newscaster.source_hunter.get_llm_response", return_value="NO_ANSWER"):
        result = answer_with_source_hunter("What is the count?", topic="count", mode="standard")

    assert result.status == "no_evidence"
    assert result.sources, "validated sources should still be reported on a non-answer"
    assert result.metadata.get("synthesis_non_answer") is True


def test_query_variants_leads_with_question():
    # A pointed question (even a longish ~22-word one) must drive the search, not the broad
    # topic — and the topic must still be searched within the iteration cap.
    from newscaster.source_hunter import _query_variants
    q = ("What did the Federal Reserve decide at its June 2026 FOMC meeting regarding "
         "interest rates, and what is the current target range?")
    variants = _query_variants(q, "Recent U.S. economic news and the Fed outlook", "June 18, 2026")
    assert variants[0] == q
    assert "Recent U.S. economic news and the Fed outlook" in variants[:3]


def test_query_variants_leads_with_topic_for_long_research_prompt():
    # A long research *prompt* (Tier-2 style, ~98 words) is not query-like, so the clean
    # topic leads.
    from newscaster.source_hunter import _query_variants
    long_prompt = "Research this headline thoroughly and report the facts. " + " ".join(["instruction"] * 50)
    variants = _query_variants(long_prompt, "Riverside city council water rate vote", "June 18, 2026")
    assert variants[0] == "Riverside city council water rate vote"
