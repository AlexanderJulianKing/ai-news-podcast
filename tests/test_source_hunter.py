from unittest.mock import patch

import newscaster.config as cfg
from newscaster.source_hunter import SourceHunterResult, answer_with_source_hunter, answer_with_escalation
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


def test_source_hunter_logs_uncapturable_urls(monkeypatch):
    # A URL that fails to fetch (a rejected source carrying an `error`) is logged to the dedicated
    # fetch-failures jsonl for coverage tracking; a validation rejection (fetched fine, just didn't
    # support the claim) is not.
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_MAX_ITERATIONS", 1)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_NEARBY_SOURCE_DEPTH", 0)
    rejected = [
        {"url": "https://paywalled.example/article", "error": "HTTP 403"},               # fetch failed
        {"url": "https://offtopic.example/page", "validation": {"reason": "off-topic"}},  # validation rejection
    ]
    with patch("newscaster.source_hunter.search_web", return_value=[
        {"headline": "h", "url": "https://x.example", "snippet": "s"}
    ]), \
         patch("newscaster.source_hunter.fetch_discovered_evidence", return_value={"sources": [{"ok": False}]}), \
         patch("newscaster.source_hunter.filter_validated_evidence", return_value={
             "sources": [], "rejected_sources": rejected,
         }), \
         patch("newscaster.source_hunter._generate_evidence_contract", return_value={}), \
         patch("newscaster.source_hunter.get_llm_response"), \
         patch("newscaster.source_hunter.write_jsonl_log") as mock_log:
        answer_with_source_hunter("did the strait close?", topic="strait", formatted_date="June 21, 2026")

    failures = [call.args[1] for call in mock_log.call_args_list
                if call.args and call.args[0] == "source_hunter_fetch_failures"]
    assert len(failures) == 1                                   # only the fetch failure, not the off-topic rejection
    assert failures[0]["url"] == "https://paywalled.example/article"
    assert failures[0]["error"] == "HTTP 403"


def test_tier2_brief_uses_web_search_not_source_hunter():
    # Tier-2 only ranks headlines by importance, so it uses one cheap web-grounded call
    # (Gemma 4 + OpenRouter web search), not the heavier fetch-validate source hunter.
    with patch("newscaster.scrapers.topic_finder.openrouter_web_brief",
               return_value="web brief") as mock_brief, \
         patch("newscaster.scrapers.topic_finder.answer_with_escalation") as mock_hunter:
        brief = topic_finder._research_headline_brief("headline", "June 18, 2026")

    assert brief == "web brief"
    mock_brief.assert_called_once()
    mock_hunter.assert_not_called()


def test_tier2_brief_marks_unverified_when_web_brief_fails():
    with patch("newscaster.scrapers.topic_finder.openrouter_web_brief",
               side_effect=RuntimeError("web down")):
        brief = topic_finder._research_headline_brief("headline", "June 18, 2026")

    assert brief.startswith("UNVERIFIED:")


def test_summarize_headline_does_not_rerun_apparatus_when_unverifiable(monkeypatch):
    """An unverifiable headline must not trigger a second full source-hunter pass.

    The helper already escalates standard -> advanced internally, so a no-evidence
    result should mark UNVERIFIED after exactly those two invocations rather than
    rerunning the entire apparatus on the near-identical retry prompt.
    """
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_ENABLED", True)
    with patch("newscaster.source_hunter.answer_with_source_hunter", return_value=SourceHunterResult(
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
    with patch("newscaster.source_hunter.answer_with_source_hunter", side_effect=[
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


def test_source_hunter_returns_partial_findings_as_success(monkeypatch):
    """A validated source that only partially answers now returns its FINDINGS + GAPS as
    success, so the caller (the research agent's Opus) can target the gap, not discard it."""
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_MAX_ITERATIONS", 1)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_NEARBY_SOURCE_DEPTH", 0)
    raw_evidence = {"sources": [{
        "ok": True, "url": "https://example.com/report", "final_url": "https://example.com/report",
        "title": "Report", "content_type": "text/html", "char_count": 100, "excerpt": "The Fed held rates.",
    }]}
    validated = {"sources": [raw_evidence["sources"][0] | {"validation": {"score": 7}}], "rejected_sources": []}

    partial = "FINDINGS: The Fed held rates steady.\nGAPS: The exact inflation figure is not stated."
    with patch("newscaster.source_hunter._generate_evidence_contract", return_value={}), \
         patch("newscaster.source_hunter.search_web", return_value=[
             {"headline": "Report", "url": "https://example.com/report", "snippet": "s"}
         ]), \
         patch("newscaster.source_hunter.fetch_discovered_evidence", return_value=raw_evidence), \
         patch("newscaster.source_hunter.filter_validated_evidence", return_value=validated), \
         patch("newscaster.source_hunter.get_llm_response", return_value=partial):
        result = answer_with_source_hunter("What rate and what inflation figure?", topic="fed", mode="standard")

    assert result.status == "success"
    assert "FINDINGS" in result.answer and "GAPS" in result.answer
    assert result.sources, "validated sources should be reported"


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


def test_answer_with_escalation_escalates_to_advanced_on_non_success():
    with patch("newscaster.source_hunter.answer_with_source_hunter", side_effect=[
        SourceHunterResult(answer="No evidence", status="no_evidence"),
        SourceHunterResult(answer="advanced answer", sources=[{"url": "https://example.com"}], status="success"),
    ]) as mock_hunter:
        result = answer_with_escalation("q", topic="t", formatted_date="June 18, 2026")

    assert result.status == "success"
    assert result.answer == "advanced answer"
    assert mock_hunter.call_count == 2
    assert [c.kwargs["mode"] for c in mock_hunter.call_args_list] == ["standard", "advanced"]


def test_answer_with_escalation_stops_at_standard_on_success():
    with patch("newscaster.source_hunter.answer_with_source_hunter", side_effect=[
        SourceHunterResult(answer="standard answer", status="success"),
    ]) as mock_hunter:
        result = answer_with_escalation("q", topic="t")

    assert result.answer == "standard answer"
    assert mock_hunter.call_count == 1


def test_evidence_contract_advisory_for_news_research_only(monkeypatch):
    """The contract is advisory for news_research, a hard gate for every other category.

    A source that clears the base date/entity/topic checks but trips a contract-derived
    rejection (here a reject_if rule) must PASS under news_research (contract advisory ->
    ranking only) and FAIL under any other category. This is the fix for breaking-news
    recall collapse: a confirmed UK-PM resignation was rejected on every source because the
    generated contract's source-preference and reject_if rules vetoed real coverage.
    """
    from newscaster import source_hunter_primitives as shp

    # Force a contract-derived rejection regardless of the contract's internal slot logic,
    # so the test pins the GATE behavior (advisory vs hard), not contract generation.
    monkeypatch.setattr(shp, "_contract_reject_reasons",
                        lambda *a, **k: ["contract_reject:forced_for_test"])

    question = "What layoffs did Acme Corporation announce on June 22, 2026?"
    source = {
        "ok": True,
        "url": "https://www.nbcnews.com/business/acme-layoffs",
        "title": "Acme Corporation announces layoffs",
        "excerpt": ("Acme Corporation announced on June 22, 2026 that it will lay off "
                    "employees. The Acme Corporation layoffs were confirmed in an official "
                    "company statement."),
    }
    base = {"id": "t", "question": question, "evidence_contract": {}}
    news = shp.validate_source_for_question({**base, "category": "news_research"}, source)
    other = shp.validate_source_for_question({**base, "category": "local_government"}, source)

    # The contract rejection is recorded both ways (forensics still see it)...
    assert "contract_reject:forced_for_test" in news["reasons"]
    assert "contract_reject:forced_for_test" in other["reasons"]
    # ...but it only vetoes outside news_research.
    assert news["passed"] is True, news["reasons"]
    assert other["passed"] is False, other["reasons"]
