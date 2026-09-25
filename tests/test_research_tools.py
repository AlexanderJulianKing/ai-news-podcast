"""The tool-using researcher: quote verification, no-evidence, fallback, and no rerun."""
import json
from unittest.mock import patch

import newscaster.config as cfg
from newscaster import research_tools as rt
from newscaster import source_hunter as sh

PAGE = {"url": "https://gov.ca.gov/bills", "title": "Governor signs election bills", "content_type": "text/html",
        "char_count": 120, "links": [],
        "text": "SACRAMENTO - Governor Newsom signed AB 282 by Assemblymember Gail Pellerin, which makes "
                "pre-certification seizures of ballots a felony."}


def _script(final_facts):
    """A fake model: search, open the page, then answer with the given facts."""
    replies = [
        {"tool_calls": [{"id": "1", "function": {"name": "web_search", "arguments": json.dumps({"query": "Newsom ballot bill"})}}]},
        {"tool_calls": [{"id": "2", "function": {"name": "open_page", "arguments": json.dumps({"url": PAGE["url"]})}}]},
        {"content": json.dumps({"answer": "AB 282.", "facts": final_facts, "gaps": ["effective date"]})},
    ]
    calls = iter(replies)
    return lambda payload: {"choices": [{"message": next(calls)}], "usage": {"cost": 0.001}}


def _run(facts):
    with patch.object(rt, "_chat", side_effect=_script(facts)), \
         patch.object(rt, "search_web", return_value=[{"headline": "h", "url": PAGE["url"], "snippet": "s"}]), \
         patch.object(rt, "fetch_source_text", return_value=PAGE):
        return rt.research_with_tools("Which ballot bills did Newsom sign?", "September 25, 2026")


def test_keeps_only_facts_whose_quote_is_on_the_page():
    out = _run([
        {"fact": "AB 282 is by Gail Pellerin", "quote": "AB 282 by Assemblymember Gail Pellerin", "url": PAGE["url"]},
        {"fact": "Invented", "quote": "takes effect January 1, 2027", "url": PAGE["url"]},
        {"fact": "From a page never opened", "quote": "anything", "url": "https://elsewhere.com"},
    ])
    assert out["status"] == "success"
    assert out["metadata"]["facts_kept"] == 1 and out["metadata"]["facts_dropped"] == 2
    assert "Gail Pellerin" in out["answer"] and "January 1, 2027" not in out["answer"]
    assert "GAPS:" in out["answer"] and "effective date" in out["answer"]
    assert "AB 282" in out["sources"][0]["excerpt"]      # the fact-checker reads real page text


def test_no_verified_fact_means_no_evidence():
    out = _run([{"fact": "Invented", "quote": "takes effect January 1, 2027", "url": PAGE["url"]}])
    assert out["status"] == "no_evidence" and out["sources"] == []


def test_quote_matching_tolerates_punctuation_and_ellipses():
    assert rt.quote_on_page("Newsom signed AB 282 … a felony", PAGE["text"])
    assert rt.quote_on_page("“Governor Newsom signed AB 282”", PAGE["text"])
    assert not rt.quote_on_page("Newsom vetoed AB 282", PAGE["text"])


def test_source_hunter_uses_the_tool_loop_and_does_not_rerun_it(monkeypatch):
    monkeypatch.setattr(cfg, "RESEARCH_TOOL_LOOP_ENABLED", True)
    out = {"status": "no_evidence", "answer": "No accepted source evidence was found.", "sources": [],
           "metadata": {"engine": "tools"}}
    with patch("newscaster.research_tools.research_with_tools", return_value=out) as loop:
        result = sh.answer_with_escalation("Q?", topic="T", formatted_date="September 25, 2026")
    assert result.status == "no_evidence" and loop.call_count == 1


def test_tool_loop_failure_falls_back_to_the_fixed_pipeline(monkeypatch):
    monkeypatch.setattr(cfg, "RESEARCH_TOOL_LOOP_ENABLED", True)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_QUESTION_QUERIES", False)
    monkeypatch.setattr(cfg, "SOURCE_HUNTER_NEARBY_SOURCE_DEPTH", 0)
    with patch("newscaster.research_tools.research_with_tools", side_effect=RuntimeError("model down")), \
         patch("newscaster.source_hunter.search_web", return_value=[]) as search, \
         patch("newscaster.source_hunter._generate_evidence_contract", return_value={}):
        result = sh.answer_with_source_hunter("Q?", topic="T")
    assert search.called and result.status == "no_evidence"
