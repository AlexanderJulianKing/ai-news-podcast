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
    """The OpenRouter (Luna) loop, which is now the backup."""
    with patch.object(cfg, "RESEARCH_TOOL_PROVIDER", "openrouter"), \
         patch.object(rt, "_chat", side_effect=_script(facts)), \
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
    # 2026-10-10: only the verified quote reaches the controller, never the model's own wording
    assert "AB 282 is by Gail Pellerin" not in out["answer"] and "ANSWER:" not in out["answer"]
    assert '"AB 282 by Assemblymember Gail Pellerin"' in out["answer"]
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


def test_quote_survives_pdf_page_headers_and_line_numbers():
    opinion = ("In any event, the Attorney General has not issued any such order here, having directed only a pause in the "
               "14 BONTA v. BIANCO Opinion of the Court by Guerrero, C. J. investigation into the ballot count.")
    declaration = "At the end of the day the Riverside County Sheriff's Office opened and 21 counted 22 boxes from the pallet."
    assert rt.quote_on_page("In any event, the Attorney General has not issued any such order here, having directed "
                            "only a pause in the investigation into the ballot count.", opinion)
    assert rt.quote_on_page("At the end of the day the Riverside County Sheriff's Office opened and counted 22 boxes "
                            "from the pallet.", declaration)


def test_gap_tolerance_does_not_accept_a_changed_or_invented_quote():
    page = "The court ordered the sheriff to return the ballots to county election officials within ten days."
    assert not rt.quote_on_page("The court ordered the sheriff to destroy the ballots within ten days.", page)
    assert not rt.quote_on_page("The court fined the sheriff for seizing the ballots.", page)
    assert not rt.quote_on_page("ordered the sheriff", "the court refused and ordered nothing of the sheriff")   # short: exact only


def test_gap_tolerance_never_skips_a_negation():
    page = "The court did not order the sheriff to return the ballots to county election officials."
    assert not rt.quote_on_page("The court did order the sheriff to return the ballots to county election officials.", page)
    assert not rt.quote_on_page("The court order the sheriff to return the ballots to county election officials.", page)
    assert rt.quote_on_page("The court did not order the sheriff to return the ballots to county election officials.", page)


# --- Claude Haiku 5.5 as the researcher (2026-10-10, benchmarks/haiku_vs_luna/) ---------

class _Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, *blocks):
        self.content = list(blocks)
        self.usage = None


def _haiku_script(final_facts):
    replies = iter([
        _Msg(_Block("tool_use", id="t1", name="web_search", input={"query": "Newsom ballot bill"})),
        _Msg(_Block("tool_use", id="t2", name="open_page", input={"url": PAGE["url"]})),
        _Msg(_Block("text", text=json.dumps({"answer": "AB 282.", "facts": final_facts,
                                              "gaps": ["Searched for the effective date; not found."]}))),
    ])
    sent = []

    def create(client, **kwargs):
        sent.append(kwargs)
        return next(replies)
    return create, sent


def test_haiku_researches_with_tools_and_only_verified_quotes_come_back(monkeypatch):
    create, sent = _haiku_script([
        {"fact": "AB 282 is by Gail Pellerin", "quote": "AB 282 by Assemblymember Gail Pellerin", "url": PAGE["url"]},
        {"fact": "Invented", "quote": "takes effect January 1, 2027", "url": PAGE["url"]},
    ])
    monkeypatch.setattr(cfg, "RESEARCH_TOOL_PROVIDER", "anthropic")
    with patch.object(rt, "_create_message", side_effect=create), \
         patch.object(rt, "search_web", return_value=[{"headline": "h", "url": PAGE["url"], "snippet": "s"}]), \
         patch.object(rt, "fetch_source_text", return_value=PAGE):
        out = rt.research_with_tools("Which ballot bills did Newsom sign?", "September 25, 2026")
    assert sent[0]["model"] == "claude-haiku-5-5"
    assert "Searched for" in sent[0]["system"]                       # gaps are searches, not claims
    assert any(b.get("type") == "tool_result" for b in sent[2]["messages"][-1]["content"])
    assert out["status"] == "success" and out["metadata"]["model"] == "claude-haiku-5-5"
    assert '"AB 282 by Assemblymember Gail Pellerin"' in out["answer"] and "January 1, 2027" not in out["answer"]
    assert "AB 282 is by Gail Pellerin" not in out["answer"] and "ANSWER:" not in out["answer"]
    assert "Searched for the effective date; not found." in out["answer"].split("GAPS:")[1]


def test_a_haiku_failure_falls_back_to_luna(monkeypatch):
    monkeypatch.setattr(cfg, "RESEARCH_TOOL_PROVIDER", "anthropic")
    with patch.object(rt, "_create_message", side_effect=RuntimeError("anthropic down")), \
         patch.object(rt, "_chat", side_effect=_script([
             {"fact": "f", "quote": "AB 282 by Assemblymember Gail Pellerin", "url": PAGE["url"]}])), \
         patch.object(rt, "time") as fake_time, \
         patch.object(rt, "search_web", return_value=[{"headline": "h", "url": PAGE["url"], "snippet": "s"}]), \
         patch.object(rt, "fetch_source_text", return_value=PAGE):
        fake_time.time.return_value = 0.0
        out = rt.research_with_tools("Which ballot bills did Newsom sign?", "September 25, 2026")
    assert out["status"] == "success" and out["metadata"]["model"] == cfg.RESEARCH_TOOL_FALLBACK_MODEL
