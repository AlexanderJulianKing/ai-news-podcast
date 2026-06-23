import json
from unittest.mock import patch

from newscaster.review import (
    extract_quotes, verify_quotes, build_source_corpus, faithfulness_flags, stable_fact_flags,
    verified_stable_fact_flags,
)


def test_extract_pulls_marked_quotes():
    script = ("Grace: He said, quote, the program starts June first, endquote. "
              "Later she noted, quote, funding is secure, endquote, before moving on.")
    assert extract_quotes(script) == ["the program starts June first", "funding is secure"]


def test_extract_ignores_paraphrase():
    # Indirect speech with no quote/endquote markers is not a checkable direct quote
    # (this is why the Sybiha "fuel for the fire" paraphrase needs the LLM pass, not this one).
    assert extract_quotes("He added that this is simply the fuel for the fire of the conflict.") == []


def test_verify_flags_distinctive_quote_absent_from_sources():
    script = "The group called it a, quote, vindication of personal freedom, endquote."
    corpus = "The group supported the ruling but issued no such statement."
    verdicts = verify_quotes(script, corpus)
    assert len(verdicts) == 1
    assert verdicts[0].grounded is False


def test_verify_passes_quote_present_in_sources_despite_case_and_punct():
    script = "He was, quote, awkwardly positioned, endquote, on the issue."
    corpus = "Officials said the agency is **Awkwardly Positioned**, given the new policy."
    assert verify_quotes(script, corpus)[0].grounded is True


def test_verify_skips_too_short_quote():
    # Very short, common quotes (an attribution problem like "not enough") are skipped
    # rather than false-flagged; the companion LLM plausibility pass owns those.
    script = "The offer was, quote, not enough, endquote, to satisfy them."
    verdict = verify_quotes(script, "the offer was rejected")[0]
    assert verdict.grounded is True
    assert "short" in verdict.reason


def test_build_source_corpus_reads_audit_excerpts_for_the_day(tmp_path, monkeypatch):
    # The gate's ground truth is the source-hunter excerpts persisted in the audit during
    # gather (no re-fetch), scoped to the given day.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "source_hunter_audit.jsonl").write_text(
        json.dumps({"timestamp": "2026-06-19T04:00:00",
                    "sources": [{"excerpt": "The coalition warned of 145,000 lost jobs."}]}) + "\n"
        + json.dumps({"timestamp": "2026-06-18T04:00:00",
                      "sources": [{"excerpt": "yesterday unrelated excerpt"}]}) + "\n"
    )
    corpus = build_source_corpus("2026_06_19")
    assert "145,000 lost jobs" in corpus       # today's excerpt is in the corpus
    assert "yesterday" not in corpus            # other days are filtered out


def test_build_source_corpus_includes_scraped_article_text(tmp_path, monkeypatch):
    # The corpus also includes the RAW scraped article text the writer read (persisted at gather),
    # not just source-hunter excerpts — so the faithfulness pass stops false-flagging well-sourced
    # claims as "not in source material".
    monkeypatch.chdir(tmp_path)
    (tmp_path / "segment_summaries").mkdir()
    (tmp_path / "segment_summaries" / "2026_06_20_segment0_article0_source.txt").write_text(
        "The FTC and four states sued WPATH over allegedly deceptive claims about pediatric care.",
        encoding="utf-8",
    )
    corpus = build_source_corpus("2026_06_20")
    assert "FTC and four states sued WPATH" in corpus


def test_build_segment_corpus_scopes_to_one_segment(tmp_path, monkeypatch):
    # A segment's corpus = only THAT segment's scraped articles (+ excerpts), not the other segments'
    # — scoping out the noise is what restores the faithfulness pass's recall on subtle contradictions.
    from newscaster.review import build_segment_corpus
    monkeypatch.chdir(tmp_path)
    (tmp_path / "segment_summaries").mkdir()
    (tmp_path / "segment_summaries" / "2026_06_21_segment0_article0_source.txt").write_text(
        "Iran and U.S. negotiators met in Switzerland.", encoding="utf-8")
    (tmp_path / "segment_summaries" / "2026_06_21_segment1_article0_source.txt").write_text(
        "California is slow to count mail-in ballots.", encoding="utf-8")
    corp0 = build_segment_corpus("2026_06_21", 0)
    assert "Iran and U.S. negotiators met in Switzerland" in corp0
    assert "California is slow to count" not in corp0      # segment 1's sources are scoped out


def test_corpus_for_script_routes_segment_vs_overview(tmp_path, monkeypatch):
    from newscaster.review import _corpus_for_script
    monkeypatch.chdir(tmp_path)
    (tmp_path / "segment_summaries").mkdir()
    (tmp_path / "segment_summaries" / "2026_06_21_segment0_article0_source.txt").write_text(
        "SEGZERO SOURCE", encoding="utf-8")
    full = "WHOLE DAY CORPUS"
    assert "SEGZERO SOURCE" in _corpus_for_script("2026_06_21", "2026_06_21_segment_0.txt", full)
    assert _corpus_for_script("2026_06_21", "2026_06_21_overview.txt", full) == full   # not a segment
    assert _corpus_for_script("2026_06_21", "2026_06_21_segment_9.txt", full) == full  # no sources -> full


def test_faithfulness_flags_parses_flag_lines():
    out = "FLAG: He said X — not in sources\nsome prose here\nFLAG: the 50% figure — absent from sources"
    with patch("newscaster.review.get_llm_response", return_value=out):
        flags = faithfulness_flags("a script", "some corpus text")
    assert len(flags) == 2
    assert all(f.startswith("FLAG:") for f in flags)


def test_faithfulness_flags_none_when_supported():
    with patch("newscaster.review.get_llm_response", return_value="NONE"):
        assert faithfulness_flags("a script", "some corpus text") == []


def test_faithfulness_flags_skips_without_corpus_and_does_not_call_llm():
    # No corpus -> can't ground-check -> don't false-flag, and don't waste an LLM call.
    with patch("newscaster.review.get_llm_response") as mock_llm:
        assert faithfulness_flags("a script", "") == []
    mock_llm.assert_not_called()


def test_faithfulness_flags_fails_open_on_error():
    # The gate must never block the pipeline, so an LLM error yields no flags, not an exception.
    with patch("newscaster.review.get_llm_response", side_effect=RuntimeError("llm down")):
        assert faithfulness_flags("a script", "some corpus text") == []


def test_stable_fact_flags_parses_flags():
    out = "FLAG: Google CEO Eric Schmidt — former Google CEO (Pichai leads Google)\nsome prose"
    with patch("newscaster.review.get_llm_response", return_value=out):
        flags = stable_fact_flags("...script naming Eric Schmidt as Google CEO...")
    assert len(flags) == 1 and "Schmidt" in flags[0]


def test_stable_fact_flags_none_when_clean():
    with patch("newscaster.review.get_llm_response", return_value="NONE"):
        assert stable_fact_flags("a clean script") == []


def test_stable_fact_flags_fails_open_on_error():
    with patch("newscaster.review.get_llm_response", side_effect=RuntimeError("llm down")):
        assert stable_fact_flags("a script") == []


def test_verified_stable_fact_keeps_only_search_confirmed_errors():
    # Memory proposes two suspects; the search confirms Schmidt is wrong but clears Doerr.
    memory = "FLAG: Google CEO Eric Schmidt — former CEO\nFLAG: Kleiner Perkins chairman John Doerr — partner"
    def fake_brief(q):
        return "WRONG: Sundar Pichai is the current CEO of Google." if "Schmidt" in q else "CORRECT"
    with patch("newscaster.review.get_llm_response", return_value=memory), \
         patch("newscaster.review.openrouter_web_brief", side_effect=fake_brief):
        flags = verified_stable_fact_flags("a script naming Eric Schmidt and John Doerr")
    assert len(flags) == 1
    assert "Eric Schmidt" in flags[0] and "Pichai" in flags[0]


def test_verified_stable_fact_does_not_flag_when_search_unavailable():
    # If the verifying search can't run, don't flag — never accuse without confirmation.
    with patch("newscaster.review.get_llm_response", return_value="FLAG: Google CEO Eric Schmidt — former CEO"), \
         patch("newscaster.review.openrouter_web_brief", side_effect=RuntimeError("search down")):
        assert verified_stable_fact_flags("a script") == []
