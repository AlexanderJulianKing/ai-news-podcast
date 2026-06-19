import json
from unittest.mock import patch

from newscaster.review import (
    extract_quotes, verify_quotes, build_source_corpus, faithfulness_flags, stable_fact_flags,
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
