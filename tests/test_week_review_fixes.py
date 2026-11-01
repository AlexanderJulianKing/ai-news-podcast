"""Fixes from the 2026-10-29 week review (episodes Oct 22-29).

- Oct 24: the script writer wrote a note to the pipeline ("Note: The update context describes earlier
  coverage of OpenAI's ... I've written this as a standalone report") and Ethan read it on air.
- The editor saw ~45k of 200-250k characters of evidence on 8 of 25 stories, partly because the
  verified quotes repeated what the page excerpts already held.
- Unsupported specifics aired (a wrong full name for "Chu", "Cellebrite", an invented caveat): the
  editor fixes contradictions with a known correct value, never an unsupported claim.
- The audit log rotates every run, so no weekly cost could be measured.
"""
import json
import sys
import types
from unittest.mock import patch

try:                                       # broken on the Mac (protobuf mismatch), fine on the Pi
    from google.cloud import texttospeech  # noqa: F401
except Exception:
    import google.cloud
    _stub = types.ModuleType("google.cloud.texttospeech")
    google.cloud.texttospeech = _stub
    sys.modules["google.cloud.texttospeech"] = _stub

from newscaster import costs, review, text_utils
from newscaster.audio import tts
from newscaster.script import segments


def test_producer_notes_are_removed_from_the_script_and_logged(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(text_utils, "write_jsonl_log", lambda prefix, rec: logged.append((prefix, rec)))
    text = ("Grace: Good morning.\n[[NOTE: The earlier-coverage note describes a different story, "
            "so I wrote this one on its own.]]\nEthan: Thanks, Grace.")
    clean = text_utils.extract_producer_notes(text, where="2026_10_24 segment 1")
    assert clean == "Grace: Good morning.\nEthan: Thanks, Grace."
    assert logged and logged[0][0] == "producer_notes" and "different story" in logged[0][1]["note"]


def _good(reporter):
    return f"Grace: Good morning. Here is the story.\n{reporter}: Thanks, Grace. It happened today.\nGrace: Thank you."


def test_a_script_with_a_line_no_one_speaks_is_rewritten(tmp_path, monkeypatch):
    # The Oct 24 note had a colon ("Note: ..."), so it passed as a speaker line.
    monkeypatch.chdir(tmp_path)
    replies = iter(["Note: The update context doesn't match this story.\n" + _good("Ethan")])
    with patch.object(segments, "get_llm_response", side_effect=lambda *a, **k: next(replies, _good("Ethan"))) as llm, \
         patch.object(segments.time, "sleep", lambda s: None):
        segments.segments_writer({0: "A story."}, "2026_10_24", ["Ethan", "Chloe"], "October 24, 2026")
    script = (tmp_path / "output_scripts/2026_10_24_segment_0.txt").read_text()
    assert llm.call_count >= 2 and "Note" not in script
    assert all(line.split(":", 1)[0] in ("Grace", "Ethan") for line in script.splitlines())


def test_text_to_speech_never_voices_an_unlabeled_line():
    voices = ["Grace", "Ethan", "Opinion"]
    assert tts.split_speaker("Note: The update context describes another story.", voices, "Ethan", strict=True) is None
    assert tts.split_speaker("Ethan: It happened today.", voices, "Grace", strict=True) == ("Ethan", "It happened today.")


def test_verified_quotes_already_in_the_page_excerpt_are_not_repeated():
    followup = {"answer": 'FINDINGS:\n- "The county certifies by December 3." [https://x.gov/a]\n'
                          '- "Results go to the state on December 4." [https://x.gov/a]\n',
                "source_hunter_sources": [{"title": "County page", "url": "https://x.gov/a",
                                           "excerpt": "Calendar. The county certifies by December 3. More text."}]}
    blocks = review._verified_findings(followup)
    assert len(blocks) == 1
    assert "December 4" in blocks[0] and "certifies by December 3" not in blocks[0]


def test_unsupported_claims_are_fixed_only_with_a_quote_and_cut_otherwise():
    script = ("Chloe: State Senator Susan Rubio Chu called the support essential. "
              "The tool used was Cellebrite. Prices rose.")
    evidence = 'SOURCE: CalMatters — https://calmatters.org/x\n"Chu called the support essential to survivors."'
    flags = ["FLAG: Susan Rubio Chu — the source says only Chu", "FLAG: Cellebrite — not in any source",
             "FLAG: Prices rose — unsupported"]
    verdicts = {"verdicts": [
        {"flag": 1, "verdict": "fix", "find": "State Senator Susan Rubio Chu called the support essential.",
         "replace": "Chu called the support essential.", "quote": "Chu called the support essential to survivors."},
        {"flag": 2, "verdict": "fix", "find": "The tool used was Cellebrite.", "replace": ""},
        {"flag": 3, "verdict": "fix", "find": "Prices rose.", "replace": "Prices doubled.", "quote": "invented"}]}
    with patch.object(review, "get_llm_response", return_value=json.dumps(verdicts)):
        out, audit = review.rule_on_flags(script, evidence, flags, review.SOURCING_PROMPT, label="test")
    assert "Rubio" not in out and "Chu called the support essential." in out       # fixed, with a backing quote
    assert "Cellebrite" not in out                                                # cut
    assert "Prices rose." in out and "doubled" not in out                         # a fix whose quote isn't there is refused
    assert audit["applied"] == 2 and len(audit["skipped"]) == 1


def test_each_run_writes_one_cost_line(tmp_path, monkeypatch):
    rows = []
    monkeypatch.setattr(costs, "write_jsonl_log", lambda prefix, rec: rows.append((prefix, rec)))
    costs.reset()
    costs.record("llm", "claude-opus-5-5", 0.25)
    costs.record("llm", "claude-opus-5-5", 0.50)
    costs.record("tts", "Chirp3-HD", 0.03, units=1000, unit="chars")
    costs.write_run_summary("2026_10_29")
    prefix, rec = rows[0]
    assert prefix == "run_costs" and rec["key"] == "2026_10_29"
    assert rec["total_usd"] == 0.78 and rec["by_item"]["llm: claude-opus-5-5"]["calls"] == 2


def test_the_fact_check_cuts_an_unsupported_claim_from_a_main_story(tmp_path, monkeypatch):
    # Through the public entry point: the editor declines (no known correct value), the ruling cuts it.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output_scripts").mkdir()
    (tmp_path / "segment_summaries").mkdir()
    script = "Grace: What tools did DHS use?\nChloe: It used Palantir's Falcon. It also used Cellebrite.\nGrace: Thanks."
    (tmp_path / "output_scripts/2026_10_29_segment_0.txt").write_text(script)
    research = {"articles": [], "followups": [{"answer": "", "source_hunter_sources": [
        {"title": "WIRED", "url": "https://wired.com/x", "excerpt": "DHS used Palantir's Falcon and PenLink."}]}]}
    (tmp_path / "segment_summaries/2026_10_29_segment0_research.json").write_text(json.dumps(research))
    monkeypatch.setattr(review._config, "FACT_FINDER_SOURCING_RULING_ENABLED", True)
    ruling = {"verdicts": [{"flag": 1, "verdict": "fix", "find": "It also used Cellebrite.", "replace": ""}]}

    class NoEdit:
        changed, applied, rejected, text = False, [], [], script
    with patch.object(review, "review_scripts", return_value=[("2026_10_29_segment_0.txt", "faithfulness",
                                                               "FLAG: Cellebrite — no source names it")]), \
         patch.object(review, "revise_script", return_value=NoEdit()), \
         patch.object(review, "get_llm_response", return_value=json.dumps(ruling)), \
         patch.object(review, "write_jsonl_log", lambda *a, **k: None):
        result = review.review_and_revise_scripts("2026_10_29")
    aired = (tmp_path / "output_scripts/2026_10_29_segment_0.txt").read_text()
    assert "Cellebrite" not in aired and "Palantir's Falcon" in aired and result["ruled"] == 1
