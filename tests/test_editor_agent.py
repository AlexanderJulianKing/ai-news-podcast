from unittest.mock import patch

import newscaster.config as cfg
from newscaster import editor_agent
from newscaster.editor_agent import Edit, EditOutcome, revise_script, _apply_edits


def _enable(monkeypatch, rounds=3):
    monkeypatch.setattr(cfg, "FACT_FINDER_AUTOEDIT_ENABLED", True)
    monkeypatch.setattr(cfg, "FACT_FINDER_AUTOEDIT_MAX_ROUNDS", rounds)


# --- _apply_edits: pure, deterministic ---

def test_apply_replaces_unique_occurrence():
    text = "Reporting from the Google CEO Eric Schmidt event."
    out, applied, skipped = _apply_edits(
        text, [Edit(find="Google CEO Eric Schmidt", replace="former Google CEO Eric Schmidt")]
    )
    assert out == "Reporting from the former Google CEO Eric Schmidt event."
    assert len(applied) == 1 and not skipped


def test_apply_skips_ambiguous_multiple_occurrences():
    # Two identical spans -> we can't tell which the flag meant -> leave it for a human, don't guess.
    text = "Google CEO Eric Schmidt spoke. Google CEO Eric Schmidt again."
    out, applied, skipped = _apply_edits(
        text, [Edit(find="Google CEO Eric Schmidt", replace="former Google CEO Eric Schmidt")]
    )
    assert out == text  # unchanged
    assert not applied and skipped and "ambiguous" in skipped[0]["reason"]


def test_apply_skips_find_not_present():
    out, applied, skipped = _apply_edits("hello world", [Edit(find="nonexistent span", replace="x")])
    assert out == "hello world" and not applied
    assert skipped and "not present" in skipped[0]["reason"]


def test_apply_skips_noop_and_empty_find():
    out, applied, skipped = _apply_edits("abc", [Edit(find="abc", replace="abc"), Edit(find="", replace="y")])
    assert out == "abc" and not applied and len(skipped) == 2


# --- _propose_edits / _verify_edits: JSON parsing ---

def test_propose_parses_edits():
    raw = '{"edits": [{"find": "A", "replace": "B", "wrong": "w", "correct": "c", "basis": "src"}]}'
    with patch.object(editor_agent, "get_llm_response", return_value=raw):
        edits = editor_agent._propose_edits("A script", "report", "corpus", [])
    assert len(edits) == 1 and edits[0].find == "A" and edits[0].replace == "B" and edits[0].basis == "src"


def test_propose_drops_edits_without_find():
    raw = '{"edits": [{"replace": "B"}, {"find": "ok", "replace": "fixed"}]}'
    with patch.object(editor_agent, "get_llm_response", return_value=raw):
        edits = editor_agent._propose_edits("ok script", "r", "c", [])
    assert len(edits) == 1 and edits[0].find == "ok"


def test_verify_keeps_only_approved():
    e0, e1 = Edit(find="A", replace="B"), Edit(find="C", replace="D")
    raw = '{"verdicts": [{"index": 0, "approve": true, "reason": "ok"}, {"index": 1, "approve": false, "reason": "stylistic"}]}'
    with patch.object(editor_agent, "get_llm_response", return_value=raw):
        approved, rejected = editor_agent._verify_edits([e0, e1], "script", "report", "corpus")
    assert approved == [e0]
    assert len(rejected) == 1 and rejected[0]["find"] == "C" and rejected[0]["reason"] == "stylistic"


# --- revise_script: loop + safety (propose/verify patched) ---

def test_disabled_returns_unchanged(monkeypatch):
    monkeypatch.setattr(cfg, "FACT_FINDER_AUTOEDIT_ENABLED", False)
    out = revise_script("a script", "FLAG: x — WRONG: y", "sources")
    assert out.changed is False and out.text == "a script"


def test_empty_report_returns_unchanged(monkeypatch):
    _enable(monkeypatch)
    out = revise_script("a script", "", "sources")
    assert out.changed is False and out.text == "a script"


def test_approved_edit_is_applied(monkeypatch):
    _enable(monkeypatch)
    script = "Reporting from the Google CEO Eric Schmidt event."
    edit = Edit(find="Google CEO Eric Schmidt", replace="former Google CEO Eric Schmidt",
                wrong="Schmidt is current CEO", correct="Pichai is CEO", basis="verified search")
    with patch.object(editor_agent, "_propose_edits", side_effect=[[edit], []]), \
         patch.object(editor_agent, "_verify_edits", return_value=([edit], [])):
        out = revise_script(script, "FLAG: Google CEO Eric Schmidt — WRONG: Pichai is CEO", "sources", label="seg")
    assert out.changed is True
    assert out.text == "Reporting from the former Google CEO Eric Schmidt event."
    assert len(out.applied) == 1


def test_rejected_edit_not_applied(monkeypatch):
    _enable(monkeypatch)
    script = "A stylistic choice here."
    edit = Edit(find="stylistic choice", replace="different wording")
    with patch.object(editor_agent, "_propose_edits", side_effect=[[edit], []]), \
         patch.object(editor_agent, "_verify_edits", return_value=([], [{"find": edit.find, "reason": "stylistic"}])):
        out = revise_script(script, "FLAG: stylistic choice — reword", "sources")
    assert out.changed is False and out.text == script and out.rejected


def test_propose_error_fails_open(monkeypatch):
    _enable(monkeypatch)
    with patch.object(editor_agent, "_propose_edits", side_effect=RuntimeError("opus down")):
        out = revise_script("a script", "FLAG: x", "sources")
    assert out.changed is False and out.text == "a script"


def test_adversary_error_applies_nothing(monkeypatch):
    _enable(monkeypatch)
    edit = Edit(find="x marks", replace="y marks")
    with patch.object(editor_agent, "_propose_edits", side_effect=[[edit], []]), \
         patch.object(editor_agent, "_verify_edits", side_effect=RuntimeError("adversary down")):
        out = revise_script("x marks the spot", "FLAG: x", "sources")
    assert out.changed is False and out.text == "x marks the spot"


def test_length_guard_discards_catastrophic_edit(monkeypatch):
    _enable(monkeypatch)
    script = "A" * 1000
    edit = Edit(find="A" * 1000, replace="A" * 100)  # would gut the script
    with patch.object(editor_agent, "_propose_edits", side_effect=[[edit], []]), \
         patch.object(editor_agent, "_verify_edits", return_value=([edit], [])):
        out = revise_script(script, "FLAG: ...", "sources")
    assert out.changed is False and out.text == script


def test_loop_terminates_without_repeating(monkeypatch):
    _enable(monkeypatch, rounds=5)
    e1 = Edit(find="Alpha", replace="Gamma")
    with patch.object(editor_agent, "_propose_edits", side_effect=[[e1], []]) as prop, \
         patch.object(editor_agent, "_verify_edits", return_value=([e1], [])):
        out = revise_script("Alpha and Beta.", "FLAG: ...", "sources")
    assert out.text == "Gamma and Beta."
    assert prop.call_count == 2  # round 0 applied, round 1 proposed nothing -> stop


# --- review.review_and_revise_scripts: orchestration + write-back ---

def test_review_and_revise_writes_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output_scripts").mkdir()
    (tmp_path / "logs").mkdir()
    sp = tmp_path / "output_scripts" / "2026_06_19_segment0.txt"
    sp.write_text("Google CEO Eric Schmidt spoke today.", encoding="utf-8")

    from newscaster import review
    flags = [("2026_06_19_segment0.txt", "stable-fact", "FLAG: Google CEO Eric Schmidt — WRONG: Pichai is CEO")]
    outcome = EditOutcome(
        text="former Google CEO Eric Schmidt spoke today.",
        applied=[Edit(find="Google CEO Eric Schmidt", replace="former Google CEO Eric Schmidt", correct="Pichai")],
        rejected=[], changed=True,
    )
    with patch.object(review, "build_source_corpus", return_value="sources"), \
         patch.object(review, "review_scripts", return_value=flags), \
         patch.object(review, "revise_script", return_value=outcome):
        result = review.review_and_revise_scripts("2026_06_19")
    assert sp.read_text(encoding="utf-8") == "former Google CEO Eric Schmidt spoke today."
    assert result["edited_scripts"] == 1 and result["edits"] == 1


def test_review_and_revise_no_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from newscaster import review
    with patch.object(review, "build_source_corpus", return_value=""), \
         patch.object(review, "review_scripts", return_value=[]):
        result = review.review_and_revise_scripts("2026_06_19")
    assert result == {"flags": 0, "edited_scripts": 0, "edits": 0}


def test_review_and_revise_leaves_file_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output_scripts").mkdir()
    (tmp_path / "logs").mkdir()
    sp = tmp_path / "output_scripts" / "2026_06_19_overview.txt"
    sp.write_text("original text", encoding="utf-8")
    from newscaster import review
    flags = [("2026_06_19_overview.txt", "faithfulness", "FLAG: something — unsupported")]
    unchanged = EditOutcome(text="original text", applied=[], rejected=[{"find": "x", "reason": "absent not wrong"}], changed=False)
    with patch.object(review, "build_source_corpus", return_value="sources"), \
         patch.object(review, "review_scripts", return_value=flags), \
         patch.object(review, "revise_script", return_value=unchanged):
        result = review.review_and_revise_scripts("2026_06_19")
    assert sp.read_text(encoding="utf-8") == "original text"
    assert result["edited_scripts"] == 0
