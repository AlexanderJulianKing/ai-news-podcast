"""Unit tests for lead-pick harvesting and label storage.

These pin the parsing that decides which brief is which and which are barred
from leading. A silent bug there would mislabel every case.
"""
import json

import pytest

from benchmarks.editorial_decisions.harvest_lead_pick import (
    EXTRACT_PREFIX,
    TIER3_PREFIX,
    essay_answer,
    harvest,
    match_pick,
    parse_briefs,
    split_tags,
)
from benchmarks.editorial_decisions.label_server import (
    cases_payload,
    display_order,
    load_labels,
    save_label,
)

DOC = """--- Brief 1 ---
Headline: [UPDATE: ukraine_war] Trump calls on Ukraine to halt strikes
Reported by: The Associated Press, NPR

Memo text one.

--- Brief 2 ---
Headline: [SIDE-COVERED: ai_safety] AI labs ask for a slowdown

Memo text two.

--- Brief 3 ---
Headline: Fed holds rates steady

Memo text three.

=== Coverage notes: stories previously mentioned only in the side-story roundup ===
ai_safety: mentioned Sept 12."""


def test_split_tags_strips_prefix_and_keeps_slug():
    assert split_tags("[UPDATE: ukraine_war] Trump calls") == ("Trump calls", [("UPDATE", "ukraine_war")])
    assert split_tags("[MAJOR ESCALATION] Strike") == ("Strike", [("MAJOR ESCALATION", None)])
    assert split_tags("No tag here") == ("No tag here", [])


def test_parse_briefs_fields_and_eligibility():
    briefs, preamble, trailer = parse_briefs(DOC)
    assert [b["index"] for b in briefs] == [1, 2, 3]
    assert briefs[0]["headline"] == "Trump calls on Ukraine to halt strikes"
    assert briefs[0]["reported_by"] == ["The Associated Press", "NPR"]
    assert briefs[0]["brief"] == "Memo text one."
    assert [b["eligible"] for b in briefs] == [False, True, True]  # only [UPDATE] is barred
    assert preamble == ""
    assert trailer.startswith("=== Coverage notes")
    assert "Coverage notes" not in briefs[2]["brief"]


def test_parse_briefs_keeps_degraded_notice_as_preamble():
    briefs, preamble, _ = parse_briefs("RESEARCH NOTICE: degraded today.\n\n" + DOC)
    assert preamble == "RESEARCH NOTICE: degraded today."
    assert len(briefs) == 3


def test_match_pick_exact_fuzzy_and_miss():
    briefs, _, _ = parse_briefs(DOC)
    assert match_pick("Fed holds rates steady", briefs) == 3
    assert match_pick("AI labs ask for a slowdown in development", briefs) == 2
    assert match_pick("Something about a hurricane", briefs) is None


def _audit_row(ts, doc, event="success", system=TIER3_PREFIX + " for the United States."):
    return json.dumps({"event": event, "timestamp": ts, "model": "claude-opus-4-8",
                       "system_prompt": system, "user_prompt": doc})


def test_harvest_groups_retagged_replays_and_reads_production_pick(tmp_path):
    retagged = DOC.replace("[UPDATE: ukraine_war]", "[DEVELOPMENT: ukraine_war]")
    audit = tmp_path / "llm_audit.jsonl"
    audit.write_text("\n".join([
        _audit_row("2026-09-10T04:10:00", DOC),
        _audit_row("2026-09-10T04:12:00", retagged),              # same briefs, new tags
        _audit_row("2026-09-10T04:13:00", DOC, event="error"),    # ignored
        _audit_row("2026-09-10T04:14:00", DOC, system="Other."),  # ignored
    ]) + "\n")
    (tmp_path / "log_26_09_10.txt").write_text(
        "04:09:00 - Answer: a Tier 1 answer that must be skipped\n"
        "04:10:00 - TIER 3: Selecting stories\nreasoning...\n**Answer: Fed holds rates steady**\n"
        "Answer: AI labs ask for a slowdown\n"
    )
    cases = harvest([str(audit)], str(tmp_path))
    assert len(cases) == 1
    case = cases[0]
    assert case["n_variants"] == 2
    assert case["briefs"][0]["eligible"] is True      # the latest run's tags win
    assert case["production_pick_index"] == 3         # first Answer after the Tier 3 marker


def test_essay_answer_reads_inline_and_heading_forms():
    assert essay_answer("Reasoning.\n\nAnswer: Fed holds rates steady") == "Fed holds rates steady"
    assert essay_answer("# Analysis\nstuff\n\n## Answer\n\nFed holds rates steady\n") == "Fed holds rates steady"
    assert essay_answer("**Answer:** Fed holds rates steady") == "Fed holds rates steady"
    assert essay_answer("No verdict here") is None


def test_harvest_recovers_each_runs_pick_from_the_extractor_call(tmp_path):
    retagged = DOC.replace("[UPDATE: ukraine_war]", "[DEVELOPMENT: ukraine_war]")
    extract = EXTRACT_PREFIX + " What is the headline?"
    audit = tmp_path / "llm_audit.jsonl"
    audit.write_text("\n".join([
        _audit_row("2026-09-10T04:10:00", DOC),
        _audit_row("2026-09-10T04:10:01", "## Answer\n\nFed holds rates steady", system=extract),
        _audit_row("2026-09-10T04:12:00", retagged),
        _audit_row("2026-09-10T04:12:01", "Answer: Trump calls on Ukraine to halt strikes", system=extract),
    ]) + "\n")
    case = harvest([str(audit)])[0]
    assert [run["pick_index"] for run in case["runs"]] == [3, 1]
    assert case["runs"][0]["eligible_indexes"] == [2, 3]      # [UPDATE] barred brief 1
    assert case["runs"][1]["eligible_indexes"] == [1, 2, 3]   # retagged run freed it
    assert case["production_pick_index"] == 1                 # latest run


def test_labels_latest_wins_and_order_is_stable(tmp_path):
    labels = tmp_path / "labels.jsonl"
    base = {"case_id": "c1", "lead_index": 2, "remembers_outcome": "no", "confidence": "low"}
    save_label(base, str(labels))
    save_label(dict(base, lead_index=3), str(labels))
    assert load_labels(str(labels))["c1"]["lead_index"] == 3
    assert len(labels.read_text().splitlines()) == 2  # history kept

    with pytest.raises(ValueError):
        save_label({"case_id": "c1"}, str(labels))

    briefs, _, _ = parse_briefs(DOC)
    case = {"case_id": "c1", "date": "2026-09-10", "briefs": briefs}
    assert display_order(case) == display_order(case)
    assert sorted(display_order(case)) == [1, 2, 3]

    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(case) + "\n")
    payload = cases_payload(str(cases_file), str(labels))
    assert payload[0]["label"]["lead_index"] == 3


def test_hypotheticals_round_trip_through_the_production_format():
    from benchmarks.editorial_decisions.hypotheticals import load_hypotheticals

    cases = load_hypotheticals()
    assert [c["set"] for c in cases].count("invented") == 10
    assert [c["set"] for c in cases].count("invented_hard") == 8
    assert [c["set"] for c in cases].count("invented_hard2") == 10
    assert [c["set"] for c in cases].count("holdout1") == 20
    assert len({c["case_id"] for c in cases}) == 48
    for c in cases:
        assert c["synthetic"] is True
        indexes = [b["index"] for b in c["briefs"]]
        assert indexes == list(range(1, len(indexes) + 1))
        assert all(b["headline"] and b["brief"] and b["reported_by"] for b in c["briefs"])
        assert all(1 <= i <= len(indexes) for i in c["probe"].values())  # probes point at real briefs
    hard = next(c for c in cases if c["case_id"].startswith("hypo_h02"))
    assert hard["coverage_notes"].startswith("=== Coverage notes")       # trailer parsed out of the last brief
    assert "Coverage notes" not in hard["briefs"][-1]["brief"]
    barred = [b["index"] for b in hard["briefs"] if not b["eligible"]]
    assert barred == [hard["probe"]["barred_update"]]                    # only the [UPDATE] brief is barred
    arc = next(c for c in cases if c["case_id"].startswith("hypo_05"))
    assert [b["eligible"] for b in arc["briefs"]] == [True, False, True, True, True]


def test_score_pick_and_summary():
    from benchmarks.editorial_decisions.run_lead_pick import format_summary, score_pick, summarize

    briefs, _, _ = parse_briefs(DOC)  # brief 1 is [UPDATE], so barred
    case = {"briefs": briefs}
    label = {"lead_index": 2, "acceptable_indexes": [3]}
    assert score_pick(2, case, label) == {"unparsed": False, "exact": True, "acceptable": True, "barred": False}
    assert score_pick(3, case, label) == {"unparsed": False, "exact": False, "acceptable": True, "barred": False}
    assert score_pick(1, case, label) == {"unparsed": False, "exact": False, "acceptable": False, "barred": True}
    assert score_pick(None, case, label) == {"unparsed": True, "exact": False, "acceptable": False, "barred": False}
    # 'none deserved': no lead, so nothing can score as exact or acceptable
    assert score_pick(2, case, {"lead_index": None, "acceptable_indexes": []})["acceptable"] is False

    rows = [dict(score_pick(p, case, label), mode="heavy", model="m", synthetic=s)
            for p, s in [(2, False), (3, False), (1, True)]]
    summary = summarize(rows)
    assert [(g["set"], g["n"], g["exact"], g["acceptable"], g["barred"]) for g in summary] == [
        ("invented", 1, 0, 0, 1), ("real", 2, 1, 2, 0)]
    assert summarize([dict(rows[0], set="invented_hard")])[0]["set"] == "invented_hard"
    assert "acceptable" in format_summary(summary)


def test_parse_pick_prefers_answer_marker_then_first_line_only():
    from benchmarks.editorial_decisions.run_lead_pick import parse_pick

    briefs, _, _ = parse_briefs(DOC)
    assert parse_pick("AI labs ask for a slowdown is big.\n\nAnswer: Fed holds rates steady", briefs) == (3, "Fed holds rates steady", "answer")
    assert parse_pick("**Headline: Fed holds rates steady**\n\nBecause rates.", briefs)[::2] == (3, "opening")
    assert parse_pick("## Selected Story\n\n**Fed holds rates steady**\n\nReasoning: rates.", briefs)[::2] == (3, "opening")
    assert parse_pick("## Decision\n\n**Selected headline:** Fed holds rates steady\n", briefs)[::2] == (3, "opening")
    # a headline mentioned after the opening must not count as the pick
    assert parse_pick("Here is my reasoning.\nIt is long.\nVery long.\nFed holds rates steady was considered.", briefs) == (None, None, "none")


def test_score_pair_counts_shared_stories_regardless_of_slot():
    from benchmarks.editorial_decisions.run_second_pick import score_pair, summarize

    label = {"lead_index": 2, "second_index": 3}
    assert score_pair(2, 3, label) == {"second_exact": True, "pair_overlap": 2, "pair_size": 2}
    assert score_pair(3, 2, label) == {"second_exact": False, "pair_overlap": 2, "pair_size": 2}  # slots swapped
    assert score_pair(2, 1, label) == {"second_exact": False, "pair_overlap": 1, "pair_size": 2}
    assert score_pair(1, None, label) == {"second_exact": False, "pair_overlap": 0, "pair_size": 2}
    # labeler said nothing deserved the second slot
    assert score_pair(2, 1, {"lead_index": 2, "second_index": None}) == {"second_exact": False, "pair_overlap": 1, "pair_size": 1}

    rows = [dict(score_pair(2, 3, label), mode="heavy", synthetic=True, second_pick_index=3),
            dict(score_pair(2, 1, label), mode="heavy", synthetic=True, second_pick_index=1)]
    g = summarize(rows)[0]
    assert (g["n"], g["second_exact"], g["pair_overlap"], g["pair_size"], g["both"]) == (2, 1, 3, 4, 1)


def test_match_pick_handles_paraphrase_but_not_vague_answers():
    briefs = [{"index": 1, "headline": "CDC counts the most measles cases in 30 years"},
              {"index": 2, "headline": "Rising Winter Heating Costs: A new forecast says families will pay more"},
              {"index": 3, "headline": "Wholesale inflation comes in hotter than expected"},
              {"index": 4, "headline": "Rising Winter Heating Costs: A new forecast says families will pay more"}]
    assert match_pick("CDC reports the most U.S. measles cases in 30 years (2,140 cases across 31 states)", briefs) == 1
    assert match_pick("Rising Winter Heating Costs - bills will rise 8.7% this winter", briefs) == 2  # duplicate: lower index
    assert match_pick("Wholesale inflation (Producer Price Index) rose 0.5%, hotter than the 0.2% forecast", briefs) == 3
    assert match_pick("An economic story about prices this winter", briefs) is None


def test_story_key_merges_duplicate_briefs_for_scoring():
    from benchmarks.editorial_decisions.run_lead_pick import score_pick, story_key
    from benchmarks.editorial_decisions.run_second_pick import score_pair

    case = {"case_id": "x", "briefs": [
        {"index": 1, "headline": "Fed holds rates", "eligible": True},
        {"index": 2, "headline": "Heating costs rise", "eligible": True},
        {"index": 3, "headline": "**Heating costs rise**", "eligible": True}]}
    assert story_key(case) == {1: 1, 2: 2, 3: 2}
    label = {"lead_index": 3, "acceptable_indexes": [], "second_index": 1}
    assert score_pick(2, case, label)["exact"] is True          # the other copy of the same story
    assert score_pair(1, 2, label, story_key(case))["pair_overlap"] == 2


def test_candidate_prompts_apply_cleanly_to_the_production_text():
    from benchmarks.editorial_decisions.candidate_prompts import prompts_for

    lead, second = prompts_for("candidate")
    old_lead, old_second = prompts_for("production")
    assert "changed in kind" in lead and "LENS 4" in lead and "with care" not in lead
    assert lead.count("prefer the one with broader implications") == 1   # the duplicated tie-break is gone
    assert "non-US country" not in second and "directly involved or directly affected" in second
    assert "{excluded_headline}" in second
    assert "with care" in old_lead and "non-US country" in old_second    # production untouched


def test_match_pick_falls_back_to_brief_text_for_a_restated_answer():
    briefs = [{"index": 1, "headline": "Judge orders the breakup of the country's largest ticket seller",
               "brief": "A federal judge ordered the company to sell its ticketing arm within 18 months and capped service fees at 15 percent."},
              {"index": 2, "headline": "Interior Department imposes Colorado River cuts",
               "brief": "Arizona loses 27 percent of its allocation and California 10 percent."}]
    answer = "A federal judge requires the ticketing giant to sell its ticketing arm within 18 months, capping service fees at 15%"
    assert match_pick(answer, briefs) == 1
    assert match_pick("A story about fees, cuts, and a judge in Arizona this year", briefs) is None


def test_parse_pick_reads_a_brief_number_only_when_the_line_announces_the_pick():
    from benchmarks.editorial_decisions.run_lead_pick import parse_pick, score_pick

    briefs, _, _ = parse_briefs(DOC)
    assert parse_pick("The most important story is Brief 3.\n\nBecause rates.", briefs)[::2] == (3, "brief_number")
    assert parse_pick("Nothing here meets the bar.\nBrief 3 describes rates and Brief 2 is about AI.", briefs) == (None, None, "none")
    # either labeling pass's lead counts as exact when alternates are supplied
    label = {"lead_index": 2, "acceptable_indexes": [3], "lead_alternates": [3]}
    assert score_pick(3, {"briefs": briefs}, label)["exact"] is True
