"""A story that led can lead again on a big enough development after a cooldown.

Background: over Aug 1 - Sep 15 the top-scored Tier 1 story was [UPDATE]-barred on
21 of 46 mornings, 13 of them a live war that had led once and then scored 8 or 9
while the show led with a 5, 6 or 7. The hard bar becomes a two-day cooldown plus
a judged DEVELOPMENT tag; consecutive days still need a MAJOR ESCALATION.
"""

from datetime import date

from newscaster.dedup import (
    DEVELOPMENT_TAG,
    apply_coverage_depth,
    build_headline_arc_map,
    format_coverage_notes,
    last_main_date,
    strip_arc_tags,
)
from newscaster.prompts import (
    TIER1_TRIAGE_PROMPT,
    TIER3_EVERYMAN_STORY_PROMPT,
    TIER3_IMPORTANT_STORY_PROMPT,
    TIER3_OVERVIEW_PICK_PROMPT,
)
from newscaster.scrapers import topic_finder as tf
from newscaster.scrapers.watchlist import WATCHLIST_EVENT_TEST_PROMPT

LEDGER = {"arcs": {"us_iran_escalation_2": {
    "audience_state": "The US and Iran have been exchanging strikes since late June; oil is above one hundred dollars.",
    "episodes": [
        {"date": "2026_08_16", "coverage": "main"},
        {"date": "2026_08_31", "coverage": "main"},
        {"date": "2026_09_02", "coverage": "side"},
        {"date": "2026_09_04", "coverage": "side"},
    ],
}}}
LINE = "[UPDATE: us_iran_escalation_2] Ansarallah completes takeover of Yemen's Red Sea coast; oil tops $108"


def test_last_main_date_is_the_most_recent_full_segment():
    assert last_main_date(LEDGER["arcs"]["us_iran_escalation_2"]) == date(2026, 8, 31)
    assert last_main_date({"episodes": [{"date": "2026_09_01", "coverage": "side"}]}) is None
    assert last_main_date({}) is None


def test_recovery_window_boundary():
    # Led 08_31. One day later: still barred. Two days later: DEVELOPMENT.
    text, counts = apply_coverage_depth(LINE, LEDGER, today=date(2026, 9, 1), recovery_days=2)
    assert text.startswith("[UPDATE: us_iran_escalation_2]") and counts["development"] == 0
    text, counts = apply_coverage_depth(LINE, LEDGER, today=date(2026, 9, 2), recovery_days=2)
    assert text.startswith("[DEVELOPMENT: us_iran_escalation_2]") and counts["development"] == 1
    # The Sep 6 morning that led with a 5 over a 9.
    text, _ = apply_coverage_depth(LINE, LEDGER, today=date(2026, 9, 6), recovery_days=2)
    assert text.startswith("[DEVELOPMENT: us_iran_escalation_2]")


def test_recovery_days_default_comes_from_config(monkeypatch):
    from newscaster import config as cfg
    monkeypatch.setattr(cfg, "MAIN_RECOVERY_DAYS", 10)
    text, _ = apply_coverage_depth(LINE, LEDGER, today=date(2026, 9, 6))
    assert text.startswith("[UPDATE:")


def test_development_note_says_when_it_led_and_what_the_audience_knows():
    text, _ = apply_coverage_depth(LINE, LEDGER, today=date(2026, 9, 6), recovery_days=2)
    notes = format_coverage_notes(build_headline_arc_map(text), LEDGER, today=date(2026, 9, 6))
    assert notes.startswith("- us_iran_escalation_2: last led 2026_08_31 (6 days ago), 2 roundup mentions since.")
    assert "Audience already knows: The US and Iran have been exchanging strikes" in notes


def test_development_tag_flows_through_parsers():
    tagged = f"[{DEVELOPMENT_TAG}: us_iran_escalation_2] Oil tops $108"
    assert strip_arc_tags(tagged) == "Oil tops $108"
    assert build_headline_arc_map(tagged) == {"oil tops 108": (DEVELOPMENT_TAG, "us_iran_escalation_2")}


def test_prompts_carry_the_development_rule_and_lens_3():
    for prompt in (TIER3_IMPORTANT_STORY_PROMPT, TIER3_EVERYMAN_STORY_PROMPT):
        assert "'[DEVELOPMENT]' stories CAN be selected, with care" in prompt
        assert "'[UPDATE]' stories MUST NOT be selected" in prompt
    assert "LENS 3 — FRONTIER CAPABILITY AND CONTROL" in TIER3_IMPORTANT_STORY_PROMPT
    assert "a company's own announcement or a benchmark score does not pass" in TIER3_IMPORTANT_STORY_PROMPT
    assert "Frontier capability and control" in TIER1_TRIAGE_PROMPT
    assert "'[DEVELOPMENT]' tag prefixes" in TIER3_OVERVIEW_PICK_PROMPT
    assert "a verified first" in WATCHLIST_EVENT_TEST_PROMPT


def test_shortlist_cap_raised_for_the_larger_pool():
    assert tf._MERGED_SHORTLIST_LIMIT == 16


# --- tagger retention guard ------------------------------------------------------

POOL = "\n".join(f"Headline number {i} about a distinct event happening" for i in range(10))


def test_tag_pool_accepts_normal_retention(monkeypatch):
    calls = []
    def fake(default, *args, **kw):
        calls.append(1)
        return "\n".join(POOL.split("\n")[:8])          # 80%: two same-story removals
    monkeypatch.setattr(tf, "call_with_default", fake)
    out = tf._tag_pool(POOL, "sys", "label")
    assert len(calls) == 1 and len(tf._pool_lines(out)) == 8


def test_tag_pool_retries_then_merges_lost_lines_back(monkeypatch):
    calls = []
    def fake(default, *args, **kw):
        calls.append(1)
        return "[UPDATE: x] " + POOL.split("\n")[0] + "\n" + POOL.split("\n")[1]   # 20% kept
    monkeypatch.setattr(tf, "call_with_default", fake)
    monkeypatch.setattr(tf._config, "TAGGER_MIN_RETENTION", 0.7, raising=False)
    out = tf._tag_pool(POOL, "sys", "label")
    assert len(calls) == 2
    lines = tf._pool_lines(out)
    assert len(lines) == 10                                  # nothing discarded
    assert lines[0].startswith("[UPDATE: x]")                # the tags it did produce survive
    assert "Headline number 9 about a distinct event happening" in out
