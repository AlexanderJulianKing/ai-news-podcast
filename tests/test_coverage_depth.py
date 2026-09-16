"""Coverage depth: a story the audience only met in the side-story roundup must not
be barred from the main slot by the [UPDATE] rule.

Background (2026-07-22/24 and 2026-09-12..15 in production): a side-story mention
created a ledger arc, the tagger then marked every later development [UPDATE], and
the Tier-3 prompt says [UPDATE] MUST NOT be selected. One sentence of roundup airtime
locked the OpenAI/Hugging Face incident and the AI-slowdown debate out of a full
segment for good. On 2026-09-14 the judge wrote that the highest-stakes briefs were
"ineligible for selection, regardless of their significance".
"""

from datetime import date

from newscaster.dedup import (
    DEVELOPMENT_TAG,
    SIDE_COVERED_TAG,
    apply_coverage_depth,
    arc_has_main_coverage,
    build_headline_arc_map,
    find_matching_arc,
    format_coverage_notes,
    recover_arc_for_headline,
    strip_arc_tags,
)
from newscaster.prompts import (
    SEGMENT_SCRIPT_SIDE_COVERED_CONTEXT,
    SEGMENT_SCRIPT_UPDATE_CONTEXT,
    TIER1_TRIAGE_PROMPT,
    TIER3_EVERYMAN_STORY_PROMPT,
    TIER3_IMPORTANT_STORY_PROMPT,
    TIER3_OVERVIEW_PICK_PROMPT,
)
from newscaster.scrapers.topic_finder import _restore_triage_arc_tags
from newscaster.script.segments import continuation_context_template


def _ledger():
    return {
        "arcs": {
            # The real September sequence: three roundup mentions, never a segment.
            "ai_safety_cyberattacks": {
                "episodes": [
                    {"date": "2026_09_12", "coverage": "side", "coverage_slot": 2},
                    {"date": "2026_09_13", "coverage": "side", "coverage_slot": 1},
                    {"date": "2026_09_14", "coverage": "side", "coverage_slot": 0},
                ],
            },
            # A story that did get a full segment.
            "us_iran_escalation_2": {
                "episodes": [
                    {"date": "2026_09_01", "coverage": "main", "coverage_slot": 0},
                    {"date": "2026_09_10", "coverage": "side", "coverage_slot": 3},
                ],
            },
        }
    }


TAGGED = (
    "NPR:\n"
    "[UPDATE: ai_safety_cyberattacks] Trump downplays need to regulate AI after CEOs urge slowdown\n"
    "[UPDATE: us_iran_escalation_2] Iranian officials postpone Gulf States meeting\n"
    "[MAJOR ESCALATION: ai_safety_cyberattacks] Congress passes AI moratorium\n"
    "[UPDATE: never_seen_slug] Some story the tagger invented an arc for\n"
    "A brand new story with no tag\n"
)


# --- the rewrite itself ------------------------------------------------------

# Iran led on 2026_09_01; on 09_02 that is one day ago, inside the recovery window.
DAY_AFTER_IRAN_LED = date(2026, 9, 2)
NO_REWRITES = {"side_covered": 0, "development": 0}


def test_side_only_arc_is_downgraded_and_recently_led_arc_is_not():
    text, counts = apply_coverage_depth(TAGGED, _ledger(), today=DAY_AFTER_IRAN_LED)
    assert counts == {"side_covered": 1, "development": 0}
    assert "[SIDE-COVERED: ai_safety_cyberattacks] Trump downplays" in text
    assert "[UPDATE: us_iran_escalation_2] Iranian officials" in text


def test_major_escalation_and_unknown_slugs_are_left_alone():
    text, _ = apply_coverage_depth(TAGGED, _ledger(), today=DAY_AFTER_IRAN_LED)
    assert "[MAJOR ESCALATION: ai_safety_cyberattacks] Congress passes" in text
    assert "[UPDATE: never_seen_slug] Some story" in text
    assert "A brand new story with no tag" in text


def test_empty_inputs_are_safe():
    assert apply_coverage_depth("", _ledger(), today=DAY_AFTER_IRAN_LED) == ("", NO_REWRITES)
    assert apply_coverage_depth(TAGGED, {}, today=DAY_AFTER_IRAN_LED) == (TAGGED, NO_REWRITES)
    assert apply_coverage_depth(TAGGED, None, today=DAY_AFTER_IRAN_LED) == (TAGGED, NO_REWRITES)


def test_arc_has_main_coverage():
    assert arc_has_main_coverage(_ledger()["arcs"]["us_iran_escalation_2"])
    assert not arc_has_main_coverage(_ledger()["arcs"]["ai_safety_cyberattacks"])
    assert not arc_has_main_coverage({})
    assert not arc_has_main_coverage(None)


# --- the new tag flows through every parser ----------------------------------

def test_parsers_accept_the_side_covered_tag():
    tagged = f"[{SIDE_COVERED_TAG}: ai_safety_cyberattacks] Trump downplays AI regulation"
    assert find_matching_arc(tagged) == (SIDE_COVERED_TAG, "ai_safety_cyberattacks")
    assert strip_arc_tags(tagged) == "Trump downplays AI regulation"
    assert strip_arc_tags("[SIDE-COVERED] old style") == "old style"


def test_map_and_restore_preserve_the_side_covered_verdict():
    text, _ = apply_coverage_depth(TAGGED, _ledger(), today=DAY_AFTER_IRAN_LED)
    arc_map = build_headline_arc_map(text)
    assert (SIDE_COVERED_TAG, "ai_safety_cyberattacks") in arc_map.values()
    assert ("UPDATE", "us_iran_escalation_2") in arc_map.values()

    # Tier 1 dropped the prefix; restore must put SIDE-COVERED back, not UPDATE.
    stripped = [{"score": 7, "headline": "Trump downplays need to regulate AI after CEOs urge slowdown", "reason": "x"}]
    restored = _restore_triage_arc_tags(stripped, arc_map)
    assert restored[0]["headline"].startswith("[SIDE-COVERED: ai_safety_cyberattacks] ")

    # And a de-tagged pick still resolves to the same arc.
    assert recover_arc_for_headline("Trump downplays need to regulate AI after CEOs urge slowdown", arc_map) == (
        SIDE_COVERED_TAG, "ai_safety_cyberattacks",
    )


# --- persistence signal ------------------------------------------------------

def test_coverage_notes_count_distinct_side_days_for_side_covered_arcs_only():
    text, _ = apply_coverage_depth(TAGGED, _ledger(), today=DAY_AFTER_IRAN_LED)
    notes = format_coverage_notes(build_headline_arc_map(text), _ledger(), today=DAY_AFTER_IRAN_LED)
    assert "ai_safety_cyberattacks: 3 side-story mentions (2026_09_12 to 2026_09_14); never a full segment." in notes
    assert "us_iran_escalation_2" not in notes      # it is an UPDATE, not side-covered


def test_coverage_notes_singular_and_empty():
    ledger = {"arcs": {"one_day": {"episodes": [{"date": "2026_09_14", "coverage": "side"}]}}}
    notes = format_coverage_notes({"k": (SIDE_COVERED_TAG, "one_day")}, ledger, today=date(2026, 9, 15))
    assert notes == "- one_day: 1 side-story mention (2026_09_14); never a full segment."
    assert format_coverage_notes({"k": ("UPDATE", "one_day")}, ledger, today=date(2026, 9, 15)) == ""
    assert format_coverage_notes({}, ledger, today=date(2026, 9, 15)) == ""


# --- the prompts say what the code now does ----------------------------------

def test_tier3_prompts_make_side_covered_selectable_and_update_not():
    for prompt in (TIER3_IMPORTANT_STORY_PROMPT, TIER3_EVERYMAN_STORY_PROMPT):
        assert "'[UPDATE]' stories MUST NOT be selected" in prompt
        assert "'[SIDE-COVERED]' stories CAN be selected" in prompt
        assert "Coverage notes" in prompt


def test_tier1_and_overview_prompts_know_the_third_tag():
    assert "[SIDE-COVERED]" in TIER1_TRIAGE_PROMPT
    assert "'[SIDE-COVERED]'" in TIER3_OVERVIEW_PICK_PROMPT and "tag prefixes" in TIER3_OVERVIEW_PICK_PROMPT


# --- script framing ----------------------------------------------------------

def test_side_covered_main_pick_gets_first_full_telling_framing():
    episodes = [
        {"date": "2026_09_12", "coverage": "side"},
        {"date": "2026_09_13", "coverage": "side"},
        {"date": "2026_09_15", "coverage": "main"},   # today's slot, being written now
    ]
    template, kind = continuation_context_template(episodes, "2026_09_15")
    assert kind == "side-covered"
    assert template is SEGMENT_SCRIPT_SIDE_COVERED_CONTEXT
    assert "first full telling" in template


def test_prior_full_segment_keeps_update_framing():
    episodes = [
        {"date": "2026_09_01", "coverage": "main"},
        {"date": "2026_09_15", "coverage": "main"},
    ]
    template, kind = continuation_context_template(episodes, "2026_09_15")
    assert kind == "update"
    assert template is SEGMENT_SCRIPT_UPDATE_CONTEXT
