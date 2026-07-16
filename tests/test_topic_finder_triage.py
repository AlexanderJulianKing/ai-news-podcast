from datetime import date

from newscaster.dedup import build_headline_arc_map
from newscaster.scrapers.topic_finder import (
    _apply_main_story_arc_cooldown,
    _merge_shortlists,
    _parse_tier1_scores,
    _restore_triage_arc_tags,
)


def test_merge_shortlists_preserves_national_then_adds_california():
    national = [f"National story {i}" for i in range(1, 11)]
    california = [
        "National story 2",
        "California housing insurance deadline",
        "California wildfire evacuation order",
        "California water allocation ruling",
        "California port labor dispute",
    ]

    merged = _merge_shortlists(national, california, limit=13)

    assert merged[:10] == national
    assert merged[10:] == [
        "California housing insurance deadline",
        "California wildfire evacuation order",
        "California water allocation ruling",
    ]


def test_merge_shortlists_dedupes_arc_tags_and_punctuation():
    national = ["[UPDATE: abc] California DMV data sharing"]
    california = ["California DMV data-sharing", "Another California story"]

    merged = _merge_shortlists(national, california, limit=3)

    assert merged == ["[UPDATE: abc] California DMV data sharing", "Another California story"]


def test_parse_tier1_scores_sorts_highest_first():
    response = """
SCORE: 5 | HEADLINE: Lower story | REASON: smaller
SCORE: 9 | HEADLINE: Higher story | REASON: bigger
"""

    parsed = _parse_tier1_scores(response)

    assert [item["headline"] for item in parsed] == ["Higher story", "Lower story"]


def test_restore_triage_arc_tags_after_model_strips_them():
    iran_headline = (
        "Escalating Conflict between the U.S. and Iran: The United States conducted "
        "airstrikes against Iranian military targets for the third consecutive weekend."
    )
    arc_map = build_headline_arc_map(
        f"[UPDATE: us_iran_escalation_2] {iran_headline}\n"
        "A genuinely new story"
    )
    scored = [
        {"score": 9, "headline": iran_headline, "reason": "Large global impact."},
        {"score": 7, "headline": "A genuinely new story", "reason": "Fresh news."},
    ]

    restored = _restore_triage_arc_tags(scored, arc_map)

    assert restored == [
        {
            "score": 9,
            "headline": f"[UPDATE: us_iran_escalation_2] {iran_headline}",
            "reason": "Large global impact.",
        },
        {"score": 7, "headline": "A genuinely new story", "reason": "Fresh news."},
    ]
    assert scored[0]["headline"] == iran_headline


def test_restore_triage_arc_tags_preserves_major_escalation_verdict():
    headline = "US strikes Iran after a new blockade in the Strait of Hormuz"
    tagged = f"[MAJOR ESCALATION: us_iran_escalation_2] {headline}"
    arc_map = build_headline_arc_map(tagged)

    restored = _restore_triage_arc_tags(
        [{"score": 9, "headline": tagged, "reason": "Major escalation."}],
        arc_map,
    )

    assert restored[0]["headline"] == tagged


def test_main_story_arc_cooldown_downgrades_recent_major_escalations():
    text = (
        "[MAJOR ESCALATION: us_iran_escalation_2] U.S. restarts its blockade\n"
        "[MAJOR ESCALATION: us_iran_escalation_2] U.S. expands its strikes\n"
        "[UPDATE: us_iran_escalation_2] Congress debates war powers\n"
        "A genuinely new story"
    )
    ledger = {
        "arcs": {
            "us_iran_escalation_2": {
                "episodes": [
                    {"date": "2026-07-14", "coverage": "main"},
                ]
            }
        }
    }

    cooled, count, slugs = _apply_main_story_arc_cooldown(
        text,
        ledger,
        date(2026, 7, 16),
    )

    assert "[MAJOR ESCALATION: us_iran_escalation_2]" not in cooled
    assert cooled.count("[UPDATE: us_iran_escalation_2]") == 3
    assert count == 2
    assert slugs == ["us_iran_escalation_2"]


def test_main_story_arc_cooldown_allows_arc_after_window_expires():
    text = "[MAJOR ESCALATION: us_iran_escalation_2] A truly new phase begins"
    ledger = {
        "arcs": {
            "us_iran_escalation_2": {
                "episodes": [
                    {"date": "2026-07-12", "coverage": "main"},
                    {"date": "2026-07-15", "coverage": "side"},
                ]
            }
        }
    }

    cooled, count, slugs = _apply_main_story_arc_cooldown(
        text,
        ledger,
        date(2026, 7, 16),
    )

    assert cooled == text
    assert count == 0
    assert slugs == []
