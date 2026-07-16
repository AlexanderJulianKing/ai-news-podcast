from newscaster.dedup import build_headline_arc_map
from newscaster.prompts import LEDGER_REPETITION_REMOVER_TEMPLATE, REPETITION_REMOVER_TEMPLATE
from newscaster.scrapers.topic_finder import (
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


def test_major_escalation_prompt_sets_a_qualitative_threshold():
    for prompt in (LEDGER_REPETITION_REMOVER_TEMPLATE, REPETITION_REMOVER_TEMPLATE):
        assert "MAJOR ESCALATION — RARE" in prompt
        assert "Another round of strikes" in prompt
        assert "renewed or enforced blockades" in prompt
        assert "congressional notifications" in prompt
        assert "not merely its intensity, scale, or latest details" in prompt
        assert "When uncertain, classify it as UPDATE" in prompt
