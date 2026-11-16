from newscaster.dedup import build_headline_arc_map
from newscaster.prompts import LEDGER_REPETITION_REMOVER_TEMPLATE, REPETITION_REMOVER_TEMPLATE
from newscaster.scrapers.topic_finder import (
    _merge_shortlists,
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


def test_triage_scores_numbered_lines_and_keeps_the_original_wording():
    # 2026-11-15: the model used to retype every headline with its score (~10k output tokens, $0.19 a
    # call) and dropped tags and "[date] (link)" suffixes on the way. Now it returns "N: score" lines.
    from unittest.mock import patch
    from newscaster import prompts
    from newscaster.scrapers import topic_finder
    cuba = "Trump considers military operation against Cuba [Nov 13, 2026 07:56 AM] (https://www.dropsitenews.com/p/x)"
    faa = "[UPDATE: faa_ground_stop] The FAA ended the nationwide ground stop that paralyzed U.S. air travel."
    rain = "Atmospheric rivers brought rain to the Pacific Northwest."
    sent = {}

    def fake(user, system_prompt=None, mode=None, **_):
        sent.update(user=user, mode=mode)
        return "3: 9\n1: 7\n2: 4\n2: 1\n99: 10"            # a repeat and an out-of-range number are ignored
    with patch.object(topic_finder, "get_llm_response", side_effect=fake):
        scored = topic_finder._triage(f"NPR:\n{faa}\n{rain}\nDrop Site News:\n{cuba}", prompts.TIER1_TRIAGE_PROMPT)
    assert sent["mode"] == "triage" and f"1. {faa}" in sent["user"]          # section headers get no number
    assert sent["user"].startswith("NPR:\n1. ") and "Drop Site News:\n3. " in sent["user"]
    assert [(s["score"], s["headline"]) for s in scored] == [(9, cuba), (7, faa), (4, rain)]


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
        assert "restart a war after a durable ceasefire" in prompt
        assert "months without active fighting" in prompt
        assert "recent daily exchanges is an UPDATE" in prompt
        assert "The same action can be a MAJOR ESCALATION or an UPDATE" in prompt
        assert "Recency and the audience's last known state matter" in prompt


def test_restore_triage_arc_tags_preserves_side_covered_verdict():
    arc_map = build_headline_arc_map(
        "[SIDE-COVERED: ai_safety_cyberattacks] Trump downplays need to regulate AI\n"
    )
    scored = [{"score": 7, "headline": "Trump downplays need to regulate AI", "reason": "r"}]
    restored = _restore_triage_arc_tags(scored, arc_map)
    assert restored[0]["headline"] == "[SIDE-COVERED: ai_safety_cyberattacks] Trump downplays need to regulate AI"


def test_merge_shortlists_collapses_same_arc_and_rephrasings():
    national = [
        "[SIDE-COVERED: trump_voting_lawsuit] The Supreme Court rejected the Trump administration's bid to impose new mail-ballot rules",
        "[SIDE-COVERED: trump_voting_lawsuit] Supreme Court justices blocked a plan by Donald Trump to restrict mail ballots",
        "NATO jets shot down a drone over Lithuania on September 15, 2026",
        "The Kennedy Center board voted to close most of the performing arts venue on September 15, 2026",
        "The Kennedy Center's board of directors voted to close the arts center during a meeting on Tuesday",
        "[SIDE-COVERED: trump_voting_lawsuit] The Supreme Court upheld a lower court ruling that bars the Postal Service change",
    ]
    merged = _merge_shortlists(national, [], limit=13)
    assert merged == [national[0], national[2], national[3]]


def test_merge_shortlists_keeps_distinct_stories_that_share_a_name():
    national = [
        "Trump calls on Ukraine to halt strikes on Russian diesel fuel",
        "Trump hits the campaign trail to rally GOP candidates for the midterms",
        "Trump renames Lake Ontario to Lake America on federal maps",
    ]
    assert _merge_shortlists(national, [], limit=13) == national


def test_merge_shortlists_does_not_merge_unrelated_events_that_share_a_date_tail():
    national = [
        "Saudi Arabia shut down its oil pipeline today, September 15, 2026",
        "Senator Mitch McConnell returned to the Senate today, September 15, 2026",
        "The Kennedy Center board voted to close the venue today, September 15, 2026",
    ]
    assert _merge_shortlists(national, [], limit=13) == national
