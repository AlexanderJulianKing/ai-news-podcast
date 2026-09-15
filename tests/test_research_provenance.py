"""Wire provenance and degraded-research detection for Tier 2/3.

Background (2026-07-22 in production): the morning after the joint OpenAI/Hugging
Face statement, all 12 Tier-2 briefs came back UNVERIFIED and the judge called the
hack "speculative ... may be hypothetical". A headline carried by AP and NPR is a
reported fact; a one-shot web brief failing to add detail is not evidence against
it, and twelve failures at once is a research outage, not twelve false stories.
"""

from newscaster.prompts import (
    RESEARCH_DEGRADED_NOTICE,
    TIER3_EVERYMAN_STORY_PROMPT,
    TIER3_IMPORTANT_STORY_PROMPT,
)
from newscaster.scrapers.topic_finder import (
    _format_research_briefs,
    attribute_headline_sources,
    brief_is_unverified,
    build_source_index,
    research_degraded,
)


# --- UNVERIFIED detection ----------------------------------------------------

def test_unverified_marker_detected_with_and_without_markdown():
    assert brief_is_unverified("UNVERIFIED: Web research did not return a usable brief.")
    assert brief_is_unverified("**UNVERIFIED:** could not confirm")
    assert brief_is_unverified("  unverified - nothing found")
    assert not brief_is_unverified("The story is UNVERIFIED by some outlets but confirmed by AP.")
    assert not brief_is_unverified("")
    assert not brief_is_unverified(None)


def test_research_degraded_needs_both_volume_and_share():
    unv = ("h", "UNVERIFIED: x")
    ok = ("h", "AP reports that ...")
    # 12/12, the July 22 morning.
    assert research_degraded([unv] * 12, min_briefs=4, fraction=0.75) == (True, 12, 12)
    # 3 of 13: normal noise.
    assert research_degraded([unv] * 3 + [ok] * 10, min_briefs=4, fraction=0.75) == (False, 3, 13)
    # 10 of 13 = 0.77: over the line.
    assert research_degraded([unv] * 10 + [ok] * 3, min_briefs=4, fraction=0.75)[0] is True
    # Too few briefs to call it an outage even at 100%.
    assert research_degraded([unv] * 3, min_briefs=4, fraction=0.75) == (False, 3, 3)
    assert research_degraded([], min_briefs=4, fraction=0.75) == (False, 0, 0)


def test_research_degraded_reads_config_defaults(monkeypatch):
    from newscaster import config as cfg
    monkeypatch.setattr(cfg, "RESEARCH_DEGRADED_MIN_BRIEFS", 2)
    monkeypatch.setattr(cfg, "RESEARCH_DEGRADED_UNVERIFIED_FRACTION", 0.5)
    assert research_degraded([("h", "UNVERIFIED: a"), ("h", "fine")])[0] is True


# --- provenance --------------------------------------------------------------

POOL = [
    ("NPR", "*   **AI Safety and Regulation:** Industry leaders are calling for a temporary slowdown "
            "in the development of advanced artificial intelligence.\n"
            "*   **Rising Winter Heating Costs:** A new forecast indicates American families should "
            "prepare to spend significantly more to heat their homes this winter."),
    ("The Associated Press", "*   **Trade:** \"US banning dairy products, most alcohol and motorcycles from Canada in growing trade war\"\n"
                             "*   **Politics:** \"Trump calls on Ukraine to halt strikes on Russian diesel fuel\""),
    ("ProPublica", "How the Trump administration used keyword screening to cut research funding"),
]


def test_verbatim_headline_is_attributed_to_its_source_only():
    idx = build_source_index(POOL)
    assert attribute_headline_sources(
        "US banning dairy products, most alcohol and motorcycles from Canada in growing trade war", idx
    ) == ["The Associated Press"]


def test_tier1_fragment_still_matches_the_longer_pool_line():
    idx = build_source_index(POOL)
    # Tier 1 often keeps the label and a truncated description.
    assert attribute_headline_sources(
        "AI Safety and Regulation: Industry leaders are calling for a temporary slowdown", idx
    ) == ["NPR"]


def test_arc_tag_does_not_break_attribution():
    idx = build_source_index(POOL)
    assert attribute_headline_sources(
        "[SIDE-COVERED: ai_safety_cyberattacks] AI Safety and Regulation: Industry leaders are calling "
        "for a temporary slowdown in the development of advanced artificial intelligence.", idx
    ) == ["NPR"]


def test_unrelated_or_tiny_headlines_get_no_attribution():
    idx = build_source_index(POOL)
    assert attribute_headline_sources("Local bakery wins pie contest in Riverside county fair", idx) == []
    assert attribute_headline_sources("Heating costs", idx) == []


def test_same_story_on_two_front_pages_lists_both_in_pool_order():
    pool = POOL + [("Democracy Now", "Trump calls on Ukraine to halt strikes on Russian diesel fuel, sources say")]
    idx = build_source_index(pool)
    assert attribute_headline_sources("Trump calls on Ukraine to halt strikes on Russian diesel fuel", idx) == [
        "The Associated Press", "Democracy Now",
    ]


# --- document assembly -------------------------------------------------------

def test_brief_document_carries_reported_by_when_sources_known():
    doc = _format_research_briefs([
        ("Headline A", "Brief A", ["NPR", "The Associated Press"]),
        ("Headline B", "Brief B", []),
        ("Headline C", "Brief C"),          # legacy 2-tuple still accepted
    ])
    assert "--- Brief 1 ---\nHeadline: Headline A\nReported by: NPR, The Associated Press\n\nBrief A" in doc
    assert "--- Brief 2 ---\nHeadline: Headline B\n\nBrief B" in doc
    assert "--- Brief 3 ---\nHeadline: Headline C\n\nBrief C" in doc


def test_prompts_tell_tier3_how_to_read_provenance_and_notice():
    for prompt in (TIER3_IMPORTANT_STORY_PROMPT, TIER3_EVERYMAN_STORY_PROMPT):
        assert "PROVENANCE" in prompt
        assert "Reported by:" in prompt
        assert "RESEARCH NOTICE" in prompt
    notice = RESEARCH_DEGRADED_NOTICE.format(n_unverified=12, n_total=12)
    assert notice.startswith("RESEARCH NOTICE: 12 of 12 briefs")
    assert "Do not penalize UNVERIFIED today" in notice
