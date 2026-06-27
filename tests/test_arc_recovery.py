"""Tests for arc-identity recovery: joining a chosen (possibly rephrased,
de-tagged) headline back to the slug the dedup tagger assigned it.

Background: the dedup tagger reliably prepends [UPDATE: slug] / [MAJOR ESCALATION:
slug] to recurring headlines, but the Tier-3 selection prompts strip that prefix,
so find_matching_arc() (which only parses the prefix) never recovers the slug and
every story spawns a fresh single-episode arc. These helpers rebuild the
{clean headline -> (tag, slug)} association from the tagger's own output and
recover it for a chosen headline, so arcs accumulate episodes again.
"""

from newscaster.dedup import build_headline_arc_map, recover_arc_for_headline


# A realistic slice of dedup-tagger output (June 27 production): some lines carry
# slugs, some are plain section headers / untagged new stories.
TAGGED_TEXT = """NPR:
[MAJOR ESCALATION: hormuz_strait_tensions] US strikes Iran in response to a drone attack on a ship.
[UPDATE: fed_chair_inflation] Inflation hits 3-year high, official says.
[UPDATE: ukraine_russia_strikes] Ukraine unleashes one of its heaviest drone bombardments of Russia.
[UPDATE: scotus_immigration_ruling] Experts Warn of Population Decline Following SCOTUS Immigration Ruling
A Trump commission urges 'bridges' between church and state in sweeping draft report
The Associated Press:
"""


def test_build_map_keeps_only_tagged_lines():
    m = build_headline_arc_map(TAGGED_TEXT)
    # Four tagged lines -> four entries; untagged headline and section headers excluded.
    assert len(m) == 4
    assert ("MAJOR ESCALATION", "hormuz_strait_tensions") in m.values()
    assert ("UPDATE", "fed_chair_inflation") in m.values()


def test_build_map_handles_empty_or_untagged_text():
    assert build_headline_arc_map("") == {}
    assert build_headline_arc_map("Just a plain headline\nAnother one") == {}


def test_recover_exact_match_modulo_case_and_punctuation():
    m = build_headline_arc_map(TAGGED_TEXT)
    # The overview picker echoed this headline without the trailing period.
    assert recover_arc_for_headline(
        "Inflation hits 3-year high, official says", m
    ) == ("UPDATE", "fed_chair_inflation")
    # Case differences must not defeat the match.
    assert recover_arc_for_headline(
        "experts warn of population decline following scotus immigration ruling", m
    ) == ("UPDATE", "scotus_immigration_ruling")


def test_recover_verbatim_main_pick():
    m = build_headline_arc_map(TAGGED_TEXT)
    assert recover_arc_for_headline(
        "US strikes Iran in response to a drone attack on a ship.", m
    ) == ("MAJOR ESCALATION", "hormuz_strait_tensions")


def test_recover_clear_subset_rephrase():
    m = build_headline_arc_map(TAGGED_TEXT)
    # A shortened rephrase that is clearly the same story (all words a subset).
    assert recover_arc_for_headline(
        "Ukraine unleashes heaviest drone bombardments of Russia", m
    ) == ("UPDATE", "ukraine_russia_strikes")


def test_recover_returns_none_for_unrelated_headline():
    m = build_headline_arc_map(TAGGED_TEXT)
    assert recover_arc_for_headline(
        "Local school board approves new cafeteria menu", m
    ) is None


def test_recover_returns_none_for_heavy_paraphrase():
    # Documents the known limitation: a heavily reworded headline for the same
    # real-world event degrades to "no match" (a fresh arc), which is no worse
    # than today's behavior — never a WRONG match (which would inject the wrong
    # audience_state).
    m = build_headline_arc_map(
        "[UPDATE: venezuela_earthquake_aftermath] Venezuelans take search for the "
        "missing into their own hands as earthquake death toll climbs\n"
    )
    assert recover_arc_for_headline(
        "Thousands still missing in Venezuela following magnitude 7.2 earthquake", m
    ) is None


def test_recover_with_empty_map_is_none():
    assert recover_arc_for_headline("anything", {}) is None
