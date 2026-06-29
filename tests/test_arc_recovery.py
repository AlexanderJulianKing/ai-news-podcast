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


# --- Real production tagger output: Gemma wraps tags in markdown (bullets,
# headings, bold, numbers), so the tag is NEVER at the literal start of a line.
# These lines are copied from the 2026-06-28 llm audit log.
REAL_GEMMA_OUTPUT = """Here are the de-duplicated headlines:

NPR:
*   **[MAJOR ESCALATION: us_iran_strikes_4] Conflict in the Middle East:** US airstrikes again hit Iran as Tehran strikes Bahrain and Kuwait, further imperiling the interim deal.
*   **[UPDATE: europe_heat_wave_2] Climate & Weather:** Central Europe sizzles as heat records are smashed in Switzerland, Denmark, and the Czech Republic.
*   **FIFA World Cup:** Egypt advances past the group stage after a 1-1 draw with Iran.
*   **[UPDATE: inflation_record_high] Economy:** A key inflation gauge surges to a 3-year high, and mortgage rates continue to climb.

The Associated Press:
### **2. [UPDATE: venezuela_earthquake_2] Desperate Search for Survivors in Venezuela**
"""


def test_build_map_parses_markdown_wrapped_tags():
    m = build_headline_arc_map(REAL_GEMMA_OUTPUT)
    # Four tagged lines; the untagged FIFA bullet and section headers excluded.
    assert ("MAJOR ESCALATION", "us_iran_strikes_4") in m.values()
    assert ("UPDATE", "europe_heat_wave_2") in m.values()
    assert ("UPDATE", "inflation_record_high") in m.values()
    assert ("UPDATE", "venezuela_earthquake_2") in m.values()
    assert len(m) == 4


def test_recover_real_main_pick_from_markdown_output():
    # The exact 2026-06-28 lead headline the picker chose, de-tagged.
    m = build_headline_arc_map(REAL_GEMMA_OUTPUT)
    chosen = ("Conflict in the Middle East: US airstrikes again hit Iran as Tehran "
              "strikes Bahrain and Kuwait, further imperiling the interim deal.")
    assert recover_arc_for_headline(chosen, m) == ("MAJOR ESCALATION", "us_iran_strikes_4")


def test_untagged_markdown_headline_is_not_swallowed_into_neighbor():
    # The untagged FIFA bullet sits between two tagged lines; it must NOT be folded
    # into the europe_heat_wave_2 key, and an Egypt/FIFA headline must not match it.
    m = build_headline_arc_map(REAL_GEMMA_OUTPUT)
    assert recover_arc_for_headline("Egypt advances past the group stage at the World Cup", m) is None


def test_build_map_splits_multiple_tags_on_one_line():
    # Gemma sometimes crams several tagged items onto one line (semicolon-joined).
    one_line = (
        "World roundup: [UPDATE: alpha_story] Alpha vote passes the House; "
        "[MAJOR ESCALATION: bravo_story] Bravo conflict erupts into open war"
    )
    m = build_headline_arc_map(one_line)
    assert recover_arc_for_headline("Alpha vote passes the House", m) == ("UPDATE", "alpha_story")
    assert recover_arc_for_headline("Bravo conflict erupts into open war", m) == ("MAJOR ESCALATION", "bravo_story")
