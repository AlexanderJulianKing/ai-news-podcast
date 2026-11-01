"""The arc-match LLM fallback may recover the tagger's verdict for a reworded headline,
but it must never overrule the tagger's verdict on the headline itself.

2026-10-22 evening and 2026-10-24 morning: the tagger judged a PHP-flaw story and a
Russian-influence story NEW, the fallback attached both to ai_safety_cyberattacks, and
both aired as false follow-ups ("update to this morning's story"; a "Note: the update
context describes ... OpenAI's offline training policy" line read on air).
"""

from unittest.mock import patch

from newscaster.dedup import build_headline_arc_map, build_new_headline_keys, resolve_arc_identity
from newscaster.scrapers import topic_finder

SLUG = "ai_safety_cyberattacks"
LEDGER = {"arcs": {SLUG: {
    "slug": SLUG,
    # The real topic: a frozen research brief that mentions cyberattacks and Russian propaganda.
    "topic": "FINDINGS: An Anthropic researcher resigned, warning artificial intelligence could kill us all. "
             "Cyberattacks and misuse: state-sponsored groups tried to use AI for cyberattacks; a swarm of AI "
             "agents broke containment. GAPS: social media propaganda campaigns beyond Russia and Iran.",
    "audience_state": "OpenAI has shifted to training its most advanced AI models without internet access.",
    "episodes": [{"date": "2026_10_23_pm", "coverage": "side", "coverage_slot": 0,
                  "headline": "OpenAI says it is training its most advanced AI models offline."}],
}}}

KREMLIN = ("Kremlin-backed groups renewed tactics to influence online discussions and sow distrust "
           "and discord in U.S. and other elections, researchers said.")
POOL = "\n".join([
    "NPR (front page):",
    "- [DEVELOPMENT: ai_safety_cyberattacks] METR published work on red-teaming OpenAI's internal "
    "agent monitoring systems on October 22, 2026 (via METR)",
    "- [DEVELOPMENT: ai_safety_cyberattacks] Anthropic said some of its AI agents had taken unintended "
    "actions on federal, state and local websites, and Philadelphia police said an Anthropic AI model "
    "submitted a false tip",
    "- " + KREMLIN,
    "- A software vulnerability detected by artificial intelligence could affect wide swaths of the internet. (via AP)",
])


def _resolve(headline):
    arc_map = build_headline_arc_map(POOL)
    keys = build_new_headline_keys(POOL)
    with patch("newscaster.llm.call_with_default", return_value=SLUG) as llm:
        out = resolve_arc_identity(headline, arc_map, LEDGER, new_headline_keys=keys)
    return out, llm


def test_fallback_cannot_overrule_the_taggers_new_verdict():
    # Oct 24: Tier 3 picked the pool line verbatim; the tagger had left it untagged.
    out, llm = _resolve(KREMLIN)
    assert out is None
    llm.assert_not_called()


def test_new_verdict_holds_when_the_pool_line_carries_an_outlet_suffix():
    # Pool lines often end in "(via AP)" or a Drop Site timestamp/URL that the pick drops.
    out, llm = _resolve("A software vulnerability detected by artificial intelligence could affect wide swaths of the internet.")
    assert out is None
    llm.assert_not_called()


def test_reworded_continuation_still_reaches_the_fallback():
    # Oct 15 pm: the headline extractor reworded a tagged line; the fallback rightly recovered it.
    out, llm = _resolve("Anthropic says its AI agents took unauthorized actions on government websites, "
                        "including submitting a false murder tip")
    assert out == ("DEVELOPMENT", SLUG)
    llm.assert_called_once()


def test_side_stories_get_the_same_veto():
    seen = []

    def fake_grounding(headline, audience_state=None):
        seen.append(audience_state)
        return "brief text"

    with patch("newscaster.llm.call_with_default", return_value=SLUG) as llm, \
         patch.object(topic_finder, "summarize_headline_with_grounding", side_effect=fake_grounding):
        text, _h, _b, arcs = topic_finder.overview_process(
            "- x", headline_arc_map=build_headline_arc_map(POOL), ledger=LEDGER,
            headlines=[KREMLIN], new_headline_keys=build_new_headline_keys(POOL))

    assert arcs == [None]
    assert seen == [None]
    assert "FOLLOW-UP" not in text
    llm.assert_not_called()


def test_the_fallback_sees_what_aired_not_a_months_old_research_brief():
    # 2026-11-01 replay of 121 recorded fallback decisions: with each arc's last three aired headlines and a short
    # topic (instead of the full first research brief, e.g. 2,637 characters for ai_safety_cyberattacks), the
    # fallback kept all 32 genuine matches and rejected the PHP and Kremlin mismatches on its own.
    from newscaster import dedup
    brief = "FINDINGS: *   **AI Safety Warnings:** a researcher resigned... propaganda campaigns by Russia " * 30
    ledger = {"arcs": {"ai_safety_cyberattacks": {"topic": brief, "episodes": [
        {"date": "2026_10_15_pm", "coverage": "main", "headline": "Anthropic says its AI agents took unauthorized actions"},
        {"date": "2026_10_21_pm", "coverage": "side", "headline": "OpenAI says its AI agents weakened safety monitoring"},
        {"date": "2026_10_22", "coverage": "main", "headline": "India's CERT-In issues an advisory on AI agents"},
        {"date": "2026_09_12", "coverage": "side", "headline": "AI safety researchers resign over existential risk"}]}}}
    cands = dedup._build_arc_candidates({"k": ("UPDATE", "ai_safety_cyberattacks")}, ledger)
    desc = cands["ai_safety_cyberattacks"][1]
    assert "FINDINGS" not in desc and len(desc) < 900
    assert desc.index("Anthropic says") < desc.index("OpenAI says") < desc.index("CERT-In")   # newest last
    assert "existential risk" in desc                                                     # the first headline stands in for the topic
