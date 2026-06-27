"""Tests for the scoped LLM fallback that recovers arc identity for headlines
the deterministic string matcher misses (heavy paraphrases of the same event).

Safety contract: a miss (None -> fresh arc) is acceptable; a WRONG match injects
the wrong audience_state, so the fallback must return None whenever it is not
confident (model says NONE, ambiguous output, or no plausible candidate).
"""

from unittest.mock import patch

from newscaster.dedup import llm_recover_arc, resolve_arc_identity


LEDGER = {
    "arcs": {
        "venezuela_earthquake_aftermath": {
            "slug": "venezuela_earthquake_aftermath",
            "topic": "Twin magnitude 7.2 and 7.5 earthquakes devastate Venezuela",
            "audience_state": "Twin quakes struck near Yumare; 235 dead.",
        },
        "scotus_immigration_ruling": {
            "slug": "scotus_immigration_ruling",
            "topic": "Supreme Court lets the administration revoke TPS and curb asylum",
            "audience_state": "6-3 ruling in Mullin v. Doe.",
        },
    }
}

# The tagger's verdicts: clean-headline-key -> (tag, slug)
ARC_MAP = {
    "venezuelans take search for the missing into their own hands as earthquake death toll climbs": (
        "UPDATE", "venezuela_earthquake_aftermath",
    ),
    "experts warn of population decline following scotus immigration ruling": (
        "UPDATE", "scotus_immigration_ruling",
    ),
}

# A heavy paraphrase of the Venezuela arc that string matching cannot catch.
PARAPHRASE = "Thousands still missing in Venezuela following magnitude 7.2 earthquake"


def test_llm_recovers_heavy_paraphrase_when_model_returns_slug():
    with patch("newscaster.llm.call_with_default", return_value="venezuela_earthquake_aftermath") as m:
        out = llm_recover_arc(PARAPHRASE, ARC_MAP, LEDGER)
    assert out == ("UPDATE", "venezuela_earthquake_aftermath")
    m.assert_called_once()


def test_llm_returns_none_when_model_says_none():
    with patch("newscaster.llm.call_with_default", return_value="NONE"):
        assert llm_recover_arc(PARAPHRASE, ARC_MAP, LEDGER) is None


def test_llm_returns_none_on_ambiguous_output():
    # Two slugs named -> not confident -> no match.
    with patch(
        "newscaster.llm.call_with_default",
        return_value="venezuela_earthquake_aftermath or maybe scotus_immigration_ruling",
    ):
        assert llm_recover_arc(PARAPHRASE, ARC_MAP, LEDGER) is None


def test_llm_returns_none_on_hallucinated_slug_not_in_candidates():
    with patch("newscaster.llm.call_with_default", return_value="some_made_up_slug"):
        assert llm_recover_arc(PARAPHRASE, ARC_MAP, LEDGER) is None


def test_llm_is_not_called_for_clearly_unrelated_headline():
    # Cheap token-overlap gate must short-circuit before paying for the LLM.
    with patch("newscaster.llm.call_with_default") as m:
        out = llm_recover_arc("Local school board approves new cafeteria menu", ARC_MAP, LEDGER)
    assert out is None
    m.assert_not_called()


def test_llm_is_not_called_when_no_candidates():
    with patch("newscaster.llm.call_with_default") as m:
        assert llm_recover_arc(PARAPHRASE, {}, LEDGER) is None
    m.assert_not_called()


def test_resolve_prefers_string_match_and_skips_llm():
    # Exact (normalized) string hit must short-circuit — no LLM call.
    with patch("newscaster.llm.call_with_default") as m:
        out = resolve_arc_identity(
            "Experts warn of population decline following SCOTUS immigration ruling",
            ARC_MAP, LEDGER,
        )
    assert out == ("UPDATE", "scotus_immigration_ruling")
    m.assert_not_called()


def test_resolve_falls_through_to_llm_on_paraphrase():
    with patch("newscaster.llm.call_with_default", return_value="venezuela_earthquake_aftermath") as m:
        out = resolve_arc_identity(PARAPHRASE, ARC_MAP, LEDGER)
    assert out == ("UPDATE", "venezuela_earthquake_aftermath")
    m.assert_called_once()


def test_resolve_can_disable_llm():
    with patch("newscaster.llm.call_with_default") as m:
        out = resolve_arc_identity(PARAPHRASE, ARC_MAP, LEDGER, use_llm=False)
    assert out is None
    m.assert_not_called()
