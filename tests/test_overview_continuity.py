"""Tests for side-story (overview) continuity: when a side story is a
continuation of a tracked arc, the overview writer must be told what the
audience already knows and instructed to report only what is new — mirroring
the SEGMENT_SCRIPT_UPDATE_CONTEXT injection that main segments already get.
"""

from unittest.mock import patch

from newscaster.scrapers import topic_finder


def test_grounding_injects_continuation_context_when_audience_state_given():
    captured = {}

    def fake_answer(prompt, topic, formatted_date):
        captured["prompt"] = prompt
        return "The death toll rose to nine hundred twenty."

    with patch.object(topic_finder, "_source_hunter_answer", side_effect=fake_answer):
        out = topic_finder.summarize_headline_with_grounding(
            "Venezuela earthquake",
            audience_state="Twin quakes of magnitude 7.2 and 7.5 struck; 235 dead.",
        )

    assert out == "The death toll rose to nine hundred twenty."
    # The prior coverage is shown to the researcher...
    assert "Twin quakes of magnitude 7.2 and 7.5" in captured["prompt"]
    # ...along with an explicit "only what's new" instruction.
    assert "new" in captured["prompt"].lower()


def test_grounding_omits_continuation_context_without_audience_state():
    captured = {}

    def fake_answer(prompt, topic, formatted_date):
        captured["prompt"] = prompt
        return "A fresh story with no prior coverage."

    with patch.object(topic_finder, "_source_hunter_answer", side_effect=fake_answer):
        topic_finder.summarize_headline_with_grounding("Brand new thing")

    assert "already know" not in captured["prompt"].lower()


def test_overview_process_passes_prior_state_for_recurring_story_only():
    overview_text = "1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
    arc_map = {
        "inflation hits 3 year high official says": ("UPDATE", "fed_chair_inflation"),
    }
    ledger = {
        "arcs": {
            "fed_chair_inflation": {
                "audience_state": "Inflation was three point one percent last month.",
            }
        }
    }

    captured = []

    def fake_grounding(headline, audience_state=None):
        captured.append((headline, audience_state))
        return "brief text"

    # First extracted headline is the recurring one; the rest are a brand-new story.
    extracted = [
        "Inflation hits 3-year high, official says",
        "A brand new local story",
        "A brand new local story",
        "A brand new local story",
        "A brand new local story",
    ]

    with patch.object(topic_finder, "get_llm_response", side_effect=extracted), \
         patch.object(topic_finder, "summarize_headline_with_grounding", side_effect=fake_grounding):
        topic_finder.overview_process(overview_text, headline_arc_map=arc_map, ledger=ledger)

    assert captured[0] == (
        "Inflation hits 3-year high, official says",
        "Inflation was three point one percent last month.",
    )
    # The unmatched headlines carry no prior state.
    assert captured[1] == ("A brand new local story", None)


def test_overview_process_backward_compatible_without_map_or_ledger():
    overview_text = "1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
    captured = []

    def fake_grounding(headline, audience_state=None):
        captured.append((headline, audience_state))
        return "brief text"

    with patch.object(topic_finder, "get_llm_response", return_value="Some headline"), \
         patch.object(topic_finder, "summarize_headline_with_grounding", side_effect=fake_grounding):
        topic_finder.overview_process(overview_text)

    # No map/ledger -> never any prior state, old behavior preserved.
    assert all(state is None for _h, state in captured)
    assert len(captured) == 5
