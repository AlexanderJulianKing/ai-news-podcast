"""Capture of article provenance in result_piper."""
from unittest.mock import patch

import newscaster.scrapers.topic_finder as tf


def test_result_piper_appends_article_record():
    articles = []
    result = {"headline": "Big Fire", "url": "https://npr.org/fire", "snippet": "...",
              "date": "2026-03-08"}
    with patch.object(tf, "determine_relevance", return_value=True), \
         patch.object(tf, "scrape_text", return_value="full article body"), \
         patch.object(tf, "summarize_text", return_value="a concise summary"), \
         patch.object(tf, "call_with_default", side_effect=["yes", "SOURCE: NPR"]):
        summary_prompt, counter = tf.result_piper(
            "", 0, "wildfire", result, 0, "2026_03_09", articles=articles
        )

    assert counter == 1
    assert len(articles) == 1
    rec = articles[0]
    assert rec["chunk_id"] == "2026_03_09_seg0_art0"
    assert rec["url"] == "https://npr.org/fire"
    assert rec["outlet"] == "NPR"
    assert rec["original_headline"] == "Big Fire"
    assert rec["published_date"] == "2026-03-08"
    assert rec["retrieved_date"] == "2026_03_09"
    assert rec["surfacing_topic"] == "wildfire"
    assert rec["summary"] == "a concise summary"


def test_result_piper_without_accumulator_still_works():
    """Backward compatible: omitting `articles` must not error."""
    result = {"headline": "H", "url": "https://x", "snippet": "s"}
    with patch.object(tf, "determine_relevance", return_value=False):
        summary_prompt, counter = tf.result_piper("seed", 0, "topic", result, 0, "2026_03_09")
    assert counter == 0
    assert summary_prompt == "seed"


import newscaster.pipeline as pipeline


def test_run_follow_up_rounds_collects_qa():
    followups = []

    def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
        # Grounded calls answer questions; ungrounded calls produce a quoted question.
        if grounding:
            return "the grounded answer"
        return '"a follow up question"'

    with patch.object(pipeline, "get_llm_response", side_effect=fake_llm):
        pipeline._run_follow_up_rounds("seed summary", "fup template", "challenging template",
                                       followups=followups)

    assert len(followups) == 8  # 4 modes x (regular + challenging)
    assert followups[0]["question"] == "a follow up question"
    assert followups[0]["answer"] == "the grounded answer"
    assert "asker" in followups[0] and "challenging" in followups[0]
