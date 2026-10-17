"""Digests and the seed lookup (2026-10-17).

- On 10-15 a Drop Site Daily recap title ("Iranian mine sinks ship in Hormuz; Trump pardons two
  at Alaska rally; Congo Ebola cases top 9,000 — Drop Site Daily: October 14, 2026") became a
  main story's topic, and half its research went to the Ebola and pardon items.
- Starting articles that are digests (newsletters, roundups) were summarized whole.
- The seed lookup's sources were never saved, so the fact check could not see the CNBC page
  whose figure the script credited to Al Jazeera.
"""
from datetime import datetime, timezone
from unittest.mock import patch

from newscaster import pipeline
from newscaster.scrapers import dropsite, topic_finder
from newscaster.source_hunter import SourceHunterResult


def test_a_digest_title_becomes_one_headline_per_story():
    item = {"title": "Iranian mine sinks ship in Hormuz; Trump pardons two at Alaska rally; Congo Ebola cases "
                     "top 9,000 — Drop Site Daily: October 14, 2026",
            "description": "Today's recap", "link": "https://www.dropsitenews.com/p/daily-oct-14",
            "published": datetime(2026, 10, 14, 22, tzinfo=timezone.utc)}
    single = dict(item, title="Israel strikes Beirut suburb; dozens killed", description="")
    text = dropsite._format_headlines([item, single], now=datetime(2026, 10, 15, 11, tzinfo=timezone.utc))
    lines = text.strip().splitlines()[1:]
    assert [l.split(" [")[0] for l in lines] == [
        "Iranian mine sinks ship in Hormuz", "Trump pardons two at Alaska rally", "Congo Ebola cases top 9,000",
        "Israel strikes Beirut suburb; dozens killed"]     # an ordinary title with a semicolon is left alone
    assert "Drop Site Daily" not in text.split("\n", 1)[1].split("Israel")[0]


def test_article_summary_is_told_which_story_to_keep(monkeypatch):
    sent = {}
    monkeypatch.setattr(topic_finder, "get_llm_response", lambda prompt, **k: sent.setdefault("p", prompt) or "ok")
    topic_finder.summarize_text("Item 1... Item 2...", "article", topic="Iranian mine sinks ship in Hormuz")
    assert "Iranian mine sinks ship in Hormuz" in sent["p"] and "digest" in sent["p"].lower()


def test_seed_lookup_is_saved_with_the_story_research(monkeypatch):
    seed = SourceHunterResult(answer="FINDINGS:\n- \"8.5 million bpd\" [https://www.cnbc.com/x]",
                              sources=[{"title": "CNBC", "url": "https://www.cnbc.com/x", "excerpt": "8.5 million bpd"}],
                              status="success")
    followups = []

    class Done:
        summary_prompt, successful_summary_counter, followups, articles, done_reason = "evidence", 1, [], [], "ok"
    monkeypatch.setattr(pipeline._config, "SOURCE_HUNTER_ENABLED", True)
    monkeypatch.setattr(pipeline._config, "AGENTIC_RESEARCH_ENABLED", True)
    with patch.object(pipeline, "_collect_articles", return_value=("Today is X.", 1)), \
         patch.object(pipeline, "answer_with_escalation", return_value=seed), \
         patch.object(pipeline, "run_adaptive_research", return_value=Done()), \
         patch.object(pipeline, "_super_summary_with_retry", return_value="summary"), \
         patch.object(pipeline, "call_with_default", return_value="no"), \
         patch.object(pipeline, "_augment_with_prior_research", side_effect=lambda s, d: s):
        pipeline._gather_one_topic("Hormuz", 0, "October 15, 2026", "2026_10_15", "", "", articles=[], followups=followups)
    assert followups and followups[0]["asker"] == "Seed lookup"
    assert followups[0]["source_hunter_sources"][0]["url"] == "https://www.cnbc.com/x"
