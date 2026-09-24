"""Beat feeds (business, science, health, courts, world) as pool sections."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from newscaster.scrapers import topic_finder as tf
from newscaster.scrapers.watchlist import BEAT_SELECTION_PROMPT, beat_scraper, select_with_llm

NOW = datetime(2026, 9, 15, 18, 0, tzinfo=timezone.utc)
FEED = b"""<rss><channel>
<item><title>Fed raises benchmark rate a quarter point</title><description>The FOMC voted 9-3.</description><pubDate>Tue, 15 Sep 2026 14:00:00 GMT</pubDate></item>
<item><title>Ten tips for your 401(k)</title><description>Personal finance.</description><pubDate>Tue, 15 Sep 2026 12:00:00 GMT</pubDate></item>
<item><title>Old story</title><pubDate>Fri, 11 Sep 2026 12:00:00 GMT</pubDate></item>
</channel></rss>"""


def _response(content):
    r = MagicMock(ok=True, content=content); r.raise_for_status.return_value = None; return r


def test_beat_prompt_asks_for_concrete_events_and_skips_noise():
    p = BEAT_SELECTION_PROMPT.format(date="September 15, 2026", group="Business and markets", max_items=8, items="x")
    assert "Choose up to 8" in p and "Business and markets sources" in p
    assert "Skip opinion, explainers" in p and "personal-finance tips" in p
    assert "add no consequences or significance of your own" in p
    assert "write exactly: NONE" in p


def test_select_with_llm_none_detection_is_a_word_not_a_prefix():
    items = [("CNBC", {"title": "t", "description": "", "published": NOW})]
    with patch("newscaster.scrapers.watchlist.call_with_default", return_value="NONE"):
        assert select_with_llm(items, "{date} {items}", "today", "l") == []
    with patch("newscaster.scrapers.watchlist.call_with_default", return_value="Nonetheless, the Fed raised rates (via CNBC)"):
        assert select_with_llm(items, "{date} {items}", "today", "l") == ["Nonetheless, the Fed raised rates (via CNBC)"]


def test_beat_scraper_windows_selects_and_labels():
    with patch("newscaster.scrapers.watchlist.requests.get", return_value=_response(FEED)), \
         patch("newscaster.scrapers.watchlist.call_with_default",
               return_value="The Federal Reserve raised its benchmark rate a quarter point on Sep 15, 2026 (via CNBC)") as llm:
        text = beat_scraper("Business and markets", [("CNBC", "u")], now=NOW, lookback_hours=24, max_items=8)
    assert text.startswith("Business and markets beat, as of September 15, 2026.")
    assert "raised its benchmark rate" in text
    prompt = llm.call_args.args[1]
    assert "Fed raises benchmark rate" in prompt and "Old story" not in prompt   # 24h window applied


def test_beats_are_appended_to_the_pool_and_never_break_it(monkeypatch):
    calls = []
    def fake_call(default, *args, _log_label=None, **kwargs):
        calls.append(_log_label); return f"stub from {_log_label}"
    monkeypatch.setattr(tf, "call_with_default", fake_call)
    monkeypatch.setattr(tf, "calmatters_scraper", lambda: "cm")
    monkeypatch.setattr(tf, "dropsite_scraper", lambda: "ds")
    monkeypatch.setattr(tf, "rss_scraper", lambda source, url: "rss")
    monkeypatch.setattr(tf, "riverside_scraper", lambda: "rv")
    monkeypatch.setattr(tf._config, "WATCHLIST_ENABLED", False, raising=False)
    monkeypatch.setattr(tf._config, "BEATS_ENABLED", True, raising=False)
    monkeypatch.setattr(tf._config, "BEAT_FEEDS", [("Health", [("STAT", "u")]), ("World", [("BBC", "u")])], raising=False)

    with patch.object(tf, "beat_scraper", side_effect=lambda g, f: f"{g} beat text\n\n"):
        sections = tf._gather_headline_sections("September 15, 2026")
    assert [name for _h, name, _t in sections][-2:] == ["Health beat (STAT)", "World beat (BBC)"]
    assert sections[-1][0] == "World (beat):"

    with patch.object(tf, "beat_scraper", side_effect=RuntimeError("feeds down")):
        assert len(tf._gather_headline_sections("September 15, 2026")) == 7
    monkeypatch.setattr(tf._config, "BEATS_ENABLED", False, raising=False)
    assert len(tf._gather_headline_sections("September 15, 2026")) == 7


def test_group_note_is_added_to_the_selection_prompt():
    seen = {}
    def fake_select(items, template, today, label, **extra):
        seen["prompt"] = template.format(date="d", items="ITEMS", **extra)
        return ["Local thing happened (via KPBS)"]
    with patch("newscaster.scrapers.watchlist.collect_watch_items", return_value=([("KPBS", {"title": "t"})], [])), \
         patch("newscaster.scrapers.watchlist.select_with_llm", side_effect=fake_select):
        beat_scraper("San Diego", [("KPBS", "u")], max_items=8, note="Choose only local events {here}.")
    assert "Choose only local events {here}.\n\nITEMS" in seen["prompt"]
