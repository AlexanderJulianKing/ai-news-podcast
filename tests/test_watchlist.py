"""Specialist watch (AI labs and evaluators): deterministic feed reads, one LLM
event test, nominate-only output that always says what happened.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from newscaster.scrapers.watchlist import (
    SECTION_NAME,
    parse_anthropic_news_listing,
    WATCHLIST_EVENT_TEST_PROMPT,
    apply_event_test,
    collect_watch_items,
    format_items_for_test,
    parse_feed,
    watchlist_scraper,
)

NOW = datetime(2026, 9, 15, 18, 0, tzinfo=timezone.utc)

RSS_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>METR</title>
  <item>
    <title>Hugging Face incident investigation report</title>
    <description><![CDATA[<p>Findings on the <b>July</b> intrusion.</p>]]></description>
    <link>https://metr.org/hf-report</link>
    <pubDate>Mon, 14 Sep 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Too old</title>
    <link>https://metr.org/old</link>
    <pubDate>Mon, 07 Sep 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Undated item</title>
    <link>https://metr.org/undated</link>
  </item>
  <item>
    <title>No title</title>
    <description>Anthropic disclosed that a weapons cell used Claude Code for missile guidance software.</description>
    <link>https://incidentdatabase.ai/cite/999</link>
    <pubDate>Sun, 13 Sep 2026 01:00:02 GMT</pubDate>
  </item>
  <item>
    <title>No title</title>
    <description>... (report_number: 7953)</description>
    <guid>44850a5f</guid>
    <pubDate>Sun, 13 Sep 2026 01:00:02 GMT</pubDate>
  </item>
</channel></rss>
"""

ATOM_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Lab blog</title>
  <entry>
    <title>Model pause announced</title>
    <summary>Two-week pause on RL training.</summary>
    <link href="https://lab.example/pause"/>
    <published>2026-09-15T09:30:00Z</published>
  </entry>
  <entry>
    <title>Ancient post</title>
    <link href="https://lab.example/old"/>
    <updated>2026-01-01T00:00:00Z</updated>
  </entry>
</feed>
"""


# --- parsing -----------------------------------------------------------------

def test_parse_rss_keeps_window_drops_old_and_undated_and_cleans_html():
    items = parse_feed(RSS_FEED, now=NOW, lookback_hours=72)
    titles = [item["title"] for item in items]
    assert titles[0] == "Hugging Face incident investigation report"
    assert "Too old" not in titles
    assert "Undated item" not in titles
    assert items[0]["description"] == "Findings on the July intrusion."
    assert items[0]["link"] == "https://metr.org/hf-report"


def test_placeholder_title_falls_back_to_description_or_drops_empty_body():
    items = parse_feed(RSS_FEED, now=NOW, lookback_hours=72)
    fallback = [item for item in items if item["link"].endswith("/999")][0]
    assert fallback["title"].startswith("Anthropic disclosed that a weapons cell used Claude Code")
    # The bare "... (report_number: N)" entry has nothing to judge and is dropped.
    assert not any("report_number" in item["title"] for item in items)
    assert len(items) == 2


def test_parse_atom_reads_published_or_updated_and_href_links():
    items = parse_feed(ATOM_FEED, now=NOW, lookback_hours=72)
    assert [item["title"] for item in items] == ["Model pause announced"]
    assert items[0]["description"] == "Two-week pause on RL training."
    assert items[0]["link"] == "https://lab.example/pause"


def test_parse_feed_ignores_future_dated_entries():
    future = RSS_FEED.replace(b"Mon, 14 Sep 2026 12:00:00 GMT", b"Fri, 25 Sep 2026 12:00:00 GMT")
    titles = [item["title"] for item in parse_feed(future, now=NOW, lookback_hours=72)]
    assert "Hugging Face incident investigation report" not in titles


# --- collection --------------------------------------------------------------

def _response(content):
    response = MagicMock(ok=True, content=content)
    response.raise_for_status.return_value = None
    return response


def test_collect_reports_failed_feeds_and_caps_items_per_feed():
    many = b"<rss><channel>" + b"".join(
        f"<item><title>Item {i}</title><pubDate>Mon, 14 Sep 2026 1{i}:00:00 GMT</pubDate></item>".encode()
        for i in range(5)
    ) + b"</channel></rss>"

    def fake_get(url, **_kw):
        if "dead" in url:
            import requests
            raise requests.RequestException("connection refused")
        return _response(many)

    with patch("newscaster.scrapers.watchlist.requests.get", side_effect=fake_get), \
         patch("newscaster.scrapers.watchlist.time.sleep"):
        items, failed = collect_watch_items(
            [("Alive", "https://alive.example/feed"), ("Dead", "https://dead.example/feed")],
            now=NOW, lookback_hours=72, max_per_feed=3,
        )
    assert failed == ["Dead"]
    assert [name for name, _ in items] == ["Alive"] * 3


def test_collect_treats_unparseable_feed_as_failed():
    with patch("newscaster.scrapers.watchlist.requests.get", return_value=_response(b"not xml")):
        items, failed = collect_watch_items([("Broken", "https://b.example/feed")], now=NOW)
    assert items == [] and failed == ["Broken"]


# --- the event test ----------------------------------------------------------

def _items():
    return [("METR", parse_feed(RSS_FEED, now=NOW, lookback_hours=72)[0])]


def test_format_items_for_test_labels_source_and_date():
    text = format_items_for_test(_items())
    assert text.startswith("[METR, Sep 14, 2026] Hugging Face incident investigation report — Findings")


def test_event_test_prompt_excludes_launches_and_forbids_added_significance():
    assert "not by itself an event" in WATCHLIST_EVENT_TEST_PROMPT
    assert "ROUTE B" in WATCHLIST_EVENT_TEST_PROMPT and "new frontier or flagship model" in WATCHLIST_EVENT_TEST_PROMPT
    assert "never state a lab's own claim as settled fact" in WATCHLIST_EVENT_TEST_PROMPT
    assert "customer stories and case studies" in WATCHLIST_EVENT_TEST_PROMPT
    assert "add no consequences or significance of your own" in WATCHLIST_EVENT_TEST_PROMPT
    assert "NONE PASS" in WATCHLIST_EVENT_TEST_PROMPT


def test_apply_event_test_parses_passing_lines_none_pass_and_degraded():
    with patch("newscaster.scrapers.watchlist.call_with_default",
               return_value="- METR reported X on Sep 14, 2026 (via METR)\n\n* OpenAI paused Y (via OpenAI)"):
        assert apply_event_test(_items(), "September 15, 2026") == [
            "METR reported X on Sep 14, 2026 (via METR)",
            "OpenAI paused Y (via OpenAI)",
        ]
    with patch("newscaster.scrapers.watchlist.call_with_default", return_value="NONE PASS"):
        assert apply_event_test(_items(), "September 15, 2026") == []
    with patch("newscaster.scrapers.watchlist.call_with_default", return_value=None):
        assert apply_event_test(_items(), "September 15, 2026") is None
    assert apply_event_test([], "September 15, 2026") == []


# --- the section text: always says what happened -----------------------------

def test_section_lists_passing_items_and_names_failed_feeds():
    def fake_get(url, **_kw):
        if "dead" in url:
            import requests
            raise requests.RequestException("nope")
        return _response(RSS_FEED)

    with patch("newscaster.scrapers.watchlist.requests.get", side_effect=fake_get), \
         patch("newscaster.scrapers.watchlist.time.sleep"), \
         patch("newscaster.scrapers.watchlist.call_with_default",
               return_value="METR published its Hugging Face incident findings on Sep 14, 2026 (via METR)"):
        text = watchlist_scraper(
            feeds=[("METR", "https://metr.org/feed.xml"), ("Anthropic", "https://dead.example/feed")],
            now=NOW, lookback_hours=72,
        )
    assert text.startswith(SECTION_NAME)
    assert "judge them by the same criteria as any other source" in text
    assert "METR published its Hugging Face incident findings" in text
    assert "Feeds that could not be fetched today: Anthropic." in text
    assert text.endswith("\n\n")


def test_section_distinguishes_quiet_week_from_degraded_llm():
    with patch("newscaster.scrapers.watchlist.requests.get", return_value=_response(RSS_FEED)), \
         patch("newscaster.scrapers.watchlist.call_with_default", return_value="NONE PASS"):
        quiet = watchlist_scraper(feeds=[("METR", "u")], now=NOW, lookback_hours=72)
    assert "No items from these sources qualified in the last 72 hours" in quiet

    with patch("newscaster.scrapers.watchlist.requests.get", return_value=_response(RSS_FEED)), \
         patch("newscaster.scrapers.watchlist.call_with_default", return_value=None):
        degraded = watchlist_scraper(feeds=[("METR", "u")], now=NOW, lookback_hours=72)
    assert "could not be run today" in degraded


# --- Anthropic: listing page instead of a feed --------------------------------

ANTHROPIC_LISTING = b"""<html><body>
<a href="/news/improving-alignment-security-efforts">Announcements Aug 31, 2026 Improving our alignment and security efforts We are conducting a review</a>
<a href="/news/enterprise-frontier-safeguards">Sep 1, 2026 Announcements Developing Enterprise Frontier Safeguards</a>
<a href="/news/enterprise-frontier-safeguards">Sep 1, 2026 Announcements Developing Enterprise Frontier Safeguards</a>
<a href="/news/claude-opus-5">Product Jul 24, 2026 Introducing Claude Opus 5</a>
<a href="https://www.anthropic.com/news/announcing-our-updated-rsp">Responsible Scaling Policy</a>
<a href="/research/some-paper">Research Sep 14, 2026 A paper</a>
</body></html>"""

ANTHROPIC_ARTICLE = b"""<html><head>
<title>Improving our alignment and security practices  \\ Anthropic</title>
<meta name="description" content="On July 30, we reported three incidents in which Claude models gained unauthorized access to real computer systems.">
</head><body></body></html>"""


def test_anthropic_listing_parses_dates_in_either_order_and_skips_undated_old_and_duplicates():
    now = datetime(2026, 9, 2, 18, 0, tzinfo=timezone.utc)
    items = parse_anthropic_news_listing(ANTHROPIC_LISTING, now=now, lookback_hours=72)
    assert [item["title"] for item in items] == [
        "Developing Enterprise Frontier Safeguards",
        "Improving our alignment and security efforts We are conducting a review",
    ]
    assert items[0]["published"].date() == datetime(2026, 9, 1).date()
    assert items[0]["link"] == "https://www.anthropic.com/news/enterprise-frontier-safeguards"
    links = [item["link"] for item in items]
    assert not any("claude-opus-5" in l or "updated-rsp" in l or "/research/" in l for l in links)


def test_anthropic_article_page_upgrades_title_and_supplies_description():
    now = datetime(2026, 9, 2, 18, 0, tzinfo=timezone.utc)
    fetched = []

    def fetch_article(url):
        fetched.append(url)
        return ANTHROPIC_ARTICLE if url.endswith("improving-alignment-security-efforts") else None

    items = parse_anthropic_news_listing(ANTHROPIC_LISTING, now=now, lookback_hours=72, fetch_article=fetch_article)
    by_link = {item["link"].rsplit("/", 1)[-1]: item for item in items}
    assert by_link["improving-alignment-security-efforts"]["title"] == "Improving our alignment and security practices"
    assert by_link["improving-alignment-security-efforts"]["description"].startswith("On July 30, we reported three incidents")
    # Fetch failed for the other in-window item: listing title stands, no description.
    assert by_link["enterprise-frontier-safeguards"]["title"] == "Developing Enterprise Frontier Safeguards"
    assert by_link["enterprise-frontier-safeguards"]["description"] == ""
    assert len(fetched) == 2          # only in-window items are fetched


def test_collect_dispatches_anthropic_news_kind_and_keeps_rss_default():
    now = datetime(2026, 9, 2, 18, 0, tzinfo=timezone.utc)

    def fake_get(url, **_kw):
        if url.endswith("/news"):
            return _response(ANTHROPIC_LISTING)
        if "/news/" in url:
            return _response(ANTHROPIC_ARTICLE)
        return _response(RSS_FEED.replace(b"Mon, 14 Sep 2026", b"Tue, 01 Sep 2026").replace(b"Sun, 13 Sep 2026", b"Tue, 01 Sep 2026"))

    with patch("newscaster.scrapers.watchlist.requests.get", side_effect=fake_get):
        items, failed = collect_watch_items(
            [("Anthropic", "https://www.anthropic.com/news", "anthropic-news"), ("METR", "https://metr.org/feed.xml")],
            now=now, lookback_hours=72, max_per_feed=8,
        )
    assert failed == []
    names = [name for name, _ in items]
    assert names.count("Anthropic") == 2 and "METR" in names


def test_anthropic_boilerplate_description_is_blanked():
    from newscaster.scrapers.watchlist import _article_title_and_description
    page = (b"<html><head><title>Expanding our support for scientists \\ Anthropic</title>"
            b"<meta name=\"description\" content=\"Anthropic is an AI safety and research company that's working to build reliable AI.\">"
            b"</head></html>")
    title, description = _article_title_and_description(page)
    assert title == "Expanding our support for scientists"
    assert description == ""


def test_anthropic_featured_cards_outside_news_path_are_read():
    # 2026-09-22: the Opus 5.5 launch card linked to /claude-opus-5-5 and a feature to
    # an absolute /features/ URL; both were skipped when only /news/ links counted.
    listing = b"""<html><body>
<a href="/claude-opus-5-5">Introducing Claude Opus 5.5 Announcements Sep 22, 2026 Opus 5.5 performs at the level of Fable 5.1</a>
<a href="https://www.anthropic.com/features/ebola-response">Features Sep 22, 2026 The Situation Report</a>
<a href="https://twitter.com/anthropicai">Sep 22, 2026 elsewhere</a>
<a href="/news/claude-discovers-novel-enzyme-system">Sep 23, 2026 Science Claude discovers a novel enzyme system</a>
</body></html>"""
    now = datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc)
    items = parse_anthropic_news_listing(listing, now=now, lookback_hours=72)
    links = [item["link"] for item in items]
    assert "https://www.anthropic.com/claude-opus-5-5" in links
    assert "https://www.anthropic.com/features/ebola-response" in links
    assert not any("twitter.com" in l for l in links)
    titles = [item["title"] for item in items]
    assert "Claude discovers a novel enzyme system" in titles
    assert "The Situation Report" in titles
