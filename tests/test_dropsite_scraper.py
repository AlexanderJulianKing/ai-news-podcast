from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from newscaster.scrapers.dropsite import (
    _extract_items,
    _format_headlines,
    dropsite_scraper,
)


SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Drop Site News</title>
    <item>
      <title><![CDATA[Newest story]]></title>
      <description><![CDATA[<p>Fresh <strong>description</strong>.</p>]]></description>
      <link>https://www.dropsitenews.com/p/newest-story</link>
      <pubDate>Thu, 18 Jun 2026 14:59:03 GMT</pubDate>
    </item>
    <item>
      <title><![CDATA[Older but current story]]></title>
      <description><![CDATA[Drop Site Daily: June 17, 2026]]></description>
      <link>https://www.dropsitenews.com/p/older-current-story</link>
      <pubDate>Wed, 17 Jun 2026 20:08:24 GMT</pubDate>
    </item>
    <item>
      <title><![CDATA[Too old story]]></title>
      <description>Outside the lookback window</description>
      <link>https://www.dropsitenews.com/p/too-old-story</link>
      <pubDate>Sun, 14 Jun 2026 20:08:24 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def test_extract_items_filters_sorts_and_cleans_feed_entries():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc)

    items = _extract_items(SAMPLE_FEED, now=now, lookback_hours=48)

    assert [item["title"] for item in items] == [
        "Newest story",
        "Older but current story",
    ]
    assert items[0]["description"] == "Fresh description."
    assert items[0]["link"] == "https://www.dropsitenews.com/p/newest-story"


def test_format_headlines_includes_source_dates_descriptions_and_links():
    now = datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc)
    items = _extract_items(SAMPLE_FEED, now=now, lookback_hours=48)

    text = _format_headlines(items, now=now)

    assert "Drop Site News, the news source" in text
    assert "Newest story" in text
    assert "Fresh description." in text
    assert "https://www.dropsitenews.com/p/newest-story" in text
    assert "Too old story" not in text


def test_dropsite_scraper_fetches_and_formats_rss():
    response = MagicMock(ok=True, content=SAMPLE_FEED)
    response.raise_for_status.return_value = None
    now = datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc)

    with patch("newscaster.scrapers.dropsite.requests.get", return_value=response):
        text = dropsite_scraper(now=now)

    assert "Newest story" in text
    assert "Older but current story" in text


def test_dropsite_scraper_handles_unreadable_rss():
    response = MagicMock(ok=True, content=b"not xml")
    response.raise_for_status.return_value = None

    with patch("newscaster.scrapers.dropsite.requests.get", return_value=response):
        text = dropsite_scraper()

    assert "unreadable RSS feed" in text


def test_rss_scraper_names_its_source(monkeypatch):
    import newscaster.scrapers.dropsite as ds

    class Resp:
        ok = True
        content = SAMPLE_FEED
        def raise_for_status(self):
            pass
    monkeypatch.setattr(ds.requests, "get", lambda *a, **k: Resp())
    out = ds.rss_scraper("ProPublica", "https://example.org/feed",
                         now=datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc), lookback_hours=48)
    assert out.startswith("ProPublica, the news source, has released the following headlines in the past 48 hours")
    assert "Drop Site" not in out.splitlines()[0]


def test_wordpress_boilerplate_description_is_dropped():
    feed = b"""<?xml version="1.0"?><rss version="2.0"><channel><item>
      <title>Collins story</title>
      <description>The post Collins story appeared first on ProPublica.</description>
      <link>https://www.propublica.org/article/x</link>
      <pubDate>Wed, 23 Sep 2026 15:34:00 +0000</pubDate></item></channel></rss>"""
    items = _extract_items(feed, now=datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc), lookback_hours=48)
    assert items[0]["title"] == "Collins story" and items[0]["description"] == ""
