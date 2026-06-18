from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
import re
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from newscaster.logging import print_and_write


DROP_SITE_FEED_URL = "https://www.dropsitenews.com/feed"


def _clean_text(value, max_chars=260):
    text = BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)
    text = " ".join(unescape(text).split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "..."
    return text


def _parse_rss_datetime(value):
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_items(feed_xml, now=None, lookback_hours=48):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)

    root = ET.fromstring(feed_xml)
    items = []
    for item in root.findall("./channel/item"):
        published = _parse_rss_datetime(item.findtext("pubDate"))
        if published is None or published < cutoff or published > now + timedelta(hours=3):
            continue

        title = _clean_text(item.findtext("title"), max_chars=220)
        description = _clean_text(item.findtext("description"), max_chars=240)
        link = (item.findtext("link") or "").strip()
        if not title:
            continue

        items.append({
            "title": title,
            "description": description,
            "link": link,
            "published": published,
        })

    items.sort(key=lambda row: row["published"], reverse=True)
    return items


def _format_headlines(items, now=None):
    now = now or datetime.now(timezone.utc)
    today = now.astimezone().strftime("%B %e, %Y")
    if not items:
        return (
            "Drop Site News, the news source, has no feed headlines from the past "
            f"48 hours as of {today}.\n\n"
        )

    lines = [
        "Drop Site News, the news source, has released the following headlines "
        f"in the past 48 hours as of {today}:"
    ]
    for item in items:
        published = item["published"].astimezone().strftime("%b %-d, %Y %I:%M %p")
        description = f" — {item['description']}" if item.get("description") else ""
        link = f" ({item['link']})" if item.get("link") else ""
        lines.append(f"{item['title']}{description} [{published}]{link}")
    return "\n".join(lines) + "\n\n"


def dropsite_scraper(feed_url=DROP_SITE_FEED_URL, now=None, lookback_hours=48):
    response = None
    for attempt in range(3):
        try:
            response = requests.get(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DropSiteScraper/1.0)"},
                timeout=(5, 20),
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            wait = attempt + 1
            print_and_write("Drop Site feed fetch failed", str(exc), f"Retrying in {wait}s")
            time.sleep(wait)

    if response is None or not response.ok:
        print_and_write("Skipping Drop Site after repeated feed failures")
        return "Drop Site News, the news source, could not be fetched today.\n\n"

    try:
        items = _extract_items(response.content, now=now, lookback_hours=lookback_hours)
    except ET.ParseError as exc:
        print_and_write("Drop Site feed parse failed", str(exc))
        return "Drop Site News, the news source, returned an unreadable RSS feed today.\n\n"

    return _format_headlines(items, now=now)
