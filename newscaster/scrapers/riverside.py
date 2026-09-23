"""City of Riverside press releases, read straight from riversideca.gov/media.

The page is plain server-rendered HTML with a dated card per release, so it is
parsed directly. On 2026-09-23 Gemini's URL reader reported "none released today
or yesterday" while the page listed releases dated Sept 23 and Sept 22.
"""
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from newscaster.logging import print_and_write

RIVERSIDE_MEDIA_URL = "https://www.riversideca.gov/media"
SOURCE = "The City of Riverside"


def _card_date(day, month, today):
    """The card shows only day and month ("23", "Sept"); pick the year that is not in the future."""
    try:
        base = datetime.strptime(f"{month.strip()[:3]} {int(day)} {today.year}", "%b %d %Y").date()
    except ValueError:
        return None
    if base > today + timedelta(days=1):
        base = base.replace(year=today.year - 1)
    return base


def parse_press_releases(html, today, lookback_days=2, base_url="https://www.riversideca.gov"):
    """[(date, department, title, link)] for releases dated within the last `lookback_days` days."""
    soup = BeautifulSoup(html or b"", "html.parser")
    items, seen = [], set()
    for card in soup.select("figure.pressCard"):
        day, month = card.select_one(".date .day"), card.select_one(".date .month")
        if not day or not month:
            continue
        when = _card_date(day.get_text(strip=True), month.get_text(strip=True), today)
        if when is None or when <= today - timedelta(days=lookback_days):
            continue
        dept = card.select_one("h4.departmentCard")
        headings = [h for h in card.select("figcaption h4") if "departmentCard" not in (h.get("class") or [])]
        title = " ".join(headings[0].get_text(" ", strip=True).split()) if headings else ""
        anchor = card.find("a", href=True)
        link = anchor["href"] if anchor else ""
        if link.startswith("/"):
            link = base_url + link
        if not title or (title, when) in seen:
            continue
        seen.add((title, when))
        items.append((when, dept.get_text(strip=True) if dept else "", title, link))
    items.sort(key=lambda row: row[0], reverse=True)
    return items


def riverside_scraper(url=RIVERSIDE_MEDIA_URL, today=None):
    today = today or datetime.now().date()
    stamp = today.strftime("%B %d, %Y")
    html = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Newscaster/1.0)"},
                                    timeout=(5, 20))
            response.raise_for_status()
            html = response.content
            break
        except requests.RequestException as exc:
            print_and_write("Riverside fetch failed", str(exc), f"Retrying in {attempt + 1}s")
            time.sleep(attempt + 1)
    if html is None:
        return f"{SOURCE} press releases could not be fetched today, {stamp}."
    items = parse_press_releases(html, today)
    if not items:
        return f"{SOURCE} released no press releases today or yesterday, as of {stamp}."
    lines = [f"{SOURCE} released these press releases today or yesterday, as of {stamp}:"]
    for when, dept, title, link in items:
        label = f"{dept}, " if dept else ""
        lines.append(f"{title} ({label}{when.strftime('%b %-d, %Y')}) ({link})")
    return "\n".join(lines)
