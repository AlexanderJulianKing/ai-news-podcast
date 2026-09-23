"""Specialist watch: AI labs and independent evaluators. Nominate-only.

Front pages miss most of what happens inside the AI field until it becomes a
political story. In 2026, METR's Hugging Face findings (Aug 26) and OpenAI's
training pause (Aug 18) never entered the headline pool at all. This module reads
a fixed set of RSS/Atom feeds deterministically, keeps items from the last
WATCHLIST_LOOKBACK_HOURS, and spends one LLM call applying an event test. Items
that pass are appended to the pool as one more source and get no special weight
anywhere downstream: Tier 1 scores them by the same criteria as an AP headline.

The event test is what keeps press releases out. A model launch, benchmark score,
pricing change or feature fails unless the post itself also reports that an
outside party was forced to act, that a developer lost control of a system or
disclosed an incident, or that an independent evaluation reported a finding.

Fetch failures are reported in the section text rather than swallowed, so a dead
feed is visible in the daily log and cannot be mistaken for a quiet day.

Two source kinds. "rss" (default) reads RSS 2.0 or Atom. "anthropic-news" parses
Anthropic's server-rendered /news listing, because Anthropic publishes no feed and
its sitemap's lastmod is a site-regeneration stamp (159 pages shared 2026-09-09),
not a publication date. Each card's anchor text carries the date, so recency is
still deterministic; the article page then supplies the real title and description.
"""

from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
import re
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from newscaster import config as _config
from newscaster.llm import call_with_default
from newscaster.logging import print_and_write

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_PLACEHOLDER_TITLES = frozenset({"no title", "untitled", "(untitled)"})


def _is_placeholder_body(text):
    """True for bodies with no judgeable content: empty, an ellipsis, or a bare report id."""
    words = re.sub(r"\(report_number:\s*\d+\)", "", text or "").replace("...", " ").split()
    return len(words) < 4
SECTION_NAME = "Specialist watch (AI labs and evaluators)"

# One judgment call per day over a few dozen titles. This is a precision filter:
# a false pass puts a press release in front of Tier 1, so it gets the heavy model.
# Route B (2026-09-23, Alex): major launches and extraordinary lab results count on
# the lab's word alone, because the most important lab news may have no outside
# confirmation yet. The sentence must attribute the claim to the lab.
WATCHLIST_EVENT_TEST_PROMPT = (
    "Today is {date}. Below are recent posts from AI labs and independent evaluators, each with its "
    "source and publication date. Keep an item only if it passes route A or route B.\n"
    "ROUTE A, outside confirmation. Something happened (an action, incident, finding, filing, or "
    "decision), and at least one of: an outside party (a regulator, court, legislature, another company, "
    "or an independent evaluator) was forced to act or has acted; a developer lost control of a system or "
    "disclosed an incident; an independent evaluation reported a finding about a frontier system's "
    "capability or safety; or a verified first: a result no system achieved before, attested by formal "
    "verification, peer review, an independent evaluator, or a prize body.\n"
    "ROUTE B, major lab news on the lab's own word. Either: a leading lab released a new frontier or "
    "flagship model, or a new named general-purpose model in its main lineup (a new generation, or a new "
    "model or tier at any size, such as a flagship or a new mid-size pair; not a minor version, fine-tune, "
    "regional rollout, narrow single-purpose model such as text-to-speech, or feature); or a lab reported something extraordinary, such as a capability or scientific result far "
    "beyond what AI systems could do before, a serious safety incident, or a major change in how it "
    "deploys or restricts its most capable models. Ask whether a well-informed listener would call it big "
    "news about AI if it is true.\n"
    "Neither route: customer stories and case studies, courses and training programs, partnerships, "
    "pricing or caching changes, small features, speeches and remarks, policy essays, hiring, or a "
    "benchmark score with no new model. A product, model, feature, or pricing announcement that is not "
    "a major release is not by itself an event.\n\n"
    "For each passing item write ONE plain sentence: WHO did WHAT, and WHEN (use the post date), naming "
    "the specific actors, systems, numbers, and places the post gives. For a route B item, attribute every "
    "claim to the lab ('Anthropic said', 'OpenAI reported'); never state a lab's own claim as settled fact. "
    "Report only what the post states; add no consequences or significance of your own. End each sentence "
    "with ' (via SOURCE)' using the source name shown. One item per line, no numbering, no bold.\n"
    "If nothing passes, write exactly: NONE PASS\n\n{items}"
)


# Beat feeds (business, science, health, courts, world) reuse the feed machinery
# but not the AI event test: one heavy call per beat picks the few concrete events
# worth putting in front of Tier 1, in the same one-sentence shape as the front pages.
BEAT_SELECTION_PROMPT = (
    "Today is {date}. Below are items published in the last day by {group} sources, each with its source "
    "and date. Choose up to {max_items} that report a specific event: something happened, with a named actor "
    "and object (an action, decision, ruling, filing, incident, result, or announcement). Prefer events that "
    "force an institution to respond, reveal misconduct by the powerful, or report a verified first in what a "
    "technology or field can do. Skip opinion, explainers, previews, live-blog headers, listicles, sports "
    "results, celebrity items, and personal-finance tips.\n"
    "Write each chosen item as ONE plain sentence: WHO did WHAT, and WHEN (use the item date). Name the "
    "specific actors, places, and numbers the item gives; report only what the item states and add no "
    "consequences or significance of your own. End each sentence with ' (via SOURCE)' using the source name "
    "shown. One per line, no numbering, no bold. If none qualify, write exactly: NONE\n\n{items}"
)

_NONE_RE = re.compile(r"^NONE(\s+PASS)?\b", re.IGNORECASE)


def _clean_text(value, max_chars=300):
    text = BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)
    text = " ".join(unescape(text).split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "..."
    return text


def _parse_datetime(value):
    """RSS (RFC 2822) or Atom (ISO 8601) timestamp to an aware UTC datetime, else None."""
    if not value:
        return None
    value = value.strip()
    parsed = None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_text(element, *paths):
    for path in paths:
        node = element.find(path, _ATOM_NS) if ":" in path else element.find(path)
        if node is None:
            continue
        text = node.text or node.get("href")
        if text and text.strip():
            return text.strip()
    return ""


def parse_feed(feed_xml, now=None, lookback_hours=72):
    """Items from an RSS 2.0 or Atom feed published within the lookback window.

    Returns dicts with title, description, link, published (aware UTC), newest first.
    Items with no parseable date are dropped: an undated item cannot be shown to be
    recent, and the whole point of this source is recency.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)

    root = ET.fromstring(feed_xml)
    entries = root.findall("./channel/item") or root.findall("atom:entry", _ATOM_NS)
    items = []
    for entry in entries:
        published = _parse_datetime(_first_text(entry, "pubDate", "atom:published", "atom:updated"))
        if published is None or published < cutoff or published > now + timedelta(hours=3):
            continue
        title = _clean_text(_first_text(entry, "title", "atom:title"), max_chars=220)
        description = _clean_text(
            _first_text(entry, "description", "atom:summary", "atom:content"), max_chars=300
        )
        # The AI Incident Database publishes some items titled "No title" with the
        # substance in the description, and some that are empty placeholders
        # ("... (report_number: 7953)"). Use the description as the title in the
        # first case; drop the second, since there is nothing to judge.
        if not title or title.lower() in _PLACEHOLDER_TITLES:
            title = "" if _is_placeholder_body(description) else description[:160]
        if not title:
            continue
        items.append({
            "title": title,
            "description": description,
            "link": _first_text(entry, "link", "atom:link"),
            "published": published,
        })
    items.sort(key=lambda row: row["published"], reverse=True)
    return items


_ANTHROPIC_DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),\s+(\d{4})\b"
)
_ANTHROPIC_BOILERPLATE_PREFIX = "anthropic is an ai safety and research company"
_ANTHROPIC_CATEGORIES = (
    "societal impacts", "economic research", "case study", "announcements", "interpretability",
    "alignment", "engineering", "education", "research", "product", "policy", "events", "event", "news",
    "science", "features", "feature",
)


def _article_title_and_description(html):
    """(title, description) from an article page's <title> and meta description."""
    soup = BeautifulSoup(html or b"", "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.split(r"\s+[\\|]\s+", title)[0].strip()      # "Title \ Anthropic"
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    description = _clean_text(meta.get("content") if meta else "", max_chars=300)
    if description.lower().startswith(_ANTHROPIC_BOILERPLATE_PREFIX):
        description = ""      # site-wide default, not an article summary
    return _clean_text(title, max_chars=220), description


def parse_anthropic_news_listing(html, now=None, lookback_hours=72, base_url="https://www.anthropic.com",
                                 fetch_article=None):
    """Items from Anthropic's /news listing whose card date is inside the window.

    The card date is day-granular, so the window is compared by calendar date.
    Undated cards (e.g. the standing Responsible Scaling Policy link) are skipped.
    fetch_article(url) -> html bytes, if given, upgrades title/description from the
    article page; on failure the listing title stands and the description is empty.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff_date = (now.astimezone(timezone.utc) - timedelta(hours=lookback_hours)).date()

    soup = BeautifulSoup(html or b"", "html.parser")
    items, seen = [], set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        # Featured cards link outside /news/ (the Opus 5.5 launch was /claude-opus-5-5,
        # a feature was /features/...), so accept any same-site link whose card is dated.
        if href.startswith("/") and not href.startswith("//"):
            link = base_url.rstrip("/") + href
        elif href.startswith(base_url.rstrip("/") + "/"):
            link = href
        else:
            continue
        if link in seen:
            continue
        seen.add(link)
        text = " ".join(anchor.get_text(" ", strip=True).split())
        match = _ANTHROPIC_DATE_RE.search(text)
        if not match:
            continue
        try:
            published = datetime.strptime(
                f"{match.group(1)[:3]} {match.group(2)} {match.group(3)}", "%b %d %Y"
            ).replace(hour=12, tzinfo=timezone.utc)
        except ValueError:
            continue
        if published.date() < cutoff_date or published.date() > now.date() + timedelta(days=1):
            continue
        title = (text[:match.start()] + " " + text[match.end():]).strip()
        lowered = title.lower()
        for category in _ANTHROPIC_CATEGORIES:
            if lowered.startswith(category + " "):
                title = title[len(category):].strip()
                break
        description = ""
        if fetch_article is not None:
            article_html = fetch_article(link)
            if article_html:
                better_title, description = _article_title_and_description(article_html)
                if better_title:
                    title = better_title
        title = _clean_text(title, max_chars=220)
        if not title:
            continue
        items.append({"title": title, "description": description, "link": link, "published": published})
    items.sort(key=lambda row: row["published"], reverse=True)
    return items


def _fetch(url, attempts=3):
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NewscasterWatchlist/1.0)"},
                timeout=(5, 20),
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            wait = attempt + 1
            print_and_write("Watchlist feed fetch failed", url, str(exc), f"Retrying in {wait}s")
            time.sleep(wait)
    return None


def collect_watch_items(feeds, now=None, lookback_hours=72, max_per_feed=8):
    """Fetch every feed. Returns (items, failed_sources).

    items: list of (source_name, item) newest first within each source.
    failed_sources: names whose feed could not be fetched or parsed.
    """
    items, failed = [], []
    for entry in feeds:
        name, url = entry[0], entry[1]
        kind = entry[2] if len(entry) > 2 else "rss"
        content = _fetch(url)
        if content is None:
            failed.append(name)
            continue
        try:
            if kind == "anthropic-news":
                parsed = parse_anthropic_news_listing(
                    content, now=now, lookback_hours=lookback_hours,
                    fetch_article=lambda link: _fetch(link, attempts=1),
                )
            else:
                parsed = parse_feed(content, now=now, lookback_hours=lookback_hours)
        except ET.ParseError as exc:
            print_and_write("Watchlist feed parse failed", name, str(exc))
            failed.append(name)
            continue
        items.extend((name, item) for item in parsed[:max_per_feed])
    return items, failed


def format_items_for_test(items):
    lines = []
    for name, item in items:
        stamp = item["published"].astimezone().strftime("%b %-d, %Y")
        desc = f" — {item['description']}" if item.get("description") else ""
        lines.append(f"[{name}, {stamp}] {item['title']}{desc}")
    return "\n".join(lines)


def select_with_llm(items, prompt_template, today_str, label, **extra):
    """One heavy-model call over formatted items. Returns chosen lines, [] for NONE, None if degraded."""
    if not items:
        return []
    prompt = prompt_template.format(date=today_str, items=format_items_for_test(items), **extra)
    response = call_with_default(None, prompt, mode="heavy", _log_label=label)
    if response is None:
        return None
    lines = [line.strip().lstrip("-*• ").strip() for line in response.split("\n")]
    lines = [line for line in lines if line]
    if not lines or any(_NONE_RE.match(line) for line in lines):
        return []
    return lines


def apply_event_test(items, today_str):
    """The AI watch's event test. Returns passing lines, [] for NONE PASS, or None if degraded."""
    return select_with_llm(items, WATCHLIST_EVENT_TEST_PROMPT, today_str, "watchlist-event-test")


def feed_group_section(group_name, feeds, prompt_template, *, intro, label, now=None,
                       lookback_hours=72, max_per_feed=8, max_items=None):
    """Pool section text for one group of feeds: fetch, window, select, and always say what happened."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone().strftime("%B %e, %Y").replace("  ", " ")
    items, failed = collect_watch_items(feeds, now=now, lookback_hours=lookback_hours, max_per_feed=max_per_feed)
    print_and_write(
        f"{group_name}: {len(items)} items from {len(feeds) - len(failed)}/{len(feeds)} feeds "
        f"in the last {lookback_hours}h" + (f"; failed: {', '.join(failed)}" if failed else "")
    )
    chosen = select_with_llm(items, prompt_template, today, label, group=group_name, max_items=max_items)
    lines = [intro.format(today=today)]
    if chosen is None:
        lines.append("The selection could not be run today (LLM call degraded); no items are nominated.")
    elif not chosen:
        lines.append(f"No items from these sources qualified in the last {lookback_hours} hours.")
    else:
        lines.extend(chosen)
    if failed:
        lines.append(f"Feeds that could not be fetched today: {', '.join(failed)}.")
    return "\n".join(lines) + "\n\n"


def beat_scraper(group_name, feeds, now=None, lookback_hours=None, max_items=None, max_per_feed=None):
    """One beat (business, science, health, courts, world) as a pool section."""
    return feed_group_section(
        group_name, feeds, BEAT_SELECTION_PROMPT,
        intro=f"{group_name} beat, as of {{today}}. Events selected from specialist feeds; judge them by the same criteria as any other source.",
        label=f"beat-{group_name.lower().replace(' ', '-')}", now=now,
        lookback_hours=lookback_hours or getattr(_config, "BEAT_LOOKBACK_HOURS", 24),
        max_per_feed=max_per_feed or getattr(_config, "BEAT_MAX_ITEMS_PER_FEED", 30),
        max_items=max_items or getattr(_config, "BEAT_MAX_ITEMS", 8),
    )


def watchlist_scraper(feeds=None, now=None, lookback_hours=None, max_per_feed=None):
    """Return the pool section text for the specialist watch.

    Always names the source, and always says what happened when nothing came
    through, so the daily log distinguishes a quiet week from a broken feed.
    """
    feeds = feeds if feeds is not None else getattr(_config, "WATCHLIST_FEEDS", [])
    return feed_group_section(
        "Watchlist", feeds, WATCHLIST_EVENT_TEST_PROMPT,
        intro=f"{SECTION_NAME}, as of {{today}}. These items passed an event test; judge them by the same criteria as any other source.",
        label="watchlist-event-test", now=now,
        lookback_hours=lookback_hours or getattr(_config, "WATCHLIST_LOOKBACK_HOURS", 72),
        max_per_feed=max_per_feed or getattr(_config, "WATCHLIST_MAX_ITEMS_PER_FEED", 8),
    )
