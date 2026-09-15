"""Event-first front-page extraction and the specialist watch as a pool source."""

from unittest.mock import patch

import pytest

from newscaster.prompts import (
    EVENT_SCRAPER_GROUNDED_TAIL,
    EVENT_SCRAPER_PROMPT,
    EVENT_SCRAPER_TIMESTAMP_RULES,
)
from newscaster.scrapers import topic_finder as tf


def test_event_prompt_asks_for_events_not_headlines_and_forbids_added_significance():
    prompt = EVENT_SCRAPER_PROMPT.format(date="September 15, 2026", max_items=20)
    assert "up to 20 items" in prompt
    assert "WHO did WHAT" in prompt
    assert "do not add consequences, reactions or significance" in prompt
    assert "Do not describe a mood" in prompt
    assert "One item per line, no numbering" in prompt
    assert "{" not in prompt  # every placeholder filled


def test_timestamp_rules_and_grounded_tail_format_cleanly():
    rules = EVENT_SCRAPER_TIMESTAMP_RULES.format(date="September 15, 2026")
    assert "'Now' or 'minutes ago'" in rules and "Today in History" in rules
    assert EVENT_SCRAPER_GROUNDED_TAIL.format(source="NPR (npr.org)") == "Source: NPR (npr.org).\n"


@pytest.fixture
def stubbed_scrapers(monkeypatch):
    """Every network/LLM call returns a stub tagged with its label; captures kwargs."""
    calls = []

    def fake_call(default, *args, _log_label=None, **kwargs):
        calls.append((_log_label, args[0] if args else "", kwargs))
        return f"stub from {_log_label}"

    monkeypatch.setattr(tf, "call_with_default", fake_call)
    monkeypatch.setattr(tf, "calmatters_scraper", lambda: "stub from calmatters")
    monkeypatch.setattr(tf, "dropsite_scraper", lambda: "stub from dropsite")
    monkeypatch.setattr(tf._config, "SCRAPE_MAX_ITEMS", 20, raising=False)
    return calls


def test_gather_sections_uses_event_prompt_with_the_right_transport_per_source(stubbed_scrapers, monkeypatch):
    monkeypatch.setattr(tf._config, "WATCHLIST_ENABLED", False, raising=False)
    sections = tf._gather_headline_sections("September 15, 2026")

    names = [name for _h, name, _t in sections]
    assert names == ["NPR", "The Associated Press", "Democracy Now", "ProPublica", "CalMatters",
                     "Drop Site News", "The City of Riverside"]

    by_label = {label: (prompt, kw) for label, prompt, kw in stubbed_scrapers}
    npr_prompt, npr_kw = by_label["scrape-npr"]
    ap_prompt, ap_kw = by_label["scrape-ap"]
    assert "up to 20 items" in npr_prompt and npr_kw.get("grounding") is True
    assert npr_prompt.endswith("Source: NPR's morning news brief and homepage (npr.org).\n")
    assert "up to 20 items" in ap_prompt and ap_kw.get("url_context") is True
    assert "Today in History" in ap_prompt and ap_prompt.endswith("https://apnews.com")
    assert by_label["scrape-dn"][1].get("grounding") is True
    assert by_label["scrape-pp"][1].get("url_context") is True
    # Riverside keeps its own, narrower prompt.
    assert "riversideca.gov" in by_label["scrape-riverside"][0]


def test_watchlist_is_appended_last_when_enabled(stubbed_scrapers, monkeypatch):
    monkeypatch.setattr(tf._config, "WATCHLIST_ENABLED", True, raising=False)
    with patch.object(tf, "watchlist_scraper", return_value="Specialist watch, as of today.\nMETR did X (via METR)\n\n"):
        sections = tf._gather_headline_sections("September 15, 2026")
    assert sections[-1][0] == "Specialist watch:"
    assert sections[-1][1] == "Specialist watch"
    assert "METR did X" in sections[-1][2]
    assert len(sections) == 8


def test_watchlist_disabled_or_failing_or_empty_never_breaks_the_pool(stubbed_scrapers, monkeypatch):
    monkeypatch.setattr(tf._config, "WATCHLIST_ENABLED", False, raising=False)
    assert len(tf._gather_headline_sections("September 15, 2026")) == 7

    monkeypatch.setattr(tf._config, "WATCHLIST_ENABLED", True, raising=False)
    with patch.object(tf, "watchlist_scraper", side_effect=RuntimeError("feed exploded")):
        assert len(tf._gather_headline_sections("September 15, 2026")) == 7
    with patch.object(tf, "watchlist_scraper", return_value="   \n"):
        assert len(tf._gather_headline_sections("September 15, 2026")) == 7


def test_pool_string_and_provenance_index_are_built_from_the_same_sections(stubbed_scrapers, monkeypatch):
    monkeypatch.setattr(tf._config, "WATCHLIST_ENABLED", False, raising=False)
    sections = tf._gather_headline_sections("September 15, 2026")
    pool = "\n\n".join(f"{h}\n{t}" for h, _n, t in sections)
    assert pool.startswith("NPR:\nstub from scrape-npr")
    assert "\n\nThe Associated Press:\nstub from scrape-ap" in pool
    index = tf.build_source_index([(n, t) for _h, n, t in sections])
    assert [n for n, _ in index] == [n for _h, n, _t in sections]
