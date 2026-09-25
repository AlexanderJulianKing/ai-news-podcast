"""Browser scraper: bot-check detection, prompt shape, and fallback to the old scrape."""
from unittest.mock import patch

import pytest

from newscaster.scrapers import browser, topic_finder

FRONT_PAGE = "LIVE Trump welcomes Xi to Washington for state visit. " * 80


def test_check_page_rejects_bot_checks_and_thin_pages():
    with pytest.raises(browser.RenderError, match="bot check"):
        browser.check_page("apnews.com Performing security verification. Verifying...")
    with pytest.raises(browser.RenderError, match="too thin"):
        browser.check_page("Menu Sign in Donate")
    browser.check_page(FRONT_PAGE)  # a real page passes


def test_scrape_rendered_sends_page_text_with_order_note():
    seen = {}

    def ask(prompt):
        seen["prompt"] = prompt
        return "Trump welcomed Xi Jinping to Washington for a state visit."
    with patch.object(browser, "render_page", return_value={"title": "AP", "text": FRONT_PAGE}):
        out = browser.scrape_rendered("https://apnews.com", "ap", "EVENT RULES. ", "TIMESTAMP RULES. ", ask)
    assert out.startswith("Trump welcomed Xi")
    assert "in the order the page shows them" in seen["prompt"]
    assert "RENDERED FRONT PAGE TEXT (https://apnews.com)" in seen["prompt"] and "Trump welcomes Xi" in seen["prompt"]


def test_front_page_falls_back_when_the_browser_fails(monkeypatch):
    monkeypatch.setattr(topic_finder._config, "BROWSER_SCRAPE_ENABLED", True)
    monkeypatch.setattr(topic_finder._config, "BROWSER_SCRAPE_SOURCES", ("ap",))
    with patch.object(topic_finder, "scrape_rendered", side_effect=browser.RenderError("bot check page")):
        out = topic_finder._front_page("ap", "https://apnews.com", "scrape-ap", lambda: "GEMINI LIST", "E", "T")
    assert out == "GEMINI LIST"
    with patch.object(topic_finder, "scrape_rendered", return_value="BROWSER LIST"):
        assert topic_finder._front_page("ap", "https://apnews.com", "scrape-ap", lambda: "GEMINI LIST", "E", "T") == "BROWSER LIST"
    # sources not listed never touch the browser
    with patch.object(topic_finder, "scrape_rendered") as sr:
        assert topic_finder._front_page("pp", "https://www.propublica.org", "scrape-pp", lambda: "GEMINI", "E", "T") == "GEMINI"
    sr.assert_not_called()


def test_render_page_without_chromium_raises(monkeypatch):
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    with pytest.raises(browser.RenderError, match="not installed"):
        browser.render_page("https://apnews.com")


def test_reap_orphans_kills_only_our_browsers(monkeypatch, tmp_path):
    killed = []
    monkeypatch.setattr(browser.tempfile, "gettempdir", lambda: str(tmp_path))
    (tmp_path / "newscaster_chromium_old").mkdir()
    (tmp_path / "someone_else").mkdir()

    class Done:
        stdout = "111 222"
    monkeypatch.setattr(browser.subprocess, "run", lambda *a, **k: Done())
    monkeypatch.setattr(browser.os, "getpgid", lambda pid: {0: 1, 111: 111, 222: 1}[pid])
    monkeypatch.setattr(browser.os, "killpg", lambda g, sig: killed.append(g))
    browser.reap_orphans()
    assert killed == [111]                                  # never our own process group
    assert not (tmp_path / "newscaster_chromium_old").exists()
    assert (tmp_path / "someone_else").exists()


def test_order_note_treats_the_live_page_as_current():
    assert "treat every story on it as current news" in browser.ORDER_NOTE
    assert "with or without a timestamp" in browser.ORDER_NOTE
