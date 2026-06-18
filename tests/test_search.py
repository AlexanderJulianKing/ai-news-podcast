from unittest.mock import MagicMock, patch

import pytest

import newscaster.config as cfg
from newscaster.search import openrouter_web_search, search_web


def test_search_web_uses_google_primary(monkeypatch):
    monkeypatch.setattr(cfg, "SEARCH_PROVIDER", "google_cse")
    monkeypatch.setattr(cfg, "SEARCH_FALLBACK_PROVIDER", "openrouter_web")
    with patch("newscaster.search.google_official_search", return_value=[
        {"headline": "A", "url": "https://example.com/a", "snippet": "s"}
    ]) as mock_google, \
         patch("newscaster.search.openrouter_web_search") as mock_or:
        results = search_web("query", num_results=3)

    assert results == [{"headline": "A", "url": "https://example.com/a", "snippet": "s"}]
    mock_google.assert_called_once_with("query", num_results=3, days_prior=1)
    mock_or.assert_not_called()


def test_search_web_falls_back_when_google_fails(monkeypatch):
    monkeypatch.setattr(cfg, "SEARCH_PROVIDER", "google_cse")
    monkeypatch.setattr(cfg, "SEARCH_FALLBACK_PROVIDER", "openrouter_web")
    with patch("newscaster.search.google_official_search", side_effect=RuntimeError("cse dead")), \
         patch("newscaster.search.openrouter_web_search", return_value=[
             {"headline": "B", "url": "https://example.com/b", "snippet": "fallback"}
         ]) as mock_or:
        results = search_web("query", num_results=2, days_prior=3)

    mock_or.assert_called_once_with("query", num_results=2, days_prior=3)
    assert results[0]["url"] == "https://example.com/b"


def test_search_web_falls_back_when_google_empty(monkeypatch):
    monkeypatch.setattr(cfg, "SEARCH_PROVIDER", "google_cse")
    monkeypatch.setattr(cfg, "SEARCH_FALLBACK_PROVIDER", "openrouter_web")
    monkeypatch.setattr(cfg, "SEARCH_FALLBACK_ON_EMPTY", True)
    with patch("newscaster.search.google_official_search", return_value=[]), \
         patch("newscaster.search.openrouter_web_search", return_value=[
             {"headline": "B", "url": "https://example.com/b", "snippet": "fallback"}
         ]) as mock_or:
        assert search_web("query", num_results=2)[0]["url"] == "https://example.com/b"
    mock_or.assert_called_once()


def test_openrouter_web_search_parses_annotations(monkeypatch):
    monkeypatch.setattr(cfg, "OPENROUTER_API_KEY", "key")
    monkeypatch.setattr(cfg, "SEARCH_OPENROUTER_MODEL", "openai/gpt-5.5")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": "[]",
                "annotations": [{
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.com/report",
                        "title": "Report",
                        "content": "excerpt",
                    },
                }],
            }
        }]
    }
    with patch("newscaster.search.requests.post", return_value=response) as mock_post:
        results = openrouter_web_search("query", num_results=5)

    assert results == [{"headline": "Report", "url": "https://example.com/report", "snippet": "excerpt"}]
    payload = mock_post.call_args.kwargs["json"]
    assert payload["plugins"][0]["id"] == "web"
    assert payload["plugins"][0]["engine"] == cfg.SEARCH_OPENROUTER_ENGINE
    assert payload["model"] == "openai/gpt-5.5"


def test_openrouter_web_search_parses_json_content(monkeypatch):
    monkeypatch.setattr(cfg, "OPENROUTER_API_KEY", "key")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": '[{"headline":"Title","url":"https://example.com/t","snippet":"s"}]',
                "annotations": [],
            }
        }]
    }
    with patch("newscaster.search.requests.post", return_value=response):
        results = openrouter_web_search("query", num_results=5)
    assert results[0]["headline"] == "Title"


def test_openrouter_web_search_raises_without_urls(monkeypatch):
    monkeypatch.setattr(cfg, "OPENROUTER_API_KEY", "key")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": "no links"}}]}
    with patch("newscaster.search.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="no URL results"):
            openrouter_web_search("query")


def test_openrouter_web_search_extracts_urls_from_prose(monkeypatch):
    # Regression: when the model returns prose (no JSON array, no annotations),
    # the fallback URL regex must still extract the links.
    monkeypatch.setattr(cfg, "OPENROUTER_API_KEY", "key")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": (
                    "Top sources: https://www.federalreserve.gov/news.htm and "
                    "https://apnews.com/article/xyz"
                ),
                "annotations": [],
            }
        }]
    }
    with patch("newscaster.search.requests.post", return_value=response):
        results = openrouter_web_search("query", num_results=5)
    urls = {r["url"] for r in results}
    assert "https://www.federalreserve.gov/news.htm" in urls
    assert "https://apnews.com/article/xyz" in urls


def test_google_official_search_honors_days_prior(monkeypatch):
    # Regression: days_prior must drive dateRestrict (it was hardcoded to 'd1', so the
    # pipeline's "widen to 3 days" retry was a no-op).
    pytest.importorskip("googleapiclient")
    import newscaster.scrapers.google_search as gs
    monkeypatch.setattr(cfg, "GOOGLE_SEARCH_API_KEY", "k")
    monkeypatch.setattr(cfg, "GOOGLE_CSE_ID", "cse")
    captured = {}

    class _FakeList:
        def execute(self):
            return {"items": [{"title": "t", "link": "https://example.gov/x", "snippet": "s"}]}

    class _FakeCse:
        def list(self, **kwargs):
            captured.update(kwargs)
            return _FakeList()

    class _FakeService:
        def cse(self):
            return _FakeCse()

    with patch("googleapiclient.discovery.build", return_value=_FakeService()):
        gs.google_official_search("federal minimum wage", num_results=5, days_prior=7)

    assert captured["dateRestrict"] == "d7"
