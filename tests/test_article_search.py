"""Main-story article search skips sites that can't be read without logging in (2026-10-12).

That morning the DOJ Civil Rights story's search came back mostly Instagram posts. They
passed the relevance check, every scrape failed, and the story was dropped from the show.
"""
from unittest.mock import patch

from newscaster import pipeline
from newscaster.search import _tinyfish_params


def test_article_search_excludes_social_sites_and_drops_any_that_slip_through():
    results = [{"headline": "IG post", "url": "https://www.instagram.com/p/abc/"},
               {"headline": "NPR story", "url": "https://www.npr.org/2026/10/12/doj"},
               {"headline": "thread", "url": "https://old.reddit.com/r/politics/x"}]
    with patch.object(pipeline, "search_web", return_value=results) as search:
        out = pipeline._article_search("DOJ overhauls Civil Rights Division", 9)
    query = search.call_args.args[0]
    assert query.startswith("DOJ overhauls Civil Rights Division") and "-site:instagram.com" in query
    assert [r["url"] for r in out] == ["https://www.npr.org/2026/10/12/doj"]
    # TinyFish gets the exclusions as exclude_domains, not as query text
    params = _tinyfish_params(query)
    assert "instagram.com" in params["exclude_domains"] and "site:" not in params["query"]
