"""Main stories start from the links their Tier-2 brief already found (2026-10-12).

That morning the DOJ Civil Rights story was dropped: its article search came back as
Instagram posts, while its brief had already cited NPR, WYPR, justice.gov and Bloomberg Law.
"""
from unittest.mock import patch

from newscaster import pipeline, search
from newscaster.scrapers import topic_finder

NPR = "https://www.npr.org/2026/10/11/g-s1-145063/justice-department-civil-rights-transformed"
DOJ_BRIEF = ("The Justice Department's Civil Rights Division has lost about 70% of its attorneys and shifted "
             "enforcement priorities under its new chief, NPR reported.")


def test_web_brief_returns_its_source_links():
    message = {"content": "Brief text.", "annotations": [
        {"type": "url_citation", "url_citation": {"url": NPR, "title": "Justice Department civil rights transformed",
                                                  "content": "The division has lost..."}},
        {"type": "url_citation", "url_citation": {"url": "https://www.justice.gov/crt", "title": "CRT"}}]}
    with patch.object(search, "_openrouter_web_chat", return_value=(message, {"usage": {}})):
        text, links = search.openrouter_web_brief("question", with_sources=True)
    assert text == "Brief text."
    assert [l["url"] for l in links] == [NPR, "https://www.justice.gov/crt"]
    assert links[0]["headline"] == "Justice Department civil rights transformed"


def test_a_picked_story_gets_the_links_of_its_own_brief():
    briefs = [
        ("The Justice Department has reshaped its civil rights mission, changing its work and workforce.", DOJ_BRIEF, []),
        ("India's chief justice refused to meet opposition leaders over voter-roll cuts.", "Chief Justice Surya Kant...", []),
    ]
    links = {briefs[0][0]: [{"url": NPR, "headline": "NPR", "snippet": ""}],
             briefs[1][0]: [{"url": "https://timesofindia.indiatimes.com/x", "headline": "TOI", "snippet": ""}]}
    picked = "Justice Department overhauls Civil Rights Division, replacing staff and shifting enforcement priorities"
    assert [l["url"] for l in topic_finder.links_for_topic(picked, briefs, links)] == [NPR]
    assert topic_finder.links_for_topic("Hurricane Simon weakens near Mexico", briefs, links) == []
    labeled = "Most Important Story for Californians: " + picked
    assert [l["url"] for l in topic_finder.links_for_topic(labeled, briefs, links)] == [NPR]


def test_gather_reads_brief_links_before_searching():
    seen = []

    def fake_piper(summary_prompt, counter, topic, result, *a, **k):
        seen.append(result["url"])
        return summary_prompt + "x", counter + 1

    links = [{"url": "https://www.instagram.com/p/x/", "headline": "IG", "snippet": ""}] + \
            [{"url": f"https://news.example/{i}", "headline": "h", "snippet": ""} for i in range(3)]
    with patch.object(pipeline, "result_piper", side_effect=fake_piper), \
         patch.object(pipeline, "_article_search", return_value=[]) as search_call:
        _prompt, count = pipeline._collect_articles("DOJ story", 1, "October 12, 2026", "2026_10_12", links=links)
    assert count == 3 and seen == [f"https://news.example/{i}" for i in range(3)]   # Instagram skipped
    search_call.assert_not_called()      # three good articles from the brief: no search needed


def test_without_brief_links_the_search_still_runs():
    found = [{"url": "https://www.npr.org/a", "headline": "h", "snippet": ""}]
    with patch.object(pipeline, "result_piper", side_effect=lambda p, c, *a, **k: (p, c + 1)), \
         patch.object(pipeline, "_article_search", return_value=found) as search_call:
        _prompt, count = pipeline._collect_articles("DOJ story", 1, "October 12, 2026", "2026_10_12", links=None)
    assert count == 1 and search_call.called
