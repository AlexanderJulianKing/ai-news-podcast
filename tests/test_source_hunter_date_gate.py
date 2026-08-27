"""The as-of date must not veto sources that never state a date.

Regression cover for the 2026-08-24 finding: the overview brief prompt carries a
`Date: <today>` line, `extract_constraints` turned it into a hard per-source
requirement, and 78% of already-fetched coverage was rejected because news pages
keep the publication date out of the extracted text.
"""
from newscaster.source_hunter_primitives import (
    DATE_WINDOW_BACK_DAYS,
    _text_states_a_date,
    _date_present_in_window,
    validate_source_for_question,
)

QUESTION = (
    "Headline: Gaza ceasefire talks resume\n"
    "Date: August 17, 2026\n"
    "Instructions:\n- Summarize the key facts.\n"
)


def _task():
    return {"id": "t", "question": QUESTION, "category": "news_research",
            "evidence_contract": {}}


def _source(excerpt, title="Gaza ceasefire talks resume"):
    return {"ok": True, "title": title,
            "final_url": "https://example.com/gaza-ceasefire-talks-resume",
            "excerpt": excerpt}


def _reasons(excerpt, **kw):
    return validate_source_for_question(_task(), _source(excerpt, **kw))["reasons"]


def test_text_states_a_date_detects_common_formats():
    assert _text_states_a_date("Published August 16, 2026 by staff")
    assert _text_states_a_date("Updated Aug. 16 at noon")
    assert _text_states_a_date("2026-08-16T09:00:00Z")
    assert _text_states_a_date("filed 8/16/2026")


def test_text_states_a_date_false_when_undated():
    assert not _text_states_a_date("Gaza ceasefire talks resumed, officials said.")
    assert not _text_states_a_date("Posted 2 hours ago")
    assert not _text_states_a_date("")


def test_window_accepts_neighbouring_days():
    target = {"month": 8, "day": 17, "year": 2026}
    # a day late, and a day ahead for time zones
    assert _date_present_in_window("filed August 16, 2026", target)
    assert _date_present_in_window("filed August 18, 2026", target)
    assert _date_present_in_window("filed August 17, 2026", target)


def test_window_rejects_clearly_stale_dates():
    target = {"month": 8, "day": 17, "year": 2026}
    assert not _date_present_in_window("filed August 1, 2026", target)
    assert not _date_present_in_window("filed August 17, 2024", target)


def test_undated_source_is_rejected_for_unknown_date():
    reasons = _reasons("Gaza ceasefire talks resumed in Doha, officials said.")
    # not an exact-date failure: we simply cannot tell how old it is
    assert "date_mismatch" not in reasons
    assert "date_unknown" in reasons


def test_source_dated_a_day_earlier_is_not_vetoed():
    reasons = _reasons("Published August 16, 2026. Gaza ceasefire talks resumed in Doha.")
    assert "date_mismatch" not in reasons


def test_stale_dated_source_is_still_vetoed():
    reasons = _reasons("Published January 3, 2019. Gaza ceasefire talks resumed in Doha.")
    assert "date_mismatch" in reasons


def test_exact_date_still_scores_as_a_match():
    result = validate_source_for_question(
        _task(), _source("Published August 17, 2026. Gaza ceasefire talks resumed."))
    assert any(result["matched"].get("dates") or [])
    assert "date_mismatch" not in result["reasons"]


def test_window_covers_the_configured_lookback():
    target = {"month": 8, "day": 17, "year": 2026}
    inside = 17 - DATE_WINDOW_BACK_DAYS
    assert _date_present_in_window(f"filed August {inside}, 2026", target)
    assert not _date_present_in_window(f"filed August {inside - 1}, 2026", target)


# --- publish-date metadata -------------------------------------------------
# 72% of pages that look "undated" in their body text do publish a machine
# readable date. Reading it is what separates today's story from a five-month
# old one that would otherwise air as current news.
from newscaster.source_hunter_primitives import (  # noqa: E402
    extract_published_date,
    _iso_within_window,
)

HARD = [{"month": 8, "day": 17, "year": 2026, "soft": False}]


def test_extract_published_date_from_meta_property():
    html = b'<html><head><meta property="article:published_time" content="2026-03-25T17:23:23Z"></head></html>'
    assert extract_published_date(html) == "2026-03-25"


def test_extract_published_date_from_time_element():
    html = b'<html><body><time datetime="2026-08-16T09:00:00-07:00">yesterday</time></body></html>'
    assert extract_published_date(html) == "2026-08-16"


def test_extract_published_date_from_json_ld():
    html = b'<html><script type="application/ld+json">{"datePublished":"2026-08-15","x":1}</script></html>'
    assert extract_published_date(html) == "2026-08-15"


def test_extract_published_date_none_when_absent():
    assert extract_published_date(b"<html><body><p>no date here</p></body></html>") is None


def test_extract_published_date_ignores_impossible_dates():
    assert extract_published_date(
        b'<html><head><meta name="date" content="2026-13-45"></head></html>') is None


def test_iso_window_accepts_recent_and_rejects_stale():
    assert _iso_within_window("2026-08-17", HARD)
    assert _iso_within_window("2026-08-16", HARD)
    assert _iso_within_window("2026-08-18", HARD)   # time zones
    assert not _iso_within_window("2026-03-25", HARD)
    assert not _iso_within_window("2022-03-12", HARD)


def test_stale_metadata_date_vetoes_even_when_body_is_undated():
    """The exact 2026-08-13 failure: a March story airing as today's news."""
    source = {"ok": True, "title": "Jury finds Meta and YouTube liable",
              "final_url": "https://apnews.com/article/social-media-addiction-trial",
              "excerpt": "A jury found the platforms liable, with no date in the body.",
              "published_date": "2026-03-25"}
    result = validate_source_for_question(_task(), source)
    assert "date_mismatch" in result["reasons"]
    assert not result["passed"]


def test_fresh_metadata_date_passes():
    source = {"ok": True, "title": "Gaza ceasefire talks resume",
              "final_url": "https://example.com/gaza-ceasefire-talks-resume",
              "excerpt": "Talks resumed in Doha, officials said.",
              "published_date": "2026-08-16"}
    assert "date_mismatch" not in validate_source_for_question(_task(), source)["reasons"]


def test_undated_crawled_link_is_not_treated_as_current():
    source = {"ok": True, "title": "Gaza ceasefire talks resume",
              "final_url": "https://example.com/gaza-ceasefire-talks-resume",
              "excerpt": "Talks resumed in Doha, officials said.",
              "candidate_source": "nearby_link"}
    result = validate_source_for_question(_task(), source)
    assert "date_unknown" in result["reasons"]
    assert not result["passed"]


def test_undated_search_result_is_also_rejected():
    """A search hit with no establishable date is still not evidence of today."""
    source = {"ok": True, "title": "Gaza ceasefire talks resume",
              "final_url": "https://example.com/gaza-ceasefire-talks-resume",
              "excerpt": "Talks resumed in Doha, officials said.",
              "candidate_source": "content"}
    result = validate_source_for_question(_task(), source)
    assert "date_unknown" in result["reasons"]
    assert not result["passed"]


# --- production-path propagation ------------------------------------------
# The Aug-24 fix originally landed its published_date line in
# fetch_controlled_evidence (benchmark-only); production uses
# fetch_discovered_evidence, so the metadata rescue was silently absent on
# the live path. These tests go through the REAL production function with a
# mocked fetch, so a repeat of that mis-wiring fails loudly.
from unittest.mock import patch as _patch  # noqa: E402
from newscaster.source_hunter_primitives import fetch_discovered_evidence  # noqa: E402


def _fetched(published):
    return {"url": "https://example.com/gaza-ceasefire-talks-resume",
            "title": "Gaza ceasefire talks resume",
            "content_type": "text/html", "status_code": 200,
            "text": "Talks resumed in Doha, officials said.",
            "published_date": published, "char_count": 38, "links": []}


def _discover(published):
    with _patch("newscaster.source_hunter_primitives.fetch_source_text",
                return_value=_fetched(published)):
        out = fetch_discovered_evidence(
            {"question": QUESTION, "evidence_contract": {}},
            [{"url": "https://example.com/gaza-ceasefire-talks-resume",
              "title": "t", "reason": "", "source": "content"}],
        )
    return out["sources"][0]


def test_discovered_evidence_carries_published_date():
    assert _discover("2026-08-16")["published_date"] == "2026-08-16"


def test_metadata_rescue_works_through_production_path():
    src = _discover("2026-08-16")  # body text is undated; date only in metadata
    result = validate_source_for_question(_task(), src)
    assert "date_unknown" not in result["reasons"]
    assert "date_mismatch" not in result["reasons"]


def test_stale_metadata_blocks_through_production_path():
    src = _discover("2026-03-25")
    result = validate_source_for_question(_task(), src)
    assert "date_mismatch" in result["reasons"]
    assert not result["passed"]
