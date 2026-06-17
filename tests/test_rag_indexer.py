"""Tests for research-record building and day indexing."""
import json
import os
from unittest.mock import patch

from newscaster.rag import indexer
from newscaster.rag.store import ResearchIndex


def test_build_research_record_assigns_followup_chunk_ids():
    rec = indexer.build_research_record(
        date="2026_03_09", slot=0, topic="fire", arc_slug="slug-a",
        articles=[{"chunk_id": "2026_03_09_seg0_art0", "summary": "s",
                   "url": "u", "outlet": "NPR", "original_headline": "h",
                   "published_date": None, "retrieved_date": "2026_03_09",
                   "surfacing_topic": "fire"}],
        followups=[{"asker": "Gemini Flash", "question": "why?", "answer": "because",
                    "challenging": False}],
    )
    assert rec["arc_slug"] == "slug-a"
    assert rec["followups"][0]["chunk_id"] == "2026_03_09_seg0_fu0"


def test_chunks_from_record_makes_article_and_followup_chunks():
    rec = {
        "date": "2026_03_09", "slot": 0, "topic": "fire", "arc_slug": "slug-a",
        "articles": [{"chunk_id": "2026_03_09_seg0_art0", "summary": "body",
                      "url": "u", "outlet": "NPR", "original_headline": "h",
                      "published_date": None, "retrieved_date": "2026_03_09",
                      "surfacing_topic": "fire"}],
        "followups": [{"chunk_id": "2026_03_09_seg0_fu0", "asker": "X",
                       "question": "q", "answer": "a", "challenging": False}],
    }
    specs = indexer.chunks_from_record(rec)
    by_type = {s["chunk_type"]: s for s in specs}
    assert by_type["article"]["text"] == "body"
    assert by_type["article"]["outlet"] == "NPR"
    assert by_type["followup"]["text"] == "Q: q\nA: a"
    assert by_type["followup"]["arc_slug"] == "slug-a"


def test_index_day_embeds_and_upserts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    rec = {
        "date": "2026_03_09", "slot": 0, "topic": "fire", "arc_slug": None,
        "articles": [{"chunk_id": "2026_03_09_seg0_art0", "summary": "body",
                      "url": "u", "outlet": "NPR", "original_headline": "h",
                      "published_date": None, "retrieved_date": "2026_03_09",
                      "surfacing_topic": "fire"}],
        "followups": [],
    }
    with open("segment_summaries/2026_03_09_segment0_research.json", "w") as f:
        json.dump(rec, f)

    store = ResearchIndex(db_path=tmp_path / "idx.db")
    with patch("newscaster.rag.indexer.embed_texts", return_value=[[1.0, 0.0]]) as mock_embed:
        n = indexer.index_day("2026_03_09", store=store)

    assert n == 1
    mock_embed.assert_called_once_with(["body"], task_type="RETRIEVAL_DOCUMENT")
    assert len(store.search([1.0, 0.0], min_sim=0.0)) == 1


def test_index_day_no_sidecars_returns_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    with patch("newscaster.rag.indexer.embed_texts") as mock_embed:
        assert indexer.index_day("2026_03_09", store=ResearchIndex(db_path=tmp_path / "idx.db")) == 0
    mock_embed.assert_not_called()


def test_index_day_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    rec = {"date": "2026_03_09", "slot": 0, "topic": "t", "arc_slug": None,
           "articles": [{"chunk_id": "2026_03_09_seg0_art0", "summary": "b",
                         "url": "u", "outlet": "O", "original_headline": "h",
                         "published_date": None, "retrieved_date": "2026_03_09",
                         "surfacing_topic": "t"}], "followups": []}
    with open("segment_summaries/2026_03_09_segment0_research.json", "w") as f:
        json.dump(rec, f)
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    with patch("newscaster.rag.indexer.embed_texts", return_value=[[1.0, 0.0]]):
        indexer.index_day("2026_03_09", store=store)
        indexer.index_day("2026_03_09", store=store)
    assert len(store.search([1.0, 0.0], min_sim=0.0)) == 1  # no duplicate


def test_index_day_raises_on_vector_count_mismatch(tmp_path, monkeypatch):
    import pytest
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    rec = {"date": "2026_03_09", "slot": 0, "topic": "t", "arc_slug": None,
           "articles": [
               {"chunk_id": "2026_03_09_seg0_art0", "summary": "b1", "url": "u",
                "outlet": "O", "original_headline": "h", "published_date": None,
                "retrieved_date": "2026_03_09", "surfacing_topic": "t"},
               {"chunk_id": "2026_03_09_seg0_art1", "summary": "b2", "url": "u",
                "outlet": "O", "original_headline": "h", "published_date": None,
                "retrieved_date": "2026_03_09", "surfacing_topic": "t"},
           ], "followups": []}
    with open("segment_summaries/2026_03_09_segment0_research.json", "w") as f:
        json.dump(rec, f)
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    with patch("newscaster.rag.indexer.embed_texts", return_value=[[1.0, 0.0]]):  # 1 vector, 2 specs
        with pytest.raises(RuntimeError):
            indexer.index_day("2026_03_09", store=store)


def test_index_day_skips_malformed_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("logs", exist_ok=True)
    # Article missing the required 'chunk_id' key -> KeyError -> sidecar skipped, not fatal.
    bad = {"date": "2026_03_09", "slot": 0, "topic": "t", "arc_slug": None,
           "articles": [{"summary": "b", "url": "u", "outlet": "O",
                         "original_headline": "h", "published_date": None,
                         "retrieved_date": "2026_03_09", "surfacing_topic": "t"}],
           "followups": []}
    with open("segment_summaries/2026_03_09_segment0_research.json", "w") as f:
        json.dump(bad, f)
    with patch("newscaster.rag.indexer.embed_texts") as mock_embed:
        n = indexer.index_day("2026_03_09", store=ResearchIndex(db_path=tmp_path / "idx.db"))
    assert n == 0
    mock_embed.assert_not_called()
