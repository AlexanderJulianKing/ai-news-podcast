"""Tests for the SQLite + NumPy research vector store."""
from newscaster.rag.store import ResearchIndex, Chunk


def _chunk(cid, vec, date="2026_03_08", ctype="article"):
    return Chunk(
        chunk_id=cid, date=date, arc_slug=None, slot=0, chunk_type=ctype,
        outlet="NPR", headline="H", url="http://x", text="t", vector=vec,
    )


def test_upsert_and_search_orders_by_similarity(tmp_path):
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    store.upsert([
        _chunk("a", [1.0, 0.0, 0.0]),
        _chunk("b", [0.0, 1.0, 0.0]),
    ])
    hits = store.search([0.9, 0.1, 0.0], k=2, min_sim=0.0)
    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert hits[0].similarity > hits[1].similarity


def test_min_sim_filters_weak_matches(tmp_path):
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    store.upsert([_chunk("a", [1.0, 0.0, 0.0])])
    assert store.search([0.0, 1.0, 0.0], min_sim=0.5) == []


def test_exclude_date_filters_same_day(tmp_path):
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    store.upsert([
        _chunk("today", [1.0, 0.0, 0.0], date="2026_03_09"),
        _chunk("old", [1.0, 0.0, 0.0], date="2026_03_01"),
    ])
    hits = store.search([1.0, 0.0, 0.0], exclude_date="2026_03_09", min_sim=0.0)
    assert [h.chunk_id for h in hits] == ["old"]


def test_empty_store_returns_empty(tmp_path):
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    assert store.search([1.0, 0.0, 0.0]) == []


def test_upsert_is_idempotent_by_chunk_id(tmp_path):
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    store.upsert([_chunk("a", [1.0, 0.0, 0.0])])
    store.upsert([_chunk("a", [0.0, 1.0, 0.0])])  # same id, new vector
    hits = store.search([0.0, 1.0, 0.0], min_sim=0.9)
    assert len(hits) == 1 and hits[0].chunk_id == "a"


def test_meta_mismatch_raises(tmp_path, monkeypatch):
    import newscaster.config as cfg
    db = tmp_path / "idx.db"
    ResearchIndex(db_path=db)  # writes meta with current model/dim
    monkeypatch.setattr(cfg, "EMBED_MODEL", "different-model")
    import pytest
    with pytest.raises(ValueError):
        ResearchIndex(db_path=db)


def test_zero_norm_vector_is_skipped(tmp_path):
    store = ResearchIndex(db_path=tmp_path / "idx.db")
    store.upsert([_chunk("zero", [0.0, 0.0, 0.0])])
    assert store.search([1.0, 0.0, 0.0], min_sim=0.0) == []
