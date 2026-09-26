"""Unit tests for the RAG recall metric math and cosine ranking.

The retrieval/embedding/LLM paths are exercised by running the benchmark itself;
these tests pin the pure scoring functions, where a silent bug would produce a
misleading headline number.
"""
import math

import numpy as np

from benchmarks.rag_recall.rag_recall_bench import (
    average_precision,
    load_chunks,
    precision_at_k,
    rank_by_cosine,
    rank_prior_by_cosine,
    recall_at_k,
    reciprocal_rank,
    run_cross_episode_eval,
    wilson_interval,
)


# ranked = [a, b, c, d, e]; relevant = {b, d}
RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d"}


def test_recall_at_k_partial_and_full():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0          # a only
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5          # b found, d not
    assert recall_at_k(RANKED, RELEVANT, 4) == 1.0          # b and d found
    assert recall_at_k(RANKED, RELEVANT, 10) == 1.0         # k beyond list


def test_recall_single_relevant_known_item():
    assert recall_at_k(["x", "t", "y"], {"t"}, 1) == 0.0
    assert recall_at_k(["x", "t", "y"], {"t"}, 2) == 1.0


def test_recall_empty_relevant_is_nan():
    assert math.isnan(recall_at_k(RANKED, set(), 5))


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0       # 0 of top 1
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5       # 1 of top 2
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5       # 2 of top 4
    assert math.isnan(precision_at_k(RANKED, RELEVANT, 0))


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5         # first relevant at rank 2
    assert reciprocal_rank(["b"], RELEVANT) == 1.0
    assert reciprocal_rank(["a", "c", "e"], RELEVANT) == 0.0


def test_average_precision():
    # relevant at ranks 2 and 4: AP = (1/2 + 2/4) / 2 = 0.5
    assert average_precision(RANKED, RELEVANT) == 0.5
    # both relevant at the very top: AP = (1/1 + 2/2) / 2 = 1.0
    assert average_precision(["b", "d", "a"], RELEVANT) == 1.0
    assert math.isnan(average_precision(RANKED, set()))


def test_rank_by_cosine_orders_by_similarity_and_excludes():
    chunks = [
        {"chunk_id": "same", "vec": np.array([1.0, 0.0])},
        {"chunk_id": "near", "vec": np.array([0.9, 0.1])},
        {"chunk_id": "far", "vec": np.array([0.0, 1.0])},
        {"chunk_id": "self", "vec": np.array([1.0, 0.0])},
    ]
    query = np.array([1.0, 0.0])
    ranked, sims = rank_by_cosine(query, chunks, exclude_ids=["self"])
    assert "self" not in ranked
    assert ranked[0] == "same"        # cosine 1.0
    assert ranked[-1] == "far"        # cosine 0.0
    assert sims["far"] == 0.0


def test_rank_by_cosine_zero_query_returns_empty():
    chunks = [{"chunk_id": "a", "vec": np.array([1.0, 0.0])}]
    ranked, sims = rank_by_cosine(np.array([0.0, 0.0]), chunks)
    assert ranked == [] and sims == {}


def test_rank_prior_excludes_same_day_and_future_and_applies_threshold():
    chunks = [
        {"chunk_id": "old", "date": "2026_01_01", "vec": np.array([0.8, 0.6])},
        {"chunk_id": "same", "date": "2026_01_02", "vec": np.array([1.0, 0.0])},
        {"chunk_id": "future", "date": "2026_01_03", "vec": np.array([1.0, 0.0])},
    ]
    ranked, _ = rank_prior_by_cosine(
        np.array([1.0, 0.0]), chunks, "2026_01_02", min_sim=0.75
    )
    assert ranked == ["old"]
    ranked, _ = rank_prior_by_cosine(
        np.array([1.0, 0.0]), chunks, "2026_01_02", min_sim=0.85
    )
    assert ranked == []


def test_cross_episode_query_uses_articles_not_post_retrieval_followups():
    base = {
        "slot": 0, "outlet": None, "headline": None, "url": None, "text": "x",
    }
    chunks = [
        {**base, "chunk_id": "prior", "date": "2026_01_01", "arc_slug": "arc",
         "chunk_type": "article", "vec": np.array([1.0, 0.0])},
        {**base, "chunk_id": "article", "date": "2026_01_02", "arc_slug": "arc",
         "chunk_type": "article", "vec": np.array([1.0, 0.0])},
        {**base, "chunk_id": "followup", "date": "2026_01_02", "arc_slug": "arc",
         "chunk_type": "followup", "vec": np.array([-1.0, 0.0])},
    ]
    result = run_cross_episode_eval(chunks, [1], min_sim=0.65)
    assert result["n_queries"] == 1
    assert result["hit_rate@1"] == 1.0


def test_cross_episode_reviewed_aliases_join_fragmented_ledger_arcs():
    base = {
        "slot": 0, "outlet": None, "headline": None, "url": None, "text": "x",
        "chunk_type": "article", "vec": np.array([1.0, 0.0]),
    }
    chunks = [
        {**base, "chunk_id": "prior", "date": "2026_01_01", "arc_slug": "story_v1"},
        {**base, "chunk_id": "current", "date": "2026_01_02", "arc_slug": "story_v2"},
    ]
    raw = run_cross_episode_eval(chunks, [1])
    reviewed = run_cross_episode_eval(
        chunks, [1], arc_aliases={"story_v1": "story", "story_v2": "story"}
    )
    assert raw["n_queries"] == 0
    assert reviewed["n_queries"] == 1
    assert reviewed["hit_rate@1"] == 1.0


def test_wilson_interval_is_bounded_and_not_falsely_certain():
    lo, hi = wilson_interval(9, 9)
    assert 0.70 < lo < 0.71
    assert hi == 1.0


def test_load_chunks_roundtrip(tmp_path):
    import sqlite3
    db = tmp_path / "idx.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "create table chunks(chunk_id TEXT, date TEXT, arc_slug TEXT, slot INTEGER,"
        " chunk_type TEXT, outlet TEXT, headline TEXT, url TEXT, text TEXT, vector BLOB)"
    )
    v = np.array([3.0, 4.0], dtype=np.float32)  # norm 5 -> expect unit vector
    con.execute(
        "insert into chunks values (?,?,?,?,?,?,?,?,?,?)",
        ("c1", "2026_06_20", "arc_x", 0, "article", "NPR", "h", "u", "body", v.tobytes()),
    )
    con.commit()
    con.close()
    chunks = load_chunks(db)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "c1"
    assert abs(np.linalg.norm(chunks[0]["vec"]) - 1.0) < 1e-9
    np.testing.assert_allclose(chunks[0]["vec"], [0.6, 0.8], atol=1e-7)
