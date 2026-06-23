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
    recall_at_k,
    reciprocal_rank,
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
