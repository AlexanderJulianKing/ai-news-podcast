# RAG Research Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture each pulled source document as a structured, provenance-rich record, build a semantic index over that research, and at gather time fold relevant prior coverage into the synthesis via a retrieve-then-refine pass.

**Architecture:** A new `newscaster/rag/` package owns embeddings (Gemini `embed_content`), a SQLite + NumPy brute-force vector store, and an indexer. The existing gather pipeline gains (a) capture accumulators that turn discarded provenance + follow-up Q&A into per-slot `_research.json` sidecars, and (b) a gated, fail-safe refine step. Capture and indexing ship enabled; the behavior-changing refine pass is gated behind `RAG_AUGMENT_ENABLED` (default off).

**Tech Stack:** Python 3.11, `google-genai` (already a dep), `numpy` (new direct dep), stdlib `sqlite3`, `pytest` + `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-06-16-rag-research-augmentation-design.md`

---

## File Structure

**New files:**
- `newscaster/rag/__init__.py` — package marker (empty)
- `newscaster/rag/embeddings.py` — `embed_texts()`; Gemini adapter mirroring `llm/gemini.py` error discipline
- `newscaster/rag/store.py` — `Chunk`, `Hit` dataclasses; `ResearchIndex` (SQLite + NumPy cosine)
- `newscaster/rag/indexer.py` — `build_research_record()`, `chunks_from_record()`, `index_day()`
- `newscaster/rag/retrieve.py` — `retrieve_prior_research()`
- `tests/test_rag_store.py`, `tests/test_rag_embeddings.py`, `tests/test_rag_indexer.py`, `tests/test_rag_capture.py`, `tests/test_rag_gather_sidecar.py`, `tests/test_rag_refine.py`

**Modified files:**
- `requirements.txt` — add `numpy`
- `newscaster/config.py` — add RAG tunables (module-level constants)
- `newscaster/scrapers/topic_finder.py` — `result_piper` gains optional `articles` accumulator
- `newscaster/pipeline.py` — `_run_follow_up_rounds` gains `followups` accumulator; `_gather_one_topic` threads accumulators + refine hook; `gather_news` writes sidecars + indexes; new `_augment_with_prior_research()`
- `newscaster/prompts.py` — add `RAG_REFINE_PROMPT`

**Shared contracts (used across tasks — keep identical):**
- `Chunk(chunk_id, date, arc_slug, slot, chunk_type, outlet, headline, url, text, vector)`
- `Hit(chunk_id, date, chunk_type, outlet, headline, url, text, similarity)`
- Research record dict: `{date, slot, topic, arc_slug, articles: [...], followups: [...]}`
  - article item: `{chunk_id, url, outlet, original_headline, published_date, retrieved_date, surfacing_topic, summary}`
  - followup item: `{chunk_id, asker, question, answer, challenging}`
- `embed_texts(texts, *, task_type="RETRIEVAL_DOCUMENT", model=None, dimension=None) -> list[list[float]]`
- `ResearchIndex(db_path=None)`; `.upsert(chunks) -> int`; `.search(query_vec, k=None, exclude_date=None, min_sim=None) -> list[Hit]`
- `index_day(date, store=None) -> int`

---

## Task 1: Config tunables + numpy dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `newscaster/config.py:51-55` (constants block)
- Test: `tests/test_config.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_rag_constants_present():
    """RAG tunables are module-level constants, available without init()."""
    import newscaster.config as cfg
    assert cfg.EMBED_MODEL == "gemini-embedding-2"
    assert cfg.EMBED_DIM == 1536
    assert cfg.RAG_TOP_K == 6
    assert isinstance(cfg.RAG_MIN_SIM, float)
    assert cfg.RAG_AUGMENT_ENABLED is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py::test_rag_constants_present -v`
Expected: FAIL with `AttributeError: module 'newscaster.config' has no attribute 'EMBED_MODEL'`

- [ ] **Step 3: Add the constants**

In `newscaster/config.py`, after the `FALLBACK_MODEL = "openai/gpt-5.5"` line, add:

```python
# --- RAG / embeddings tunables ---
EMBED_MODEL = "gemini-embedding-2"   # verified current; space incompatible with -001
EMBED_DIM = 1536                     # pinned; changing requires a full re-embed
RAG_TOP_K = 6                        # chunks retrieved per refine
RAG_MIN_SIM = 0.65                   # cosine floor; below -> inject nothing (tune empirically)
RAG_AUGMENT_ENABLED = False          # gates the retrieve-then-refine pass
```

- [ ] **Step 4: Add numpy to requirements**

Append `numpy` as a new line at the end of `requirements.txt`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: PASS (all config tests)

- [ ] **Step 6: Commit**

```bash
git add newscaster/config.py requirements.txt tests/test_config.py
git commit -m "Add RAG config tunables and numpy dependency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Vector store (`store.py`)

**Files:**
- Create: `newscaster/rag/__init__.py` (empty)
- Create: `newscaster/rag/store.py`
- Test: `tests/test_rag_store.py`

- [ ] **Step 1: Create the empty package marker**

Create `newscaster/rag/__init__.py` with a single line:

```python
"""Retrieval-augmented generation: embeddings, vector store, indexing, retrieval."""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_rag_store.py`:

```python
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
    # Orthogonal query -> cosine 0, below floor -> nothing.
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rag_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newscaster.rag.store'`

- [ ] **Step 4: Implement the store**

Create `newscaster/rag/store.py`:

```python
"""SQLite + NumPy brute-force vector store for research chunks.

Small-N by design (a few thousand chunks ~= a year of output): load all vectors,
compute cosine in NumPy, return top-k. Switch to FAISS only above ~50k vectors.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import newscaster.config as _config

DEFAULT_DB_PATH = Path("stories_chosen/research_index.db")  # relative to CWD (repo root in prod), like the pipeline's other outputs; keeps tests that chdir(tmp_path) hermetic


@dataclass
class Chunk:
    chunk_id: str
    date: str
    arc_slug: str | None
    slot: int
    chunk_type: str   # 'article' | 'followup'
    outlet: str | None
    headline: str | None
    url: str | None
    text: str
    vector: list      # list[float], length == EMBED_DIM


@dataclass
class Hit:
    chunk_id: str
    date: str
    chunk_type: str
    outlet: str | None
    headline: str | None
    url: str | None
    text: str
    similarity: float


class ResearchIndex:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks(
                   chunk_id TEXT PRIMARY KEY, date TEXT, arc_slug TEXT, slot INTEGER,
                   chunk_type TEXT, outlet TEXT, headline TEXT, url TEXT,
                   text TEXT, vector BLOB)"""
        )
        row = self._conn.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO meta(key,value) VALUES('embed_model',?)", (_config.EMBED_MODEL,))
            self._conn.execute("INSERT INTO meta(key,value) VALUES('embed_dim',?)", (str(_config.EMBED_DIM),))
            self._conn.commit()
        else:
            existing_model = row["value"]
            existing_dim = self._conn.execute("SELECT value FROM meta WHERE key='embed_dim'").fetchone()["value"]
            if existing_model != _config.EMBED_MODEL or existing_dim != str(_config.EMBED_DIM):
                raise ValueError(
                    f"Index built with {existing_model}/{existing_dim} but config is "
                    f"{_config.EMBED_MODEL}/{_config.EMBED_DIM}; spaces are incompatible — re-embed required"
                )

    def upsert(self, chunks):
        for c in chunks:
            blob = np.asarray(c.vector, dtype=np.float32).tobytes()
            self._conn.execute(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id,date,arc_slug,slot,chunk_type,outlet,headline,url,text,vector)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (c.chunk_id, c.date, c.arc_slug, c.slot, c.chunk_type,
                 c.outlet, c.headline, c.url, c.text, blob),
            )
        self._conn.commit()
        return len(chunks)

    def search(self, query_vec, k=None, exclude_date=None, min_sim=None):
        k = _config.RAG_TOP_K if k is None else k
        min_sim = _config.RAG_MIN_SIM if min_sim is None else min_sim
        rows = self._conn.execute(
            "SELECT chunk_id,date,chunk_type,outlet,headline,url,text,vector FROM chunks"
        ).fetchall()
        if not rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        hits = []
        for r in rows:
            if exclude_date is not None and r["date"] == exclude_date:
                continue
            v = np.frombuffer(r["vector"], dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            sim = float(np.dot(q, v / vn))
            if sim < min_sim:
                continue
            hits.append(Hit(
                chunk_id=r["chunk_id"], date=r["date"], chunk_type=r["chunk_type"],
                outlet=r["outlet"], headline=r["headline"], url=r["url"],
                text=r["text"], similarity=sim,
            ))
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:k]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rag_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add newscaster/rag/__init__.py newscaster/rag/store.py tests/test_rag_store.py
git commit -m "Add SQLite+NumPy research vector store

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Embeddings adapter (`embeddings.py`)

**Files:**
- Create: `newscaster/rag/embeddings.py`
- Test: `tests/test_rag_embeddings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_embeddings.py`:

```python
"""Tests for the Gemini embeddings adapter (client mocked)."""
from unittest.mock import patch, MagicMock
import pytest

from newscaster.rag import embeddings
from newscaster.llm.errors import LLMError


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


def _fake_response(vectors):
    resp = MagicMock()
    resp.embeddings = [_FakeEmbedding(v) for v in vectors]
    return resp


def test_embed_texts_returns_vectors():
    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = _fake_response([[0.1, 0.2], [0.3, 0.4]])
    with patch("newscaster.rag.embeddings.genai.Client", return_value=fake_client):
        out = embeddings.embed_texts(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    # contents passed through as a list
    _, kwargs = fake_client.models.embed_content.call_args
    assert kwargs["contents"] == ["a", "b"]


def test_embed_texts_empty_input_skips_api():
    with patch("newscaster.rag.embeddings.genai.Client") as ctor:
        assert embeddings.embed_texts([]) == []
    ctor.assert_not_called()


def test_embed_texts_maps_api_error_to_llmerror():
    fake_client = MagicMock()
    fake_client.models.embed_content.side_effect = RuntimeError("boom")
    with patch("newscaster.rag.embeddings.genai.Client", return_value=fake_client):
        with pytest.raises(LLMError):
            embeddings.embed_texts(["a"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rag_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newscaster.rag.embeddings'`

- [ ] **Step 3: Implement the adapter**

Create `newscaster/rag/embeddings.py`:

```python
"""Gemini embeddings adapter.

Mirrors the error discipline of newscaster/llm/gemini.py: SDK exceptions are
classified into the typed LLMError hierarchy so callers can decide fallback.
"""
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import newscaster.config as _config
from newscaster.llm.errors import LLMMalformedResponseError, classify


def embed_texts(texts, *, task_type="RETRIEVAL_DOCUMENT", model=None, dimension=None):
    """Embed a list of strings. Returns list[list[float]] parallel to `texts`.

    Empty input returns [] without an API call. Raises a typed LLMError on failure.
    """
    if not texts:
        return []
    model = model or _config.EMBED_MODEL
    dimension = dimension or _config.EMBED_DIM
    try:
        client = genai.Client(api_key=_config.GOOGLE_GENAI_API_KEY)
        response = client.models.embed_content(
            model=model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                output_dimensionality=dimension,
                task_type=task_type,
                # NOTE: no auto_truncate — that parameter is Vertex-only and the
                # Gemini API rejects it. Callers must keep inputs under ~8192 tokens.
            ),
        )
    except genai_errors.APIError as e:
        status_code = getattr(e, "code", None)
        cls = classify(e, status_code=status_code)
        raise cls(str(e), provider="google", model=model, status_code=status_code) from e
    except Exception as e:
        cls = classify(e)
        raise cls(str(e), provider="google", model=model) from e

    out = getattr(response, "embeddings", None)
    if not out:
        raise LLMMalformedResponseError(
            "embed_content returned no embeddings", provider="google", model=model
        )
    return [list(e.values) for e in out]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rag_embeddings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Manual smoke test (verify real API shape; requires keys.txt + network)**

Run:
```bash
python3 -c "
import newscaster.config as c; c.init()
from newscaster.rag.embeddings import embed_texts
v = embed_texts(['hello world'])
print('dim', len(v[0]))
"
```
Expected: `dim 1536`. If this errors on the response shape (`.embeddings` / `.values`), fix `embed_texts` to match the installed SDK and re-run Step 4. If the model id `gemini-embedding-2` is rejected, confirm the current id at <https://ai.google.dev/gemini-api/docs/embeddings> and update `EMBED_MODEL`.

- [ ] **Step 6: Commit**

```bash
git add newscaster/rag/embeddings.py tests/test_rag_embeddings.py
git commit -m "Add Gemini embeddings adapter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Indexer (`indexer.py`)

**Files:**
- Create: `newscaster/rag/indexer.py`
- Test: `tests/test_rag_indexer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_indexer.py`:

```python
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
    mock_embed.assert_called_once()
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rag_indexer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newscaster.rag.indexer'`

- [ ] **Step 3: Implement the indexer**

Create `newscaster/rag/indexer.py`:

```python
"""Build per-slot research records and index a day's research into the store."""
import glob
import json

from newscaster.logging import print_and_write
from newscaster.rag.embeddings import embed_texts
from newscaster.rag.store import Chunk, ResearchIndex


def build_research_record(date, slot, topic, arc_slug, articles, followups):
    """Assemble the sidecar dict. Stamps followup chunk_ids (articles already have them)."""
    stamped_followups = []
    for j, fu in enumerate(followups or []):
        item = dict(fu)
        item["chunk_id"] = f"{date}_seg{slot}_fu{j}"
        stamped_followups.append(item)
    return {
        "date": date,
        "slot": slot,
        "topic": topic,
        "arc_slug": arc_slug,
        "articles": list(articles or []),
        "followups": stamped_followups,
    }


def chunks_from_record(record):
    """Return a list of chunk-spec dicts (all Chunk fields except `vector`)."""
    date = record["date"]
    slot = record["slot"]
    arc_slug = record.get("arc_slug")
    specs = []
    for art in record.get("articles", []):
        specs.append({
            "chunk_id": art["chunk_id"], "date": date, "arc_slug": arc_slug, "slot": slot,
            "chunk_type": "article", "outlet": art.get("outlet"),
            "headline": art.get("original_headline"), "url": art.get("url"),
            "text": art.get("summary", ""),
        })
    for fu in record.get("followups", []):
        specs.append({
            "chunk_id": fu["chunk_id"], "date": date, "arc_slug": arc_slug, "slot": slot,
            "chunk_type": "followup", "outlet": None, "headline": None, "url": None,
            "text": f"Q: {fu.get('question', '')}\nA: {fu.get('answer', '')}",
        })
    return [s for s in specs if s["text"].strip()]


def index_day(date, store=None):
    """Embed and upsert every chunk in the day's `_research.json` sidecars. Idempotent."""
    paths = sorted(glob.glob(f"segment_summaries/{date}_segment*_research.json"))
    specs = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print_and_write(f"index_day: skipping unreadable sidecar {path}: {e}")
            continue
        specs.extend(chunks_from_record(record))
    if not specs:
        return 0
    vectors = embed_texts([s["text"] for s in specs], task_type="RETRIEVAL_DOCUMENT")
    store = store or ResearchIndex()
    chunks = [Chunk(vector=vectors[i], **specs[i]) for i in range(len(specs))]
    return store.upsert(chunks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rag_indexer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add newscaster/rag/indexer.py tests/test_rag_indexer.py
git commit -m "Add research record builder and day indexer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Article provenance capture in `result_piper`

**Files:**
- Modify: `newscaster/scrapers/topic_finder.py:98-148` (`result_piper`)
- Test: `tests/test_rag_capture.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_capture.py`:

```python
"""Capture of article provenance in result_piper."""
from unittest.mock import patch

import newscaster.scrapers.topic_finder as tf


def test_result_piper_appends_article_record():
    articles = []
    result = {"headline": "Big Fire", "url": "https://npr.org/fire", "snippet": "...",
              "date": "2026-03-08"}
    with patch.object(tf, "determine_relevance", return_value=True), \
         patch.object(tf, "scrape_text", return_value="full article body"), \
         patch.object(tf, "summarize_text", return_value="a concise summary"), \
         patch.object(tf, "call_with_default", side_effect=["yes", "NPR"]):
        summary_prompt, counter = tf.result_piper(
            "", 0, "wildfire", result, 0, "2026_03_09", articles=articles
        )

    assert counter == 1
    assert len(articles) == 1
    rec = articles[0]
    assert rec["chunk_id"] == "2026_03_09_seg0_art0"
    assert rec["url"] == "https://npr.org/fire"
    assert rec["outlet"] == "NPR"
    assert rec["original_headline"] == "Big Fire"
    assert rec["published_date"] == "2026-03-08"
    assert rec["retrieved_date"] == "2026_03_09"
    assert rec["surfacing_topic"] == "wildfire"
    assert rec["summary"] == "a concise summary"


def test_result_piper_without_accumulator_still_works():
    """Backward compatible: omitting `articles` must not error."""
    result = {"headline": "H", "url": "https://x", "snippet": "s"}
    with patch.object(tf, "determine_relevance", return_value=False):
        summary_prompt, counter = tf.result_piper("seed", 0, "topic", result, 0, "2026_03_09")
    assert counter == 0
    assert summary_prompt == "seed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rag_capture.py -v`
Expected: FAIL with `TypeError: result_piper() got an unexpected keyword argument 'articles'`

- [ ] **Step 3: Modify `result_piper`**

In `newscaster/scrapers/topic_finder.py`, change the signature (line 98) to add the accumulator:

```python
def result_piper(summary_prompt, successful_summary_counter, topic, result, i, formatted_date2, articles=None):
```

Then, inside the `if 'yes' in response.lower():` block, immediately after the existing
`outfile.write(summary)` / `outfile.close()` lines and **before** `successful_summary_counter += 1`, insert:

```python
            if articles is not None:
                articles.append({
                    "chunk_id": f"{formatted_date2}_seg{i}_art{successful_summary_counter}",
                    "url": url,
                    "outlet": (news_source_response or "").strip(),
                    "original_headline": result.get("headline"),
                    "published_date": result.get("date"),
                    "retrieved_date": formatted_date2,
                    "surfacing_topic": topic,
                    "summary": summary,
                })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rag_capture.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Verify existing callers still pass**

Run: `python3 -m pytest tests/ -v -k "gather or topic or pipeline"`
Expected: PASS (no regressions — `articles` defaults to `None`)

- [ ] **Step 6: Commit**

```bash
git add newscaster/scrapers/topic_finder.py tests/test_rag_capture.py
git commit -m "Capture article provenance records in result_piper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Follow-up Q&A capture + thread accumulators through `_gather_one_topic`

**Files:**
- Modify: `newscaster/pipeline.py:30-75` (`_run_follow_up_rounds`)
- Modify: `newscaster/pipeline.py:268-382` (`_gather_one_topic`)
- Test: `tests/test_rag_capture.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_capture.py`:

```python
import newscaster.pipeline as pipeline


def test_run_follow_up_rounds_collects_qa():
    followups = []

    def fake_llm(prompt, system_prompt=None, mode="light", grounding=False, url_context=False):
        # Grounded calls answer questions; ungrounded calls produce a quoted question.
        if grounding:
            return "the grounded answer"
        return '"a follow up question"'

    with patch.object(pipeline, "get_llm_response", side_effect=fake_llm):
        pipeline._run_follow_up_rounds("seed summary", "fup template", "challenging template",
                                       followups=followups)

    assert len(followups) == 8  # 4 modes x (regular + challenging)
    assert followups[0]["question"] == "a follow up question"
    assert followups[0]["answer"] == "the grounded answer"
    assert "asker" in followups[0] and "challenging" in followups[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rag_capture.py::test_run_follow_up_rounds_collects_qa -v`
Expected: FAIL with `TypeError: _run_follow_up_rounds() got an unexpected keyword argument 'followups'`

- [ ] **Step 3: Modify `_run_follow_up_rounds`**

In `newscaster/pipeline.py`, change the signature (line 30):

```python
def _run_follow_up_rounds(summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text, followups=None):
```

Then, in the loop, after the line that appends to `summary_prompt`
(`summary_prompt = summary_prompt + '\n' + f'{asker_name} asked:' + ...`), add:

```python
        if followups is not None:
            followups.append({
                "asker": asker_name,
                "question": follow_up_question,
                "answer": response,
                "challenging": is_challenging,
            })
```

- [ ] **Step 4: Thread accumulators through `_gather_one_topic`**

Change the `_gather_one_topic` signature (line 268) to accept the accumulators:

```python
def _gather_one_topic(topic, topic_index, formatted_date, formatted_date2,
                     follow_up_prompt_text, challenging_follow_up_prompt_text,
                     articles=None, followups=None):
```

At its three `result_piper(...)` call sites (currently lines ~293, ~310, ~337), add the
`articles=articles` keyword argument, e.g.:

```python
        summary_prompt, successful_summary_counter = result_piper(
            summary_prompt, successful_summary_counter, topic, result, topic_index,
            formatted_date2, articles=articles)
```

(Apply the same `articles=articles` addition to all three call sites.)

At the `_run_follow_up_rounds(...)` call (line ~355), pass the accumulator:

```python
    summary_prompt = _run_follow_up_rounds(
        summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text,
        followups=followups,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rag_capture.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add newscaster/pipeline.py tests/test_rag_capture.py
git commit -m "Capture follow-up Q&A and thread research accumulators through gather

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Write sidecars + index in `gather_news`

**Files:**
- Modify: `newscaster/pipeline.py:216-265` (`gather_news` topic loop + post-loop writes)
- Test: `tests/test_rag_gather_sidecar.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_gather_sidecar.py`:

```python
"""gather_news writes per-slot research sidecars and calls index_day."""
import json
import os
from unittest.mock import patch

from newscaster.scrapers.topic_finder import TopicFinderResult
import newscaster.pipeline as pipeline


def _tf(topics, arc_context):
    return TopicFinderResult(
        topics=topics, overview="ov", follow_up_prompt_text="f",
        challenging_follow_up_prompt_text="c", arc_context=arc_context,
        ledger={"arcs": {}}, side_story_briefs=[],
    )


def test_gather_writes_sidecar_with_arc_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("segment_summaries", "output_scripts", "logs"):
        os.makedirs(d)

    tf_result = _tf(["topic-zero"], arc_context=[{"slug": "arc-zero"}])

    def fake_one_topic(topic, idx, *a, articles=None, followups=None, **k):
        if articles is not None:
            articles.append({"chunk_id": f"2026_11_05_seg{idx}_art0", "url": "u",
                             "outlet": "NPR", "original_headline": "h",
                             "published_date": None, "retrieved_date": "2026_11_05",
                             "surfacing_topic": topic, "summary": "captured summary"})
        return f"summary for {topic}"

    with patch("newscaster.pipeline.topic_finder", return_value=tf_result), \
         patch("newscaster.pipeline._gather_one_topic", side_effect=fake_one_topic), \
         patch("newscaster.pipeline.index_day", return_value=1) as mock_index:
        pipeline.gather_news("November 5, 2026", "2026_11_05")

    sidecar = "segment_summaries/2026_11_05_segment0_research.json"
    assert os.path.exists(sidecar)
    with open(sidecar) as f:
        rec = json.load(f)
    assert rec["arc_slug"] == "arc-zero"
    assert rec["articles"][0]["summary"] == "captured summary"
    mock_index.assert_called_once_with("2026_11_05")


def test_gather_index_failure_does_not_break_gather(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("segment_summaries", "output_scripts", "logs"):
        os.makedirs(d)
    tf_result = _tf(["t"], arc_context=[None])
    with patch("newscaster.pipeline.topic_finder", return_value=tf_result), \
         patch("newscaster.pipeline._gather_one_topic", side_effect=lambda topic, idx, *a, **k: f"s {topic}"), \
         patch("newscaster.pipeline.index_day", side_effect=RuntimeError("index boom")):
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")
    # Gather still completed: summary written, marker present.
    assert os.path.exists("segment_summaries/2026_11_05_segment0_summary.txt")
    assert os.path.exists("segment_summaries/2026_11_05_GATHER_COMPLETE.flag")
    assert result is tf_result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rag_gather_sidecar.py -v`
Expected: FAIL — `index_day` is not importable in `newscaster.pipeline` (AttributeError on patch) / sidecar not written.

- [ ] **Step 3: Import `index_day` lazily and capture per slot**

In `newscaster/pipeline.py`, add a module-level import near the other `newscaster.*`
imports so it is patchable (the `newscaster.rag.indexer` module only pulls stdlib +
numpy + google-genai, all already importable):

```python
from newscaster.rag.indexer import index_day, build_research_record
```

Replace the topic loop body (lines ~216-240) so each freshly-gathered slot gets
accumulators, and stash them for sidecar writing:

```python
    stories: dict[int, str] = {}
    slot_records: dict[int, tuple] = {}  # slot -> (articles, followups)

    arc_context = getattr(tf_result, "arc_context", None) or []

    for topic_index, topic in enumerate(topics):
        topic_OG = topic
        existing_summary = "segment_summaries/{}_segment{}_summary.txt".format(formatted_date2, topic_index)
        if os.path.exists(existing_summary):
            with open(existing_summary, 'r', encoding='utf-8') as f:
                existing_text = f.read()
            if existing_text.strip():
                stories[topic_index] = existing_text
                print_and_write(f"Slot {topic_index} already has a summary on disk; reusing (skipping gather)")
                continue
            print_and_write(f"Slot {topic_index} summary on disk is empty/whitespace; will re-gather")
        articles, followups = [], []
        try:
            stories[topic_index] = _gather_one_topic(
                topic, topic_index, formatted_date, formatted_date2,
                follow_up_prompt_text, challenging_follow_up_prompt_text,
                articles=articles, followups=followups,
            )
        except LLMError as e:
            print_and_write(
                f'GATHER FAILURE: topic "{topic_OG}" (slot {topic_index}) failed: {e}; '
                f'slot will be empty and skipped downstream'
            )
            continue
        slot_records[topic_index] = (articles, followups)
```

- [ ] **Step 4: Write sidecars + index after the summary writes**

Immediately after the existing block that writes summary files
(`for i, summary_text in stories.items(): _atomic_write_text(...)`, lines ~242-246),
add:

```python
    # Write per-slot research sidecars (provenance + Q&A) for freshly-gathered slots.
    for slot, (articles, followups) in slot_records.items():
        arc = arc_context[slot] if slot < len(arc_context) else None
        arc_slug = arc.get("slug") if isinstance(arc, dict) else None
        topic_str = topics[slot] if slot < len(topics) else ""
        record = build_research_record(
            formatted_date2, slot, topic_str, arc_slug, articles, followups
        )
        _atomic_write_text(
            "segment_summaries/{}_segment{}_research.json".format(formatted_date2, slot),
            json.dumps(record, ensure_ascii=False, indent=2),
        )

    # Index the day's research (non-critical: never let it break gather).
    try:
        indexed = index_day(formatted_date2)
        print_and_write(f"Indexed {indexed} research chunks for {formatted_date2}")
    except Exception as e:
        print_and_write(f"RAG index_day failed for {formatted_date2}: {e}; continuing")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rag_gather_sidecar.py tests/test_pipeline_gather_isolation.py -v`
Expected: PASS (new sidecar tests + all existing gather-isolation tests — accumulators are absorbed by their mocks, `index_day` no-ops on empty sidecars)

- [ ] **Step 6: Commit**

```bash
git add newscaster/pipeline.py tests/test_rag_gather_sidecar.py
git commit -m "Write research sidecars and index them in gather_news

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Retrieve-then-refine (gated augmentation)

**Files:**
- Modify: `newscaster/prompts.py` (append `RAG_REFINE_PROMPT`)
- Create: `newscaster/rag/retrieve.py`
- Modify: `newscaster/pipeline.py` (add `_augment_with_prior_research`; hook into `_gather_one_topic`)
- Test: `tests/test_rag_refine.py`

- [ ] **Step 1: Add the refine prompt**

Append to `newscaster/prompts.py`:

```python
RAG_REFINE_PROMPT = (
    "You are refining today's news synthesis using BACKGROUND from prior coverage of "
    "related stories. The background is dated and may be outdated.\n\n"
    "RULES:\n"
    "1. Today's synthesis is authoritative. If the background conflicts with it, today's "
    "synthesis wins — never replace a current fact with an older one.\n"
    "2. Never present background facts as current. When you use a background detail, mark "
    "its time explicitly (e.g. 'as of <date>').\n"
    "3. Only add background that genuinely deepens or contextualizes today's story "
    "(history, prior developments, earlier figures). Ignore anything irrelevant.\n"
    "4. Preserve strict sourcing: do not fuse separate facts into implied relationships "
    "the sources did not assert.\n"
    "5. Keep today's synthesis intact; you are adding context, not rewriting it.\n\n"
    "Return only the enriched synthesis."
)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_rag_refine.py`:

```python
"""Retrieve-then-refine augmentation (gated, fail-safe)."""
from unittest.mock import patch

import newscaster.config as cfg
import newscaster.pipeline as pipeline
from newscaster.rag.store import Hit


def _hit(text):
    return Hit(chunk_id="c", date="2026_03_01", chunk_type="article",
               outlet="NPR", headline="h", url="u", text=text, similarity=0.9)


def test_augment_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", False)
    with patch("newscaster.pipeline.retrieve_prior_research") as mock_ret:
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "draft summary"
    mock_ret.assert_not_called()


def test_augment_no_hits_returns_draft(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)
    with patch("newscaster.pipeline.retrieve_prior_research", return_value=[]), \
         patch("newscaster.pipeline.get_llm_response") as mock_llm:
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "draft summary"
    mock_llm.assert_not_called()


def test_augment_with_hits_calls_refine(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)
    with patch("newscaster.pipeline.retrieve_prior_research", return_value=[_hit("old context")]), \
         patch("newscaster.pipeline.get_llm_response", return_value="enriched") as mock_llm:
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "enriched"
    # Background text and the draft are both in the user prompt.
    user_prompt = mock_llm.call_args[0][0]
    assert "old context" in user_prompt and "draft summary" in user_prompt


def test_augment_failure_falls_back_to_draft(monkeypatch):
    monkeypatch.setattr(cfg, "RAG_AUGMENT_ENABLED", True)
    with patch("newscaster.pipeline.retrieve_prior_research", side_effect=RuntimeError("boom")):
        out = pipeline._augment_with_prior_research("draft summary", "2026_03_09")
    assert out == "draft summary"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rag_refine.py -v`
Expected: FAIL — `_augment_with_prior_research` / `retrieve_prior_research` not defined.

- [ ] **Step 4: Implement `retrieve.py`**

Create `newscaster/rag/retrieve.py`:

```python
"""Query-side retrieval: embed a draft and pull similar prior-coverage chunks."""
import newscaster.config as _config
from newscaster.rag.embeddings import embed_texts
from newscaster.rag.store import ResearchIndex


def retrieve_prior_research(draft_text, exclude_date, store=None):
    """Return Hits for prior research most similar to `draft_text`.

    Truncates the query defensively; excludes `exclude_date` (today) so we never
    retrieve the current run's own chunks. Returns [] when the store is empty or
    no chunk clears RAG_MIN_SIM.
    """
    query = (draft_text or "").strip()
    if not query:
        return []
    query = query[:24000]  # ~6k tokens, safely under the 8192-token limit (no auto_truncate on the Gemini API)
    vecs = embed_texts([query], task_type="RETRIEVAL_QUERY")
    if not vecs:
        return []
    store = store or ResearchIndex()
    return store.search(vecs[0], exclude_date=exclude_date)
```

- [ ] **Step 5: Implement `_augment_with_prior_research` and hook it in**

In `newscaster/pipeline.py`, add to the imports:

```python
from newscaster.rag.retrieve import retrieve_prior_research
from newscaster.prompts import RAG_REFINE_PROMPT
```

(Add `RAG_REFINE_PROMPT` to the existing `from newscaster.prompts import (...)` block,
and add the `retrieve_prior_research` import alongside the other `newscaster.*` imports.)

Add the function (place it just above `_gather_one_topic`):

```python
def _augment_with_prior_research(super_summary, formatted_date2):
    """Fold dated prior coverage into the draft summary. Gated + fail-safe:
    returns the un-augmented draft when disabled, on no hits, or on any error."""
    import newscaster.config as _config
    if not _config.RAG_AUGMENT_ENABLED:
        return super_summary
    try:
        hits = retrieve_prior_research(super_summary, exclude_date=formatted_date2)
        if not hits:
            return super_summary
        context = "\n\n".join(
            f"[Prior coverage — {h.outlet or 'unknown'}, {h.date}]\n{h.text}" for h in hits
        )
        user_prompt = (
            f"TODAY'S SYNTHESIS:\n{super_summary}\n\n"
            f"BACKGROUND FROM PRIOR COVERAGE:\n{context}"
        )
        enriched = get_llm_response(user_prompt, system_prompt=RAG_REFINE_PROMPT, mode="standard")
        print_and_write(f"RAG: augmented summary with {len(hits)} prior-coverage chunks")
        return enriched
    except Exception as e:
        print_and_write(f"RAG augment failed: {e}; using un-augmented summary")
        return super_summary
```

Then hook it into `_gather_one_topic`: replace the final `return super_summary`
(line ~382) with:

```python
    super_summary = _augment_with_prior_research(super_summary, formatted_date2)
    return super_summary
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rag_refine.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests, including the pre-existing suite)

- [ ] **Step 8: Commit**

```bash
git add newscaster/prompts.py newscaster/rag/retrieve.py newscaster/pipeline.py tests/test_rag_refine.py
git commit -m "Add gated retrieve-then-refine research augmentation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation notes

- **Rollout:** capture + indexing are now live; let the index accumulate real content
  across several daily runs before flipping `RAG_AUGMENT_ENABLED = True`.
- **Tune `RAG_MIN_SIM`:** once the index has content, log observed similarities (or
  inspect `research_index.db`) and set the floor so only genuinely related prior
  coverage clears it.
- **Backfill (optional):** the existing ~14 days predate the sidecars; a one-off script
  could index the bare `_segment{i}_article{j}_summary.txt` files with limited
  metadata. Not required for the feature to work going forward.
- **`research_index.db`** lives in `stories_chosen/` — add it to `.gitignore` if that
  directory is not already fully ignored.
