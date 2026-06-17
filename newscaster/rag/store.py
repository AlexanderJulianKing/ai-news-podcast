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
