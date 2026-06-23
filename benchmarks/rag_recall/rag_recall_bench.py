"""Recall benchmark for the Newscaster RAG embedding-retrieval index.

Measures how well the from-scratch retriever (Gemini embeddings -> SQLite/NumPy
cosine store, see newscaster/rag/) actually finds relevant chunks. Two tasks:

1. ARC-COHESION (free; no API). Each indexed chunk is used as a query (its own
   stored vector); the relevant set is the other chunks sharing its arc_slug
   (i.e. the same ongoing story). This is a genuine MULTI-relevant retrieval
   task, so recall@k, precision@k, MRR and MAP are all meaningful. A
   "cross-day" variant restricts relevance to same-arc chunks on OTHER dates --
   that mirrors the production memory use (retrieve.py excludes today) but only
   becomes measurable once the index contains arcs that span >1 day.

2. KNOWN-ITEM (needs the Gemini API + an LLM). For each chunk, an LLM writes a
   paraphrased question with deliberately low lexical overlap; we embed it as a
   RETRIEVAL_QUERY and check whether the source chunk is returned. Single
   relevant item, so we report recall@k and MRR. We also report the mean
   query/source word-overlap so the reader can see the queries are genuinely
   paraphrased (semantic retrieval, not keyword matching).

The benchmark is read-only and never writes to the production index. Point it
at a copy of the index with --db.

Honesty notes (see README.md): the production index is young, so N is small and
the cross-day variant may be empty. The numbers describe THIS index; rerun as it
grows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "pi_research_index.db"
DEFAULT_OUT = HERE / "outputs"
DEFAULT_K_VALUES = (1, 3, 5, 6, 10)

# Production retrieval defaults, for the threshold-sensitivity readout.
PROD_TOP_K = 6
PROD_MIN_SIM = 0.65


# --------------------------------------------------------------------------- #
# Pure ranking metrics. ranked_ids is a list of ids in descending score order;
# relevant is the set of ids that count as correct. These are unit-tested.
# --------------------------------------------------------------------------- #
def recall_at_k(ranked_ids, relevant, k):
    """Fraction of relevant items appearing in the top k."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / len(relevant)


def precision_at_k(ranked_ids, relevant, k):
    """Fraction of the top k that are relevant."""
    if k <= 0:
        return float("nan")
    relevant = set(relevant)
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hits / k


def reciprocal_rank(ranked_ids, relevant):
    """1 / rank of the first relevant item (rank is 1-based); 0 if none."""
    relevant = set(relevant)
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked_ids, relevant):
    """Average precision over the relevant set (the per-query term behind MAP)."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    hits = 0
    score = 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def _nanmean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- #
# Data loading + cosine ranking
# --------------------------------------------------------------------------- #
def load_chunks(db_path):
    """Return list of dicts with a unit-normalized float32 vector per chunk."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "select chunk_id, date, arc_slug, slot, chunk_type, outlet, "
            "headline, url, text, vector from chunks"
        ).fetchall()
    finally:
        con.close()
    chunks = []
    for cid, date, arc, slot, ctype, outlet, headline, url, text, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        chunks.append({
            "chunk_id": cid, "date": date, "arc_slug": arc, "slot": slot,
            "chunk_type": ctype, "outlet": outlet, "headline": headline,
            "url": url, "text": text or "", "vec": vec / norm,
        })
    return chunks


def rank_by_cosine(query_vec, chunks, exclude_ids=()):
    """Return (ranked_ids, sims_by_id). Vectors are unit-norm so cosine = dot."""
    q = np.asarray(query_vec, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return [], {}
    q = q / n
    exclude = set(exclude_ids)
    scored = [
        (c["chunk_id"], float(np.dot(q, c["vec"])))
        for c in chunks if c["chunk_id"] not in exclude
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in scored], dict(scored)


# --------------------------------------------------------------------------- #
# Task 1: arc-cohesion (no API)
# --------------------------------------------------------------------------- #
def run_arc_eval(chunks, k_values, cross_day=False):
    by_id = {c["chunk_id"]: c for c in chunks}
    per_query = []
    for c in chunks:
        same_arc = [
            o["chunk_id"] for o in chunks
            if o["arc_slug"] == c["arc_slug"] and o["chunk_id"] != c["chunk_id"]
            and (not cross_day or o["date"] != c["date"])
        ]
        if not same_arc:
            continue  # singleton arc (or no cross-day partner): no positives
        ranked, sims = rank_by_cosine(c["vec"], chunks, exclude_ids=[c["chunk_id"]])
        relevant = set(same_arc)
        rec = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in k_values}
        prec = {f"precision@{k}": precision_at_k(ranked, relevant, k) for k in k_values}
        # how many same-arc partners clear the production similarity floor
        cleared = sum(1 for cid in same_arc if sims.get(cid, 0.0) >= PROD_MIN_SIM)
        per_query.append({
            "chunk_id": c["chunk_id"], "arc_slug": c["arc_slug"],
            "n_relevant": len(relevant),
            **rec, **prec,
            "rr": reciprocal_rank(ranked, relevant),
            "ap": average_precision(ranked, relevant),
            "relevant_above_min_sim_frac": cleared / len(relevant),
        })
    return _aggregate(per_query, k_values, n_total=len(chunks))


def _aggregate(per_query, k_values, n_total):
    if not per_query:
        return {"n_queries": 0, "n_chunks": n_total, "note": "no queries with positive pairs"}
    agg = {"n_queries": len(per_query), "n_chunks": n_total}
    for k in k_values:
        agg[f"recall@{k}"] = _nanmean([q.get(f"recall@{k}") for q in per_query])
    for k in k_values:
        if any(f"precision@{k}" in q for q in per_query):
            agg[f"precision@{k}"] = _nanmean([q.get(f"precision@{k}") for q in per_query])
    agg["mrr"] = _nanmean([q["rr"] for q in per_query])
    if all("ap" in q for q in per_query):
        agg["map"] = _nanmean([q["ap"] for q in per_query])
    if all("relevant_above_min_sim_frac" in q for q in per_query):
        agg["mean_relevant_above_min_sim"] = _nanmean(
            [q["relevant_above_min_sim_frac"] for q in per_query]
        )
    return agg


# --------------------------------------------------------------------------- #
# Task 2: known-item (LLM paraphrase queries + Gemini embeddings)
# --------------------------------------------------------------------------- #
_QUERY_SYSTEM = (
    "You write retrieval-test queries. Given a news fact, output ONE natural "
    "question a reader might ask whose answer is that fact. Paraphrase: avoid "
    "reusing the fact's distinctive nouns and numbers verbatim wherever a "
    "paraphrase works, so the question tests meaning rather than keyword "
    "overlap. Output only the question, no preamble."
)


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def generate_known_item_queries(chunks, cache_path, refresh=False):
    """Map chunk_id -> paraphrased query string. Cached to disk for reproducibility."""
    from newscaster.llm import get_llm_response  # lazy: only when known-item runs

    cache = {}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text())
    out = {}
    for c in chunks:
        cid = c["chunk_id"]
        if cid in cache and cache[cid]:
            out[cid] = cache[cid]
            continue
        source = c["text"][:1500]
        try:
            q = get_llm_response(source, system_prompt=_QUERY_SYSTEM, mode="light").strip()
        except Exception as e:  # noqa: BLE001 - benchmark should not die on one bad call
            print(f"  query-gen failed for {cid}: {e}", file=sys.stderr)
            q = ""
        out[cid] = q
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2))
    return out


def run_known_item_eval(chunks, k_values, cache_path, refresh=False):
    from newscaster.rag.embeddings import embed_texts  # lazy

    queries = generate_known_item_queries(chunks, cache_path, refresh=refresh)
    items = [(c, queries.get(c["chunk_id"], "")) for c in chunks]
    items = [(c, q) for c, q in items if q]
    if not items:
        return {"n_queries": 0, "n_chunks": len(chunks), "note": "no queries generated"}

    qvecs = embed_texts([q for _, q in items], task_type="RETRIEVAL_QUERY")
    per_query = []
    for (c, q), qvec in zip(items, qvecs):
        ranked, _ = rank_by_cosine(qvec, chunks)  # include self; self is the target
        relevant = {c["chunk_id"]}
        rec = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in k_values}
        overlap = len(_tokens(q) & _tokens(c["text"])) / max(1, len(_tokens(q)))
        per_query.append({
            "chunk_id": c["chunk_id"], "query": q, **rec,
            "rr": reciprocal_rank(ranked, relevant),
            "query_source_word_overlap": overlap,
        })
    agg = _aggregate(per_query, k_values, n_total=len(chunks))
    agg["mean_query_source_word_overlap"] = _nanmean(
        [q["query_source_word_overlap"] for q in per_query]
    )
    agg["_per_query"] = per_query
    return agg


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, float):
        return "n/a" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def print_report(results, k_values):
    print("\n" + "=" * 70)
    print("RAG EMBEDDING-RETRIEVAL RECALL BENCHMARK")
    print("=" * 70)
    print(f"index: {results['db']}")
    print(f"chunks: {results['n_chunks']}  |  arcs: {results['n_arcs']}  "
          f"|  dates: {results['n_dates']}  |  embed: {results['embed_model']} "
          f"({results['embed_dim']}d)")
    print(f"production retrieval config: top_k={PROD_TOP_K}, min_sim={PROD_MIN_SIM}")

    for key, title in (("arc", "ARC-COHESION (same-story retrieval, multi-relevant)"),
                       ("arc_cross_day", "ARC CROSS-DAY (production memory use)"),
                       ("known_item", "KNOWN-ITEM (paraphrased-query single-target)")):
        block = results.get(key)
        if not block:
            continue
        print(f"\n-- {title} --")
        if block.get("n_queries", 0) == 0:
            print(f"   n/a ({block.get('note', 'no data')})")
            continue
        print(f"   queries: {block['n_queries']}")
        cols = [f"recall@{k}" for k in k_values]
        print("   " + "  ".join(f"{c}={_fmt(block[c])}" for c in cols if c in block))
        if "mrr" in block:
            line = f"   MRR={_fmt(block['mrr'])}"
            if "map" in block:
                line += f"  MAP={_fmt(block['map'])}"
            print(line)
        for k in (PROD_TOP_K,):
            pk = f"precision@{k}"
            if pk in block:
                print(f"   precision@{k}={_fmt(block[pk])}")
        if "mean_relevant_above_min_sim" in block:
            print(f"   mean fraction of same-arc partners clearing min_sim="
                  f"{PROD_MIN_SIM}: {_fmt(block['mean_relevant_above_min_sim'])}")
        if "mean_query_source_word_overlap" in block:
            print(f"   mean query/source word overlap: "
                  f"{_fmt(block['mean_query_source_word_overlap'])} "
                  f"(low => genuinely paraphrased, semantic retrieval)")
    print("=" * 70 + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB), help="path to a research_index.db copy")
    ap.add_argument("--k", default=",".join(str(k) for k in DEFAULT_K_VALUES),
                    help="comma-separated k values")
    ap.add_argument("--known-item", action="store_true",
                    help="also run the LLM/embedding known-item task (uses the API)")
    ap.add_argument("--refresh-queries", action="store_true",
                    help="regenerate cached known-item queries")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--label", default="pi", help="label for the output filename")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        ap.error(f"index not found: {db_path}")
    k_values = [int(x) for x in args.k.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(db_path)
    if not chunks:
        ap.error(f"index {db_path} has no chunks")

    embed_model = embed_dim = None
    con = sqlite3.connect(str(db_path))
    try:
        meta = dict(con.execute("select key, value from meta").fetchall())
        embed_model, embed_dim = meta.get("embed_model"), meta.get("embed_dim")
    finally:
        con.close()

    results = {
        "db": str(db_path),
        "n_chunks": len(chunks),
        "n_arcs": len({c["arc_slug"] for c in chunks}),
        "n_dates": len({c["date"] for c in chunks}),
        "embed_model": embed_model, "embed_dim": embed_dim,
        "k_values": k_values,
        "arc": run_arc_eval(chunks, k_values, cross_day=False),
        "arc_cross_day": run_arc_eval(chunks, k_values, cross_day=True),
    }

    if args.known_item:
        import newscaster.config as config
        config.init()
        results["known_item"] = run_known_item_eval(
            chunks, k_values, out_dir / "known_item_queries.json",
            refresh=args.refresh_queries,
        )

    print_report(results, k_values)
    # strip bulky per-query detail from the saved summary
    saveable = json.loads(json.dumps(results))
    if "known_item" in saveable:
        saveable["known_item"].pop("_per_query", None)
    out_path = out_dir / f"rag_recall_results_{args.label}.json"
    out_path.write_text(json.dumps(saveable, indent=2))
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
