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
import math
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
DEFAULT_RELEVANCE_GROUPS = HERE / "relevance_groups.json"
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


def rank_prior_by_cosine(query_vec, chunks, query_date, min_sim=PROD_MIN_SIM):
    """Rank only chunks that production could have seen before ``query_date``.

    This prevents two optimistic benchmark leaks: same-episode chunks and future
    episodes. It also applies the production similarity floor before measuring
    the final returned list.
    """
    prior = [c for c in chunks if c["date"] < query_date]
    ranked, sims = rank_by_cosine(query_vec, prior)
    ranked = [cid for cid in ranked if sims[cid] >= min_sim]
    return ranked, sims


def wilson_interval(successes, total, z=1.96):
    """95% Wilson confidence interval for a binary success rate."""
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [centre - half, centre + half]


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


def load_arc_aliases(path):
    """Return ledger slug -> reviewed canonical story mapping."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    aliases = {}
    for canonical, slugs in payload.get("groups", {}).items():
        for slug in slugs:
            if slug in aliases and aliases[slug] != canonical:
                raise ValueError(f"arc slug {slug!r} appears in multiple relevance groups")
            aliases[slug] = canonical
    return aliases


def _cross_episode_cases(chunks, arc_aliases=None):
    arc_aliases = arc_aliases or {}
    episodes = {}
    for chunk in chunks:
        if not chunk["arc_slug"]:
            continue
        canonical = arc_aliases.get(chunk["arc_slug"], chunk["arc_slug"])
        episodes.setdefault((canonical, chunk["date"]), []).append(chunk)
    cases = []
    for (arc_slug, date), current in sorted(episodes.items(), key=lambda item: item[0][1]):
        relevant = {
            c["chunk_id"] for c in chunks
            if arc_aliases.get(c["arc_slug"], c["arc_slug"]) == arc_slug
            and c["date"] < date
        }
        if not relevant:
            continue
        query_chunks = [c for c in current if c["chunk_type"] == "article"]
        if not query_chunks:
            query_chunks = current  # defensive fallback for malformed/legacy episodes
        cases.append({
            "arc_slug": arc_slug, "date": date, "current": current,
            "query_chunks": query_chunks, "relevant": relevant,
        })
    return cases


def _score_cross_episode_cases(chunks, cases, query_vecs, k_values, min_sim):
    per_query = []
    for case, query_vec in zip(cases, query_vecs):
        arc_slug, date = case["arc_slug"], case["date"]
        current, query_chunks, relevant = (
            case["current"], case["query_chunks"], case["relevant"]
        )
        ranked, _ = rank_prior_by_cosine(query_vec, chunks, date, min_sim=min_sim)
        row = {
            "episode": f"{date}:{arc_slug}",
            "date": date,
            "arc_slug": arc_slug,
            "n_current_chunks": len(current),
            "n_query_article_chunks": len(query_chunks),
            "n_relevant": len(relevant),
            "n_returned": min(PROD_TOP_K, len(ranked)),
            "rr": reciprocal_rank(ranked, relevant),
            "ap": average_precision(ranked, relevant),
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            row[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
            row[f"hit@{k}"] = float(any(cid in relevant for cid in ranked[:k]))
        per_query.append(row)

    agg = _aggregate(per_query, k_values, n_total=len(chunks))
    if not per_query:
        return agg
    for k in k_values:
        hits = int(sum(q[f"hit@{k}"] for q in per_query))
        agg[f"hit_rate@{k}"] = hits / len(per_query)
        agg[f"hit_rate@{k}_95ci"] = wilson_interval(hits, len(per_query))
        ceilings = [min(k, q["n_relevant"]) / q["n_relevant"] for q in per_query]
        agg[f"recall@{k}_ceiling"] = _nanmean(ceilings)
        agg[f"recall@{k}_ceiling_fraction"] = _nanmean([
            q[f"recall@{k}"] / ceiling
            for q, ceiling in zip(per_query, ceilings)
        ])
    agg["mean_returned@production_k"] = _nanmean([q["n_returned"] for q in per_query])
    agg["n_continuing_episodes"] = len(per_query)
    agg["n_cross_episode_arcs"] = len({q["arc_slug"] for q in per_query})
    agg["min_sim"] = min_sim
    agg["_per_query"] = per_query
    return agg


def run_cross_episode_eval(chunks, k_values, min_sim=PROD_MIN_SIM, arc_aliases=None):
    """Free chronological proxy using current-episode article vectors only.

    Follow-up chunks are deliberately excluded from each query to avoid using
    evidence created after production retrieval. Candidates are strictly older.
    ``hit@k`` asks whether memory surfaced any relevant prior coverage; recall@k
    measures how much of all prior coverage returned.
    """
    cases = _cross_episode_cases(chunks, arc_aliases=arc_aliases)
    query_vecs = [
        np.mean([c["vec"] for c in case["query_chunks"]], axis=0)
        for case in cases
    ]
    return _score_cross_episode_cases(chunks, cases, query_vecs, k_values, min_sim)


def run_production_query_cross_episode_eval(
        chunks, k_values, min_sim=PROD_MIN_SIM, arc_aliases=None):
    """Cross-episode evaluation with queries shaped like the production query."""
    from newscaster.rag.embeddings import embed_texts  # lazy: this task uses the API

    cases = _cross_episode_cases(chunks, arc_aliases=arc_aliases)
    if not cases:
        return {"n_queries": 0, "n_chunks": len(chunks),
                "note": "no continuing episodes"}
    queries = []
    for case in cases:
        evidence = "\n\n".join(
            f"Headline: {c.get('headline') or '(unknown)'}\n{c['text']}"
            for c in case["query_chunks"]
        )[:12000]
        queries.append(
            f"Topic: {case['arc_slug'].replace('_', ' ')}\n"
            f"Today: {case['date']}\n\nCurrent seed evidence:\n{evidence}"
        )
    query_vecs = embed_texts(queries, task_type="RETRIEVAL_QUERY")
    if len(query_vecs) != len(cases):
        raise RuntimeError(
            f"embed_texts returned {len(query_vecs)} vectors for {len(cases)} queries"
        )
    result = _score_cross_episode_cases(chunks, cases, query_vecs, k_values, min_sim)
    result["query_shape"] = "production topic/date/current seed evidence"
    return result


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
        ranked, sims = rank_by_cosine(qvec, chunks)  # include self; self is the target
        production_ranked = [cid for cid in ranked if sims[cid] >= PROD_MIN_SIM]
        relevant = {c["chunk_id"]}
        rec = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in k_values}
        overlap = len(_tokens(q) & _tokens(c["text"])) / max(1, len(_tokens(q)))
        per_query.append({
            "chunk_id": c["chunk_id"], "query": q, **rec,
            "rr": reciprocal_rank(ranked, relevant),
            "query_source_word_overlap": overlap,
            "target_similarity": sims[c["chunk_id"]],
            "production_hit": float(c["chunk_id"] in production_ranked[:PROD_TOP_K]),
            "n_returned": min(PROD_TOP_K, len(production_ranked)),
        })
    agg = _aggregate(per_query, k_values, n_total=len(chunks))
    agg["mean_query_source_word_overlap"] = _nanmean(
        [q["query_source_word_overlap"] for q in per_query]
    )
    production_hits = int(sum(q["production_hit"] for q in per_query))
    agg[f"production_recall@{PROD_TOP_K}"] = production_hits / len(per_query)
    agg[f"production_recall@{PROD_TOP_K}_95ci"] = wilson_interval(
        production_hits, len(per_query)
    )
    agg["mean_target_similarity"] = _nanmean([q["target_similarity"] for q in per_query])
    agg["mean_returned@production_k"] = _nanmean([q["n_returned"] for q in per_query])
    agg["min_sim"] = PROD_MIN_SIM
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
                       ("cross_episode_raw_ledger",
                        "CROSS-EPISODE (raw ledger labels)"),
                       ("cross_episode", "CROSS-EPISODE (reviewed relevance labels)"),
                       ("production_query_cross_episode",
                        "CROSS-EPISODE (production-shaped query)"),
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
        if f"hit_rate@{PROD_TOP_K}" in block:
            lo, hi = block[f"hit_rate@{PROD_TOP_K}_95ci"]
            print(f"   production hit-rate@{PROD_TOP_K}={_fmt(block[f'hit_rate@{PROD_TOP_K}'])} "
                  f"(95% CI {_fmt(lo)}-{_fmt(hi)}; min_sim={block['min_sim']})")
            print(f"   recall@{PROD_TOP_K} ceiling={_fmt(block[f'recall@{PROD_TOP_K}_ceiling'])}  "
                  f"fraction of attainable recall={_fmt(block[f'recall@{PROD_TOP_K}_ceiling_fraction'])}")
        if f"production_recall@{PROD_TOP_K}" in block:
            lo, hi = block[f"production_recall@{PROD_TOP_K}_95ci"]
            print(f"   thresholded production recall@{PROD_TOP_K}="
                  f"{_fmt(block[f'production_recall@{PROD_TOP_K}'])} "
                  f"(95% CI {_fmt(lo)}-{_fmt(hi)}; min_sim={block['min_sim']})")
    print("=" * 70 + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB), help="path to a research_index.db copy")
    ap.add_argument("--k", default=",".join(str(k) for k in DEFAULT_K_VALUES),
                    help="comma-separated k values")
    ap.add_argument("--known-item", action="store_true",
                    help="also run the LLM/embedding known-item task (uses the API)")
    ap.add_argument("--production-query-cross-episode", action="store_true",
                    help="embed production-shaped queries for continuing episodes (uses API)")
    ap.add_argument("--refresh-queries", action="store_true",
                    help="regenerate cached known-item queries")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--label", default="pi", help="label for the output filename")
    ap.add_argument(
        "--relevance-groups", default=str(DEFAULT_RELEVANCE_GROUPS),
        help="reviewed JSON groups that canonicalize fragmented story-arc slugs",
    )
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

    aliases = load_arc_aliases(args.relevance_groups) if args.relevance_groups else {}
    results = {
        "db": str(db_path),
        "n_chunks": len(chunks),
        "n_arcs": len({c["arc_slug"] for c in chunks}),
        "n_dates": len({c["date"] for c in chunks}),
        "embed_model": embed_model, "embed_dim": embed_dim,
        "k_values": k_values,
        "arc": run_arc_eval(chunks, k_values, cross_day=False),
        "relevance_groups": args.relevance_groups if aliases else None,
        "n_reviewed_arc_aliases": len(aliases),
        "cross_episode_raw_ledger": run_cross_episode_eval(chunks, k_values),
        "cross_episode": run_cross_episode_eval(chunks, k_values, arc_aliases=aliases),
    }

    if args.known_item or args.production_query_cross_episode:
        import newscaster.config as config
        config.init()
    if args.production_query_cross_episode:
        results["production_query_cross_episode"] = (
            run_production_query_cross_episode_eval(
                chunks, k_values, arc_aliases=aliases
            )
        )
    if args.known_item:
        results["known_item"] = run_known_item_eval(
            chunks, k_values, out_dir / "known_item_queries.json",
            refresh=args.refresh_queries,
        )

    print_report(results, k_values)
    # strip bulky per-query detail from the saved summary
    saveable = json.loads(json.dumps(results))
    if "known_item" in saveable:
        saveable["known_item"].pop("_per_query", None)
    if "cross_episode" in saveable:
        saveable["cross_episode"].pop("_per_query", None)
    if "cross_episode_raw_ledger" in saveable:
        saveable["cross_episode_raw_ledger"].pop("_per_query", None)
    if "production_query_cross_episode" in saveable:
        saveable["production_query_cross_episode"].pop("_per_query", None)
    out_path = out_dir / f"rag_recall_results_{args.label}.json"
    out_path.write_text(json.dumps(saveable, indent=2))
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
