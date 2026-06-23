# RAG embedding-retrieval recall benchmark

Measures how well the Newscaster retrieval memory (`newscaster/rag/`) actually
finds relevant chunks. The retriever is from-scratch: Gemini embeddings
(`gemini-embedding-2`, 1536-d) stored as float32 blobs in SQLite, retrieved by
brute-force cosine in NumPy (`newscaster/rag/store.py`).

This exists for two reasons: (1) to verify retrieval works, and (2) because
"RAG without recall metrics" is a named hiring red flag — a retriever you can't
measure is a retriever you can't trust.

The benchmark is **read-only** and never touches the production index. Point it
at a copy with `--db`.

## Tasks

The hard part of a recall benchmark is defining "relevant" without hand-labeling.
We use two complementary, label-free definitions.

### 1. Arc-cohesion (multi-relevant, no API)
Each indexed chunk is used as a query (its own stored vector). The relevant set
is the **other chunks sharing its `arc_slug`** — i.e. the same ongoing story, as
already tracked by the pipeline's story ledger. Because each query has several
relevant partners, `recall@k`, `precision@k`, `MRR` and `MAP` are all meaningful.
This measures whether the embedding space clusters same-story content. It needs
no API calls (it reuses stored vectors).

A **cross-day** variant restricts relevance to same-arc chunks on *other* dates.
That mirrors the real production memory use (`retrieve.py` excludes today), and
becomes measurable once the index holds arcs that span more than one day.

### 2. Known-item (single-relevant, uses the API)
For each chunk, an LLM writes a paraphrased question with deliberately low
lexical overlap; we embed it as a `RETRIEVAL_QUERY` and check whether the source
chunk is returned. One relevant item, so we report `recall@k` and `MRR`. We also
report the mean query/source word overlap so you can confirm the queries are
genuinely paraphrased — i.e. that this measures *semantic* retrieval, not keyword
matching. This is the stronger generalization signal: it asks "given a natural
question, does the store surface the right document?"

## Running

```bash
# arc-cohesion only (free, no API):
python3 -m benchmarks.rag_recall.rag_recall_bench --db benchmarks/rag_recall/data/research_index.db

# add the known-item task (LLM paraphrase queries + Gemini embeddings; needs keys.txt):
python3 -m benchmarks.rag_recall.rag_recall_bench --known-item

# metric math is unit-tested:
python3 -m pytest benchmarks/rag_recall/test_rag_recall_bench.py
```

Generated queries are cached to `outputs/known_item_queries.json` for
reproducibility (`--refresh-queries` to regenerate). Results are written to
`outputs/rag_recall_results_<label>.json`. Both `outputs/` and `data/` are
git-ignored — the index contains scraped article text and is not committed.

## Results — production index snapshot 2026-06-22

Index: 44 chunks (21 article + 23 follow-up), 7 story arcs, 4 days (2026-06-19
to 06-22). Production retrieval config: `top_k=6`, `min_sim=0.65`.

| Task | recall@1 | recall@3 | recall@5 | recall@6 | recall@10 | MRR | MAP | precision@6 |
|---|---|---|---|---|---|---|---|---|
| Arc-cohesion (multi-relevant) | 0.18 | 0.53 | 0.77 | 0.81 | 0.91 | **0.97** | 0.88 | 0.71 |
| Known-item (paraphrased query) | 0.43 | 0.73 | 0.80 | **0.84** | 0.91 | 0.61 | — | — |
| Arc cross-day (production memory) | — | — | — | — | — | — | — | — |

Mean query/source word overlap (known-item): **0.44** — queries are genuinely
reworded, so retrieval is semantic, not lexical. Mean fraction of same-arc
partners clearing `min_sim=0.65`: **0.83**.

### Reading the numbers
- **MRR 0.97 (arc-cohesion)**: for nearly every chunk, the single nearest
  neighbor is from the same story. The embedding space cleanly separates stories.
- **Known-item recall@6 0.84 / recall@10 0.91**: given a paraphrased question,
  the exact source chunk lands in the production top-6 84% of the time. This is
  the headline generalization result.
- **`recall@1` looks low on arc-cohesion (0.18) by construction**: with ~6
  relevant partners per query, `recall@k` is capped at `k / n_relevant`, so
  `recall@1` maxes out near 1/6 ≈ 0.17 even when the top hit is always correct
  (hence MRR 0.97). Use `recall@6`/MRR/MAP, not `recall@1`, for the multi-relevant
  task; `precision@k` is only meaningful here (the known-item task has one target).

### Honest caveats
- **Young index (N=44, 4 days).** These numbers describe this snapshot; rerun as
  the index grows for tighter estimates.
- **Cross-day is the production-realistic metric and is not yet measurable** —
  the current index has no arc spanning more than one day. It will populate as
  multi-day stories accumulate; the code computes it automatically.
- **Arc-cohesion is an intra-story signal** (same-day same-story chunks are
  textually similar), so it is an easier task than cross-day retrieval. Known-item
  is the more demanding, more generalizable measure.
