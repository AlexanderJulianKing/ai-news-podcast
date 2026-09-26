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

# embed the small set of continuing episodes using the production query shape:
python3 -m benchmarks.rag_recall.rag_recall_bench --production-query-cross-episode

# metric math is unit-tested:
python3 -m pytest benchmarks/rag_recall/test_rag_recall_bench.py
```

Generated queries are cached to `outputs/known_item_queries.json` for
reproducibility (`--refresh-queries` to regenerate). Results are written to
`outputs/rag_recall_results_<label>.json`. Both `outputs/` and `data/` are
git-ignored — the index contains scraped article text and is not committed.

## Results — production index snapshot 2026-07-20

Index: 394 chunks, 54 story arcs, 32 days (2026-06-19 to 2026-07-20).
Production retrieval config: `top_k=6`, `min_sim=0.65`.

| Task | queries | hit@1 | hit@3 | hit@6 | recall@6 | precision@6 | MRR | MAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-episode, reviewed labels | 12 | **1.00** | **1.00** | **1.00** | 0.25 | **0.93** | **1.00** | 0.84 |
| Cross-episode, raw ledger labels | 9 | 0.78 | 1.00 | 1.00 | 0.25 | 0.76 | 0.87 | 0.68 |
| Arc-cohesion | 394 | — | — | — | 0.62 | 0.67 | **0.92** | 0.75 |

The cross-episode benchmark groups chunks by story arc and date, then uses only
the current episode's article vectors as a proxy for the production query.
Follow-up chunks are excluded because they are created after retrieval. Candidate
chunks must be strictly older than the query episode and must clear the real
similarity threshold. This prevents same-day and future-episode leakage. The
reviewed result additionally uses `relevance_groups.json` to join ledger slugs
that a human review confirmed describe the same continuing story. The raw-ledger
result remains visible so this correction is auditable.

### Reading the numbers

- **Cross-episode hit@1 and MRR 1.00:** all 12 continuing episodes surfaced
  relevant prior coverage as the first result. The hit-rate 95% Wilson interval
  is 0.76–1.00, reflecting the small sample.
- **Cross-episode precision@6 0.93:** 93% of the six returned chunks belonged to
  the correct continuing story, not merely one lucky hit among irrelevant results.
- **Cross-episode recall@6 0.25 is not a failure:** later episodes can have dozens
  of relevant historical chunks, while production intentionally returns only six.
  The average mathematical recall ceiling at `k=6` is only 0.30. The retriever
  achieves 0.25, or **93% of attainable recall**. The operational requirement is
  to surface useful prior coverage, so hit rate, precision, and ceiling-normalized
  recall are the primary metrics; raw full-set recall answers a different question.
- **Arc-cohesion MRR 0.92 across 394 queries:** a same-story chunk is usually the
  first relevant neighbor across the full production snapshot. This remains an
  easier, partly same-day task and is supporting evidence rather than the headline.

### Honest caveats

- The 12 continuing episodes come from only 2 reviewed multi-day stories, so they are correlated
  and do not yet demonstrate broad story diversity. Keep accumulating episodes and
  rerun before treating 100% as a stable population estimate.
- The free cross-episode query is an article-vector centroid, not a fresh embedding
  of the exact production prompt. Use `--production-query-cross-episode` to run the
  closer API-backed version when sending the episode evidence to the configured
  embedding provider is approved.
- The older 2026-06-22 known-item run scored recall@6 0.84 on 44 paraphrased
  queries. It is retained as historical evidence, not the current headline result.
