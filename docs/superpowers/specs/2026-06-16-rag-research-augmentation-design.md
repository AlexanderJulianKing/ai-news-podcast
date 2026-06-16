# RAG: Research Augmentation via Retrieve-Then-Refine

**Date:** 2026-06-16
**Status:** Design — approved shape, pending spec review
**Branch:** `rag-research-augmentation`

## 1. Problem & Goal

The pipeline produces deep, expensive research every day — per-topic synthesized
"super summaries" plus per-article summaries and multi-round follow-up Q&A
(`_gather_one_topic` / `_run_follow_up_rounds` in `newscaster/pipeline.py`). Two
problems:

1. **The research is write-once.** `segment_summaries/{date}_segment{i}_summary.txt`
   is only ever read back for the *same day's* downstream stages (`story_gatherer`
   at `newscaster/script/headlines.py:13`, `_extract_audience_learned` at
   `newscaster/pipeline.py:395`). There is no cross-day read anywhere. When a topic
   recurs, gather restarts from a blank search instead of building on prior depth.

2. **Provenance is discarded on capture.** When an article is pulled and summarized
   in `result_piper` (`newscaster/script/headlines.py:98`), the persisted file gets
   *only the summary text* (`headlines.py:140-142`). The URL is dropped entirely;
   the outlet is computed via an LLM call but only lives in the in-memory prompt
   string. The follow-up Q&A is concatenated into one `summary_prompt` blob and
   never stored as discrete records.

**Goal:** (a) capture each pulled document as a well-documented, structured record;
(b) build a semantic index over that research; (c) at gather time, retrieve relevant
*prior* research and fold it into the synthesis via a retrieve-then-refine pass.

### Non-goals (YAGNI)

- Semantic deduplication of stories (a separate, lower-value feature given the
  existing time-windowed dedup already works at current scale).
- A dedicated vector database (FAISS/pgvector/Pinecone) — over-engineering at this
  corpus size; see §5 for the upgrade trigger.
- Storing raw scraped article text — marginal retrieval gain over the summaries we
  already produce, plus storage/copyright considerations.
- Multimodal embeddings — text-only is sufficient.

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Purpose | Augment research (retrieval feeds generation) |
| Trigger | Retrieve-then-refine: draft → embed → top-k → extra synthesis pass |
| Capture scope | Provenance **+** follow-up Q&A, as structured per-slot records |
| Index unit | Article-summary chunks + Q&A-pair chunks, with provenance metadata |
| Store | SQLite + NumPy brute-force cosine |
| Embedding model | `gemini-embedding-2` (verified current), pinned dimension |

## 3. Embedding model (verified against live docs)

- **Model:** `gemini-embedding-2` (GA April 2026; multimodal, used here text-only).
- **Input limit:** 8,192 tokens/request. Our chunks (article summaries ~500–800
  tokens, Q&A pairs smaller, draft super-summaries ~1.5–2k tokens) are all well
  under this — no chunk truncation needed. Query text is defensively truncated to
  the input limit before embedding.
- **Output dimension:** configurable 128–3072 (default 3072). **We pin 1536** — a
  strong quality/size balance; at our corpus size storage is trivial either way, but
  1536 keeps the on-Pi NumPy matrix lean. Configurable via `EMBED_DIM`.
- **SDK call:** `client.models.embed_content(model=..., contents=..., config=...)`
  using the existing `genai.Client(api_key=_config.GOOGLE_GENAI_API_KEY)` from
  `newscaster/llm/gemini.py`. No new dependency.
- **Compatibility constraint:** the `gemini-embedding-2` and `gemini-embedding-001`
  embedding spaces are incompatible. The store records the model ID + dimension in a
  `meta` table; changing either requires a full re-embed (see §6 backfill).

Sources: <https://ai.google.dev/gemini-api/docs/embeddings>,
<https://developers.googleblog.com/building-with-gemini-embedding-2/>.

## 4. Data capture — the research record

A new structured sidecar per topic slot, written during gather:

```
segment_summaries/{date}_segment{i}_research.json
{
  "date": "2026_03_09",
  "slot": 0,
  "topic": "<the topic string gathered>",
  "arc_slug": "<from tf_result.arc_context[slot]>" | null,
  "articles": [
    {
      "chunk_id": "2026_03_09_seg0_art0",
      "url": "https://...",
      "outlet": "NPR",
      "original_headline": "<search-result headline>",
      "published_date": "2026-03-08" | null,
      "retrieved_date": "2026_03_09",
      "surfacing_topic": "<query that surfaced this article>",
      "summary": "<article summary text>"
    }
  ],
  "followups": [
    {
      "chunk_id": "2026_03_09_seg0_fu0",
      "asker": "Gemini Flash Lite",
      "question": "<follow-up question>",
      "answer": "<grounded answer>",
      "challenging": false
    }
  ]
}
```

**Capture touchpoints (both already compute the data; they just discard it):**

- `result_piper` (`headlines.py:98`) — already derives URL, outlet, and summary.
  It will append an article record to an accumulator. Because `result_piper` returns
  `(summary_prompt, counter)` today, it gains a third accumulator argument (a list or
  a small `SlotResearch` dataclass) rather than changing its return contract in a way
  that ripples. `published_date` is best-effort (extracted from the result snippet/
  page where present; `null` otherwise — not fabricated).
- `_run_follow_up_rounds` (`pipeline.py:30`) — already produces
  `(asker, question, answer, is_challenging)` per round. It will collect those into
  the slot's `followups[]` rather than only concatenating into `summary_prompt`.

**Assembly & persistence:** `_gather_one_topic` accumulates the article and Q&A
records and returns them alongside the super-summary — its return becomes a small
result object (e.g. `GatheredTopic(summary, articles, followups)`) rather than a bare
string. `gather_news` — which already writes the slot `_summary.txt` and holds
`tf_result.arc_context` — stamps `arc_slug`/`topic` from `arc_context[slot]`, writes
the `_research.json` sidecar via the existing `_atomic_write_text`, then calls
`index_day`. This keeps `arc_context` plumbing out of `_gather_one_topic`. The
article `_summary.txt` files keep being written inside `result_piper` as today.

**Backward compatibility:** the existing `_segment{i}_summary.txt` and
`_segment{i}_article{j}_summary.txt` files are unchanged; the `_research.json` sidecar
is purely additive, so current reads and idempotency logic are untouched.

## 5. Vector store (`newscaster/rag/store.py`)

A single SQLite file `stories_chosen/research_index.db` (co-located with
`story_ledger.json`).

```
TABLE meta(key TEXT PRIMARY KEY, value TEXT)      -- embed_model, embed_dim
TABLE chunks(
  chunk_id   TEXT PRIMARY KEY,   -- e.g. 2026_03_09_seg0_art0
  date       TEXT,               -- YYYY_MM_DD (for exclude/recency filters)
  arc_slug   TEXT,               -- nullable
  slot       INTEGER,
  chunk_type TEXT,               -- 'article' | 'followup'
  outlet     TEXT,               -- nullable
  headline   TEXT,               -- nullable
  url        TEXT,               -- nullable
  text       TEXT,               -- the embedded text (for injection/citation)
  vector     BLOB                -- float32 array, length = embed_dim
)
```

- **Search:** load all `(chunk_id, date, vector, text, metadata)` rows once, build a
  NumPy `float32` matrix, compute cosine similarity against the query vector, return
  top-k filtered rows. `gemini-embedding-2` normalizes truncated dimensions, so
  cosine is appropriate.
- **Why brute force, not a vector DB:** at a few thousand chunks (≈1 year of output)
  the matrix is a few MB and search is sub-millisecond. **Upgrade trigger:** switch
  to FAISS (`IndexFlatIP`, then `IndexHNSW`) only above ~50k vectors, where linear
  scan latency becomes noticeable.
- **API:**
  - `upsert(chunks: list[Chunk]) -> int` — INSERT OR REPLACE by `chunk_id`.
  - `search(query_vec, k=RAG_TOP_K, exclude_date=None, min_sim=RAG_MIN_SIM) -> list[Hit]`
  - On open, verify `meta.embed_model`/`embed_dim` match config; mismatch raises a
    clear error directing to re-embed (prevents silently mixing incompatible spaces).

## 6. Indexing — the write path (`newscaster/rag/indexer.py`)

- `index_day(formatted_date2) -> int`: read every `_segment{i}_research.json` sidecar
  for the day, build `Chunk`s (one per article summary, one per Q&A pair), embed them
  in batch via `embeddings.embed_texts()`, and `upsert`. Returns count indexed.
- **Idempotent:** keyed by `chunk_id`; re-running a day replaces rather than
  duplicates. Safe under the pipeline's existing rerun model.
- **Call site:** once at the end of `gather_news`, after all slot summaries and
  sidecars are written.
- **Cold start:** empty DB is valid; first run simply has nothing to retrieve.
- **Backfill (one-off script, not pipeline):** existing pre-feature days have no
  `_research.json`. A backfill indexer falls back to the bare
  `_segment{i}_article{j}_summary.txt` files with limited metadata (no URL/outlet).
  Documented as best-effort; primary value accrues from new runs forward.

## 7. Retrieve-then-refine — the read path (`newscaster/rag/retrieve.py` + hook)

Inside `_gather_one_topic`, after the draft `super_summary` is built and *before* it
is returned:

1. Embed the draft (truncated to the input limit) as the query.
2. `store.search(query_vec, k=RAG_TOP_K, exclude_date=today, min_sim=RAG_MIN_SIM)`.
   - **`exclude_date=today` is mandatory** — it prevents retrieving the current run's
     own chunks on a rerun where today was already partially indexed.
   - **`min_sim` floor** means weak/irrelevant matches return nothing, so the refine
     pass injects *no* stale context rather than forcing it.
3. If there are hits, run one refine synthesis pass (`mode='standard'`) with a new
   prompt (`RAG_REFINE_PROMPT` in `newscaster/prompts.py`) that:
   - presents retrieved material as **dated "prior coverage" background**;
   - instructs that **today's sources win on any conflict** and old facts must
     **never be stated as current** (attribute with "as of <date>");
   - preserves the existing strict no-fact-welding sourcing discipline.
4. If there are no hits, return the un-augmented draft (no-op).

**Failure isolation:** the entire retrieve-then-refine block is wrapped so that any
embedding/store/LLM error is logged via `print_and_write` and falls back to the
un-augmented draft. RAG can never break or block gather.

**Idempotency:** the existing gather idempotency (skip slot if its `_summary.txt`
exists, `pipeline.py:221`) already prevents double-refine, since refine happens
inside `_gather_one_topic` before the summary is written.

## 8. Module layout

```
newscaster/rag/
  __init__.py
  embeddings.py   # embed_texts(texts) -> list[list[float]]; batches; typed errors
  store.py        # SQLite vector store: upsert(), search(), meta checks
  indexer.py      # index_day(); chunk-building from research.json sidecars
  retrieve.py     # retrieve_prior_research(draft, exclude_date) -> list[Hit]
```

- `embeddings.py` mirrors `newscaster/llm/gemini.py`'s error discipline: catch
  `genai_errors.APIError`, classify via `newscaster/llm/errors.py`, raise typed
  `LLMError`. Retry is light (embeddings are cheap); failures bubble to the §7
  fallback.
- Integration edits: `result_piper` + `_run_follow_up_rounds` (capture),
  `_gather_one_topic` (refine hook; return becomes a `GatheredTopic` result object
  carrying the research records), `gather_news` (stamp `arc_slug`, write sidecar,
  call `index_day`), `newscaster/prompts.py` (`RAG_REFINE_PROMPT`),
  `newscaster/config.py` (tunables).

## 9. Configuration (`newscaster/config.py`, module globals like `MAX_RETRIES`)

| Constant | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `"gemini-embedding-2"` | embedding model id |
| `EMBED_DIM` | `1536` | output dimensionality (pinned) |
| `RAG_TOP_K` | `6` | chunks retrieved per refine |
| `RAG_MIN_SIM` | `0.65` | cosine floor; below → inject nothing (**tune empirically**) |
| `RAG_AUGMENT_ENABLED` | `False` | gates the refine pass (§7) for safe rollout |

**Rollout staging:** capture (§4) and indexing (§6) are harmless and ship enabled, so
the corpus builds from day one. The behavior-changing refine pass (§7) is gated by
`RAG_AUGMENT_ENABLED`, flipped on only after the index has real content and `RAG_MIN_SIM`
is tuned against observed similarities.

## 10. Failure modes & safety (design intent)

- Empty or below-threshold index → refine is a no-op; output identical to today.
- Any embedding/store/LLM failure in the RAG path → logged, un-augmented draft used.
- Self-retrieval prevented by `exclude_date=today`.
- Staleness handled in `RAG_REFINE_PROMPT` (dated background, today-wins).
- Indexing idempotent by `chunk_id`; pipeline reruns safe.
- Model/dimension pinned in `meta`; incompatible-space mixing raises a clear error.
- `published_date` and other best-effort fields are `null` when unknown, never
  fabricated (per project sourcing rules).

## 11. Testing

**Unit:**
- `store`: cosine correctness & top-k ordering; `exclude_date` filter; `min_sim`
  threshold (returns empty below floor); empty-store returns empty; meta-mismatch
  raises.
- `indexer`: idempotency (run twice → no duplicate `chunk_id`s); builds both
  `article` and `followup` chunks from a fixture sidecar.
- `embeddings`: mocked `genai.Client` returns vectors; batches inputs; maps API
  error → typed `LLMError`.
- capture: `result_piper` populates an article record (url/outlet/summary);
  `_run_follow_up_rounds` collects Q&A tuples.

**Integration:**
- Seeded store + draft → refine receives retrieved context and produces enriched text.
- Empty store → refine skipped, draft returned unchanged.
- RAG path raises → gather still returns the un-augmented draft.

Tests mock all network/LLM calls, mirroring the existing `tests/` style.

## 12. Open items (resolve during implementation, not blocking)

- Final `RAG_MIN_SIM` value — tune against real observed cosine distributions.
- Whether to also index the coarse `_segment{i}_summary.txt` as a topic-level chunk.
  Start with article + Q&A chunks only; add later if topic-level recall is weak.
- Backfill of the existing ~14 days (limited-metadata best-effort) — optional.
