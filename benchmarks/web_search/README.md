# Web Search Model Benchmark

This benchmark compares model behavior on hard, newsroom-shaped web research tasks:
local agenda retrieval, freshness traps, exact dollar/date extraction, primary-source
discipline, and regulatory/status synthesis.

The default model matrix is:

- `gemini_flash_lite`: `google/gemini-3.1-flash-lite`
- `gemini_flash_3`: `google/gemini-3-flash-preview`
- `glm_5_2`: `z-ai/glm-5.2`
- `gemma_4_31b`: `google/gemma-4-31b-it`

The runner uses OpenRouter's `web` plugin and forces one engine across all models
for a fair first pass. The default is `parallel` with `max_results=5`.

## Dry Plan

```bash
python3 benchmarks/web_search/run_web_bench.py
```

## Smoke Run

This spends API calls. The key is read from `OPENROUTER_API_KEY` or from
`keys.txt` as `openrouter_api`.

```bash
python3 benchmarks/web_search/run_web_bench.py --execute
```

## Controlled Source Run

This variant does not use the OpenRouter web plugin for the model call. It fetches
the benchmark's preferred source URLs directly, extracts HTML/PDF text, selects
relevant excerpts from the question, and asks the models to answer only from that
controlled evidence.

```bash
python3 benchmarks/web_search/run_web_bench.py \
  --execute \
  --strategy controlled_fetch \
  --out benchmarks/web_search/outputs/controlled_YYYYMMDD.jsonl
```

## Discover Then Fetch Run

This variant tests the missing production step: finding source URLs without being
given the benchmark's preferred URLs. It uses one cheap OpenRouter web-plugin
scout call per task, extracts candidate URLs from annotations/content, fetches
those URLs directly, then asks the answer models to respond only from fetched
evidence.

```bash
python3 benchmarks/web_search/run_web_bench.py \
  --execute \
  --strategy discover_then_fetch \
  --engine parallel \
  --out benchmarks/web_search/outputs/discover_YYYYMMDD_parallel.jsonl

python3 benchmarks/web_search/grade_web_bench.py \
  benchmarks/web_search/outputs/discover_YYYYMMDD_parallel.jsonl

python3 benchmarks/web_search/summarize_discovery_bench.py \
  benchmarks/web_search/outputs/discover_YYYYMMDD_parallel.jsonl
```

Smaller run:

```bash
python3 benchmarks/web_search/run_web_bench.py --execute --limit 3
```

Single task and model:

```bash
python3 benchmarks/web_search/run_web_bench.py \
  --execute \
  --task cdc_measles_2026 \
  --model glm_5_2
```

## Grade And Report

```bash
python3 benchmarks/web_search/grade_web_bench.py benchmarks/web_search/outputs/run_YYYYMMDD_HHMMSS.jsonl
```

This creates:

- `*.scores.json`: deterministic checklist scores
- `*.report.html`: model summary, raw answers, citations, usage, and latency

## Scoring

The grader is intentionally literal. It checks for required facts, exact dates,
amounts, source domains, and forbidden stale claims. It is not meant to replace
human review, but it quickly surfaces which models miss facts, use stale sources,
or fail to cite primary material.

After a first run, inspect the low-scoring answers manually. If a model is
semantically correct but missed a phrase variant, update only the relevant task
check, then re-run the grader without re-running model calls.
