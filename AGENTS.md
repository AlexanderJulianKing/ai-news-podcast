# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

Newscaster is an automated daily news podcast generation system. It scrapes news from multiple sources, uses LLMs to select and analyze stories, generates dialogue scripts, synthesizes speech, creates video content, and uploads to YouTube. All of the code actually runs on a raspberry pi, so you will need to ssh into there via tailscale if you want to analyze previous days' outputs or change how things work in production.

## Commands

```bash
# Run full pipeline (scheduler with retry logic, triggers at 4 AM)
./main2.bash

# Run individual components
python3 main.py           # Core pipeline: scrape → analyze → script → audio
python3 moviemaker.py     # Create video from audio + image
python3 uploader2.py      # Upload to YouTube

# Run tests
python3 -m pytest tests/

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Entry Points

- `main.py` — Calls `config.init()` then `pipeline.main()`
- `moviemaker.py` — Calls `config.init()` then `video.create_and_export_video()`
- `uploader2.py` — Calls `config.init()` then `upload.main()`
- `main2.bash` — Scheduler: runs all three sequentially at 4 AM with retry logic

### Pipeline Stages (pipeline.py)

1. **`gather_news()`** — Discovers headlines from NPR, AP News, Democracy Now, ProPublica, CalMatters, Drop Site News, and City of Riverside (LLM grounding/url_context for most sources; a BeautifulSoup scraper for CalMatters and an RSS scraper for Drop Site). Story research runs through the controlled **source hunter** (URL discovery → controlled fetch/PDF extraction → source validation → answer only from accepted evidence) driven by an adaptive **research agent** (a LangGraph controller + adversary loop); the fixed multi-round follow-up path (light → standard → plus → heavy) remains as fallback. Writes to `segment_summaries/`.
2. **`write_scripts()`** — Selects stories via `headline_maker()`, generates dialogue scripts between anchor "Grace" and reporters (Ethan, Chloe, Elias). Writes to `output_scripts/`.
3. **`generate_audio()`** — Google Cloud TTS with multiple voices, intro music overlay via PyDub, final assembly. Writes to `segment_audio/` and `output_audio/`.
4. **Video** (`moviemaker.py`) — MoviePy combines `output_audio/{date}_HQ.mp3` with `image copy.png`
5. **Upload** (`uploader2.py`) — YouTube OAuth upload, reads title from `episode_titles/`

Each stage is idempotent: it checks if its output files already exist and skips if so. Output directories are auto-created by `_ensure_output_dirs()` at pipeline start.

### Multi-LLM Routing (llm/router.py)

`get_llm_response(prompt, system_prompt, mode, grounding, url_context)` is the central LLM call. Routing:

| Mode | No grounding | With grounding/url_context |
|---|---|---|
| `light` | Gemini 3.1 Flash Lite (Google) | Gemini 3 Flash (Google) |
| `standard` | Gemma 4 31B (OpenRouter) | Gemini 3 Flash (Google) |
| `advanced` | GLM 5.2, medium reasoning (OpenRouter) | Gemini 3 Flash (Google) |
| `adversary` | GPT-5.5, high reasoning (OpenRouter) | Gemini 3.1 Flash Lite (Google) |
| `plus` | Gemini 3.1 Pro (Google) | Gemini 3.1 Pro (Google) |
| `heavy` | Claude Opus 4.8 (Anthropic) | Gemini 3.1 Pro (Google) |

Only Gemini supports `grounding` (Google Search) and `url_context` (page fetching), so grounded modes route to Gemini regardless of tier. Model IDs are defined as constants in `config.py` (`LIGHT_MODEL`, `STANDARD_MODEL`, `ADVANCED_MODEL`, `ADVERSARY_MODEL`, `HEAVY_MODEL`); `FALLBACK_MODEL` (GPT-5.5 via OpenRouter) is the router's global fallback on retry exhaustion or auth failure.

### Key Patterns

- **Config init**: `config.init()` must be called before any API key is used. Keys are loaded from `keys.txt` into module-level globals in `config.py`. Other modules access keys via `import newscaster.config as _config` and read `_config.ANTHROPIC_API_KEY` etc. at call time (not import time).
- **Logging**: `print_and_write()` from `newscaster/logging.py` writes to both console and a daily log file in `logs/`.
- **Circular dependency**: `dedup.py` uses a lazy import (`from newscaster.llm import get_llm_response`) inside `summarize_story_for_archive()` to avoid circular imports at module load time.
- **All prompts** are centralized in `newscaster/prompts.py` as template constants.
- **Deduplication**: `stories_chosen/` stores JSON summaries per day; `load_recent_story_descriptions()` checks the past 7 days to avoid repeating stories.
- **Controlled research**: `source_hunter.py` + `source_hunter_primitives.py` implement evidence-gated research (discover → fetch → validate → answer from accepted evidence only); `research_agent.py` (LangGraph) orchestrates adaptive rounds with a controller and adversary; `search.py` abstracts web search (Google CSE primary → OpenRouter web-search fallback). Gated by `SOURCE_HUNTER_ENABLED` / `AGENTIC_RESEARCH_ENABLED` in `config.py`. Note: `source_hunter_primitives.py` is kept in sync with `benchmarks/web_search/web_bench_lib.py` — fixes must be applied to both.

## Configuration

API keys loaded from `keys.txt` (gitignored):
- `google_genai_api`, `anthropic_api`, `openrouter_api`, `XI_API_KEY`, `google_search_api`, `openweathermap_api`, `google_cse_id`

Also requires (gitignored): `client_secrets.json` (Google OAuth), `newscaster1-03dc16232821.json` (GCP service account)

## Output Directories

All auto-created, all gitignored. Files use `YYYY_MM_DD` date format:
`stories_chosen/`, `segment_summaries/`, `output_scripts/`, `segment_audio/`, `output_audio/`, `episode_titles/`, `logs/`
