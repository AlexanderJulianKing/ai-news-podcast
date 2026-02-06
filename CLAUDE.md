# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Newscaster is an automated daily news podcast generation system. It scrapes news from multiple sources, uses LLMs to select and analyze stories, generates dialogue scripts, synthesizes speech, creates video content, and uploads to YouTube.

## Commands

```bash
# Run full pipeline (recommended)
./main2.bash

# Run individual components
python3 main.py           # Core pipeline: scrape → analyze → script → audio
python3 moviemaker.py     # Create video from audio + image
python3 uploader2.py      # Upload to YouTube

# Scheduled execution
./main2.bash              # Bash scheduler with retry logic

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Package Structure

```
newscaster/                # Main package
├── config.py              # API key loading, constants, PROJECT_ROOT
├── logging.py             # print_and_write() log function
├── dates.py               # Date formatting helpers (spoken dates)
├── text_utils.py          # text_cleaner(), find_quoted_strings(), grounding retry
├── prompts.py             # All prompt template constants
├── dedup.py               # Story deduplication against past 7 days
├── weather.py             # OpenWeatherMap integration
├── video.py               # MoviePy video creation
├── upload.py              # YouTube OAuth upload
├── pipeline.py            # main() and super_main() orchestrators
├── llm/                   # LLM provider wrappers
│   ├── router.py          # get_llm_response() — mode-based routing
│   ├── gemini.py          # Google Gemini with grounding/url_context
│   ├── claude.py          # Anthropic Claude with thinking
│   └── openrouter.py      # OpenRouter multi-provider with retries
├── scrapers/              # News gathering
│   ├── topic_finder.py    # topic_finder(), headline extraction, overview
│   ├── calmatters.py      # CalMatters scraper
│   ├── google_search.py   # Google Custom Search API
│   └── web.py             # Generic webpage scraper
├── script/                # Script generation
│   ├── intro.py           # intro_writer() — titles, intro segments
│   ├── segments.py        # segments_writer() — dialogue generation
│   └── headlines.py       # headline_maker(), story_gatherer()
└── audio/                 # TTS and audio assembly
    ├── tts.py             # google_speak(), text2speech()
    ├── overview.py        # Overview audio splitting/merging
    ├── intro_music.py     # Theme song overlay
    └── assembly.py        # Final podcast assembly
```

### Entry Points

- `main.py` — Thin shim → `newscaster.pipeline.super_main()`
- `moviemaker.py` — Thin shim → `newscaster.video.create_and_export_video()`
- `uploader2.py` — Thin shim → `newscaster.upload.main()`

### Pipeline Stages

1. **News Gathering** — Scrapes NPR, AP News, Democracy Now, ProPublica, CalMatters, local Riverside news using LLM grounding and BeautifulSoup
2. **Story Selection** — Multi-LLM evaluation selects top US story, California-relevant story, and 5 minor stories; deduplicates against past 7 days
3. **Script Writing** — Generates dialogue scripts between anchor "Grace" and reporters with source attribution
4. **Audio Synthesis** — Google Cloud TTS with multiple voices (Grace, Ethan, Chloe, Elias); combines segments with PyDub
5. **Video Creation** — MoviePy combines audio with static image
6. **Distribution** — YouTube upload with OAuth

### Multi-LLM Orchestration

- Light mode (fast): Grok via OpenRouter
- Standard mode: GPT-5 or Gemini Flash
- Heavy mode: Gemini Pro or Claude Sonnet
- Routing handled by `get_llm_response()` in `newscaster/llm/router.py`

## Configuration

API keys are loaded from `keys.txt` (gitignored). See `keys.txt.example` for required entries:
- `google_genai_api` — Google Gemini
- `anthropic_api` — Anthropic Claude
- `openrouter_api` — OpenRouter
- `XI_API_KEY` — ElevenLabs (legacy)
- `google_search_api` — Google Custom Search
- `openweathermap_api` — OpenWeatherMap
- `google_cse_id` — Google Custom Search Engine ID

Also requires:
- `client_secrets.json` — Google OAuth client (gitignored)
- `newscaster1-03dc16232821.json` — GCP service account (gitignored)

## Output Structure

All outputs use date format `YYYY_MM_DD`:
- `stories_chosen/` — JSON files with selected stories and summaries
- `segment_summaries/` — Individual story summary text files
- `output_scripts/` — Dialogue scripts (intro, segments, overview, outro)
- `segment_audio/` — WAV files per segment
- `output_audio/` — Final MP3 files (standard and HQ)
- `episode_titles/` — YouTube video titles
- `logs/` — Daily execution logs
