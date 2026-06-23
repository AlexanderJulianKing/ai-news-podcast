# Newscaster

**A fully automated daily AI news podcast.** Newscaster scrapes the day's news, uses a from-scratch multi-provider LLM pipeline to select, research, and script the stories, synthesizes multi-voice audio, renders a video, and publishes to YouTube — unattended, every morning.

**See it in action:** [**@NewsFromAlex** on YouTube](https://www.youtube.com/@NewsFromAlex) — 525+ daily episodes, live since September 2024

**How it works (deep dive):** [**Newscaster explainer**](https://alexanderjulianking.github.io/newscaster_overview.html)

Solo project, running in production on a Raspberry Pi.

---

## What it does

Each morning, a single pipeline run:

1. **Discovers** stories from 7 sources (NPR, AP, Democracy Now, ProPublica, CalMatters, Drop Site News, City of Riverside) via Gemini grounding, dedicated scrapers, and Google Custom Search.
2. **Selects** stories with a 3-tier LLM editorial pass (triage scoring → grounded research briefs → final picks).
3. **Researches** each story with an agentic, source-grounded loop (LangGraph) that fetches and validates its own sources, with an adversarial counter-evidence check.
4. **Writes** a multi-voice dialogue script between an anchor and reporters, with heuristic quality scoring and retries.
5. **Fact-checks** the script against the gathered sources and auto-corrects confirmed factual errors before anything is voiced.
6. **Synthesizes** audio with Google Cloud TTS (distinct per-character voices, music overlay, clipping detection).
7. **Renders** a video (MoviePy: audio + background image), and
8. **Uploads** to YouTube (OAuth2) with an LLM-generated title and tags.

## Architecture highlights

The interesting engineering is the orchestration and reliability layer, all hand-built:

- **Multi-provider LLM router** — one dispatcher routes each call across capability tiers and providers (Google Gemini, Anthropic Claude, plus Gemma / GLM / GPT-5.5 via OpenRouter), grounding-aware, with a typed error taxonomy, retry/backoff with jitter, and cross-provider fallback.
- **Agentic research loop** — a LangGraph state machine decides what to ask next about each story; a controlled "source hunter" fetches pages itself (requests + BeautifulSoup), validates them against an evidence contract, and answers only from validated excerpts — returning "no evidence" rather than guessing.
- **Retrieval-augmented memory** — each day's research is embedded (Gemini embeddings) into a from-scratch SQLite + NumPy cosine vector store; later episodes retrieve relevant prior coverage. Retrieval quality is measured with a recall benchmark (`benchmarks/rag_recall/`).
- **Self-correcting fact-checker** — a pre-TTS editor grounded against the raw scraped sources: an LLM proposes find/replace fixes, an independent adversary model vets each one, and only verified, unambiguous edits are applied (with a JSONL audit log).
- **Crash-safe pipeline** — atomic writes, idempotent stages with completion markers, per-story fault isolation, and graceful degradation.
- **Evaluation harnesses** — `benchmarks/` grades news-discovery quality across a model matrix and measures embedding-retrieval recall.

## Tech stack

Python 3 · Google Gemini / Anthropic Claude / OpenRouter · Google Cloud TTS · YouTube Data API v3 · Google Custom Search · LangGraph · NumPy + SQLite (vector store) · BeautifulSoup · PyDub · MoviePy · pytest

## Setup

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. System dependencies
#    ffmpeg   — audio/video encoding (PyDub, MoviePy)
#    poppler  — provides `pdftotext`, used by the source hunter to read PDFs
#    macOS:  brew install ffmpeg poppler
#    Debian/Pi:  sudo apt install ffmpeg poppler-utils

# 3. API keys — copy the template and fill in your own keys
cp keys.txt.example keys.txt
#    google_genai_api, anthropic_api, openrouter_api,
#    google_search_api, openweathermap_api, google_cse_id

# 4. Google credentials (NOT included in the repo)
#    client_secrets.json   — OAuth client for the YouTube upload
#    <service-account>.json — GCP service account for Cloud TTS
```

`keys.txt` and the credential JSON files are gitignored — never commit them.

## Running

```bash
python3 main.py          # scrape → select → research → script → fact-check → audio
python3 moviemaker.py    # render the video from the audio + background image
python3 uploader2.py     # upload to YouTube

./main2.bash             # scheduler: runs all three daily, with retry logic

python3 -m pytest        # test suite
```

Each stage is idempotent — it skips work whose output already exists, so a re-run resumes cleanly rather than duplicating it.

## Repository layout

```
newscaster/            core package
  pipeline.py          orchestrates the daily run
  llm/                 multi-provider router, typed errors, retry/fallback
  research_agent.py    LangGraph agentic research loop
  source_hunter.py     controlled fetch → validate → synthesize
  rag/                 embeddings + vector store + retrieval
  review.py            pre-TTS fact-finder (quote / faithfulness / stable-fact passes)
  editor_agent.py      propose → vet → verify-then-apply auto-editor
  dedup.py             persistent story-arc ledger + cross-episode memory
  scrapers/            per-source news discovery
  script/  audio/      dialogue generation · TTS, assembly, music
benchmarks/            evaluation harnesses (web-search quality, RAG recall)
main.py · moviemaker.py · uploader2.py   entry points
main2.bash             daily scheduler
```

## Notes

The specific LLM model choices and routing table evolve over time; see the [explainer](https://alexanderjulianking.github.io/newscaster_overview.html) for the design narrative. This is solo, end-to-end personal-project work — there is no team CI/CD or cloud-scale deployment; it runs as a scheduled job on a Raspberry Pi.
