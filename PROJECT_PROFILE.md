# Newscaster 3.5 — Project Profile

> Automated daily AI news podcast: LLM-grounded research, editorial selection, multi-voice script generation, speech synthesis, video production, and YouTube publishing.

## What It Does

Newscaster is a fully automated end-to-end news podcast pipeline that runs daily at 4 AM. Each day it:

1. **Discovers stories** from 7 news sources (NPR, AP News, Democracy Now, ProPublica, CalMatters, Drop Site News, City of Riverside) — via Gemini's grounding and URL-context features for most sources, a dedicated BeautifulSoup scraper for CalMatters, an RSS scraper for Drop Site News, and article-body extraction from search-result URLs
2. **Selects stories** using a 3-tier LLM editorial pipeline (triage scoring → deep research → final selection) with deduplication against the past 7 days
3. **Researches** each story through a controlled **source hunter** (URL discovery → controlled fetch/PDF extraction → source validation → answer only from accepted evidence) driven by an adaptive **research agent** — a LangGraph controller that chooses each next research action, with a second-perspective adversary that poses the strongest skeptical question before synthesis. A fixed multi-round follow-up path (light → standard → plus → heavy, with adversarial "challenging" questions) remains as fallback when the agent is disabled or errors
4. **Generates** natural multi-voice dialogue scripts between anchor "Grace" and reporters Ethan and Chloe, with heuristic quality scoring and automatic retries
5. **Synthesizes speech** using Google Cloud TTS (Chirp3 HD) with distinct voices per character, intro music overlay, clipping detection, and automatic normalization
6. **Produces video** by packaging audio with branding imagery (static image + audio mux via MoviePy)
7. **Uploads** to YouTube with LLM-generated episode titles and Gemini-generated tags

The system maintains a **persistent story ledger** that tracks multi-episode narrative arcs and audience knowledge state — extracting what the audience learned each episode and reinjecting that context into future scripts for coherent follow-up coverage. Stories are tagged as `[UPDATE]` or `[MAJOR ESCALATION]` when they continue an existing arc.

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3 (~8,000 lines across 42 files) |
| **LLM APIs** | Google Gemini (3 Flash, 3.1 Flash Lite, 3.1 Pro) with grounding/url_context + extended thinking; Anthropic Claude Opus 4.8; GLM 5.2 and Gemma 4 31B via OpenRouter; GPT-5.5 as the adversary reviewer and the router's OpenRouter fallback; extended thinking/reasoning effort enabled across providers |
| **Web/Scraping** | BeautifulSoup4, requests (article body extraction, CalMatters scraper); Google Custom Search API with LLM query rewriting |
| **Audio** | Google Cloud Text-to-Speech (Chirp3 HD), PyDub, wave/struct (clipping detection) |
| **Video** | MoviePy (static image + audio packaging) |
| **Cloud/Infra** | Google Cloud Platform (TTS, YouTube Data API v3), OAuth2 with persistent token refresh, Google Custom Search Engine |
| **Scheduling** | Bash scheduler with 3-pass retry logic and intermediate output cleanup |
| **Testing** | pytest — 213 tests spanning config bootstrapping, router dispatch, the search abstraction, source-hunter validation, research-agent control flow, and a fully mocked web-search benchmark |

## Architecture Highlights

### Multi-LLM Router

Central `get_llm_response()` dispatcher routes requests across 4 model tiers based on task complexity and whether real-time web grounding is needed:

| Mode | Without Grounding | With Grounding |
|---|---|---|
| **light** | Gemini 3.1 Flash Lite | Gemini 3 Flash |
| **standard** | Gemma 4 31B (OpenRouter) | Gemini 3 Flash |
| **advanced** | GLM 5.2 (OpenRouter, medium reasoning) | Gemini 3 Flash |
| **adversary** | GPT-5.5 (OpenRouter, high reasoning) | Gemini 3.1 Flash Lite |
| **plus** | Gemini 3.1 Pro | Gemini 3.1 Pro |
| **heavy** | Claude Opus 4.8 (Anthropic) | Gemini 3.1 Pro |

This balances cost, latency, and quality — light tasks use fast/cheap models, while dialogue generation and deep synthesis use the most capable models. Only Gemini supports grounding (Google Search) and URL fetching, so the router automatically selects the right provider. Extended thinking is enabled across providers (Gemini: 8k token budget; Claude: 10k token budget; the OpenRouter models use reasoning-effort levels). The OpenRouter client has sophisticated retry logic with transport variant cycling and provider fallbacks.

### Tiered Story Selection Pipeline

Three-stage LLM-powered editorial process:
1. **Triage** (standard/Gemini Flash) — Scores all discovered stories 1-10 for newsworthiness using structured output parsing
2. **Research** (controlled source hunter) — Evidence-gated investigation on the shortlisted candidates, producing background briefs attributed to validated sources; headlines with no accepted current evidence are marked `UNVERIFIED` rather than summarized from model knowledge
3. **Selection** (heavy/Claude Opus for main picks, standard/Gemini for overview) — Final picks: 1 national "important" story, 1 California/everyday story, plus a 5-story overview segment. Each search result goes through double relevance filtering (pre-scrape headline check + post-summarization content check)

### Controlled Source Hunter

Story research is evidence-gated rather than recalled from the model. For each question the hunter generates a soft "evidence contract" (the facts an answer should recover, plus any source preferences), discovers candidate URLs via search, fetches and extracts them (HTML and PDF, with a reader fallback for bot-walled pages), classifies each source as primary-ish (`.gov`/`.mil`, official record systems) or secondary-ish (news/social), validates against the contract, and synthesizes an answer **only from accepted evidence** — returning `no_evidence` instead of guessing when nothing validates. Rejected sources become rejection memory for the next search round, and a bounded same-domain expansion can hop from a discovered official page to the exact evidence page.

### Adaptive Research Agent

A LangGraph loop orchestrates research for each selected story: a controller (heavy tier) decides the next action — a precise grounded question, a broader article search, or "done" — within min/max iteration budgets; an adversary (GPT-5.5, high reasoning) poses the single strongest skeptical question before the story is allowed to synthesize. Optional RAG research-memory injects prior coverage as background. If the agent is disabled or errors, the pipeline falls back to the fixed multi-round follow-up path.

### Story Arc Tracking & Audience Memory

A persistent JSON ledger tracks stories across episodes with a full continuity loop:
- Auto-creates story arcs with LLM-generated slugs (e.g., `israel_gaza_humanitarian_aid`)
- Tags continuing stories as `[UPDATE]` or `[MAJOR ESCALATION]`
- Extracts `audience_learned` facts from each episode and reinjects them into future script prompts
- Prevents redundant recaps while maintaining narrative continuity
- Auto-prunes arcs after 45 days

### Audio Quality Assurance

- Clipping detection on all WAV output (flags >0.05% samples at 99% max amplitude)
- Automatic normalization with ±3dB headroom
- Up to 2 regeneration attempts before falling back to normalization
- Dynamic intro music fitting — selects from 7 BGM variants of different lengths, picking the shortest one that exceeds the voice intro duration
- TTS text chunking to handle length limits on long overview narration
- TTS failure recovery: every 3rd failure, an LLM rewrites the text to fit under 5000 chars / 700 chars per sentence before retrying

### Script Quality Scoring

Heuristic validation of generated dialogue scripts:
- Word count within target range
- Voice balance across characters
- Presence of attribution phrases and journalistic standards
- Automatic retry on formatting failures

### Resilience & Partial Idempotency

- Core pipeline stages (news gathering, script writing, segment TTS) check if output files already exist and skip if so
- Partial failures can be re-run without fully duplicating work
- Bash scheduler retries up to 3x, cleaning intermediate outputs between attempts
- YouTube upload has exponential backoff retry (max 10 attempts)
- Intros, outros, overview, and final audio assembly always regenerate (intentional — these depend on all segments being complete)

## Package Structure

```
newscaster/
  __init__.py
  config.py             # API key management via deferred initialization
  pipeline.py            # Main orchestrator: gather → write → generate
  search.py              # Web search abstraction (Google CSE → OpenRouter web fallback)
  source_hunter.py       # Evidence-gated research: discover → fetch → validate → answer
  source_hunter_primitives.py  # Stateless source-hunter toolkit (synced with benchmarks/)
  research_agent.py      # LangGraph adaptive research loop (controller + adversary)
  prompts.py             # Centralized editorial/system prompt templates
  dedup.py               # Story dedup + persistent arc ledger + audience memory
  logging.py             # Dual console + file logging
  dates.py               # Spoken date formatting (e.g., "March seventeenth, twenty twenty-six")
  text_utils.py          # Text cleaning, quote extraction, grounding retry detection
  weather.py             # OpenWeatherMap daily temperature for intro scripts
  video.py               # MoviePy audio+image packaging
  upload.py              # YouTube OAuth2 upload + Gemini tag generation
  llm/
    router.py            # Multi-LLM dispatcher (mode × grounding routing)
    gemini.py            # Google Gemini with grounding/url_context/thinking
    claude.py            # Direct Anthropic Claude client with extended thinking + GPT-5 fallback
    openrouter.py        # OpenRouter client (provider overrides, transport fallbacks, retry, usage capture)
  scrapers/
    topic_finder.py      # Tiered story selection + deep research + follow-up rounds
    calmatters.py        # Dedicated CalMatters scraper (BeautifulSoup)
    dropsite.py          # Drop Site News RSS scraper (48h lookback)
    web.py               # Article body extraction from URLs
    google_search.py     # Google CSE with LLM query rewriting on empty results
  script/
    intro.py             # LLM-generated episode titles + intro scripts
    segments.py          # Dialogue script generation with arc context injection
    headlines.py         # Headline extraction from summaries
  audio/
    tts.py               # Google Cloud TTS with clipping detection + normalization
    assembly.py          # Final audio mixing + dual-bitrate export
    overview.py          # Overview narration with TTS text chunking
    intro_music.py       # Music overlay mixing
```

## Key Engineering Decisions

- **Centralized prompt management**: 24 key editorial prompts live in `prompts.py` as template constants — single source of truth for editorial voice, easy to iterate (some utility prompts remain inline)
- **Deferred config initialization**: `config.init()` loads API keys at runtime, not import time — solves circular dependency issues and enables clean testing
- **Lazy imports**: Strategic use in `dedup.py` and `claude.py` to break circular dependencies without restructuring the package
- **Dual-bitrate export**: 124k standard + 248k HQ for YouTube — optimizes for both file size and platform quality requirements
- **Modular refactoring**: Evolved from a monolithic 2,100+ line `main.py` into a clean package with 6 subpackages
- **Evidence-gated research over open grounding**: Headline discovery still uses Gemini grounding/url_context per source, but story research moved to the controlled source hunter — it answers only from fetched, validated sources, returning `UNVERIFIED`/`no_evidence` rather than guessing from model knowledge
- **OpenRouter with provider overrides**: Transport fallbacks, retry strategy, and usage capture for reliable Claude access
- **Adversarial follow-up questions**: Each story gets both regular and "challenging" follow-ups that question the premise — e.g., if a story criticizes a government response, the system asks whether the criticism is actually valid; the adaptive research agent adds a dedicated adversary pass (GPT-5.5) that poses the single strongest skeptical question before a story may synthesize
- **Grounding citation extraction**: Gemini responses automatically have source citations extracted from grounding metadata and appended
- **Multi-provider fallback chain**: `heavy` requests go directly to the Anthropic Claude client; on retry exhaustion or auth failure the router falls back to GPT-5.5 via OpenRouter, which has its own retry logic with transport-variant cycling and provider overrides

## Deployment

- **Schedule**: Bash loop (`main2.bash`) triggers the full pipeline daily at 4 AM — runs `main.py` → `moviemaker.py` → `uploader2.py` sequentially
- **Credentials**: OAuth tokens persist across runs; 7 API keys loaded from gitignored `keys.txt`; GCP service account for TTS
- **Output**: Daily episodes auto-published to YouTube with LLM-generated titles and tags

## Skills Demonstrated

- **AI/ML Systems Engineering**: Multi-provider LLM orchestration with intelligent tier-based routing, extended thinking, evidence-gated retrieval with source validation, audience-memory extraction, output quality scoring, adversarial fact-checking, and prompt engineering across centralized templates
- **End-to-End Automation Pipeline**: Full pipeline from news discovery through media production to platform publishing, with retry logic and partial idempotency
- **Audio Processing**: TTS integration with multiple voices, clipping detection, automatic normalization, music overlay mixing, text chunking for API limits, dual-bitrate export
- **API Integration**: Google Cloud (TTS, YouTube Data API v3), Anthropic Claude, OpenRouter (with 20+ provider overrides), Google Custom Search Engine, Google Gemini grounding/url_context, OpenWeatherMap, OAuth2 token management
- **Automation Engineering**: Idempotent stages, exponential backoff retry, transport-level fallback cycling, state management via persistent JSON ledger (atomic writes via tempfile + os.replace), bash scheduling with cleanup between retries
- **Software Architecture**: Monolith-to-package refactoring, centralized configuration with deferred init, dependency management via lazy imports, provider abstraction layer
