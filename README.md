# Newscaster

**A fully automated daily news podcast.** Every morning, Newscaster reads the day's
news and picks the stories. It researches and scripts them with a multi-model LLM
pipeline, voices them with several speakers, renders a video, and publishes to
YouTube, all with no one at the keyboard.

**See it in action:** [**@NewsFromAlex** on YouTube](https://www.youtube.com/@NewsFromAlex), daily since September 2024.

**Full walkthrough of one run:** [`docs/PIPELINE.md`](docs/PIPELINE.md)

Solo project, running in production on a Raspberry Pi.

---

## What it does

Each morning, one run:

1. **Gathers** about 150 headlines from 16 sections:
   - **Front pages.** NPR, AP and Democracy Now are read in a real browser.
   - **Feeds and pages.** RSS feeds and direct page parsing cover ProPublica, Drop
     Site News, CalMatters and the City of Riverside.
   - **AI lab watch.** OpenAI, Anthropic, Google DeepMind, METR and others.
   - **Eight topic groups:** business, science and tech, health, courts, world,
     San Diego and Temecula, AI and tech press, and official records.
2. **Marks repeats** against a ledger of every story arc the show has covered, so
   yesterday's lead doesn't lead again unless something new happened.
3. **Picks** a main story, an everyday story, and five side stories. This takes
   two scoring passes, a web-searched brief per shortlisted headline, and final
   editorial calls.
4. **Researches** the main stories with an agent loop. Opus decides what to look
   up next and chases open gaps, a tool-using researcher does each lookup, and a
   second model challenges the result before it stops. Stories are researched in
   parallel.
5. **Writes** the script as a conversation between an anchor and two reporters,
   plus a roundup, intro and outro.
6. **Fact-checks** every quote and claim against the saved sources and fixes
   confirmed errors before anything is voiced.
7. **Voices** it with Google Cloud Text-to-Speech, with every line matched to the
   same loudness, and adds the theme music.
8. **Renders** a video and **uploads** it to YouTube.

## Models

Most LLM calls name a tier, and one router maps tiers to models. As of 2026-09-23:

| Tier | Model | Main jobs |
|---|---|---|
| heavy | Claude Opus 5.5 | Story selection, research controller, segment scripts, script fixes |
| standard / advanced | GPT-6 Luna | Reading front pages, summaries, research answers, web briefs, faithfulness check |
| tagger / adversary / fallback | GPT-6 Sol | Marking repeats, challenging research, approving fixes, backup for any failed call |
| light | Gemini 3.1 Flash-Lite | Small yes/no checks, the spoken intro, YouTube tags |

Web briefs (GPT-6 Luna) and the search fallback (GPT-6 Sol) skip the router.
Gemini 3 Flash is the backup front-page reader, Gemini 3.1 Pro answers two backup
research questions, and `gemini-embedding-2` makes the embeddings.

Model choices are tested before they change. `benchmarks/` holds the harnesses.
The comments in `newscaster/config.py` record why each model was chosen.

## Engineering highlights

- **Multi-provider LLM router.** It covers Anthropic, OpenRouter and Google. It
  has typed errors, retries with backoff, and a fallback model, and every attempt
  is written to an audit log.
- **Honest front-page reading.** A normal, visible Chromium runs on a virtual
  screen, and the rendered page is read over the DevTools protocol. Nothing is
  disguised. A screenshot of each page is kept for checking.
- **Tool-using researcher.** GPT-6 Luna searches, opens pages, searches inside
  them and follows links until a question is answered. Every fact must carry a
  quote that code finds on a page it fetched; otherwise the answer is "no
  evidence". A fixed fetch-and-validate pipeline is the fallback.
- **Agentic research loop.** Built with LangGraph. Opus is the controller, GPT-6
  Sol is the adversary, and memory from past episodes comes from embeddings.
- **Structured repeat tagging.** The model returns a verdict per numbered
  headline, and code applies it. A headline is never silently dropped.
- **Self-correcting fact check.** One model proposes each fix, a second model must
  approve it, and every change is logged.
- **Retrieval memory.** Research is embedded with Gemini embeddings into a
  from-scratch SQLite and NumPy vector store. Recall is measured by
  `benchmarks/rag_recall/`.
- **Safe to resume.** The ledger, run manifest, research summaries and stage
  markers are written atomically. Finished stages are skipped on a rerun, and a
  failed story is isolated from the rest.

## Tech stack

Python 3 · Anthropic, OpenRouter and Google Gemini APIs · Chromium + Xvfb (DevTools protocol) · LangGraph · Google Cloud TTS · YouTube Data API v3 · Google Custom Search · OpenWeatherMap · NumPy + SQLite · BeautifulSoup · PyDub · MoviePy · pytest

## Setup

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. System dependencies
#    ffmpeg            audio and video encoding (PyDub, MoviePy)
#    poppler           provides pdftotext, used by the source hunter to read PDFs
#    chromium, xvfb    the browser reader for NPR, AP and Democracy Now
#                      (without them those three fall back to Gemini readers)
#    macOS:      brew install ffmpeg poppler
#    Debian/Pi:  sudo apt install ffmpeg poppler-utils chromium xvfb

# 3. API keys: copy the template and fill in your own
cp keys.txt.example keys.txt
#    google_genai_api, anthropic_api, openrouter_api,
#    google_search_api, openweathermap_api, google_cse_id

# 4. Google credentials (not in the repo)
#    client_secrets.json     OAuth client for the YouTube upload
#    service-account JSON    GCP service account for Cloud TTS (file name is set in newscaster/audio/tts.py)
```

`keys.txt` and the credential files are gitignored. Never commit them.

## Running

```bash
python3 main.py          # gather, pick, research, script, fact-check, audio
python3 moviemaker.py    # render the video
python3 uploader2.py --file=output_video.mp4 --category=25 --privacyStatus=public

./main2.bash             # the daily loop on the Pi: waits for 4 a.m., runs all three, retries
                         # (Linux only: it uses ./venv and GNU date)

pip install pytest && python3 -m pytest   # tests never launch a browser or write to logs/
```

Each stage leaves a completion marker, so a rerun picks up where the last one
stopped.

## Repository layout

```
newscaster/              core package
  pipeline.py            runs the day, stage by stage
  config.py              models, feeds and switches, with the reasons for each
  llm/                   router, providers, typed errors, retries and fallback
  scrapers/              topic_finder (gathering, tagging, picking), browser reader,
                         RSS, CalMatters, Riverside, AI watch and beat feeds
  tagger.py              structured repeat tagging
  dedup.py               story-arc ledger and coverage depth
  research_agent.py      LangGraph research loop
  source_hunter.py       contract, fetch, validate, answer
  search.py              Google search and web briefs
  rag/                   embeddings, vector store, retrieval
  script/                titles, intro, segments
  review.py              pre-TTS fact check
  editor_agent.py        propose, approve, apply script fixes
  weather.py             commute weather for the intro
  audio/                 TTS, loudness, music, assembly
  video.py  upload.py    video render, YouTube upload
benchmarks/              evaluation harnesses (web search, RAG recall, editor brain)
docs/PIPELINE.md         one run, front to back
docs/archive/            earlier design notes (June and July 2026)
main.py · moviemaker.py · uploader2.py   entry points
main2.bash               daily scheduler
```

## Notes

This is solo personal-project work. It runs as a scheduled job on a Raspberry Pi,
with no team CI or cloud deployment. The
[explainer page](https://alexanderjulianking.github.io/newscaster_overview.html)
tells the design story; for how the code works today, trust
[`docs/PIPELINE.md`](docs/PIPELINE.md).
