# How a daily episode is made

This walks through one production run, front to back, as the code stands on
2026-09-23. File references point at the code that does each step. Line numbers
drift, so search for the function name if a reference is off.

## Schedule

- The Raspberry Pi runs `main2.bash` in a tmux session called `newscaster`.
- The script checks the clock every minute. Once a day, in the 4 a.m. hour, it runs:
  1. `main.py`: everything from gathering headlines to the finished MP3.
  2. `moviemaker.py`: the video.
  3. `uploader2.py`: the public YouTube upload.
- If `main.py` finishes without an MP3, the script deletes that day's scripts and
  segment audio and runs `main.py` again, up to two more times. A known gap is
  listed at the end.
- A full run publishes to YouTube. Never run `main2.bash` to test something.

## Models

Every LLM call names a mode. `newscaster/llm/router.py` (`_select_primary`) maps
each mode to a model, and the models are set in `newscaster/config.py`.

| Mode | Model | Used for |
|---|---|---|
| light | Gemini 3.1 Flash-Lite | Small chores: yes/no checks, pulling a headline out of text, tags |
| standard | GPT-6 Luna, low reasoning | Reading pages, summaries, the roundup script, evidence contracts |
| advanced | GPT-6 Luna, medium reasoning | Harder synthesis, the faithfulness fact check |
| heavy | Claude Opus 5.5 | Story selection, the research controller, segment scripts, script fixes |
| adversary | GPT-6 Sol, high reasoning | Challenges the research loop and approves script fixes |
| tagger | GPT-6 Sol, medium reasoning | Marks headlines as new, update or repeat |
| plus | Gemini 3.1 Pro | 2 of the 8 backup follow-up questions (only if the research loop crashes) |
| fallback | GPT-6 Sol, low reasoning | Any routed call that fails all its retries |

- **Retries.** Each routed call gets up to 5 attempts with backoff before it falls
  back to Sol.
  - A bad request (HTTP 400, 404 or 422) is raised at once, with no fallback.
  - An auth error skips the retries and goes straight to Sol.
  - A refusal or safety block comes back as an empty reply. That counts as
    retryable, so it is tried 5 times and then sent to Sol.
- **Google search tools.** Calls that ask for Google Search grounding or Google's
  URL reader go to Gemini models instead (`TOOL_LIGHT_STANDARD_MODEL`,
  `TOOL_PLUS_HEAVY_MODEL`). Today that means the backup readers for three front
  pages. Embeddings use `gemini-embedding-2`.
- **Web briefs.** These are GPT-6 Luna with OpenRouter's web search
  (`newscaster/search.py`, `openrouter_web_brief`). They bypass the router, so they
  get no retries and no fallback.
- **Plain search.** Google Custom Search runs first. An empty result is
  rephrased by Flash-Lite and retried, up to 5 tries in total. If that still
  fails, OpenRouter web search takes over, run by GPT-6 Sol at low reasoning.
  Web briefs and this fallback are logged to `search_audit.jsonl`, not the LLM
  audit log.

## 1. Gather headlines

`newscaster/scrapers/topic_finder.py`, `_gather_headline_sections`. It takes about
3 to 4 minutes and yields roughly 150 lines from 16 sections.

| Source | How it is read |
|---|---|
| NPR, AP, Democracy Now | A normal, visible Chromium loads the page on a virtual screen (Xvfb). GPT-6 Luna lists the stories in page order. A screenshot is kept for 14 days. If the browser fails, the older Gemini readers run instead. (`newscaster/scrapers/browser.py`) |
| ProPublica | RSS feed, last 48 hours. The front page has no dates. (`scrapers/dropsite.py`, `rss_scraper`) |
| Drop Site News | RSS feed, last 48 hours |
| CalMatters | Direct fetch of 7 section pages, today's and yesterday's stories (`scrapers/calmatters.py`) |
| City of Riverside | Direct parse of the dated press-release cards, today and yesterday (`scrapers/riverside.py`) |
| AI watch | OpenAI, Anthropic, Google DeepMind, METR, AI Incident Database and Import AI, last 72 hours. Opus keeps items that pass an event test (`scrapers/watchlist.py`) |
| 8 beat groups | RSS feeds, last 24 hours. Opus picks up to 8 events per group (`scrapers/watchlist.py`, `beat_scraper`) |

**How each line is written.** Every front-page line must be one sentence stating
who did what. Teasers with no event are left out, and so are lines that comment
on the page itself (`EVENT_SCRAPER_PROMPT` in `newscaster/prompts.py`).

**The AI watch test** has two ways to pass:
- **Route A, outside confirmation:** a regulator, court or other company acted,
  a developer lost control of a system or disclosed an incident, an independent
  evaluation found something, or a verified first.
- **Route B, major lab news on the lab's own word:** a new general-purpose model
  in a lab's main lineup, or an extraordinary result or incident. The lab gets
  the credit for the claim.

Case studies, rollouts, small features, pricing and speeches fail both routes.

**The beat groups** (`BEAT_FEEDS` in `newscaster/config.py`):

| Group | Outlets |
|---|---|
| Business and markets | CNBC, CNBC Business |
| Science and technology | Quanta Magazine, Ars Technica |
| Health | STAT News, KFF Health News |
| Courts | SCOTUSblog |
| World | BBC World, Al Jazeera, The Guardian |
| San Diego and Temecula | KPBS, Times of San Diego, Voice of San Diego, Valley News, City of Temecula. Local events only. |
| AI and tech press | The Verge AI, Simon Willison, 404 Media, Hacker News (300+ points) |
| Official records | Federal Register presidential documents, FDA press releases |

A group can carry one extra instruction. For example, the San Diego group skips
national stories that local outlets also run.

## 2. Mark repeats

`newscaster/tagger.py` (`tag_pool`), called from `topic_finder._tag_pool`.

- **Story memory.** The story ledger (`stories_chosen/story_ledger.json`) remembers
  every story arc the show has covered. Arcs expire after 45 days. The tagger
  compares against arcs covered in the last 14 days.
- **The tagger.** GPT-6 Sol reads the numbered headlines 10 at a time. It returns a
  verdict per line: new, update to a known arc, major escalation, or same as
  something already covered.
- **Applying the verdicts.** Code applies them, so a headline is only dropped on
  an explicit "same". Skipped line numbers are asked again, and a line still
  missing is kept untagged. If this structured tagger fails, the older rewrite
  tagger runs instead.
- **Coverage depth** (`dedup.py`, `apply_coverage_depth`). An update to a story
  that only ever appeared in the roundup becomes `[SIDE-COVERED]` and may lead. A
  story that last led 2 or more days ago becomes `[DEVELOPMENT]` and may lead
  again. Only a recent lead keeps the `[UPDATE]` bar.

## 3. Pick the stories

All in `topic_finder.topic_finder`.

1. **Triage.** Opus scores every headline from 1 to 10 for newsworthiness and
   keeps the top 10.
2. **California recall.** A second Opus pass scores relevance to an average
   Californian and keeps the top 5. The two lists merge to at most 16, with
   duplicates removed.
3. **Research briefs.** Each shortlisted headline gets one web brief from GPT-6
   Luna. A brief the web cannot confirm is marked UNVERIFIED.
   - Each brief carries a "Reported by:" line naming the sources that ran the
     story, plus coverage notes from the ledger.
   - If at least 4 briefs exist and 75% or more are UNVERIFIED, the final pickers
     are told the web search degraded, so they don't penalize those stories.
4. **Final picks.**
   - Opus picks the most important story.
   - Opus then picks the story that matters most to an average Californian (or,
     if nothing affects California, an everyday American), excluding the first.
   - GPT-6 Luna picks 5 side stories for the roundup.
5. **Ledger update.** Right after the picks, the ledger creates or updates an arc
   for each main and side story.

## 4. Research

**Parallel work.** The 5 side stories are researched at the same time, and so are
the 2 main stories (`PARALLEL_STORY_RESEARCH`). Within one main story the rounds run
in order, because each question depends on the last answer.

**Side stories** (`topic_finder.overview_process`):
- A research lookup answers a question about each one.
- GPT-6 Luna writes the roundup script from those findings.

**Research lookups** (`newscaster/research_tools.py`, called through
`newscaster/source_hunter.py`). Every research question, from a side story, the
seed pass or the agent loop, goes to a tool-using researcher:
1. **GPT-6 Luna (medium) researches with tools,** the way Claude Code does: web
   search, open a page (full text, in parts, with its links), and search inside a
   page. It chooses what to open, follows links to primary sources, and rewrites
   its searches from what it reads. It has up to 30 tool calls and 5 minutes.
2. **Every fact needs an exact quote.** Code keeps a fact only when its quote
   appears in the text of a page the researcher actually fetched. A lookup with no
   verified fact returns "no evidence", and is not re-run.
3. **The answer** lists the verified facts with their quotes and pages, the open
   gaps, and the sources. Each source keeps the page text around its quotes, which
   the fact check uses later.

Why: on the 32 real lookups of Sept 24 and 25, a blind Opus comparison preferred
this researcher's answers 31 to 1 over the fixed pipeline below and 32 to 0 over
what aired (`benchmarks/agentic_search/`). Each lookup costs about half a cent and
takes about 2 minutes. Switch: `RESEARCH_TOOL_LOOP_ENABLED`.

**Fallback: the fixed source hunter.** If the tool loop itself fails (the model
call errors out), the older pipeline runs instead: Luna writes up to 3 search
queries from the question; pages are fetched and checked in code for date,
entities and topic; and Luna answers from the accepted excerpts.

**The two main stories** (`newscaster/pipeline.py`, `_gather_one_topic`):
1. **Articles.** Google search finds articles. Up to 3 are kept after relevance
   checks and summarized.
   - A first source-hunter pass, the "seed", answers a starting question. Its
     answer is fed into the loop.
2. **Agent loop** (`newscaster/research_agent.py`, built with LangGraph):
   - **Controller.** Opus decides what to look up next, for 2 to 8 rounds. Each
     round runs a research lookup or fetches another article. It sees every earlier
     answer's open GAPS and is told to chase an answerable gap with a narrower
     question aimed at the primary source (a bill page, a court opinion, an agency
     release) before moving to a new topic.
   - **Memory.** Before the loop, relevant research from past episodes is pulled
     from the embedding index (`newscaster/rag/`).
   - **Adversary.** If the controller tries to stop early, Sol checks what's
     missing and can force one more search.
   - **Backup.** If the loop itself crashes, 8 fixed follow-up questions run
     instead.
3. **Summary.** GPT-6 Luna writes one research summary per story. If the output
   looks broken (empty, very short, repetitive, or looping on one word), it tries
   Luna medium, then Opus (`pipeline.py`, degeneracy guard). Flash-Lite then
   checks the summary for off-topic material, and Luna refines it if needed. The
   refined text is not re-checked by the guard.
4. **Saved for later.** The research is saved and embedded so later episodes can
   use it.

## 5. Write the script

`pipeline.write_scripts` and `newscaster/script/`.

- **Titles.** GPT-6 Luna writes one per main story. A title over 80 characters
  is regenerated up to 5 times, and can still end up longer. Titles are saved to
  `episode_titles/<date>.txt`.
- **Weather** (`newscaster/weather.py`). La Jolla daytime numbers and Temecula
  evening and overnight numbers, from OpenWeatherMap. Rain is mentioned only at
  30% chance or more.
- **Intro.** A fixed opening, then a short intro written by Flash-Lite that opens
  with the weather.
- **Segments.** Opus writes each main story as a conversation: Grace, the anchor,
  with Ethan on the first story and Chloe on the second.
  - **Framing.** A story that already had a full segment is framed as an update.
    A story heard only in the roundup is introduced as a first full telling.
  - **Draft scoring.** Code scores the first draft on length near 1,500 words,
    speaker balance and attribution. Below 0.55, two more drafts are written and
    the best of the three is kept.
  - **Missing speakers.** A reply with no speaker lines is asked again, up to 5
    times.
- **Roundup.** Grace reads up to 5 side stories. Stories marked UNVERIFIED are
  left out. The writer sees each story's headline, and a FOLLOW-UP note for any
  story the show covered before (when, whether it led or was in the roundup, and
  what listeners already know). A follow-up opens with a short tie-back such as
  "Following up on yesterday's story about…" instead of sounding like a new story.
- **Audience memory.** After the scripts, Flash-Lite records what the audience
  now knows about each arc, for tomorrow's framing. A side story left out of the roundup
  (UNVERIFIED) is recorded nowhere: not as coverage, and not as something the
  audience learned.
- **Outro.** A fixed text that credits the models.

## 6. Fact-check before any audio

`newscaster/review.py`, `review_and_revise_scripts`. A failure here never stops
the show.

- **Quotes.** Code checks every quote word for word against the saved source text.
- **Faithfulness.** GPT-6 Luna (medium) checks the segments and roundup against
  their sources. For the roundup, the "sources" are its research briefs plus the
  matching source-hunter excerpts.
- **General knowledge.** GPT-6 Luna flags suspect facts, and a web brief checks
  each one. These flags are only logged and never change the script.
- **Auto-fix** (`newscaster/editor_agent.py`). For quote and faithfulness
  problems, Opus proposes find-and-replace fixes and Sol approves or rejects each
  one, for up to 3 rounds. Applied changes are logged to
  `logs/fact_finder_edits.jsonl`. Switch: `FACT_FINDER_AUTOEDIT_ENABLED`.
- **Flags with nothing to fix.** A flag that only says a claim is missing from the
  sources, with no matching text, skips the editor.
- **Malformed replies.** A reply that doesn't follow the expected format is asked
  again once, then logged.

## 7. Audio

`newscaster/audio/`.

- **Voices.** Google Cloud Text-to-Speech, Chirp3-HD:
  - Grace: Aoede
  - Chloe: Leda
  - Ethan: Fenrir
- **One line at a time.** Clipped lines are re-synthesized.
- **Loudness.** Every line is matched to -23 LUFS, because the voices differ by
  about 2.5 dB (`audio/loudness.py`).
- **Music.** The intro goes over the theme music. The pick is the shortest theme
  version that fits the intro voice.
- **Output.** Intro, segments, roundup and outro are joined with 2-second gaps
  into `output_audio/<date>.mp3` and a higher-quality `_HQ.mp3`.

## 8. Video and upload

- **Video.** `moviemaker.py` puts the HQ audio over a still image
  (`image copy.png`).
- **Upload.** `uploader2.py` uploads to YouTube as public:
  - **Title:** the date plus the first episode title. A title over 100 characters
    is shortened by Flash-Lite, then cut at a word as a last resort.
  - **Tags:** written by Flash-Lite.
- **Retries.** Within one session, failed upload chunks are retried up to 10
  times. `main2.bash` then tries a fresh session up to 3 times, and logs to
  `logs/upload_failures.log` if all fail.

## Resuming a failed run

- Each stage writes a marker file when it finishes:
  - `segment_summaries/<date>_GATHER_COMPLETE.flag`
  - `output_scripts/<date>_SCRIPTS_COMPLETE.flag`
  - `output_scripts/<date>_UPLOAD_COMPLETE.flag`
- A rerun skips gathering and script writing when their markers exist.
- **Reuse within a stage.** The day's picks are stored in
  `segment_summaries/<date>_GATHER_MANIFEST.json`, and existing per-story
  summaries, scripts and segment MP3s are reused.
- **Stages that always rerun.** The fact check, audio and video always run
  again. Audio re-voices the intro, outro and roundup.
- **The upload marker.** Only `main2.bash` checks it. Running `uploader2.py` by
  hand uploads again.

## Logs

| File | What it records |
|---|---|
| `logs/log_<yy_mm_dd>.txt` | Everything printed during the run (`print_and_write`) |
| `logs/llm_audit.jsonl` | One row per routed attempt (success, error, or retries used up), with provider, model, phase, and a `call_id` that groups one call's rows |
| `logs/search_audit.jsonl` | Web briefs and search calls, which skip the router |
| `logs/source_hunter_audit.jsonl`, `logs/source_hunter_fetch_failures.jsonl` | Source hunter questions, accepted pages and failed fetches |
| `logs/fact_finder_edits.jsonl` | Applied script fixes, plus rejected proposals for any script that got at least one fix |
| `logs/fact_check_malformed.jsonl` | Fact-check replies that broke format twice |
| `logs/upload_failures.log` | Uploads that failed all 3 sessions |
| `screenshots/` | Front-page screenshots from the browser reader, 14 days |

Tests write to a temporary folder instead, through `NEWSCASTER_LOG_DIR`. JSONL
logs rotate at 10 MB and keep 5 old files.

## Known problems (found 2026-09-23, not yet fixed)

1. **TTS can hang.** The speech-synthesis retry loop in `audio/tts.py` has no
   limit, so a line Google keeps rejecting can hang the run.
2. **The retry after a late failure fails too.** `main2.bash` deletes the
   scripts but not the `SCRIPTS_COMPLETE` marker. The rerun skips writing
   scripts, then crashes opening the deleted `intro1.txt`.
3. **A comma in a title breaks the YouTube title.** Titles are saved
   comma-separated and split on commas at upload. The title prompt says "No
   commas", so this bites only when the model ignores that.
4. **Some calls are wasted.**
   - `script/headlines.py`, `headline_maker`, makes a call per story whose output
     is never used. It has no error handling, so if it fails it stops the whole
     script stage.
   - Some script files (segments, intro, outro, titles) are plain writes, so a
     crash mid-write can leave a half file that the next run reuses.
   - The general-knowledge fact check runs web searches that cannot change
     anything.
5. **Web briefs have no retry or fallback.**
