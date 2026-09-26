# LLM call inventory by tier

Rebuilt 2026-09-23 from the working tree on branch `story-selection-eligibility` (tip `f92db0a`; `upload.py` has uncommitted edits, and its line numbers come from the working tree).
The first version was built 2026-09-20. Model routing changed on 2026-09-23 (`af88524`, `e034fc2`, `ac8a966`, `93c489d`, `57ae2c9`, `f92db0a`).
Tier to model mapping comes from `newscaster/config.py` and `newscaster/llm/router.py:_select_primary`.
Line numbers drift. Re-grep `get_llm_response(`, `call_with_default(`, `openrouter_web_brief(`, `select_with_llm(` and `ask=` before relying on them.
Paths are relative to `newscaster/`.

"Shape" says what the pipeline does with the answer:

- **decide**: the code reads a yes/no, a pick from a list, or an accept/reject. A wrong answer changes what airs.
- **write**: the answer is prose that gets used as text.
- **mixed**: a decision wrapped in prose or JSON that the code parses.

How a call's tier is set: `get_llm_response` defaults to `mode="light"`, and `call_with_default` passes its arguments straight through, so a call with no `mode` is light.
`grounding=True` or `url_context=True` sends the call to Gemini whatever the mode (see "Tool tiers" below).

## light: `gemini-3.1-flash-lite` (Google direct)

| Call | Where | Shape | What it decides or writes |
|---|---|---|---|
| determine-relevance | `scrapers/topic_finder.py:93` | decide | Is this search result a news article relevant to the topic? yes/no |
| article-relevance | `scrapers/topic_finder.py:147` | decide | Does the article summary hold anything relevant? yes/no |
| irrelevance-check | `pipeline.py:569` (no mode) | decide | Does the super summary contain off-topic material? yes triggers a rewrite |
| arc-match | `dedup.py:496` | decide | Which tracked arc a paraphrased headline belongs to, or NONE. Only runs when the headline shares a word with some arc |
| selection parse | `scrapers/topic_finder.py:85` (no mode), called at `:857`, `:864` | mixed | Pulls the chosen headline out of Opus's Tier-3 lead and everyman essays |
| overview story extraction | `scrapers/topic_finder.py:279` | mixed | "Find story number N" over the overview pick. Five calls, one per side story |
| controller JSON repair | `research_agent.py:157` | mixed | Repairs malformed research-controller JSON. Only on a parse failure |
| follow-up question, rounds 1-2 | `pipeline.py:56` | write | Next research question. Fallback path only (see note under heavy) |
| slug generation | `dedup.py:121` | write | New arc slug |
| archive summary | `dedup.py:581` | write | Two-sentence covered-story summary |
| audience-learned extraction | `pipeline.py:623`, `:640` | write | JSON of facts the audience has now heard (main stories, then side stories) |
| news-source name | `scrapers/topic_finder.py:157` | write | Outlet name from a URL |
| search query rephrase | `pipeline.py:467` (no mode) | write | New search query when a topic found zero articles |
| search query rephrase | `scrapers/google_search.py:64` | write | New query when Google CSE returns nothing |
| headline | `script/headlines.py:26` (no mode) | write | Segment headline. Output is never used (see last section) |
| intro narration | `script/intro.py:36` (no mode) | write | The "on the program today" intro text read on air |
| title shorten, tags | `upload.py:106`, `:163` | write | YouTube title fit (up to 3 tries, only when the title is over 100 characters) and tags |
| TTS length fix | `audio/tts.py:143` | write | Trims text to TTS limits after a failed synthesis |

## standard: `openai/gpt-6-luna`, reasoning low (OpenRouter)

| Call | Where | Shape | What it decides or writes |
|---|---|---|---|
| front-page read (NPR, AP, Democracy Now) | `scrapers/topic_finder.py:596` via `scrapers/browser.py:186` | mixed | Turns the rendered page text into a list of today's events. One call per page. Falls back to the Gemini scrape on any failure |
| overview pick | `scrapers/topic_finder.py:872` | mixed | Picks the side stories for the roundup, excluding the two main stories |
| evidence contract | `source_hunter.py:164` | mixed | Required facts and reject rules for sources (advisory for news_research). Runs once per source-hunter attempt, so twice when it escalates |
| stable-fact pass | `review.py:506` via `:446` | mixed | Flags background-fact errors from memory. Advisory only; never reaches the editor. One retry on a malformed reply |
| retype tagger (fallback) | `scrapers/topic_finder.py:571` | mixed | Old tagger that retypes the whole pool. Runs only if the structured tagger raises |
| source-hunter answer, first try | `source_hunter.py:146`, mode set at `:363` | write | FINDINGS plus GAPS from fetched sources |
| article summarize | `scrapers/topic_finder.py:121` | write | Article summary |
| super summary, first try | `pipeline.py:368` | write | Story summary. Escalates to advanced, then heavy, on degenerate output |
| super summary refinement | `pipeline.py:577` | write | Rewrite after irrelevance-check says yes |
| RAG refine | `pipeline.py:408` | write | Off (`RAG_AUGMENT_ENABLED = False`) |
| research memory note | `research_agent.py:194` | write | Distils retrieved prior coverage |
| follow-up question, rounds 3-4 | `pipeline.py:56` | write | Next research question. Fallback path only |
| overview anchor | `scrapers/topic_finder.py:880` | write | Roundup script text |
| episode title | `script/intro.py:11`, `:15` | write | One title per slot. Re-asked up to 5 times while it is over 80 characters |

## advanced: `openai/gpt-6-luna`, reasoning medium (OpenRouter)

| Call | Where | Shape | What it decides or writes |
|---|---|---|---|
| faithfulness pass | `review.py:479` via `:446` | mixed | Flags script claims the source corpus does not support. Feeds the auto-editor. One retry on a malformed reply; if both are malformed, the script is logged as unchecked |
| source-hunter answer, escalation | `source_hunter.py:146`, mode set at `:369` | write | Second try when the standard attempt finds no accepted evidence |
| super summary, second try | `pipeline.py:368` | write | Fallback after degenerate standard output |

## tagger: `openai/gpt-6-sol`, reasoning medium (OpenRouter)

New since 2026-09-20.

| Call | Where | Shape | What it decides or writes |
|---|---|---|---|
| structured dedup tagger | `scrapers/topic_finder.py:549` via `tagger.py:126` | mixed | A verdict per numbered headline, in batches of 10 (`TAGGER_BATCH_SIZE`). Code applies `[UPDATE]` / `[MAJOR ESCALATION]` tags; `dedup.apply_coverage_depth` then turns them into `[SIDE-COVERED]` or `[DEVELOPMENT]`. Controls eligibility to lead. Up to 3 calls per batch (retry on bad JSON, then a follow-up for skipped numbers) |

## adversary: `openai/gpt-6-sol`, reasoning high (OpenRouter)

| Call | Where | Shape | What it decides or writes |
|---|---|---|---|
| edit vet | `editor_agent.py:152` | decide | Approve or reject each of Opus's proposed script edits |
| research adversary | `research_agent.py:296` | mixed | Challenges whether the research is done. Can force more rounds |

## heavy: `claude-opus-5-5` (Anthropic direct)

| Call | Where | Shape | What it decides or writes |
|---|---|---|---|
| tier-1 triage, national | `scrapers/topic_finder.py:749` | mixed | Scores every pool headline; top 10 go to research |
| tier-1B triage, California | `scrapers/topic_finder.py:775` | mixed | Separate California and everyday-life scoring; top 5 merged in (16 max) |
| tier-3 important story | `scrapers/topic_finder.py:852` | mixed | Picks the lead story |
| tier-3 everyman story | `scrapers/topic_finder.py:862` | mixed | Picks the second main story |
| AI watch event test | `scrapers/watchlist.py:331` via `watchlist_scraper` | mixed | Which lab and evaluator feed items are real events worth nominating. One call |
| beat selection | `scrapers/watchlist.py:331` via `beat_scraper` | mixed | Picks up to 8 concrete events per beat. One call per beat group, 8 groups in `config.BEAT_FEEDS`: business, science and technology, health, courts, world, San Diego and Temecula, AI and tech press, official records. A group with no items in its window makes no call |
| research controller | `research_agent.py:327` | mixed | Next question, or stop |
| edit propose | `editor_agent.py:119` | mixed | Proposes find/replace fixes for corpus-grounded flags |
| follow-up question, rounds 7-8 | `pipeline.py:56` | write | Next research question. Fallback path only |
| segment script | `script/segments.py:55` | write | The spoken script. Re-asked up to 5 times if the Grace or reporter lines are missing |
| super summary, last try | `pipeline.py:368` | write | Final fallback |

Note on the follow-up rounds: `pipeline.py:56` runs only when the research agent raises, or when `AGENTIC_RESEARCH_ENABLED` or `SOURCE_HUNTER_ENABLED` is off (`pipeline.py:528-560`). Both flags are on, so on a normal morning these 8 calls do not happen.

## Tool tiers (Google only, because grounding and URL context are provider tools)

Router rule: light, standard or advanced with a tool flag goes to `TOOL_LIGHT_STANDARD_MODEL`; plus or heavy goes to `TOOL_PLUS_HEAVY_MODEL`.

| Call | Model | Where | Shape |
|---|---|---|---|
| NPR front page, Gemini fallback (grounding) | `gemini-3-flash-preview` | `scrapers/topic_finder.py:624` | mixed: which items are today's events. Runs only if the browser read fails |
| AP front page, Gemini fallback (url_context) | `gemini-3-flash-preview` | `scrapers/topic_finder.py:629` | mixed: same, AP |
| Democracy Now front page, Gemini fallback (grounding) | `gemini-3-flash-preview` | `scrapers/topic_finder.py:634` | mixed: same, Democracy Now |
| follow-up question, rounds 5-6 (`plus`, no tools) | `gemini-3.1-pro-preview` | `pipeline.py:56` | write. Fallback path only |

ProPublica is now read from RSS (`rss_scraper`, `scrapers/topic_finder.py:641`) and Riverside by a direct HTML parser (`scrapers/riverside.py`). Neither calls an LLM.

## Calls that bypass the router

| Call | Model | Where | Shape | What it decides or writes |
|---|---|---|---|---|
| tier-2 web brief | `WEB_BRIEF_MODEL` = `openai/gpt-6-luna` + OpenRouter web plugin, no reasoning setting sent | `scrapers/topic_finder.py:497` via `search.py:185` | write | Sourced memo per shortlisted headline (up to 16). Tier 3 reads these |
| stable-fact search-verify | same | `review.py:518` via `search.py:185` | decide | WRONG or CORRECT for each stable-fact suspect. Advisory only |
| web search fallback | `SEARCH_OPENROUTER_MODEL` = `openai/gpt-6-sol`, reasoning low, web plugin | `search.py:140` | mixed | URL list, only when Google CSE fails or returns nothing |

These go straight to OpenRouter with `requests`, so they get no router retry and no backup model. A failed Tier-2 brief becomes an UNVERIFIED marker.

## Fallback

Any router call that exhausts its retries (or fails auth) falls back to `openai/gpt-6-sol`, reasoning low, on OpenRouter (`router.py:_call_fallback`). Grounded or url_context calls get OpenRouter's `web_search` or `web_fetch` tool attached.

## Dead or wasteful calls

Only what was checked in the code on 2026-09-23.

- **headline_maker** (`script/headlines.py:26`, light): one call per slot on disk. Its output is `successful_topics` (`pipeline.py:677`), passed to `intro_writer` as `topics` (`pipeline.py:683`). `intro_writer` never reads `topics` (`script/intro.py:6-42`); the intro uses the episode titles instead. Pure waste.
- **RAG refine** (`pipeline.py:408`, standard): disabled by `RAG_AUGMENT_ENABLED = False`. No cost today.
- **retype tagger fallback** (`scrapers/topic_finder.py:571`, standard): `tag_pool` catches every error from a model call inside its batch loop (`tagger.py:125-130`), so an LLM failure never reaches this fallback. Only a code error in `tagger.py` would. No cost today.
- **apply_event_test** (`scrapers/watchlist.py:341`): used only by `tests/test_watchlist.py`. Production goes through `feed_group_section`. No cost, just unused code.
- **stable-fact pass plus search-verify** (`review.py:506`, `:518`): runs on every script, but its flags are logged and never edited (`review.py:672`). This is by design (see the stable-fact memory note), but it is spend with no effect on what airs.
