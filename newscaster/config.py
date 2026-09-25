import sys
import io
from typing import Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def load_keys(key_filename: str = "keys.txt") -> Dict[str, str]:
    """Load API keys from the given file into a dictionary."""
    key_path = (PROJECT_ROOT / key_filename).resolve()
    keys: Dict[str, str] = {}
    if not key_path.exists():
        raise FileNotFoundError(f"Key file not found at {key_path}")
    with key_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            keys[name.strip()] = value.strip()
    return keys


def require_key(keys: Dict[str, str], name: str) -> str:
    """Return a key value or raise a clear error if it's missing."""
    value = keys.get(name)
    if not value:
        raise RuntimeError(f"Missing '{name}' entry in keys.txt")
    return value


KEYS = None
GOOGLE_GENAI_API_KEY = None
ANTHROPIC_API_KEY = None
OPENROUTER_API_KEY = None
GOOGLE_SEARCH_API_KEY = None
OPENWEATHERMAP_API_KEY = None
GOOGLE_CSE_ID = None

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
_SECOND = 1000

# Backup when a tier fails its retries. GPT-6 Sol from 2026-09-23.
FALLBACK_MODEL = "openai/gpt-6-sol"
FALLBACK_REASONING_EFFORT = "low"

# --- LLM mode routing ---
LIGHT_MODEL = "gemini-3.1-flash-lite"
# GPT-6 Luna replaced Gemma 4 31B (standard) and GLM 5.2 (advanced) on 2026-09-23, after a blind
# pairwise test on 107 real production inputs (benchmarks/tier_swap/): better on research answers
# and the faithfulness check, about 13x cheaper than GLM, a little dearer than Gemma.
STANDARD_MODEL = "openai/gpt-6-luna"
STANDARD_REASONING_EFFORT = "low"
ADVANCED_MODEL = "openai/gpt-6-luna"
ADVANCED_REASONING_EFFORT = "medium"
HEAVY_MODEL = "claude-opus-5-5"
# The adversary vets Opus's script edits and challenges the research loop; it also sets the
# search fallback model below. GPT-6 Sol from 2026-09-23 (GPT-5.5 before).
ADVERSARY_MODEL = "openai/gpt-6-sol"
ADVERSARY_REASONING_EFFORT = "high"
TOOL_LIGHT_STANDARD_MODEL = "gemini-3-flash-preview"
TOOL_PLUS_HEAVY_MODEL = "gemini-3.1-pro-preview"

# --- Search provider routing ---
SEARCH_PROVIDER = "google_cse"
SEARCH_FALLBACK_PROVIDER = "openrouter_web"
SEARCH_FALLBACK_ON_EMPTY = True
SEARCH_OPENROUTER_MODEL = ADVERSARY_MODEL
# Web-search briefs (Tier 2 memos, background fact checks) moved to GPT-6 Luna on 2026-09-23 after a
# blind test on 48 real questions (benchmarks/tier_swap/run_web_brief.py): Tier 2 19-4, fact checks 11-7.
WEB_BRIEF_MODEL = "openai/gpt-6-luna"
SEARCH_OPENROUTER_ENGINE = "parallel"
SEARCH_OPENROUTER_MAX_RESULTS = 8

# --- Audit logging ---
LLM_AUDIT_LOG_ENABLED = True
LLM_AUDIT_LOG_PROMPTS = True
LLM_AUDIT_LOG_RESPONSES = False  # response bodies bloat logs on the Pi; prompts still captured
SEARCH_AUDIT_LOG_ENABLED = True
SOURCE_HUNTER_AUDIT_LOG_ENABLED = True

# --- RAG / embeddings tunables ---
EMBED_MODEL = "gemini-embedding-2"   # verified current; space incompatible with -001
EMBED_DIM = 1536                     # pinned; changing requires a full re-embed
RAG_TOP_K = 6                        # chunks retrieved per refine
RAG_MIN_SIM = 0.65                   # cosine floor; below -> inject nothing (tune empirically)
RAG_AUGMENT_ENABLED = False          # gates the retrieve-then-refine pass

# --- Agentic selected-story research ---
AGENTIC_RESEARCH_ENABLED = True
AGENTIC_RESEARCH_MAX_ITERATIONS = 5
AGENTIC_RESEARCH_MIN_ITERATIONS = 2
AGENTIC_RESEARCH_ADVERSARY_ENABLED = True
RAG_RESEARCH_MEMORY_ENABLED = True

# --- Fact-finder auto-edit (agentic editor: fix confirmed factual errors before TTS) ---
FACT_FINDER_AUTOEDIT_ENABLED = True
FACT_FINDER_AUTOEDIT_MAX_ROUNDS = 3

# --- Controlled source-hunter research ---
SOURCE_HUNTER_ENABLED = True
SOURCE_HUNTER_MAX_ITERATIONS = 3          # headline/fallback searches, on top of the question queries
SOURCE_HUNTER_QUESTION_QUERIES = True     # write short search queries from the question and search them first
SOURCE_HUNTER_MIN_QUESTION_SOURCES = 2    # stop searching question queries once this many of their pages validate
SOURCE_HUNTER_QUESTION_WINDOW_DAYS = 30   # follow-up questions search and accept pages up to this old (headline lookups: 1 and 3 days)
SOURCE_HUNTER_CANDIDATE_LIMIT = 8
SOURCE_HUNTER_NEARBY_SOURCE_LIMIT = 5
SOURCE_HUNTER_NEARBY_SOURCE_DEPTH = 4
SOURCE_HUNTER_MAX_SOURCE_CHARS = 9000

# --- Per-line loudness normalization (equalises the TTS voices before mixdown) ---
# Google's Chirp3-HD voices arrive at different levels: Chloe (Leda) measured
# ~2.5 LU below Grace (Aoede) in the same interview segment, which is audible
# when they alternate. Every synthesised line is pulled to LOUDNESS_TARGET_LUFS.
LOUDNESS_NORMALIZE_ENABLED = True
LOUDNESS_TARGET_LUFS = -23.0         # broadcast reference; the voices already sit near it
LOUDNESS_PEAK_CEILING_DB = -1.0      # a boost must never push peaks into the last dB
LOUDNESS_MAX_GAIN_DB = 12.0          # refuse to "rescue" a degenerate or near-silent render
LOUDNESS_MIN_GAIN_DB = 0.1           # below this the re-encode is not worth doing

# --- Story selection: eligibility and research robustness ---
# A Tier-2 morning where nearly every brief is UNVERIFIED is a research outage, not
# a dozen false stories (2026-07-22: 12/12 UNVERIFIED, the OpenAI/Hugging Face hack
# was judged "hypothetical"). Above this share, Tier 3 is told not to penalize it.
RESEARCH_DEGRADED_MIN_BRIEFS = 4
RESEARCH_DEGRADED_UNVERIFIED_FRACTION = 0.75
# Overlap coefficient a scraped pool line must reach against a shortlisted headline
# to be credited as that headline's source ('Reported by:' in the brief).
SOURCE_ATTRIBUTION_MIN_OVERLAP = 0.6
# A story that led becomes eligible to lead again ([DEVELOPMENT]) once this many
# days have passed since its last full segment; sooner needs a MAJOR ESCALATION.
MAIN_RECOVERY_DAYS = 2
# The repetition tagger re-emits the whole pool. If it returns fewer than this share
# of the lines, it is dropping stories, not deduplicating; retry, then merge back.
TAGGER_MIN_RETENTION = 0.7
# The tagger returns a verdict per numbered headline and code applies the tags (newscaster/tagger.py).
# False reverts to the old retype-the-pool tagger.
TAGGER_STRUCTURED = True
# 2026-09-23 test on six real mornings, each output audited alone by Opus 5.5 (serious errors):
# Luna batch 40 low 8, batch 10 low 4, medium 3, high 3; Sol batch 10 medium 1 (~$0.13/day).
TAGGER_BATCH_SIZE = 10
TAGGER_MODEL = "openai/gpt-6-sol"
TAGGER_REASONING_EFFORT = "medium"

# --- Headline ingestion ---
# Front pages are read with an event-first prompt (one sentence: who did what,
# when) and asked for up to this many items. The old "latest headlines" prompt let
# the model choose, and it chose about eight; that is where Navier-Stokes (09-08)
# and METR's Hugging Face findings (08-26) were lost before Tier 1 ever saw them.
SCRAPE_MAX_ITEMS = 20
# Front pages read by a real, visible Chromium on a virtual screen, then listed by the
# standard model (newscaster/scrapers/browser.py). Each falls back to the older Gemini
# scrape on any failure. Checked against Alex's own screenshots 2026-09-23.
BROWSER_SCRAPE_ENABLED = True
BROWSER_SCRAPE_SOURCES = ("npr", "ap", "dn")
BROWSER_BINARY = "chromium"
BROWSER_WAIT_SECONDS = 20
BROWSER_CONNECT_GRACE = 30   # extra seconds to keep trying the DevTools port after the wait
BROWSER_WINDOW = (1366, 3000)
BROWSER_MIN_WORDS = 300
BROWSER_TEXT_CHARS = 60000
BROWSER_SCREENSHOT_DIR = "screenshots"
BROWSER_SCREENSHOT_KEEP_DAYS = 14

# Specialist watch: AI labs and independent evaluators, read from RSS and passed
# through an event test (newscaster/scrapers/watchlist.py). Nominate-only: items
# join the pool as one more source with no special weight. Entries are (name, url)
# for RSS/Atom, or (name, url, kind); Anthropic publishes no feed, so its /news
# listing page is parsed directly (kind "anthropic-news").
WATCHLIST_ENABLED = True
WATCHLIST_LOOKBACK_HOURS = 72
# OpenAI posts many customer stories a day; at 8, the GPT-6 Sol and Luna launch
# (Sep 22, 2026) was pushed out of the window by next-day case studies.
WATCHLIST_MAX_ITEMS_PER_FEED = 25
WATCHLIST_FEEDS = [
    ('OpenAI', 'https://openai.com/news/rss.xml'),
    ('Anthropic', 'https://www.anthropic.com/news', 'anthropic-news'),
    ('Google DeepMind', 'https://deepmind.google/blog/rss.xml'),
    ('METR', 'https://metr.org/feed.xml'),
    ('AI Incident Database', 'https://incidentdatabase.ai/rss.xml'),
    ('Import AI', 'https://importai.substack.com/feed'),
]

# Beat feeds: business, science and technology, health, courts, world. Each entry is
# (group, [(outlet, url), ...]) with an optional third item: one extra instruction
# for that group's selection prompt. Front pages
# never carried Navier-Stokes (Quanta, Science, CNBC did) and the pool had no
# markets, health or courts source at all. Each beat is read from RSS with a 24h
# window, and one heavy-model call picks up to BEAT_MAX_ITEMS concrete events, so
# roughly 160 feed items a day become about 40 pool lines. Probed 2026-09-15:
# MarketWatch, Lawfare, Reuters, Science.org and Nature feeds were dead or empty.
BEATS_ENABLED = True
BEAT_LOOKBACK_HOURS = 24
BEAT_MAX_ITEMS = 8
BEAT_MAX_ITEMS_PER_FEED = 30
BEAT_FEEDS = [
    ('Business and markets', [
        ('CNBC', 'https://www.cnbc.com/id/100003114/device/rss/rss.html'),
        ('CNBC Business', 'https://www.cnbc.com/id/10001147/device/rss/rss.html'),
    ]),
    ('Science and technology', [
        ('Quanta Magazine', 'https://www.quantamagazine.org/feed/'),
        ('Ars Technica', 'https://feeds.arstechnica.com/arstechnica/index'),
    ]),
    ('Health', [
        ('STAT News', 'https://www.statnews.com/feed/'),
        ('KFF Health News', 'https://kffhealthnews.org/feed/'),
    ]),
    ('Courts', [
        ('SCOTUSblog', 'https://www.scotusblog.com/feed/'),
    ]),
    ('World', [
        ('BBC World', 'http://feeds.bbci.co.uk/news/world/rss.xml'),
        ('Al Jazeera', 'https://www.aljazeera.com/xml/rss/all.xml'),
        ('The Guardian World', 'https://www.theguardian.com/world/rss'),
    ]),
    # Added 2026-09-23 after probing ~35 feeds from the Pi. Local: Alex moved to
    # Temecula and commutes to La Jolla; the Union-Tribune feed is 403 and the
    # Press-Enterprise feed is gone. AI press: Meta, xAI, Mistral and DeepSeek publish
    # no feeds, so the lab watch cannot see them (it missed Meta Connect 2026).
    # Official records: the primary documents behind stories AP reports second-hand.
    ('San Diego and Temecula', [
        ('KPBS', 'https://www.kpbs.org/index.rss'),
        ('Times of San Diego', 'https://timesofsandiego.com/feed/'),
        ('Voice of San Diego', 'https://voiceofsandiego.org/feed/'),
        ('Valley News', 'https://myvalleynews.com/feed/'),
        ('City of Temecula', 'https://temeculaca.gov/RSSFeed.aspx?ModID=1&CID=All-newsflash.xml'),
    ], "Choose only local events: things that happened in San Diego County, Riverside County or Temecula, "
       "or California state actions aimed at them. Skip national and world stories even when a local outlet "
       "carries them; other sources cover those."),
    ('AI and tech press', [
        ('The Verge AI', 'https://www.theverge.com/rss/ai-artificial-intelligence/index.xml'),
        ('Simon Willison', 'https://simonwillison.net/atom/everything/'),
        ('404 Media', 'https://www.404media.co/rss/'),
        ('Hacker News (300+ points)', 'https://hnrss.org/frontpage?points=300'),
    ]),
    ('Official records', [
        ('Federal Register (presidential documents)',
         'https://www.federalregister.gov/api/v1/documents.rss?conditions%5Btype%5D%5B%5D=PRESDOCU'),
        ('FDA press releases', 'https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml'),
    ]),
]


def init():
    """Load API keys from keys.txt. Must be called before using any key constants."""
    global KEYS, GOOGLE_GENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    global GOOGLE_SEARCH_API_KEY, OPENWEATHERMAP_API_KEY, GOOGLE_CSE_ID
    KEYS = load_keys()
    GOOGLE_GENAI_API_KEY = require_key(KEYS, "google_genai_api")
    ANTHROPIC_API_KEY = require_key(KEYS, "anthropic_api")
    OPENROUTER_API_KEY = require_key(KEYS, "openrouter_api")
    GOOGLE_SEARCH_API_KEY = require_key(KEYS, "google_search_api")
    OPENWEATHERMAP_API_KEY = require_key(KEYS, "openweathermap_api")
    GOOGLE_CSE_ID = require_key(KEYS, "google_cse_id")
