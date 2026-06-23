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

FALLBACK_MODEL = "openai/gpt-5.5"

# --- LLM mode routing ---
LIGHT_MODEL = "gemini-3.1-flash-lite"
STANDARD_MODEL = "google/gemma-4-31b-it"
ADVANCED_MODEL = "z-ai/glm-5.2"
ADVANCED_REASONING_EFFORT = "medium"
HEAVY_MODEL = "claude-opus-4-8"
ADVERSARY_MODEL = FALLBACK_MODEL
ADVERSARY_REASONING_EFFORT = "high"
TOOL_LIGHT_STANDARD_MODEL = "gemini-3-flash-preview"
TOOL_PLUS_HEAVY_MODEL = "gemini-3.1-pro-preview"

# --- Search provider routing ---
SEARCH_PROVIDER = "google_cse"
SEARCH_FALLBACK_PROVIDER = "openrouter_web"
SEARCH_FALLBACK_ON_EMPTY = True
SEARCH_OPENROUTER_MODEL = FALLBACK_MODEL
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
SOURCE_HUNTER_MAX_ITERATIONS = 3
SOURCE_HUNTER_CANDIDATE_LIMIT = 8
SOURCE_HUNTER_NEARBY_SOURCE_LIMIT = 5
SOURCE_HUNTER_NEARBY_SOURCE_DEPTH = 4
SOURCE_HUNTER_MAX_SOURCE_CHARS = 9000


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
