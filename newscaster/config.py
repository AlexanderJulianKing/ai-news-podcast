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
XI_API_KEY = None
GOOGLE_SEARCH_API_KEY = None
OPENWEATHERMAP_API_KEY = None
GOOGLE_CSE_ID = None

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
_SECOND = 1000

FALLBACK_MODEL = "openai/gpt-5.5"

# --- RAG / embeddings tunables ---
EMBED_MODEL = "gemini-embedding-2"   # verified current; space incompatible with -001
EMBED_DIM = 1536                     # pinned; changing requires a full re-embed
RAG_TOP_K = 6                        # chunks retrieved per refine
RAG_MIN_SIM = 0.65                   # cosine floor; below -> inject nothing (tune empirically)
RAG_AUGMENT_ENABLED = False          # gates the retrieve-then-refine pass


def init():
    """Load API keys from keys.txt. Must be called before using any key constants."""
    global KEYS, GOOGLE_GENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    global XI_API_KEY, GOOGLE_SEARCH_API_KEY, OPENWEATHERMAP_API_KEY, GOOGLE_CSE_ID
    KEYS = load_keys()
    GOOGLE_GENAI_API_KEY = require_key(KEYS, "google_genai_api")
    ANTHROPIC_API_KEY = require_key(KEYS, "anthropic_api")
    OPENROUTER_API_KEY = require_key(KEYS, "openrouter_api")
    XI_API_KEY = require_key(KEYS, "XI_API_KEY")
    GOOGLE_SEARCH_API_KEY = require_key(KEYS, "google_search_api")
    OPENWEATHERMAP_API_KEY = require_key(KEYS, "openweathermap_api")
    GOOGLE_CSE_ID = require_key(KEYS, "google_cse_id")
