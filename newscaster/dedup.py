from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from newscaster.logging import print_and_write

LEDGER_PATH = Path(__file__).resolve().parent.parent / "stories_chosen" / "story_ledger.json"
ARC_EXPIRY_DAYS = 45

_EMPTY_LEDGER = {"schema_version": 1, "last_updated": None, "arcs": {}}


def load_ledger() -> dict:
    """Read ledger JSON, return empty skeleton if missing or corrupt."""
    if not LEDGER_PATH.exists():
        return json.loads(json.dumps(_EMPTY_LEDGER))
    try:
        with LEDGER_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "arcs" not in data:
            print_and_write("Ledger file has unexpected structure; starting fresh")
            return json.loads(json.dumps(_EMPTY_LEDGER))
        return data
    except (IOError, json.JSONDecodeError) as e:
        print_and_write(f"Failed to read ledger ({e}); starting fresh")
        return json.loads(json.dumps(_EMPTY_LEDGER))


def save_ledger(ledger: dict):
    """Write ledger atomically (temp file + os.replace), set last_updated to today."""
    ledger["last_updated"] = date.today().strftime("%Y_%m_%d")
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(LEDGER_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(LEDGER_PATH))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def prune_ledger(ledger: dict) -> dict:
    """Remove arcs with last_covered older than ARC_EXPIRY_DAYS."""
    today = date.today()
    to_remove = []
    for slug, arc in ledger.get("arcs", {}).items():
        last_covered = arc.get("last_covered")
        if not last_covered:
            to_remove.append(slug)
            continue
        try:
            last_date = datetime.strptime(last_covered, "%Y_%m_%d").date()
            if (today - last_date).days > ARC_EXPIRY_DAYS:
                to_remove.append(slug)
        except ValueError:
            to_remove.append(slug)
    for slug in to_remove:
        del ledger["arcs"][slug]
        print_and_write(f"Pruned expired arc: {slug}")
    return ledger


def format_arcs_for_dedup(ledger: dict, window_days: int = 14) -> str:
    """Format active arcs for the repetition remover prompt."""
    today = date.today()
    lines = []
    for slug, arc in ledger.get("arcs", {}).items():
        last_covered = arc.get("last_covered")
        if not last_covered:
            continue
        try:
            last_date = datetime.strptime(last_covered, "%Y_%m_%d").date()
            if (today - last_date).days > window_days:
                continue
        except ValueError:
            continue
        audience_state = arc.get("audience_state", "")
        topic = arc.get("topic", "")
        line = f"[ARC: {slug}] Topic: {topic} | Last covered: {last_covered} | Audience knows: {audience_state}"
        lines.append(line)
    return "\n".join(lines)


def _generate_slug_fallback(headline: str) -> str:
    """Generate a deterministic slug from the first 4 words of a headline."""
    words = re.sub(r"[^a-z0-9\s]", "", headline.lower()).split()
    return "_".join(words[:4]) if words else "unknown_story"


def create_arc(ledger, headline, coverage, coverage_slot, date_str, topic_summary) -> str:
    """Create a new arc entry. Returns the slug."""
    from newscaster.llm import call_with_default
    from newscaster.prompts import SLUG_GENERATION_PROMPT

    fallback_slug = _generate_slug_fallback(headline)
    slug_response = call_with_default(
        fallback_slug,
        f"Headline: {headline}",
        system_prompt=SLUG_GENERATION_PROMPT, mode="light",
        _log_label=f'slug-generation[{headline[:40]}]',
    )
    slug = slug_response.strip().lower().replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    if not slug:
        slug = fallback_slug

    # Handle slug collisions
    base_slug = slug
    counter = 2
    while slug in ledger["arcs"]:
        slug = f"{base_slug}_{counter}"
        counter += 1

    ledger["arcs"][slug] = {
        "slug": slug,
        "topic": topic_summary or headline,
        "first_covered": date_str,
        "last_covered": date_str,
        "episodes": [
            {
                "date": date_str,
                "coverage": coverage,
                "coverage_slot": coverage_slot,
                "headline": headline,
                "audience_learned": [],
            }
        ],
        "audience_state": "",
    }
    return slug


def update_arc(ledger, slug, headline, coverage, coverage_slot, date_str):
    """Upsert episode entry, keyed by date + coverage_slot to prevent duplicates."""
    arc = ledger["arcs"].get(slug)
    if not arc:
        print_and_write(f"update_arc: slug '{slug}' not found in ledger")
        return

    # Check for existing episode with same date + coverage_slot (idempotent)
    for ep in arc["episodes"]:
        if ep["date"] == date_str and ep["coverage_slot"] == coverage_slot:
            ep["headline"] = headline
            ep["coverage"] = coverage
            arc["last_covered"] = date_str
            return

    arc["episodes"].append({
        "date": date_str,
        "coverage": coverage,
        "coverage_slot": coverage_slot,
        "headline": headline,
        "audience_learned": [],
    })
    arc["last_covered"] = date_str


def update_audience_learned(ledger, slug, date_str, coverage_slot, learned_bullets, new_audience_state):
    """Find episode by date + coverage_slot, set audience_learned, update audience_state."""
    arc = ledger["arcs"].get(slug)
    if not arc:
        return
    for ep in arc["episodes"]:
        if ep["date"] == date_str and ep["coverage_slot"] == coverage_slot:
            ep["audience_learned"] = learned_bullets
            break
    arc["audience_state"] = new_audience_state


def find_matching_arc(headline: str):
    """Parse [UPDATE: slug] or [MAJOR ESCALATION: slug] from headline.

    Returns (tag_type, slug) or None.
    """
    match = re.match(r"\[(UPDATE|MAJOR ESCALATION):\s*([^\]]+)\]", headline)
    if match:
        return (match.group(1), match.group(2).strip())
    return None


def strip_arc_tags(headline: str) -> str:
    """Remove [UPDATE: slug] / [MAJOR ESCALATION: slug] prefixes, return clean headline."""
    cleaned = re.sub(r"\[(UPDATE|MAJOR ESCALATION):\s*[^\]]+\]\s*", "", headline)
    # Also strip old-style tags without slugs
    for tag in ("[UPDATE]", "[MAJOR ESCALATION]"):
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


# --- Arc-identity recovery -------------------------------------------------
# The dedup tagger reliably tags recurring headlines with [UPDATE: slug] /
# [MAJOR ESCALATION: slug], but the downstream selection prompts strip that
# prefix, so find_matching_arc() (which only parses the prefix) never recovers
# the slug. These helpers rebuild the {clean headline -> (tag, slug)} map from
# the tagger's own output and recover it for a chosen headline that may have
# been de-tagged and lightly rephrased by the selection LLMs.

# Words too common to carry topic identity; dropped before fuzzy comparison.
_HEADLINE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "and", "for", "as", "is", "are",
    "at", "by", "with", "from", "its", "it", "after", "amid", "over", "into",
    "his", "her", "their", "new",
})

# Fuzzy fallback is deliberately strict: a wrong match injects the wrong
# audience_state (worse than no match), so we only accept near-identical or
# clear subset/superset headlines.
_FUZZY_OVERLAP_THRESHOLD = 0.8
_FUZZY_MIN_SHARED_TOKENS = 3


def _normalize_headline_words(headline: str) -> list:
    """Lowercase, drop the arc tag and punctuation, return the word list."""
    cleaned = strip_arc_tags(headline or "")
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned.lower())
    return cleaned.split()


def _headline_key(headline: str) -> str:
    """Whitespace-normalized, punctuation-free, lowercase key for exact matching."""
    return " ".join(_normalize_headline_words(headline))


def _content_tokens(headline: str) -> set:
    """Distinctive content tokens (stopwords removed, trailing plural 's' folded)."""
    return {
        w.rstrip("s")
        for w in _normalize_headline_words(headline)
        if w not in _HEADLINE_STOPWORDS and len(w) > 2
    }


_ARC_TAG_RE = re.compile(r"\[(UPDATE|MAJOR ESCALATION):\s*([^\]]+)\]")


def build_headline_arc_map(tagged_text: str) -> dict:
    """Parse the dedup tagger's output into {clean-headline-key: (tag_type, slug)}.

    The tagger (Gemma) decorates its output with markdown — bullets ('* **'),
    headings ('### **2.'), bold, numbering — so a tag is almost never at the
    literal start of a line. We therefore find every [UPDATE: slug] /
    [MAJOR ESCALATION: slug] tag wherever it sits, and take its headline as the
    text running from the end of that tag up to the NEXT tag or the next newline,
    whichever comes first (so neighbouring untagged headlines aren't swallowed and
    several tagged items crammed onto one line each get their own entry).

    This only reads the tagger's own verdicts; it does not decide sameness — that
    judgment stays with the tagger.
    """
    text = tagged_text or ""
    matches = list(_ARC_TAG_RE.finditer(text))
    mapping = {}
    for idx, m in enumerate(matches):
        tag_type, slug = m.group(1), m.group(2).strip()
        if not slug:
            continue
        start = m.end()
        next_tag = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        nl = text.find("\n", start)
        next_nl = nl if nl != -1 else len(text)
        headline_text = text[start:min(next_tag, next_nl)]
        key = _headline_key(headline_text)
        if key:
            mapping[key] = (tag_type, slug)
    return mapping


def recover_arc_for_headline(headline: str, headline_arc_map: dict, threshold: float = _FUZZY_OVERLAP_THRESHOLD):
    """Recover (tag_type, slug) for a chosen, possibly de-tagged headline.

    Exact normalized match first; then a strict overlap-coefficient fuzzy match
    against the (small) set of tagged headlines. Returns None when nothing clears
    the bar — the caller then creates a fresh arc, exactly as before this fix.
    """
    if not headline_arc_map:
        return None
    key = _headline_key(headline)
    if not key:
        return None
    if key in headline_arc_map:
        return headline_arc_map[key]

    chosen_tokens = _content_tokens(headline)
    if len(chosen_tokens) < _FUZZY_MIN_SHARED_TOKENS:
        return None
    best_info = None
    best_overlap = 0.0
    for cand_key, info in headline_arc_map.items():
        cand_tokens = _content_tokens(cand_key)
        if not cand_tokens:
            continue
        shared = chosen_tokens & cand_tokens
        if len(shared) < _FUZZY_MIN_SHARED_TOKENS:
            continue
        # Overlap coefficient: rewards subset/superset rephrasings without
        # penalizing length differences the way Jaccard would.
        overlap = len(shared) / min(len(chosen_tokens), len(cand_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_info = info
    return best_info if best_overlap >= threshold else None


def _build_arc_candidates(headline_arc_map: dict, ledger: dict) -> dict:
    """{slug: (tag_type, topic)} for the day's tagged arcs, deduped by slug.

    Topic comes from the ledger when available (more descriptive for the LLM),
    falling back to the slug itself.
    """
    candidates = {}
    for tag_type, slug in (headline_arc_map or {}).values():
        if slug in candidates:
            continue
        arc = (ledger or {}).get("arcs", {}).get(slug) or {}
        topic = arc.get("topic") or slug.replace("_", " ")
        candidates[slug] = (tag_type, topic)
    return candidates


def _parse_arc_slug(response: str, candidate_slugs) -> str | None:
    """Return the single candidate slug named in the response, else None.

    Zero or multiple matches both yield None — the safe (no-link) outcome.
    """
    text = (response or "").lower()
    hits = [s for s in candidate_slugs if re.search(r"\b" + re.escape(s.lower()) + r"\b", text)]
    return hits[0] if len(hits) == 1 else None


def llm_recover_arc(headline: str, headline_arc_map: dict, ledger: dict):
    """Scoped LLM fallback for heavy paraphrases the string matcher misses.

    Asks one cheap light-model call to match the headline to exactly one of the
    day's tracked arcs, or NONE. Gated by a token-overlap pre-filter so clearly
    unrelated (brand-new) headlines never pay for the call. Returns (tag, slug)
    only on a confident, in-candidate match; None otherwise.
    """
    candidates = _build_arc_candidates(headline_arc_map, ledger)
    if not candidates:
        return None

    # Cheap gate: only consult the LLM when the headline shares a distinctive
    # token with some candidate (its topic or slug words). A brand-new story
    # shares nothing and short-circuits to None without a wasted NONE call.
    chosen_tokens = _content_tokens(headline)
    if not chosen_tokens:
        return None

    def _candidate_tokens(slug, topic):
        return _content_tokens(topic) | _content_tokens(slug.replace("_", " "))

    if not any(chosen_tokens & _candidate_tokens(slug, topic) for slug, (_t, topic) in candidates.items()):
        return None

    from newscaster.llm import call_with_default  # lazy: avoids import cycle
    from newscaster.prompts import ARC_MATCH_PROMPT

    listing = "\n".join(f"- {slug}: {topic}" for slug, (_t, topic) in candidates.items())
    user_prompt = f"Tracked arcs:\n{listing}\n\nHeadline: {strip_arc_tags(headline).strip()}\n\nAnswer:"
    response = call_with_default(
        "NONE", user_prompt, system_prompt=ARC_MATCH_PROMPT, mode="light",
        _log_label=f"arc-match[{strip_arc_tags(headline).strip()[:40]}]",
    )
    slug = _parse_arc_slug(response, list(candidates.keys()))
    if slug:
        return (candidates[slug][0], slug)
    return None


def resolve_arc_identity(headline: str, headline_arc_map: dict, ledger: dict = None, *, use_llm: bool = True):
    """Full arc-identity resolution: deterministic string match first, then the
    scoped LLM fallback for paraphrases, then any tag that survived on the
    headline. Returns (tag_type, slug) or None."""
    info = recover_arc_for_headline(headline, headline_arc_map)
    if info:
        return info
    if use_llm:
        info = llm_recover_arc(headline, headline_arc_map, ledger)
        if info:
            return info
    return find_matching_arc(headline)


def load_recent_story_descriptions(window_days: int = 7):
    """Return aggregated summaries of recent stories for deduping."""
    history_entries = []
    base_dir = Path(__file__).resolve().parent.parent / "stories_chosen"
    today_date = date.today()
    for offset in range(1, window_days + 1):
        day = today_date - timedelta(days=offset)
        date_key = day.strftime("%Y_%m_%d")
        summary_path = base_dir / f"{date_key}_story_summaries.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    day_payload = json.load(handle)
                if not isinstance(day_payload, dict):
                    print_and_write('Skipping unexpected summary payload structure', str(summary_path))
                    continue
                stories = day_payload.get("stories", [])
                if not isinstance(stories, list):
                    print_and_write('Skipping malformed stories list in summary payload', str(summary_path))
                    continue
                for story in stories:
                    if not isinstance(story, dict):
                        continue
                    headline = (story.get("headline") or "").strip()
                    summary = (story.get("summary") or "").strip()
                    if summary and headline:
                        history_entries.append(f"{headline}: {summary}")
                    elif summary:
                        history_entries.append(summary)
                    elif headline:
                        history_entries.append(headline)
                continue
            except (IOError, json.JSONDecodeError):
                pass
        fallback_path = base_dir / f"{date_key}_stories_chosen.txt"
        if fallback_path.exists():
            try:
                with fallback_path.open("r", encoding="utf-8") as handle:
                    content = handle.read().strip()
                if content:
                    history_entries.append(content)
            except IOError:
                pass
    if not history_entries:
        return "", False
    return "\n".join(history_entries), True


def summarize_story_for_archive(headline: str, context: str) -> str:
    """Produce a short summary used to detect future duplicates."""
    # Lazy import to avoid circular dependency at module load time
    from newscaster.llm import call_with_default

    headline = (headline or "").strip()
    context = (context or "").strip()
    summary_prompt = (
        "You maintain a log of stories already covered. Given the headline and context, "
        "write a neutral two-sentence summary capturing the core event and why it matters. "
        "Do not add commentary beyond the facts."
    )
    input_payload = f"Headline: {headline}\nContext:\n{context}"
    summary = call_with_default(
        context or headline,
        input_payload, system_prompt=summary_prompt, mode="light",
        _log_label=f'archive-summary[{headline[:40]}]',
    )
    summary = (summary or "").strip()
    if not summary:
        summary = headline
    return summary
