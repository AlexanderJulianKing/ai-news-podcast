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
    from newscaster.llm import get_llm_response
    from newscaster.prompts import SLUG_GENERATION_PROMPT

    try:
        slug_response = get_llm_response(
            f"Headline: {headline}", system_prompt=SLUG_GENERATION_PROMPT, mode="light"
        )
        slug = slug_response.strip().lower().replace(" ", "_")
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        if not slug:
            slug = _generate_slug_fallback(headline)
    except Exception:
        slug = _generate_slug_fallback(headline)

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
    from newscaster.llm import get_llm_response

    headline = (headline or "").strip()
    context = (context or "").strip()
    summary_prompt = (
        "You maintain a log of stories already covered. Given the headline and context, "
        "write a neutral two-sentence summary capturing the core event and why it matters. "
        "Do not add commentary beyond the facts."
    )
    input_payload = f"Headline: {headline}\nContext:\n{context}"
    try:
        summary = get_llm_response(input_payload, system_prompt=summary_prompt, mode="light")
    except Exception:
        summary = context or headline
    summary = (summary or "").strip()
    if not summary:
        summary = headline
    return summary
