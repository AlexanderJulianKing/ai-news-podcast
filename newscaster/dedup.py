import json
from datetime import date, timedelta
from pathlib import Path

from newscaster.logging import print_and_write


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
