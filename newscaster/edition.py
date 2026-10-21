"""Which show this run makes: the morning edition or the weekday evening edition.

main2.bash sets NEWSCASTER_EDITION for main.py, moviemaker.py and uploader2.py.
Every file, flag and ledger entry is keyed by the episode key: "2026_10_09" for the
morning (unchanged since 2024, so all history still reads) and "2026_10_09_pm" for
the evening. Keys sort in air order, and key_date() reads the date from either.
"""
import os
from datetime import date, datetime

MORNING, EVENING = "morning", "evening"
EVENING_SUFFIX = "_pm"


def current():
    value = (os.environ.get("NEWSCASTER_EDITION") or MORNING).strip().lower()
    if value not in (MORNING, EVENING):
        raise ValueError(f"NEWSCASTER_EDITION must be 'morning' or 'evening', not {value!r}")
    return value


def is_evening(edition=None):
    return (edition or current()) == EVENING


def episode_key(day=None, edition=None):
    day = day or date.today()
    return day.strftime("%Y_%m_%d") + (EVENING_SUFFIX if is_evening(edition) else "")


def key_date(key):
    """The calendar date of an episode key (either edition), or None."""
    try:
        return datetime.strptime(str(key)[:10], "%Y_%m_%d").date()
    except (TypeError, ValueError):
        return None


def key_edition(key):
    return EVENING if str(key).endswith(EVENING_SUFFIX) else MORNING


def belongs_to(filename, key):
    """True if a file named '<key>_...' belongs to this episode and not the other edition."""
    name = os.path.basename(filename)
    if not name.startswith(key + "_"):
        return False
    return key_edition(key) == EVENING or not name.startswith(key + EVENING_SUFFIX + "_")


def reference_phrase(prior_key, current_key):
    """The exact words for an earlier episode's story, as a host would say them on air:
    "this morning's story", "yesterday's story", "last night's story", "Monday's story",
    "Monday evening's story", or "the story we covered on October 7". None if unreadable.

    2026-10-21: given only "yesterday" or "2 days ago", the roundup writer aired "this
    morning's story" for earlier coverage and "Tuesday's story" for Monday's.
    """
    then, now = key_date(prior_key), key_date(current_key)
    if then is None or now is None:
        return None
    days = (now - then).days
    evening = key_edition(prior_key) == EVENING
    if days == 0:
        return "this morning's story"
    if days == 1:
        return "last night's story" if evening else "yesterday's story"
    if days < 7:
        return f"{then.strftime('%A')}{' evening' if evening else ''}'s story"
    return f"the story we covered on {then.strftime('%B')} {then.day}"


def when_spoken(prior_key, current_key):
    """How a script refers back to an earlier episode: 'this morning', 'yesterday evening',
    'yesterday', or 'N days ago'. None if the dates can't be read."""
    then, now = key_date(prior_key), key_date(current_key)
    if then is None or now is None:
        return None
    days = (now - then).days
    if days == 0:
        return "this morning"
    evening = " evening" if key_edition(prior_key) == EVENING else ""
    if days == 1:
        return "yesterday" + evening
    return f"{days} days ago" + (", in the evening edition" if evening else "")
