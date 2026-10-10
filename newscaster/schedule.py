"""When main2.bash should start a show.

The morning edition starts in the 4 a.m. hour every day. The evening edition starts
at 3:45 p.m. on weekdays (Alex leaves La Jolla about 5; a run takes about an hour)
and is skipped once it is too late to be ready for the drive.

`python3 -m newscaster.schedule` prints the edition due now, or nothing. A show
counts as done once main2.bash writes its started flag, so a failed run is not
restarted every minute (the old loop's `last_ran` did the same for the morning).
"""
import os
import sys
from datetime import datetime, time

from newscaster import edition

MORNING_HOUR = 4
EVENING_START = time(15, 45)
EVENING_LAST_START = time(17, 0)
FLAG_DIR = "logs"


def started_flag(key):
    return os.path.join(FLAG_DIR, f"{key}_STARTED.flag")


def due_edition(now, done):
    """'morning', 'evening' or None. `done` holds the episode keys already started."""
    today = now.date()
    if now.hour == MORNING_HOUR and edition.episode_key(today, edition.MORNING) not in done:
        return edition.MORNING
    if (now.weekday() < 5 and EVENING_START <= now.time() < EVENING_LAST_START
            and edition.episode_key(today, edition.EVENING) not in done):
        return edition.EVENING
    return None


def main():
    now = datetime.now()
    keys = [edition.episode_key(now.date(), e) for e in (edition.MORNING, edition.EVENING)]
    done = {k for k in keys if os.path.exists(started_flag(k))}
    due = due_edition(now, done)
    if due:
        print(due)
    return 0


if __name__ == "__main__":
    sys.exit(main())
