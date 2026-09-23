import os
import json
from datetime import datetime


def _log_dir():
    """Where logs go. NEWSCASTER_LOG_DIR overrides it; the test suite points it at a
    temporary folder so test calls never land in the production audit logs."""
    return os.environ.get("NEWSCASTER_LOG_DIR", "logs")


os.makedirs(_log_dir(), exist_ok=True)


def print_and_write(*args):
    current_date = datetime.now()
    os.makedirs(_log_dir(), exist_ok=True)
    file_name = os.path.join(_log_dir(), current_date.strftime("log_%y_%m_%d.txt"))
    current_time = current_date.strftime("%H:%M:%S")

    with open(file_name, 'a', encoding='utf-8') as file:
        for arg in args:
            log_entry = f"{current_time} - {arg}"
            file.write(log_entry + '\n')
            try:
                print(log_entry)
            except Exception as e:
                print(f'Failed to print log entry: {e}')


# Audit logs roll over by size so they can't fill the Pi's SD card. Each stream is
# bounded at roughly (LOG_BACKUP_COUNT + 1) * LOG_MAX_BYTES (~60 MB at these values).
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per active file before rotating
LOG_BACKUP_COUNT = 5               # rolled files kept (.1 .. .5); the oldest is dropped


def _rotate_log(path):
    """Roll ``path`` -> ``path.1`` -> ... -> ``path.N``, dropping the oldest."""
    for i in range(LOG_BACKUP_COUNT - 1, 0, -1):
        src = f"{path}.{i}"
        if os.path.exists(src):
            os.replace(src, f"{path}.{i + 1}")   # .N-1 -> .N overwrites the oldest
    if os.path.exists(path):
        os.replace(path, f"{path}.1")


def write_jsonl_log(prefix, payload):
    os.makedirs(_log_dir(), exist_ok=True)
    file_name = os.path.join(_log_dir(), f"{prefix}.jsonl")
    record = dict(payload)
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    line = json.dumps(record, ensure_ascii=False, default=str) + '\n'
    if os.path.exists(file_name) and os.path.getsize(file_name) + len(line.encode("utf-8")) > LOG_MAX_BYTES:
        _rotate_log(file_name)
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(line)
