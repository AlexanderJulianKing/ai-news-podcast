"""What one run spent, written as one line of logs/run_costs.jsonl at the end of main.py (2026-11-01).

The LLM audit log rotates about once a run (its prompts make it large), so the 2026-10-29 week review
could measure only that morning's cost. Every paid call records here as it happens: LLM calls through
the router, the research tool's lookups, Anthropic web searches, and both TTS voices. Costs are the
estimates the code already computes (price tables, or OpenRouter's reported cost); calls with no known
price are counted with a cost of 0.
"""
import threading

from newscaster.logging import print_and_write, write_jsonl_log

_LOCK = threading.Lock()
_ITEMS = {}


def reset():
    with _LOCK:
        _ITEMS.clear()


def record(category, name, cost=None, units=None, unit=None):
    """Add one call. cost in US dollars (None if unknown); units, e.g. TTS characters, are summed per item."""
    with _LOCK:
        item = _ITEMS.setdefault(f"{category}: {name}", {"calls": 0, "usd": 0.0, "units": 0, "unit": unit})
        item["calls"] += 1
        item["usd"] += float(cost or 0.0)
        item["units"] += int(units or 0)


def write_run_summary(key):
    with _LOCK:
        items = {k: dict(v, usd=round(v["usd"], 4)) for k, v in _ITEMS.items()}
    total = round(sum(v["usd"] for v in items.values()), 2)
    by_category = {}
    for k, v in items.items():
        cat = k.split(":", 1)[0]
        by_category[cat] = round(by_category.get(cat, 0.0) + v["usd"], 2)
    write_jsonl_log("run_costs", {"key": key, "total_usd": total, "by_category": by_category, "by_item": items})
    print_and_write(f"RUN COST {key}: ${total:.2f} " + ", ".join(f"{c} ${u:.2f}" for c, u in sorted(by_category.items())))
    return total
