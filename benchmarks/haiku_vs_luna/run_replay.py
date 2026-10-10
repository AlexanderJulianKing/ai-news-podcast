"""Claude Haiku 5.5 vs GPT-6 Luna on Luna's routine jobs (2026-10-10).

    python3 -m benchmarks.haiku_vs_luna.run_replay outputs/luna_calls_1006_1010.json outputs/replay.jsonl

Inputs are real Luna calls from the Pi's audit log (Oct 6-10). The log keeps prompts but
not answers, so each input is replayed through both models now, at matching effort:
standard -> low, advanced -> medium, topic groups -> high. Judging is blind and done
separately (make_packets.py), so this script only generates.
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("NEWSCASTER_LOG_DIR", "/tmp/nc_haiku_bench")
import newscaster.config as config  # noqa: E402

config.init()
config.LLM_AUDIT_LOG_ENABLED = False
from benchmarks.tier_swap.run_tier_swap import call, degenerate  # noqa: E402

HAIKU = "claude-haiku-5-5"
LUNA = "openai/gpt-6-luna"

# (task, how to recognise it, how many to sample, what a better answer does)
TASKS = [
    ("article_summary", lambda s, u: s.startswith("You are an intelligent assistant.") and u.startswith("Summarize this news article"), 6,
     "Better: accurate to the article, names the outlet, keeps the specific facts (names, numbers, dates), adds nothing the article does not say."),
    ("front_page", lambda s, u: s.startswith("You are an intelligent assistant.") and u.startswith("List every distinct news item"), 5,
     "Better: lists the real distinct news items on the page, each a specific event, none invented, none duplicated, following the format asked for."),
    ("synthesis", lambda s, u: s.startswith("Given the topic of"), 5,
     "Better: keeps as much on-topic information from the summaries as possible, drops off-topic material, invents nothing, stays faithful to what each source says."),
    ("faithfulness", lambda s, u: s.startswith("You are a fact-checking editor"), 6,
     "Better: flags the script's real discrepancies with the source material, does not flag statements the sources support, follows the required format."),
    ("side_picks", lambda s, u: s.startswith("Given the following research briefs"), 5,
     "Better: picks the stories a thoughtful editor would choose under the instructions, with no duplicates and no unverified or stale picks, in the required format."),
    ("roundup_script", lambda s, u: s.startswith("Please give a text of the following information as if you were a"), 5,
     "Better: reads well aloud, covers every story it was given accurately, adds nothing beyond the input, follows the instructions."),
    ("memory_note", lambda s, u: s.startswith("You are preparing a dated memory note"), 4,
     "Better: correctly separates what past coverage established from what is new or stale, with dates, and invents nothing."),
    ("headline", lambda s, u: s.startswith("Shorten the following story to a headline"), 4,
     "Better: a short, accurate, specific headline in the requested style."),
    ("stable_fact", lambda s, u: s.startswith("You verify ONLY well-established background facts"), 4,
     "Better: flags only real errors in well-established background facts, does not flag recent news it cannot know, follows the format."),
    ("topic_groups", lambda s, u: s.startswith("You are an intelligent assistant.") and "Choose up to 8" in u[:400], 5,
     "Better: keeps the items that report a specific new event under the instructions and leaves out the rest, without inventing items."),
]


def task_of(row):
    s, u = row.get("system_prompt") or "", row.get("user_prompt") or ""
    for name, match, _n, _r in TASKS:
        if match(s, u):
            return name
    return None


def effort_of(row):
    name = row.get("name") or ""
    if "topic groups" in name:
        return "high"
    if "advanced" in name:
        return "medium"
    return "low"


def sample(rows, seed=10):
    rng = random.Random(seed)
    picked = []
    for name, _m, n, _r in TASKS:
        pool = [r for r in rows if task_of(r) == name]
        rng.shuffle(pool)
        picked += [dict(r, task=name) for r in pool[:n]]
    return picked


def run_one(row, model):
    effort = effort_of(row)
    if model == HAIKU:
        spec = {"provider": "anthropic", "model": HAIKU, "effort": effort}
    else:
        spec = {"provider": "openrouter", "model": LUNA, "name": "luna", "reasoning": effort}
    try:
        text, secs, usage = call(spec, row["user_prompt"], row["system_prompt"])
    except Exception as exc:
        return {"text": "", "error": repr(exc)[:300], "seconds": None, "cost": None, "output_tokens": None}
    cost = usage.get("cost") or usage.get("estimated_cost_usd")
    out_tok = usage.get("output_tokens") or usage.get("completion_tokens")
    return {"text": text or "", "error": None, "seconds": secs, "cost": cost, "output_tokens": out_tok,
            "problem": degenerate(text or "")}


def run(row):
    out = {"id": row.get("call_id", "")[:12], "task": row["task"], "effort": effort_of(row),
           "system": row["system_prompt"], "user": row["user_prompt"]}
    out["luna"] = run_one(row, LUNA)
    out["haiku"] = run_one(row, HAIKU)
    return out


if __name__ == "__main__":
    rows = json.load(open(sys.argv[1], encoding="utf-8"))
    items = sample(rows)
    print(f"{len(items)} items:", {t: sum(1 for i in items if i['task'] == t) for t, *_ in TASKS})
    with open(sys.argv[2], "w", encoding="utf-8") as f, ThreadPoolExecutor(8) as ex:
        for r in ex.map(run, items):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            L, H = r["luna"], r["haiku"]
            print(f"{r['task']:16} luna {L['seconds']}s ${L['cost'] or 0:.4f} {L['error'] or L['problem'] or 'ok'} | "
                  f"haiku {H['seconds']}s ${H['cost'] or 0:.4f} {H['error'] or H['problem'] or 'ok'}", flush=True)
