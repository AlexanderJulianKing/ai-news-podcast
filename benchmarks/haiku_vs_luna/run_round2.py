"""Round 2 (2026-10-10): more samples, after Alex flagged round 1's 4-6 per job as too few.

    python3 -m benchmarks.haiku_vs_luna.run_round2

Adds calls not used in round 1 (from the Pi's full Luna log, Sept 27 - Oct 10) to bring
each job to about 15, and 30 new research lookups sampled from the controller's real
questions of Oct 3-9. Writes replay_r2.jsonl and research_r2.jsonl next to round 1.
"""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from benchmarks.haiku_vs_luna import run_replay, run_research

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TARGET = 15


def replay_items():
    rows = json.load(open(os.path.join(OUT, "luna_calls_all.json"), encoding="utf-8"))
    done = [json.loads(l) for l in open(os.path.join(OUT, "replay.jsonl"), encoding="utf-8")]
    used = {r["id"] for r in done}
    have = {}
    for r in done:
        have[r["task"]] = have.get(r["task"], 0) + 1
    rng = random.Random(20261010)
    items = []
    for name, *_rest in run_replay.TASKS:
        pool = [r for r in rows if run_replay.task_of(r) == name and r.get("call_id", "")[:12] not in used]
        rng.shuffle(pool)
        items += [dict(r, task=name) for r in pool[:max(0, TARGET - have.get(name, 0))]]
    return items


def research_items(n=30):
    qs = json.load(open(os.path.join(OUT, "research_qs.json"), encoding="utf-8"))
    rng = random.Random(20261010)
    rng.shuffle(qs)
    out = []
    for q in qs[:n]:
        day = datetime.strptime(q["date"][:10], "%Y_%m_%d")
        out.append({"question": q["question"], "topic": q.get("topic"), "date": day.strftime("%B %-d, %Y")})
    return out


def _write(path, fn, items, workers, label):
    with open(path, "w", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, items), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{label} {i}/{len(items)}", flush=True)


if __name__ == "__main__":
    rep, res = replay_items(), research_items()
    print(len(rep), "replay items:", {t: sum(1 for i in rep if i["task"] == t) for t, *_ in run_replay.TASKS})
    print(len(res), "research lookups")
    with ThreadPoolExecutor(2) as ex:
        a = ex.submit(_write, os.path.join(OUT, "replay_r2.jsonl"), run_replay.run, rep, 8, "replay")
        b = ex.submit(_write, os.path.join(OUT, "research_r2.jsonl"), run_research.run, res, 4, "research")
        a.result(), b.result()
