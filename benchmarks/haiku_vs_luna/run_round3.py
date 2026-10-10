"""Round 3 (2026-10-10): Haiku research with the quote-discipline rule and prompt caching,
against production Luna, on 40 research questions from Oct 3-9 not used in rounds 1-2.

    python3 -m benchmarks.haiku_vs_luna.run_round3
Fresh questions keep the rule (written after reading round 1-2 errors) from being tested
on the cases that inspired it.
"""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from benchmarks.haiku_vs_luna import run_research as rr
from newscaster import research_tools as rt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def fresh_questions(n=40):
    qs = json.load(open(os.path.join(OUT, "research_qs.json"), encoding="utf-8"))
    used = {json.loads(l)["question"] for l in open(os.path.join(OUT, "research_r2.jsonl"), encoding="utf-8")}
    pool = [q for q in qs if q["question"] not in used]
    rng = random.Random(3)
    rng.shuffle(pool)
    out = []
    for q in pool[:n]:
        day = datetime.strptime(q["date"][:10], "%Y_%m_%d")
        out.append({"question": q["question"], "topic": q.get("topic"), "date": day.strftime("%B %-d, %Y")})
    return out


def run(row):
    out = {"question": row["question"], "topic": row.get("topic"), "date": row["date"]}
    try:
        res = rt.research_with_tools(row["question"], row["date"])
        md = res["metadata"]
        out["luna"] = {"text": res["answer"], "kept": md["facts_kept"], "dropped": md["facts_dropped"],
                       "steps": len(md["steps"]), "cost": md["cost"], "seconds": md["seconds"]}
    except Exception as exc:
        out["luna"] = {"text": f"ERROR {exc!r}"[:300], "kept": 0, "dropped": 0, "steps": 0, "cost": 0, "seconds": 0}
    try:
        final, pages, steps, cost, secs = rr.haiku_research(row["question"], row["date"],
                                                             extra_rules=rr.QUOTE_DISCIPLINE, cache=True)
        text, kept, dropped = rr.as_answer(final, pages)
        out["haiku"] = {"text": text, "kept": kept, "dropped": dropped, "steps": len(steps),
                        "cost": round(cost, 4), "seconds": round(secs)}
    except Exception as exc:
        out["haiku"] = {"text": f"ERROR {exc!r}"[:300], "kept": 0, "dropped": 0, "steps": 0, "cost": 0, "seconds": 0}
    return out


if __name__ == "__main__":
    rows = fresh_questions()
    print(len(rows), "fresh lookups")
    with open(os.path.join(OUT, "research_r3.jsonl"), "w", encoding="utf-8") as f, ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(run, rows), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            L, H = r["luna"], r["haiku"]
            print(f"{i}/{len(rows)} luna {L['kept']} facts ${L['cost']:.4f} | haiku {H['kept']} facts ({H['dropped']} dropped) "
                  f"{H['steps']} steps {H['seconds']}s ${H['cost']:.4f}", flush=True)
