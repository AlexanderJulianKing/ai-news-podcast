"""Unblind the subagent verdicts and tally Haiku vs Luna per task.

    python3 -m benchmarks.haiku_vs_luna.score outputs/key_do_not_open.json outputs/verdicts/*.json
Each verdict file is a JSON list of {"id": "P001", "winner": "A"|"B"|"TIE", "reason": "...",
"invented_or_wrong": "A"|"B"|"both"|"none"}.
"""
import json
import sys
from collections import defaultdict


def main(key_path, verdict_paths):
    key = json.load(open(key_path))
    tally = defaultdict(lambda: {"haiku": 0, "luna": 0, "tie": 0, "haiku_errors": 0, "luna_errors": 0})
    seen = set()
    for path in verdict_paths:
        for v in json.load(open(path)):
            pid = v["id"]
            if pid not in key or pid in seen:
                continue
            seen.add(pid)
            k = key[pid]
            other = {"A": "B", "B": "A"}
            haiku_label = "A" if k["A"] == "haiku" else "B"
            t = tally[k["kind"]]
            w = str(v.get("winner", "")).upper()
            if w == "TIE":
                t["tie"] += 1
            elif w == haiku_label:
                t["haiku"] += 1
            elif w == other[haiku_label]:
                t["luna"] += 1
            bad = str(v.get("invented_or_wrong", "none")).upper()
            if bad in (haiku_label, "BOTH"):
                t["haiku_errors"] += 1
            if bad in (other[haiku_label], "BOTH"):
                t["luna_errors"] += 1
    missing = sorted(set(key) - seen)
    total = defaultdict(int)
    print(f"{'task':16} haiku  luna  tie | wrong/invented: haiku luna")
    for kind, t in sorted(tally.items()):
        print(f"{kind:16} {t['haiku']:5} {t['luna']:5} {t['tie']:4} | {t['haiku_errors']:5} {t['luna_errors']:5}")
        for f, n in t.items():
            total[f] += n
    print(f"{'TOTAL':16} {total['haiku']:5} {total['luna']:5} {total['tie']:4} | {total['haiku_errors']:5} {total['luna_errors']:5}")
    if missing:
        print("no verdict for:", ", ".join(missing))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
