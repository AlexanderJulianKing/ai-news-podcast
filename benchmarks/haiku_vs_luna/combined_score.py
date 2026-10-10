"""Both judges, both rounds: Haiku vs Luna per job.

    python3 -m benchmarks.haiku_vs_luna.combined_score

Opus (Claude Code subagents) and GPT-6 Astra (Codex) judged the same blind pairs. Round-1
pairs whose input the judges saw cut off (synthesis, faithfulness, memory notes) are
dropped; they were re-judged with full inputs in round 2.
"""
import glob
import json
import os
from collections import defaultdict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CUT_OFF_IN_ROUND1 = {"synthesis", "faithfulness", "memory_note"}


def verdicts(pattern):
    return {v["id"]: v for f in glob.glob(os.path.join(OUT, pattern)) for v in json.load(open(f))}


def who(key, pid, label):
    label = str(label).upper()
    if label in ("A", "B"):
        haiku_a = key[pid]["A"] == "haiku"
        return "haiku" if (label == "A") == haiku_a else "luna"
    return "tie" if label == "TIE" else ("both" if label == "BOTH" else "none")


def main():
    key = {**json.load(open(os.path.join(OUT, "key_do_not_open.json"))),
           **json.load(open(os.path.join(OUT, "key_r2_do_not_open.json")))}
    opus = {**verdicts("verdicts/*.json"), **verdicts("verdicts_r2/*.json")}
    codex = {**verdicts("codex_verdicts/*.json"), **verdicts("codex_verdicts_r2/*.json")}
    rows = defaultdict(lambda: defaultdict(int))
    for pid, k in key.items():
        if pid.startswith("P") and k["kind"] in CUT_OFF_IN_ROUND1:
            continue
        if pid not in opus or pid not in codex:
            rows[k["kind"]]["missing"] += 1
            continue
        r = rows[k["kind"]]
        r["n"] += 1
        o, c = who(key, pid, opus[pid]["winner"]), who(key, pid, codex[pid]["winner"])
        r["opus_" + o] += 1
        r["codex_" + c] += 1
        if o == c:
            r["agree_" + o] += 1
        for judge, v in (("opus", opus[pid]), ("codex", codex[pid])):
            bad = who(key, pid, v.get("invented_or_wrong", "none"))
            for m in ("haiku", "luna"):
                if bad in (m, "both"):
                    r[f"{judge}_wrong_{m}"] += 1
    head = f"{'job':16} {'n':>3} | {'Opus H-L-T':>11} | {'Codex H-L-T':>11} | {'both agree H-L':>14} | wrong H/L (Opus; Codex)"
    print(head)
    print("-" * len(head))
    tot = defaultdict(int)
    for kind in sorted(rows):
        r = rows[kind]
        for f, v in r.items():
            tot[f] += v
        print(f"{kind:16} {r['n']:3} | {r['opus_haiku']:3}-{r['opus_luna']:3}-{r['opus_tie']:3} | "
              f"{r['codex_haiku']:3}-{r['codex_luna']:3}-{r['codex_tie']:3} | {r['agree_haiku']:6}-{r['agree_luna']:<7} | "
              f"{r['opus_wrong_haiku']}/{r['opus_wrong_luna']}; {r['codex_wrong_haiku']}/{r['codex_wrong_luna']}"
              + (f"  (missing {r['missing']})" if r["missing"] else ""))
    print("-" * len(head))
    print(f"{'TOTAL':16} {tot['n']:3} | {tot['opus_haiku']:3}-{tot['opus_luna']:3}-{tot['opus_tie']:3} | "
          f"{tot['codex_haiku']:3}-{tot['codex_luna']:3}-{tot['codex_tie']:3} | {tot['agree_haiku']:6}-{tot['agree_luna']:<7} | "
          f"{tot['opus_wrong_haiku']}/{tot['opus_wrong_luna']}; {tot['codex_wrong_haiku']}/{tot['codex_wrong_luna']}")


if __name__ == "__main__":
    main()
