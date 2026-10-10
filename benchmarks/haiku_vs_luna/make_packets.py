"""Blind judge packets for Claude Code subagents (judging runs on the subscription, not the API).

    python3 -m benchmarks.haiku_vs_luna.make_packets outputs/replay.jsonl outputs/research.jsonl OUTDIR

Each packet holds a few pairs. Within a pair the two answers are labeled A and B in a random
order, with no model names. The key that unblinds them goes to OUTDIR/../key_<run>.json,
which the judges are told not to open. score.py joins verdicts with the key.
"""
import json
import os
import random
import sys

from benchmarks.haiku_vs_luna.run_replay import TASKS

RESEARCH_RUBRIC = (
    "Which answer better answers the question: more of what was asked is actually answered, with specific facts "
    "(names, numbers, bill or case numbers, dates), and fewer \"the sources do not say\" gaps? Ignore length, "
    "format and style. A \"no evidence\" answer loses to any answer with relevant findings. Both answers were "
    "written only from web pages the researcher opened, and every quoted fact was checked against that page."
)
# Characters of task input shown to the judge. Round 1 used 30,000, which cut off every
# synthesis, faithfulness and memory-note input (up to 97,000) and hid evidence from the
# judges (P015: the quote that supported Haiku sat at character 34,836). Round 2: 100,000.
MAX_INPUT = 100000


def _pairs(replay_path, research_path):
    rubric = {t: r for t, _m, _n, r in TASKS}
    for row in map(json.loads, open(replay_path, encoding="utf-8")):
        if row["luna"]["error"] or row["haiku"]["error"]:
            continue    # counted separately as failures
        user = row["user"] if len(row["user"]) <= MAX_INPUT else row["user"][:MAX_INPUT] + "\n[... input truncated for the judge ...]"
        yield {"kind": row["task"], "rubric": rubric[row["task"]], "instructions": row["system"], "input": user,
               "luna": row["luna"]["text"], "haiku": row["haiku"]["text"]}
    for row in map(json.loads, open(research_path, encoding="utf-8")):
        yield {"kind": "research", "rubric": RESEARCH_RUBRIC, "instructions": "", "input": row["question"],
               "luna": row["luna"]["text"], "haiku": row["haiku"]["text"]}


def main(replay_path, research_path, outdir, per_packet_chars=120000, id_prefix="P", key_name="key_do_not_open.json"):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(2026_10_10)
    items, key = [], {}
    for n, p in enumerate(_pairs(replay_path, research_path), 1):
        pid = f"{id_prefix}{n:03d}"
        haiku_is_a = rng.random() < 0.5
        key[pid] = {"kind": p["kind"], "A": "haiku" if haiku_is_a else "luna"}
        a, b = (p["haiku"], p["luna"]) if haiku_is_a else (p["luna"], p["haiku"])
        block = [f"## {pid} ({p['kind']})", "", "### What makes an answer better", p["rubric"], ""]
        if p["instructions"]:
            block += ["### Instructions the model was given", p["instructions"], ""]
        block += ["### Task input" if p["instructions"] else "### Question", p["input"], "",
                  "### Answer A", a or "(empty)", "", "### Answer B", b or "(empty)", ""]
        items.append((pid, "\n".join(block)))
    rng.shuffle(items)   # mix tasks across packets
    packets, cur, size = [], [], 0
    for pid, text in items:
        if cur and size + len(text) > per_packet_chars:
            packets.append(cur)
            cur, size = [], 0
        cur.append((pid, text))
        size += len(text)
    if cur:
        packets.append(cur)
    for i, pk in enumerate(packets, 1):
        with open(os.path.join(outdir, f"packet_{i:02d}.md"), "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(t for _p, t in pk))
    with open(os.path.join(os.path.dirname(outdir.rstrip("/")), key_name), "w") as f:
        json.dump(key, f, indent=1)
    print(f"{len(items)} pairs in {len(packets)} packets:", [len(p) for p in packets])


if __name__ == "__main__":
    main(*sys.argv[1:4])
