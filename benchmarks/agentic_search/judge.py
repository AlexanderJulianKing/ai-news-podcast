"""Blind pairwise judging of two answer fields in a results JSONL, with Opus.

    python3 -m benchmarks.agentic_search.judge FILE FIELD_X FIELD_Y
Prints per-row verdicts and totals in terms of FIELD_X vs FIELD_Y.
"""
import json
import os
import random
import sys
from collections import Counter

os.environ.setdefault("NEWSCASTER_LOG_DIR", "/tmp/nc_judge_logs")
import newscaster.config as config  # noqa: E402

config.init()
config.LLM_AUDIT_LOG_ENABLED = False
from newscaster.llm import get_llm_response  # noqa: E402
from newscaster.source_hunter import _focus  # noqa: E402

PROMPT = """You are grading two research answers to the same question for a news show. Both were written only from web pages the researcher fetched.

QUESTION:
{q}

ANSWER A:
{a}

ANSWER B:
{b}

Which answer better answers the question: more of what was asked is actually answered, with specific facts (names, numbers, bill or case numbers, dates), and fewer "the sources do not say" gaps? Ignore length, format and style. A "no evidence" answer loses to any answer with relevant findings.
Reply with exactly one line: "WINNER: A", "WINNER: B" or "WINNER: TIE", then one short sentence of reason."""


def main(path, fx, fy):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng = random.Random(11)
    tally, by_kind = Counter(), {"headline": Counter(), "follow-up": Counter()}
    for r in rows:
        x = r.get(fx) or "No accepted source evidence was found."
        y = r.get(fy) or "No accepted source evidence was found."
        flip = rng.random() < 0.5
        a, b = (x, y) if flip else (y, x)
        out = get_llm_response(PROMPT.format(q=_focus(r["question"])[:3000], a=a[:7000], b=b[:7000]), mode="heavy")
        line = next((l for l in out.splitlines() if "WINNER" in l.upper()), "")
        w = "TIE" if "TIE" in line.upper() else ("A" if line.strip().upper().endswith("A") else "B")
        v = "tie" if w == "TIE" else (fx if (w == "A") == flip else fy)
        kind = "headline" if r["question"].startswith("Headline:") else "follow-up"
        tally[v] += 1
        by_kind[kind][v] += 1
        print(f"{v:14} | {kind:9} | {_focus(r['question'])[:80]!r} | {out.splitlines()[-1][:150] if out else ''}", flush=True)
    print("TOTAL", dict(tally), "| follow-up", dict(by_kind["follow-up"]), "| headline", dict(by_kind["headline"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
