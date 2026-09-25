"""Run the tool-using researcher on the 32 production lookups of 2026-09-24/25.

Input: the replay file (question, topic, date, production answer, patched-source-hunter answer).
Output: JSONL with the agent's answer text (only quote-verified facts), steps, cost, seconds.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from benchmarks.agentic_search.luna_research import check_quotes, research

MODEL, EFFORT = "openai/gpt-6-luna", "medium"


def as_text(data, checks):
    lines = ["FINDINGS:"]
    kept = [f for ok, f in checks if ok]
    lines += [f"- {f.get('fact')} (\"{f.get('quote')}\") [{f.get('url')}]" for f in kept] or ["None"]
    lines += ["", "ANSWER: " + str(data.get("answer") or ""), "", "GAPS:"]
    lines += [f"- {g}" for g in data.get("gaps") or []] or ["None"]
    return "\n".join(lines)


def one(row):
    pages = {}
    facts = []
    t = time.time()
    try:
        final, steps, cost, secs = research(row["question"], MODEL, EFFORT, row["date"], pages)
        data, checks = check_quotes(final, pages)
        if data is None:
            text, ok, bad = "No usable answer (reply was not JSON).", 0, 0
        else:
            text = as_text(data, checks)
            ok, bad = sum(1 for c, _ in checks if c), sum(1 for c, _ in checks if not c)
            facts = [dict(f, quote_ok=c, page_chars=len((pages.get(f.get("url")) or {}).get("text") or "")) for c, f in checks]
    except Exception as exc:
        text, steps, cost, secs, ok, bad = f"ERROR {exc!r}", [], 0.0, time.time() - t, 0, 0
    return dict(row, agent_answer=text, agent_steps=len(steps), agent_cost=round(cost, 4),
                agent_seconds=round(secs), quotes_ok=ok, quotes_bad=bad, agent_facts=facts)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    out = open(sys.argv[2], "w", encoding="utf-8")
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(one, rows), 1):
            out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
            print(f"{i}/{len(rows)} steps={r['agent_steps']:2} ${r['agent_cost']:.4f} {r['agent_seconds']:3}s "
                  f"quotes ok/bad={r['quotes_ok']}/{r['quotes_bad']} | {r['question'][:70]!r}", flush=True)
