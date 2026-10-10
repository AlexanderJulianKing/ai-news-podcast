"""Round 4 (2026-10-10): research output with no unchecked prose, for both models.

    python3 -m benchmarks.haiku_vs_luna.run_round4

Rounds 2-3 found every flagged research error in text that code does not check: fact
sentences that do not match their own quote (R016), "I didn't find it" written as "it does
not exist" (R007), and overconfident answer lines (R004). The quotes themselves are always
real. So here code builds each answer from the verified quotes and their links only, drops
the answer line, and the model is told to write gaps as searches that found nothing.
Both models run in this format on the 39 research questions not used in rounds 2-3.
"""
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from benchmarks.haiku_vs_luna import run_research as rr
from newscaster import research_tools as rt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
GAPS_RULE = """
In "gaps", write each one as a search that found nothing, in the form "Searched for <what>; not found."
Never state that something does not exist or did not happen; only say what you looked for and could not find."""
rt.SYSTEM = rt.SYSTEM + GAPS_RULE        # both models: Luna through the production loop, Haiku through the adapter


def fresh_questions():
    qs = json.load(open(os.path.join(OUT, "research_qs.json"), encoding="utf-8"))
    used = {json.loads(l)["question"] for f in ("research_r2.jsonl", "research_r3.jsonl")
            for l in open(os.path.join(OUT, f), encoding="utf-8")}
    pool = [q for q in qs if q["question"] not in used]
    random.Random(4).shuffle(pool)
    out = []
    for q in pool:
        day = datetime.strptime(q["date"][:10], "%Y_%m_%d")
        out.append({"question": q["question"], "topic": q.get("topic"), "date": day.strftime("%B %-d, %Y")})
    return out


def render(answer_text):
    """Verified quotes with links, and gaps; drop fact sentences and the answer line."""
    if "FINDINGS:" not in answer_text:
        return answer_text, 0
    findings = answer_text.split("ANSWER:")[0].split("GAPS:")[0]
    # Each fact is '- <fact> ("<quote>") [<url>]'. Quotes from PDFs can span lines, and URLs can
    # hold spaces (PDF file names), so split into facts first and read the quote and link off the end.
    quotes = []
    for chunk in re.split(r"\n(?=- )", findings)[1:]:
        m = re.search(r'\("(.*)"\) \[(.*)\]\s*\Z', chunk, re.S)   # links can hold spaces and [brackets]
        if m:
            quotes.append((m.group(1), m.group(2)))
    gaps = answer_text.split("GAPS:", 1)[1].split("\n\nSources:")[0].strip() if "GAPS:" in answer_text else "None"
    lines = ["VERIFIED QUOTES:"] + [f'- "{q}" [{u}]' for q, u in quotes] + ["", "SEARCHES THAT FOUND NOTHING:", gaps]
    return "\n".join(lines), len(quotes)


def run(row):
    out = {"question": row["question"], "topic": row.get("topic"), "date": row["date"]}
    try:
        res = rt.research_with_tools(row["question"], row["date"])
        text, n = render(res["answer"])
        out["luna"] = {"text": text, "kept": n, "facts": res["metadata"]["facts_kept"], "raw": res["answer"], "cost": res["metadata"]["cost"],
                       "seconds": res["metadata"]["seconds"]}
    except Exception as exc:
        out["luna"] = {"text": f"ERROR {exc!r}"[:300], "kept": 0, "raw": "", "cost": 0, "seconds": 0}
    try:
        final, pages, steps, cost, secs = rr.haiku_research(row["question"], row["date"], cache=True)
        raw, kept, _dropped = rr.as_answer(final, pages)
        text, n = render(raw)
        out["haiku"] = {"text": text, "kept": n, "facts": kept, "raw": raw, "cost": round(cost, 4), "seconds": round(secs)}
    except Exception as exc:
        out["haiku"] = {"text": f"ERROR {exc!r}"[:300], "kept": 0, "raw": "", "cost": 0, "seconds": 0}
    return out


if __name__ == "__main__":
    rows = fresh_questions()
    print(len(rows), "fresh lookups")
    with open(os.path.join(OUT, "research_r4.jsonl"), "w", encoding="utf-8") as f, ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(run, rows), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i}/{len(rows)} luna {r['luna']['kept']} quotes ${r['luna']['cost']:.4f} | "
                  f"haiku {r['haiku']['kept']} quotes {r['haiku']['seconds']}s ${r['haiku']['cost']:.4f}", flush=True)
