"""Gemma vs GPT-6 Luna on the web-search brief (search.openrouter_web_brief).

    python3 -m benchmarks.tier_swap.run_web_brief --dry-run

The brief has two production jobs: the Tier 2 background memo for every candidate
story (topic_finder), and the background fact check (review). Real questions come
from the Pi's logs/search_audit.jsonl. Each is replayed now through both models
with the same web-search plugin settings production uses, then a blind Opus 5.5
judge compares the two answers, each shown with the sources that model cited.

Both models search today, not on the day of the question, so both see the same
drift in what the web says.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from datetime import datetime

from benchmarks.tier_swap.run_tier_swap import OUT_DIR, call, cost as call_cost

MODELS = {"gemma": "google/gemma-4-31b-it", "luna": "openai/gpt-6-luna"}
JUDGE = {"provider": "anthropic", "model": "claude-opus-5-5"}
OLD_T2 = ("Use Google Search to pull from multiple reputable sources. Attribute key claims to specific outlets. If you cannot "
          "verify the headline, begin your response with 'UNVERIFIED:' and explain what you tried.")
NEW_T2 = ("Use the web search results provided with this request, drawing on multiple reputable sources. Attribute key claims to "
          "specific outlets. Begin your response with 'UNVERIFIED:' only when those results do not confirm the core event in the "
          "headline, and then explain what you checked. Do not use it because a detail is unconfirmed or because you could not run "
          "a search yourself; note unconfirmed details in the memo instead.")
args_prompt_update = {"on": False}
RUBRIC = {
    "tier2": ("Better answer: a memo whose factual claims are supported by the sources it cites; covers what happened, "
              "context, impact and scale, and which outlets report it, with claims attributed to outlets; treats the date "
              "in the question as the research date; uses 'UNVERIFIED:' only when the headline truly cannot be verified. "
              "This memo is what an editor reads to judge the story's importance, so accuracy and concrete detail matter most."),
    "factcheck": ("Better answer: the correct verdict on the script's claim, as supported by the sources it cites, in the "
                  "required format ('CORRECT', or 'WRONG: <the correct current fact>'). A WRONG verdict that is itself "
                  "wrong would push a false correction toward the script, so it is a serious error."),
}
JUDGE_SYSTEM = (
    "You are grading two answers to the same research request for a news-production pipeline. Each answer was produced "
    "by a model that ran its own web search; each is shown with the sources it cited. You do not know which system wrote "
    "which, and the order is random. Check claims against the cited sources shown. Where the sources shown cannot settle "
    "a point, say so instead of guessing.\n\nReply with JSON only: {\"winner\": \"A\" | \"B\" | \"tie\", "
    "\"a_serious_errors\": integer, \"b_serious_errors\": integer, \"correct_verdict\": \"A\" | \"B\" | \"both\" | "
    "\"neither\" | \"unclear\" | \"n/a\", \"reason\": \"one or two sentences\"}. Use correct_verdict only for fact checks "
    "(which answer's CORRECT/WRONG verdict is right); otherwise \"n/a\". A serious error is a claim that is false or "
    "unsupported by the cited sources, a wrong verdict, or a broken required format."
)


def load(path, per_kind, seed="web-brief-2026-09-23"):
    seen, rows = set(), {"tier2": [], "factcheck": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") != "openrouter_web_brief" or r["question"] in seen:
                continue
            seen.add(r["question"])
            kind = "factcheck" if r["question"].startswith("A news script states:") else "tier2"
            q = r["question"]
            if args_prompt_update["on"] and kind == "tier2":  # replay under the reworded Tier 2 instruction
                q = q.replace(OLD_T2, NEW_T2)
            rows[kind].append({"kind": kind, "question": q, "max_results": r.get("max_results") or 5, "logged_question": r["question"]})
    out = []
    for kind, items in rows.items():
        items.sort(key=lambda it: it["question"])
        random.Random(seed + kind).shuffle(items)
        out += items[:per_kind]
    return out


def web_answer(model, item):
    import newscaster.config as config
    from newscaster.search import _openrouter_web_chat
    start = time.time()
    message, data = _openrouter_web_chat(
        item["question"], model=model, max_results=item["max_results"],
        search_prompt="A web search was conducted for a newsroom background brief. Prefer recent, reputable, and primary sources.",
        title="Newscaster Tier-2 Brief benchmark", timeout=(10, 150))
    notes = []
    for a in message.get("annotations") or []:
        c = a.get("url_citation") or a
        if c.get("url"):
            notes.append({"title": c.get("title") or "", "url": c["url"], "content": (c.get("content") or "")[:700]})
    return {"model": model, "text": (message.get("content") or "").strip(), "sources": notes,
            "seconds": round(time.time() - start, 1), "cost": (data.get("usage") or {}).get("cost") or 0.0}


def render(answer):
    src = "\n".join("  [{}] {} | {}\n      {}".format(i, s["title"], s["url"], s["content"].replace("\n", " "))
                    for i, s in enumerate(answer["sources"], 1)) or "  (no sources cited)"
    return "{}\n\nSOURCES CITED:\n{}".format(answer["text"], src)


def run_item(item):
    row = {"kind": item["kind"], "question": item["question"]}
    for name, model in MODELS.items():
        try:
            row[name] = web_answer(model, item)
            if not row[name]["text"]:
                row[name]["error"] = "empty completion"
        except Exception as e:
            row[name] = {"model": model, "error": "{}: {}".format(type(e).__name__, e)[:300]}
    if all("text" in row[n] and not row[n].get("error") for n in MODELS):
        luna_first = random.Random("order" + item["question"]).random() < 0.5
        a, b = ("luna", "gemma") if luna_first else ("gemma", "luna")
        user = ("RESEARCH REQUEST:\n{}\n\nWHAT MAKES AN ANSWER BETTER:\n{}\n\n=== ANSWER A ===\n{}\n\n=== ANSWER B ===\n{}").format(
            item["question"], RUBRIC[item["kind"]], render(row[a]), render(row[b]))
        try:
            text, _secs, usage = call(JUDGE, user, JUDGE_SYSTEM)
            v = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            m = {"A": a, "B": b}
            row["judge"] = {"verdict": v, "cost": call_cost(usage, JUDGE["model"]),
                            "winner": m.get(v.get("winner"), "tie" if v.get("winner") == "tie" else "unparsed"),
                            "correct_verdict": m.get(v.get("correct_verdict"), v.get("correct_verdict")),
                            "serious_errors": {a: v.get("a_serious_errors"), b: v.get("b_serious_errors")}}
        except Exception as e:
            row["judge"] = {"error": "{}: {}".format(type(e).__name__, e)[:300]}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", default="benchmarks/tier_swap/data/pi_search_audit.jsonl")
    ap.add_argument("--per-kind", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--new-tier2-prompt", action="store_true", help="replay Tier 2 questions with the reworded UNVERIFIED instruction")
    ap.add_argument("--kinds", nargs="+", default=list(RUBRIC))
    args = ap.parse_args()
    args_prompt_update["on"] = args.new_tier2_prompt
    items = [i for i in load(args.audit, args.per_kind) if i["kind"] in args.kinds]
    print("{} questions: {}".format(len(items), {k: sum(1 for i in items if i["kind"] == k) for k in RUBRIC}))
    if args.dry_run:
        return
    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    config.SEARCH_AUDIT_LOG_ENABLED = False
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "web_brief_{}.jsonl".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(run_item, items):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("{:<9} winner={:<9} {}".format(row["kind"], row.get("judge", {}).get("winner", "-"),
                  "; ".join("{}: {}".format(n, row[n]["error"][:60]) for n in MODELS if row[n].get("error"))))
    print("wrote " + out)


if __name__ == "__main__":
    main()
