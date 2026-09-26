"""Can GPT-6 Luna replace the standard (Gemma) or advanced (GLM) tier on their real jobs?

    python3 -m benchmarks.tier_swap.run_tier_swap --dry-run
    python3 -m benchmarks.tier_swap.run_tier_swap

Inputs are real calls from logs/llm_audit*.jsonl. The log keeps prompts but not
answers, so each input is replayed now through the incumbent model and through
Luna. Then:
- automatic checks: usable output, no repetition loop, and for evidence contracts,
  whether the pipeline's own parser gets any required slots out of it;
- a blind pairwise judge (Opus 5.5) that sees the task input and both answers in a
  random order, labeled A and B, and scores them against the task's own rules.

Calls bypass the router's fallback and are kept out of the audit log.
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import random
import re
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

TASKS = [  # (task, system-prompt prefix, what a good answer does)
    ("contract", "You convert newsroom research questions into evidence contracts",
     "Better answer: an evidence contract whose required facts are precise and checkable and actually needed to answer the "
     "question; sensible source preferences; reject conditions that catch real traps (previews, calendars, announcements "
     "without results) without rejecting legitimate reporting; valid structure per the instructions; no invented facts."),
    ("answer", "You are a careful newsroom research assistant. Answer only from the co",
     "Better answer: every claim is supported by the provided source excerpts (nothing added from outside them); it answers "
     "the question as fully as the sources allow; the GAPS section names exactly what the sources leave unanswered; sources "
     "listed are ones actually used."),
    ("faithfulness", "You are a fact-checking editor",
     "Better answer: flags the script claims that the source material does NOT support, with few false alarms (claims the "
     "sources do support) and few misses; follows the required output format."),
    ("faithfulness", "You are a meticulous fact-checking editor",
     "Better answer: flags the script claims that the source material does NOT support, with few false alarms and few misses; "
     "follows the required output format."),
    ("stable_fact", "You verify ONLY well-established",
     "Better answer: flags only genuine errors in well-established background facts, with few false alarms; does not flag "
     "recent news it cannot know; follows the required output format."),
    ("report", "You are a careful newsroom research assistant. Report only w",
     "Better answer: reports only what the supplied source excerpts support, with nothing added from outside them; answers the "
     "question as fully as the sources allow; names what the sources leave unanswered; keeps any required section headings."),
    ("memory_note", "You are preparing a dated memory note for today's reporting",
     "Better answer: a dated note that keeps only what the retrieved prior coverage actually says, with dates attached, and "
     "does not present old facts as today's news; follows the requested structure."),
    ("headline", "Shorten the following story to a headline",
     "Better answer: a short, accurate headline in the style the instructions request, specific to the story, with nothing "
     "the story does not say."),
    ("tagger", "You are filtering today's headlines against stories we already covered",
     "Better answer: tags each headline correctly against the stories already covered, as the instructions define the tags, "
     "and keeps every input line."),
]

INCUMBENT = {"google/gemma-4-31b-it": "standard", "z-ai/glm-5.2": "advanced"}
# Luna's reasoning effort follows the tier it would replace: Gemma runs without
# reasoning, GLM at medium.
LUNA = {"standard": {"provider": "openrouter", "model": "openai/gpt-6-luna", "name": "GPT-6 Luna", "reasoning": "low"},
        "advanced": {"provider": "openrouter", "model": "openai/gpt-6-luna", "name": "GPT-6 Luna", "reasoning": "medium"}}
JUDGE = {"provider": "anthropic", "model": "claude-opus-5-5"}


def harvest(paths):
    """Unique (task, tier, system, user) inputs from real, successful audit rows."""
    seen, out = set(), []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") != "success" or r.get("phase") == "benchmark" or r.get("model") not in INCUMBENT:
                    continue
                system, user = r.get("system_prompt") or "", r.get("user_prompt") or ""
                if user in ("p", "test prompt"):
                    continue
                task = next((t for t, prefix, _ in TASKS if system.startswith(prefix)), None)
                key = hashlib.sha1((system + "\0" + user).encode("utf-8")).hexdigest()
                if task and key not in seen:
                    seen.add(key)
                    out.append({"id": key[:12], "task": task, "tier": INCUMBENT[r["model"]], "system": system, "user": user})
    # In production the advanced tier (GLM) runs the faithfulness check and the escalated
    # research answer, on the same kind of input. Add those inputs with GLM as incumbent.
    extra = [dict(it, tier="advanced", extra=True) for it in out if it["tier"] == "standard" and it["task"] in ("faithfulness", "answer")]
    return out + extra


def degenerate(text):
    """Empty, or a repetition loop: a single word making up over a third of a long reply."""
    words = (text or "").split()
    if not words:
        return "empty"
    top = max(words.count(w) for w in set(words[:2000]))
    if len(words) > 200 and top / len(words) > 0.33:
        return "repetition loop"
    return None


_LOCAL = threading.local()


def _capture(event, *a, usage=None, **k):
    if event == "success":
        _LOCAL.usage = usage or {}


def call(spec, user, system):
    """One call with the router's retries, no fallback. Usage is kept per thread."""
    import uuid
    import newscaster.llm.router as router
    from newscaster.llm.router import _call_with_retry
    router._audit_llm_event = _capture
    _LOCAL.usage = {}
    start = time.time()
    text = _call_with_retry(spec, user, system, call_id=str(uuid.uuid4()), phase="benchmark")
    return text, round(time.time() - start, 1), dict(_LOCAL.usage)


def cost(usage, model):
    if usage.get("cost"):
        return usage["cost"]
    if usage.get("estimated_cost_usd"):
        return usage["estimated_cost_usd"]
    if model == "claude-opus-5-5":
        return (usage.get("input_tokens") or 0) * 4e-6 + (usage.get("output_tokens") or 0) * 20e-6
    return 0.0


def auto_checks(task, text):
    checks = {"degenerate": degenerate(text)}
    if task == "contract":
        from newscaster.source_hunter_primitives import parse_evidence_contract
        checks["contract_slots"] = len(parse_evidence_contract(text or "").get("required_slots") or [])
    return checks


JUDGE_SYSTEM = (
    "You are grading two answers to the same task for a news-production pipeline. You will see the task's instructions, its "
    "input, and two answers labeled A and B. You do not know which system wrote which, and the order is random. Judge only "
    "against the task's instructions and input, checking claims against the input yourself. Length and style count only "
    "when they affect usefulness to the pipeline.\n\nReply with JSON only: {\"winner\": \"A\" | \"B\" | \"tie\", "
    "\"a_serious_errors\": integer, \"b_serious_errors\": integer, \"reason\": \"one or two sentences\"}. A serious error is "
    "one that would put a false or unsupported claim on air, drop required content, or break the required output format."
)


def judge(item, first, second, rubric):
    user = ("TASK INSTRUCTIONS (the system prompt the model received):\n{}\n\nTASK INPUT:\n{}\n\nWHAT MAKES AN ANSWER BETTER:\n{}"
            "\n\n=== ANSWER A ===\n{}\n\n=== ANSWER B ===\n{}").format(item["system"], item["user"], rubric, first, second)
    text, secs, usage = call(JUDGE, user, JUDGE_SYSTEM)
    m = re.search(r"\{.*\}", text, re.S)
    verdict = json.loads(m.group(0)) if m else {"winner": "unparsed", "reason": text[:200]}
    return verdict, cost(usage, JUDGE["model"])


def run_item(item, rng_seed):
    from newscaster.llm.router import _select_primary
    rubric = next(r for t, _p, r in TASKS if t == item["task"])
    specs = {"incumbent": _select_primary(item["tier"], False, False), "luna": LUNA[item["tier"]]}
    row = {k: item[k] for k in ("id", "task", "tier")}
    for name, spec in specs.items():
        try:
            text, secs, usage = call(spec, item["user"], item["system"])
            row[name] = {"model": spec["model"], "text": text, "seconds": secs, "cost": cost(usage, spec["model"]),
                         "reasoning_tokens": usage.get("reasoning_tokens"), "checks": auto_checks(item["task"], text)}
        except Exception as e:
            row[name] = {"model": spec["model"], "error": "{}: {}".format(type(e).__name__, e)[:300]}
    if all("text" in row[n] for n in specs):
        luna_first = random.Random(rng_seed + item["id"]).random() < 0.5
        a, b = ("luna", "incumbent") if luna_first else ("incumbent", "luna")
        try:
            verdict, jcost = judge(item, row[a]["text"], row[b]["text"], rubric)
            w = verdict.get("winner")
            row["judge"] = {"order": [a, b], "verdict": verdict, "cost": jcost,
                            "winner": {"A": a, "B": b, "tie": "tie"}.get(w, "unparsed"),
                            "serious_errors": {a: verdict.get("a_serious_errors"), b: verdict.get("b_serious_errors")}}
        except Exception as e:
            row["judge"] = {"error": "{}: {}".format(type(e).__name__, e)[:300]}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", nargs="+", default=sorted(glob.glob("logs/llm_audit*.jsonl")))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-task", type=int, default=0, help="cap on inputs per (task, tier), sampled with a fixed seed; 0 = all")
    ap.add_argument("--no-extra", action="store_true", help="do not duplicate standard inputs onto the advanced tier")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = harvest(args.audit)
    if args.no_extra:
        items = [it for it in items if not it.get("extra")]
    if args.per_task:
        groups = {}
        for it in items:
            groups.setdefault((it["task"], it["tier"]), []).append(it)
        items = []
        for key in sorted(groups):
            g = sorted(groups[key], key=lambda it: it["id"])
            random.Random("sample-" + "-".join(key)).shuffle(g)
            items += g[:args.per_task]
    counts = {}
    for it in items:
        counts[(it["task"], it["tier"])] = counts.get((it["task"], it["tier"]), 0) + 1
    print("{} unique real inputs: {}".format(len(items), counts))
    if args.dry_run:
        return

    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "tier_swap_{}.jsonl".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(lambda it: run_item(it, "seed-2026-09-22"), items):
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            j = row.get("judge", {})
            print("{:<13} {:<9} winner={:<9} {}".format(row["task"], row["tier"], j.get("winner", "-"),
                  "; ".join("{}: {}".format(n, row[n].get("error", "ok")[:60]) for n in ("incumbent", "luna") if "error" in row[n])))
    print("wrote " + out)


if __name__ == "__main__":
    main()
