"""Old (retype the pool) vs new (verdict per numbered headline) tagger, both on GPT-6 Luna,
on real tagger inputs from the Pi's audit log. Reports lines lost without a verdict, and a
blind Opus 5.5 judgment of which tagged pool is more correct.

    python3 -m benchmarks.tier_swap.run_tier_swap_tagger   (see main)
"""
import concurrent.futures
import json
import os
import random
import re
from datetime import datetime

from benchmarks.tier_swap.run_tier_swap import OUT_DIR, call, cost as call_cost

LUNA = {"provider": "openrouter", "model": "openai/gpt-6-luna", "name": "GPT-6 Luna (standard)", "reasoning": "low"}
JUDGE = {"provider": "anthropic", "model": "claude-opus-5-5"}
JUDGE_SYSTEM = (
    "You are grading two outputs of a news-desk tagger. The tagger's instructions (including the log of stories already "
    "covered) and today's headline pool are shown, then two tagged pools labeled A and B in random order. Judge which "
    "applies the instructions more correctly: right lines removed as SAME STORY with no new info, right UPDATE and MAJOR "
    "ESCALATION tags with the right arc slug, new stories left untagged, and no headline lost that should have been kept. "
    "Reply with JSON only: {\"winner\": \"A\" | \"B\" | \"tie\", \"a_serious_errors\": integer, \"b_serious_errors\": "
    "integer, \"reason\": \"one or two sentences\"}. A serious error is a wrong tag or slug, a repeat left untagged, a new "
    "story wrongly removed or tagged, or a kept headline lost."
)


def inputs(path):
    seen, out = set(), []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sp = r.get("system_prompt") or ""
        if r.get("event") == "success" and sp.startswith("You are filtering today's headlines") and r["user_prompt"] not in seen:
            seen.add(r["user_prompt"])
            out.append({"system": sp, "pool": r["user_prompt"], "date": r["timestamp"][:10],
                        "ledger": "[ARC:" in sp, "slugs": re.findall(r"\[ARC:\s*([^\]\s]+)\s*\]", sp)})
    return out


def lost_lines(pool, tagged):
    """Headline lines present in the pool but in no form in the output (tags stripped)."""
    from newscaster.dedup import strip_arc_tags
    from newscaster.scrapers.topic_finder import _headline_dedupe_key
    from newscaster.tagger import is_headline_line
    kept = {_headline_dedupe_key(strip_arc_tags(l)) for l in tagged.split("\n") if is_headline_line(l)}
    return [l for l in pool.split("\n") if is_headline_line(l) and _headline_dedupe_key(l) not in kept]


def run(item):
    from newscaster.tagger import tag_pool
    old, _s, u_old = call(LUNA, item["pool"], item["system"])
    usage = {}

    def ask(user, system):
        text, _secs, u = call(LUNA, user, system)
        usage["cost"] = usage.get("cost", 0) + (u.get("cost") or 0)
        return text
    new = tag_pool(item["pool"], item["system"], ask, ledger_mode=item["ledger"], valid_slugs=item["slugs"], label="bench-tagger")
    row = {"date": item["date"], "pool_lines": sum(1 for l in item["pool"].split("\n") if len(l.strip()) > 20),
           "old": {"text": old, "cost": u_old.get("cost") or 0}, "new": {"text": new, "cost": usage.get("cost", 0)}}
    # "lost" counts every missing line; the old tagger also removes lines on purpose, so the
    # judge decides which removals were right. The new tagger removes only on a "same" verdict.
    for k in ("old", "new"):
        row[k]["missing_lines"] = len(lost_lines(item["pool"], row[k]["text"]))
    new_first = random.Random(item["pool"][:200]).random() < 0.5
    a, b = ("new", "old") if new_first else ("old", "new")
    user = "TAGGER INSTRUCTIONS:\n{}\n\nTODAY'S POOL:\n{}\n\n=== OUTPUT A ===\n{}\n\n=== OUTPUT B ===\n{}".format(
        item["system"], item["pool"], row[a]["text"], row[b]["text"])
    text, _s, u = call(JUDGE, user, JUDGE_SYSTEM)
    v = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    row["judge"] = {"winner": {"A": a, "B": b}.get(v["winner"], "tie"), "errors": {a: v.get("a_serious_errors"), b: v.get("b_serious_errors")},
                    "reason": v.get("reason"), "cost": call_cost(u, JUDGE["model"])}
    return row


def main():
    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    items = inputs("benchmarks/tier_swap/data/pi_llm_audit.jsonl")
    print(len(items), "real tagger inputs")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "tagger_{}.jsonl".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for row in pool.map(run, items):
            open(out, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["date"], "pool", row["pool_lines"], "| missing old", row["old"]["missing_lines"], "new", row["new"]["missing_lines"],
                  "| winner", row["judge"]["winner"], row["judge"]["errors"])
    print("wrote", out)


if __name__ == "__main__":
    main()
