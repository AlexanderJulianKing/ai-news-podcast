"""Structured tagger settings on real mornings, graded one output at a time.

Each output is graded alone by Opus 5.5, which must LIST every serious error with the line,
what the tagger did, and what it should have done, so the errors can be checked by hand.
"""
import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime

from benchmarks.tier_swap.run_tagger import inputs
from benchmarks.tier_swap.run_tier_swap import OUT_DIR, call, cost as call_cost

JUDGE = {"provider": "anthropic", "model": "claude-opus-5-5"}
GRADER = (
    "You are auditing the output of a news-desk tagger. You see its instructions (including the log of stories already "
    "covered), today's headline pool, and the tagged pool it produced. Check EVERY headline against the instructions. "
    "Serious errors: a repeat with no new information that was kept (it should have been removed); a new development on a "
    "tracked story left untagged; an UPDATE or MAJOR ESCALATION tag with the wrong arc slug; a story tagged as a "
    "continuation that is actually new; a headline removed that carried new information; MAJOR ESCALATION used where the "
    "instructions call for UPDATE, or the reverse. Do not count style. When a call is genuinely debatable under the "
    "instructions, do not count it as serious; list it under debatable instead.\n\nReply with JSON only: "
    "{\"serious\": [{\"headline\": \"...\", \"did\": \"...\", \"should\": \"...\"}], \"debatable\": [\"...\"]}"
)
SETTINGS = {"b40_low": (40, "low"), "b10_low": (10, "low"), "b10_medium": (10, "medium"), "b10_high": (10, "high"),
            "b10_sol": (10, "medium", "openai/gpt-6-sol"),
            "b1_low": (1, "low"), "b3_low": (3, "low"), "b3_medium": (3, "medium")}


def run(args):
    name, item = args
    from newscaster.tagger import tag_pool
    batch, effort = SETTINGS[name][:2]
    model = SETTINGS[name][2] if len(SETTINGS[name]) > 2 else "openai/gpt-6-luna"
    spec = {"provider": "openrouter", "model": model, "name": model, "reasoning": effort}
    spent = {"cost": 0.0}

    def ask(user, system):
        text, _s, u = call(spec, user, system)
        spent["cost"] += u.get("cost") or 0
        return text
    tagged = tag_pool(item["pool"], item["system"], ask, ledger_mode=item["ledger"], valid_slugs=item["slugs"], batch_size=batch, label=name, workers=8)
    user = "TAGGER INSTRUCTIONS:\n{}\n\nTODAY'S POOL:\n{}\n\nTAGGED OUTPUT:\n{}".format(item["system"], item["pool"], tagged)
    text, _s, u = call(JUDGE, user, GRADER)
    verdict = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    return {"setting": name, "date": item["date"], "tagged": tagged, "serious": verdict.get("serious") or [],
            "debatable": verdict.get("debatable") or [], "tagger_cost": spent["cost"], "judge_cost": call_cost(u, JUDGE["model"])}


def main():
    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    names = sys.argv[1:] or list(SETTINGS)
    jobs = [(n, it) for n in names for it in inputs("benchmarks/tier_swap/data/pi_llm_audit.jsonl")]
    out = os.path.join(OUT_DIR, "tagger_grid_{}.jsonl".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for row in pool.map(run, jobs):
            open(out, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["setting"], row["date"], "serious", len(row["serious"]), "debatable", len(row["debatable"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
