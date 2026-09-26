"""Score the second main slot (the 'everyman' pick) on top of a lead-pick run.

    python3 -m benchmarks.editorial_decisions.run_second_pick --from-run outputs/lead_pick_X.jsonl --dry-run

The pipeline picks the second story with TIER3_EVERYMAN_STORY_PROMPT, telling
the model which headline already leads. This replays that call for every row of
a saved lead-pick run, using the lead that same model chose, so no lead call is
paid for twice. Only cases whose label has a second pick are run.
"""
import argparse
import json
import os
from datetime import datetime

from benchmarks.editorial_decisions.run_lead_pick import OUT_DIR, call_tier, labeled_cases, parse_pick, story_key


def score_pair(model_lead, model_second, label, key=None):
    """Compare the model's two stories with the labeler's two.

    pair_overlap counts shared stories regardless of slot (0, 1 or 2): the
    episode covers the same ground even when the slots are swapped.
    second_exact is strict: same story in the second slot.
    """
    k = (lambda i: key.get(i, i)) if key else (lambda i: i)  # same-story briefs count as one
    mine = {k(i) for i in (label.get("lead_index"), label.get("second_index")) if i is not None}
    theirs = {k(i) for i in (model_lead, model_second) if i is not None}
    return {
        "second_exact": model_second is not None and label.get("second_index") is not None
                        and k(model_second) == k(label["second_index"]),
        "pair_overlap": len(mine & theirs),
        "pair_size": len(mine),
    }


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["mode"], r.get("set") or ("invented" if r["synthetic"] else "real"))
        g = groups.setdefault(key, {"mode": key[0], "set": key[1], "n": 0, "second_exact": 0, "pair_overlap": 0, "pair_size": 0, "both": 0, "unparsed": 0})
        g["n"] += 1
        g["second_exact"] += int(r["second_exact"])
        g["pair_overlap"] += r["pair_overlap"]
        g["pair_size"] += r["pair_size"]
        g["both"] += int(r["pair_size"] > 0 and r["pair_overlap"] == r["pair_size"])
        g["unparsed"] += int(r["second_pick_index"] is None)
    return [groups[k] for k in sorted(groups)]


def format_summary(summary):
    lines = ["{:<10} {:<14} {:>3} {:>13} {:>16} {:>11} {:>9}".format("mode", "set", "n", "second exact", "stories shared", "both match", "unparsed")]
    for g in summary:
        lines.append("{:<10} {:<14} {:>3} {:>13} {:>16} {:>11} {:>9}".format(
            g["mode"], g["set"], g["n"], g["second_exact"], "{}/{}".format(g["pair_overlap"], g["pair_size"]), g["both"], g["unparsed"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-run", required=True, help="a lead_pick_*.jsonl written by run_lead_pick")
    ap.add_argument("--modes", nargs="+", default=None, help="only these tiers; run one process per tier to go faster")
    ap.add_argument("--variant", default="production", help="which Tier 3 prompts to test; candidate_prompts.py holds the revisions")
    ap.add_argument("--unlabeled", action="store_true", help="the lead rows are for mornings with no label yet; record picks, score later")
    ap.add_argument("--shard", default=None, help="i/n: run only every n-th row starting at i, to split one tier across processes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from benchmarks.editorial_decisions.candidate_prompts import prompts_for
    TIER3_EVERYMAN_STORY_PROMPT = prompts_for(args.variant)[1]

    pairs = {c["case_id"]: (c, l) for c, l in labeled_cases()}
    if args.unlabeled:
        from benchmarks.editorial_decisions.run_lead_pick import unlabeled_cases
        pairs = {c["case_id"]: (c, None) for c, _ in unlabeled_cases()}
    with open(args.from_run, "r", encoding="utf-8") as f:
        lead_rows = [json.loads(line) for line in f if line.strip()]
    todo = [r for r in lead_rows
            if (not args.modes or r["mode"] in args.modes) and r["case_id"] in pairs and r.get("answer")
            and (args.unlabeled or "second_index" in pairs[r["case_id"]][1])]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
    skipped = len(lead_rows) - len(todo)
    print("{} second-slot calls; {} lead rows skipped (no second label, or no parsed lead)".format(len(todo), skipped))
    if args.dry_run:
        return

    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False  # see run_lead_pick

    # The router hands token usage to its audit hook and nowhere else. Swap the
    # hook for one that keeps the last usage, so spend can be reported.
    import newscaster.llm.router as router
    last_usage = {}

    def _capture(event, *a, usage=None, **k):
        if event == "success":
            last_usage.clear()
            last_usage.update(usage or {})
    router._audit_llm_event = _capture

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "second_pick_{}_{}{}_{}.jsonl".format(args.variant, "-".join(args.modes or ["all"]), ("_s" + args.shard.replace("/", "of")) if args.shard else "", datetime.now().strftime("%Y%m%d_%H%M%S")))

    rows = []
    for r in todo:
        case, label = pairs[r["case_id"]]
        prompt = TIER3_EVERYMAN_STORY_PROMPT.format(excluded_headline=r["answer"])
        last_usage.clear()
        try:
            essay, error = call_tier(r["mode"], case["user_prompt"], prompt), None
        except Exception as e:
            essay, error = "", "{}: {}".format(type(e).__name__, e)
        pick, answer, method = parse_pick(essay, case["briefs"])
        row = {"mode": r["mode"], "model": r.get("model"), "case_id": r["case_id"], "synthetic": r["synthetic"],
               "set": r.get("set") or case.get("set") or "real",
               "variant": args.variant, "repeat": r.get("repeat", 0), "lead_pick_index": r["pick_index"], "second_pick_index": pick, "second_answer": answer,
               "parse_method": method, "label_lead": (label or {}).get("lead_index"), "label_second": (label or {}).get("second_index"),
               "essay": essay, "error": error, "usage": dict(last_usage)}
        if label is not None:
            row.update(score_pair(r["pick_index"], pick, label, story_key(case)))
        else:
            row.update({"second_exact": False, "pair_overlap": 0, "pair_size": 0, "prelabel": True})
        rows.append(row)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if label is None:
            print("{:<10} {:<42} {} shared=-".format(r["mode"], r["case_id"][:42], "ERROR" if error else ("unparsed" if pick is None else "recorded")))
        else:
            print("{:<10} {:<42} model=({}, {}) label=({}, {}) shared={}".format(
                r["mode"], r["case_id"][:42], r["pick_index"], pick, label.get("lead_index"), label.get("second_index"), row["pair_overlap"]))

    print("\n" + format_summary(summarize(rows)))
    print("\nWrote " + out)


if __name__ == "__main__":
    main()
