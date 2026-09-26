"""Rate every brief on generic news dimensions, blind to the labels.

    python3 -m benchmarks.editorial_decisions.rate_briefs --dry-run

One call per morning asks a model to score each brief from 0 to 4 on the
DIMENSIONS below. The ratings feed fit_preferences.py, which learns how much
weight the labeler gives each dimension. The rater never sees a label, a probe
name, or an axis, so the ratings cannot leak his picks.

The dimensions are hypotheses about what could drive an editor's choice. They
are generic on purpose: no topic is named, so whatever weights come out are
rules that carry to mornings about anything.
"""
import argparse
import json
import os
import re
from datetime import datetime

from benchmarks.editorial_decisions.hypotheticals import load_hypotheticals
from benchmarks.editorial_decisions.label_server import CASES_PATH, _read_jsonl
from benchmarks.editorial_decisions.run_lead_pick import OUT_DIR, call_tier, capture_usage, usage_cost, LAST_USAGE

DIMENSIONS = {
    "reach": "How many Americans' daily lives does this change directly? 0 = almost none; 1 = thousands; 2 = hundreds of thousands; 3 = millions; 4 = tens of millions or more.",
    "tangible": "How concrete and near is the effect on people's health, money, safety, or coverage? 0 = none or purely procedural; 2 = indirect or months away; 4 = people feel it now (a bill, a benefit, a treatment, a recall, an outage).",
    "completed": "Has it actually happened? 0 = talk, a proposal, a forecast, a meeting scheduled; 2 = formally begun or officially scheduled; 4 = done and in effect (signed, ruled, approved, struck, measured).",
    "first": "Is it a genuine first or a threshold crossed, stated as a checkable fact? 0 = no; 2 = the largest or first in some years; 4 = the first ever of its kind.",
    "power": "How much governmental or military power is exercised or changed? 0 = none; 1 = local; 2 = a federal agency or a single state; 3 = Congress, the presidency, or the Supreme Court acting; 4 = war and peace, or the constitutional order.",
    "lasting": "Will this still matter in five years? 0 = forgotten in days; 2 = matters for months; 4 = changes the long-run path of a field, an institution, or how people live.",
    "turn": "For a story the audience has heard before (tagged UPDATE, DEVELOPMENT, or SIDE-COVERED): 0 = more of the same pattern; 2 = a notable new step; 4 = the story has changed in kind (started, stopped, signed, collapsed, first harm to the public). For an untagged, fresh story give 4.",
    "verified": "How solid is the sourcing in the brief? 0 = unverified or one anonymous source; 2 = one credible outlet or a company's own statement; 4 = official data, a ruling, a published trial, or several independent outlets.",
    "california": "How directly does it affect people in California in particular? 0 = not specifically; 2 = Californians among those affected; 4 = a California event affecting millions of residents.",
    "ai": "Is artificial intelligence the subject? 0 = no; 2 = AI business, politics, or statements; 4 = verified AI capability, harm, safety failure, or measured effect on jobs.",
    "abroad": "0 = a domestic US story; 2 = abroad, with US forces, citizens, or prices directly involved; 4 = abroad, with no direct US involvement.",
    "uncertainty": "Does it leave a critical system in serious doubt (who controls weapons, whether payments work, whether a government functions)? 0 = no; 4 = yes, and institutions must prepare for several outcomes.",
}

SYSTEM = (
    "You are rating news briefs for a research study. For every brief in the document, give an integer from 0 to 4 on each "
    "dimension, using the anchors exactly as written. Rate each brief on its own facts; do not rank the briefs against each other "
    "and do not decide which story is most important. Tags in square brackets at the start of a headline say whether the audience "
    "has heard the story before.\n\nDimensions:\n"
    + "\n".join("- {}: {}".format(k, v) for k, v in DIMENSIONS.items())
    + "\n\nReply with JSON only, no prose: an object whose keys are the brief numbers as strings and whose values are objects with "
    "exactly these keys: " + ", ".join(DIMENSIONS) + "."
)


def all_cases():
    return _read_jsonl(CASES_PATH) + load_hypotheticals()


def parse_ratings(text, n_briefs):
    """Return {brief_index: {dimension: int}} or raise ValueError."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    data = json.loads(match.group(0))
    out = {}
    for key, dims in data.items():
        index = int(re.sub(r"\D", "", str(key)))
        out[index] = {d: max(0, min(4, int(round(float(dims[d]))))) for d in DIMENSIONS}
    missing = [i for i in range(1, n_briefs + 1) if i not in out]
    if missing:
        raise ValueError("briefs not rated: {}".format(missing))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--repeats", type=int, default=3, help="ratings are averaged over repeats to cut noise")
    ap.add_argument("--sets", nargs="+", default=None, help="only these sets (real, invented, ...)")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = [c for c in all_cases() if not args.sets or (c.get("set") or "real") in args.sets]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cases = cases[i::n]
    print("{} mornings x {} repeats = {} calls on tier '{}'".format(len(cases), args.repeats, len(cases) * args.repeats, args.mode))
    if args.dry_run:
        return

    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    capture_usage()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "ratings_{}{}_{}.jsonl".format(
        args.mode, ("_s" + args.shard.replace("/", "of")) if args.shard else "", datetime.now().strftime("%Y%m%d_%H%M%S")))
    cost = 0.0
    for case in cases:
        for rep in range(args.repeats):
            row = {"case_id": case["case_id"], "set": case.get("set") or "real", "mode": args.mode, "repeat": rep, "error": None}
            for attempt in (1, 2):  # one retry on a malformed reply
                LAST_USAGE.clear()
                try:
                    reply = call_tier(args.mode, case["user_prompt"], SYSTEM)
                    cost += usage_cost(LAST_USAGE)
                    row["ratings"] = {str(k): v for k, v in parse_ratings(reply, len(case["briefs"])).items()}
                    row["error"] = None
                    break
                except Exception as e:
                    row["error"] = "{}: {}".format(type(e).__name__, e)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("{:<48} rep {} {}".format(case["case_id"][:48], rep, "ok" if not row["error"] else row["error"][:80]))
    print("measured cost ${:.3f} | wrote {}".format(cost, out))


if __name__ == "__main__":
    main()
