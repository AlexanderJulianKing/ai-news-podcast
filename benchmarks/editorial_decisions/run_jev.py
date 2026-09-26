"""Score TypeSafe's Jev decision model on the labeled lead and second picks.

    python3 -m benchmarks.editorial_decisions.run_jev --dry-run

Jev is not a chat model. It takes a state plus typed questions on OpenRouter's
System One endpoint and returns a choice with a probability per option, so it
needs its own runner. Each brief is one option of a `choice` question, and the
Tier 3 prompt under test goes in as the question's instructions. The second
slot is asked as a follow-up call with Jev's own lead removed from the options,
mirroring how the pipeline asks it.

Because Jev returns probabilities, two extra numbers are reported: the
probability it put on the labeler's acceptable set, and the rank it gave the
labeler's lead.
"""
import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

from benchmarks.editorial_decisions.candidate_prompts import prompts_for
from benchmarks.editorial_decisions.run_lead_pick import OUT_DIR, labeled_cases, score_pick, story_key
from benchmarks.editorial_decisions.run_second_pick import score_pair

ENDPOINT = "https://openrouter.ai/api/v1/systemone"
MODEL = "typesafe/jev-1.13"


def ask(api_key, document, instructions, options):
    """One choice question. Returns (chosen_key, probabilities, usage) or raises."""
    body = {"model": MODEL, "state": {"research_briefs": document},
            "questions": {"pick": {"type": "choice", "instructions": instructions, "criteria": options}}}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError("HTTP {}: {}".format(e.code, e.read().decode("utf-8")[:300]))
    answer = data["answers"]["pick"]
    return answer["choice"], answer.get("probabilities") or {}, data.get("usage") or {}


def _index(key):
    return int(key.split("_")[1]) if key else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="production")
    ap.add_argument("--set", dest="set_name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lead_prompt, second_template = prompts_for(args.variant)
    pairs = [(c, l) for c, l in labeled_cases() if not args.set_name or (c.get("set") or "real") == args.set_name]
    print("{} labeled mornings, up to {} calls to {}".format(len(pairs), 2 * len(pairs), MODEL))
    if args.dry_run:
        return

    import newscaster.config as config
    config.init()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "jev_{}_{}.jsonl".format(args.variant, datetime.now().strftime("%Y%m%d_%H%M%S")))
    rows, cost = [], 0.0
    for case, label in pairs:
        options = {"brief_{}".format(b["index"]): b["raw_headline"] for b in case["briefs"]}
        row = {"mode": "jev", "model": MODEL, "variant": args.variant, "case_id": case["case_id"],
               "set": case.get("set") or "real", "synthetic": bool(case.get("synthetic")), "error": None}
        try:
            choice, probs, usage = ask(config.OPENROUTER_API_KEY, case["user_prompt"], lead_prompt, options)
            cost += usage.get("cost") or 0
            lead = _index(choice)
            key = story_key(case)
            ok = {key[i] for i in (label.get("acceptable_indexes") or [])}
            if label.get("lead_index") is not None:
                ok.add(key[label["lead_index"]])
            ranked = sorted(probs, key=lambda k: -probs[k])
            row.update(pick_index=lead, probabilities=probs,
                       p_on_labeled_set=round(sum(p for k, p in probs.items() if key[_index(k)] in ok), 3),
                       rank_of_labeled_lead=(ranked.index("brief_{}".format(label["lead_index"])) + 1
                                             if label.get("lead_index") is not None and "brief_{}".format(label["lead_index"]) in ranked else None))
            row.update(score_pick(lead, case, label))

            if "second_index" in label:
                lead_headline = next(b["headline"] for b in case["briefs"] if b["index"] == lead)
                remaining = {k: v for k, v in options.items() if _index(k) != lead}
                choice2, probs2, usage2 = ask(config.OPENROUTER_API_KEY, case["user_prompt"],
                                              second_template.format(excluded_headline=lead_headline), remaining)
                cost += usage2.get("cost") or 0
                second = _index(choice2)
                row.update(second_pick_index=second, second_probabilities=probs2)
                row.update(score_pair(lead, second, label, key))
        except Exception as e:  # a failed call is a result, not a crash
            row["error"] = "{}: {}".format(type(e).__name__, e)
        rows.append(row)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("{:<46} lead={} (label {}) {} | second={} (label {}) | p(labeled set)={}".format(
            case["case_id"][:46], row.get("pick_index"), label.get("lead_index"),
            "EXACT" if row.get("exact") else "ok" if row.get("acceptable") else "BARRED" if row.get("barred") else "miss",
            row.get("second_pick_index"), label.get("second_index"), row.get("p_on_labeled_set")))

    good = [r for r in rows if not r["error"]]
    print("\n{:<16} {:>3} {:>7} {:>8} {:>7} {:>16} {:>18}".format("set", "n", "exact", "in set", "barred", "stories shared", "mean p(label set)"))
    for name in ["real", "invented", "invented_hard", "invented_hard2", "ALL"]:
        sub = [r for r in good if name == "ALL" or r["set"] == name]
        if not sub:
            continue
        two = [r for r in sub if "pair_size" in r]
        print("{:<16} {:>3} {:>7} {:>8} {:>7} {:>16} {:>18.2f}".format(
            name, len(sub), sum(r["exact"] for r in sub), sum(r["acceptable"] for r in sub), sum(r["barred"] for r in sub),
            "{}/{}".format(sum(r["pair_overlap"] for r in two), sum(r["pair_size"] for r in two)),
            sum(r["p_on_labeled_set"] for r in sub) / len(sub)))
    print("\nerrors: {} | measured cost: ${:.4f} | wrote {}".format(len(rows) - len(good), cost, out))


if __name__ == "__main__":
    main()
