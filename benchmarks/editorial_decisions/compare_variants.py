"""Compare prompt variants on saved runs. Makes no calls.

    python3 -m benchmarks.editorial_decisions.compare_variants --lead A.jsonl B.jsonl --second C.jsonl D.jsonl

Rows are re-scored against the current labels, so a label edit or a matcher fix
is picked up without paying for new calls. Rows without a 'variant' field
predate the flag and count as production. Runs per cell differ (Opus is run
once, Gemma three times), so rates are shown with their run counts.
"""
import argparse
import collections
import json

from benchmarks.editorial_decisions.run_lead_pick import labeled_cases, parse_pick, score_pick, story_key
from benchmarks.editorial_decisions.run_second_pick import score_pair


def _load(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.endswith(".json"):  # the merged re-score file: {"lead": [...], "second": [...]}
            data = json.loads(text)
            rows += data.get("lead", []) + data.get("second", [])
        else:
            rows += [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows


def merged_labels():
    """Both labeling passes as one label. The second pass's lead and also-fine
    stories join the acceptable set, and either pass's lead counts as exact. A
    model should not be marked wrong for picking what the labeler himself picked
    on another day."""
    from benchmarks.editorial_decisions.label_server import RELABELS_PATH, load_labels
    again = load_labels(RELABELS_PATH)
    out = {}
    for case, label in labeled_cases():
        label = dict(label)
        b = again.get(case["case_id"])
        if b:
            extra = [i for i in [b.get("lead_index")] + list(b.get("acceptable_indexes") or []) if i is not None]
            label["acceptable_indexes"] = sorted(set(label.get("acceptable_indexes") or []) | set(extra))
            label["lead_alternates"] = [b["lead_index"]] if b.get("lead_index") is not None else []
        out[case["case_id"]] = (case, label)
    return out


def rescore(lead_rows, second_rows, modes, union=False):
    pairs = merged_labels() if union else {c["case_id"]: (c, l) for c, l in labeled_cases()}
    lead, second = [], []
    for r in lead_rows:
        if "second_pick_index" in r or r["case_id"] not in pairs or r["mode"] not in modes:
            continue
        case, label = pairs[r["case_id"]]
        pick = parse_pick(r.get("essay") or "", case["briefs"])[0]
        row = {"variant": r.get("variant") or "production", "mode": r["mode"], "case_id": r["case_id"],
               "set": case.get("set") or "real", "pick": pick}
        row.update(score_pick(pick, case, label))
        lead.append(row)
    for r in second_rows:
        if "second_pick_index" not in r or r["case_id"] not in pairs or r["mode"] not in modes:
            continue
        case, label = pairs[r["case_id"]]
        if "second_index" not in label:
            continue
        pick = parse_pick(r.get("essay") or "", case["briefs"])[0]
        row = {"variant": r.get("variant") or "production", "mode": r["mode"], "case_id": r["case_id"],
               "set": case.get("set") or "real", "lead_pick": r["lead_pick_index"], "pick": pick}
        row.update(score_pair(r["lead_pick_index"], pick, label, story_key(case)))
        second.append(row)
    return lead, second


def table(lead, second):
    cells = collections.defaultdict(lambda: collections.Counter())
    for r in lead:
        c = cells[(r["mode"], r["set"], r["variant"])]
        c["n"] += 1
        c["exact"] += int(r["exact"])
        c["acceptable"] += int(r["acceptable"])
    for r in second:
        c = cells[(r["mode"], r["set"], r["variant"])]
        c["n2"] += 1
        c["second_exact"] += int(r["second_exact"])
        c["shared"] += r["pair_overlap"]
        c["size"] += r["pair_size"]
    lines = ["{:<9} {:<14} {:<11} {:>5} {:>12} {:>12} {:>14} {:>18}".format("mode", "set", "variant", "runs", "lead exact", "lead in set", "second exact", "two-slot stories")]
    for key in sorted(cells):
        c = cells[key]
        pct = lambda a, b: "{}/{} ({:.0f}%)".format(a, b, 100.0 * a / b) if b else "-"
        lines.append("{:<9} {:<14} {:<11} {:>5} {:>12} {:>12} {:>14} {:>18}".format(
            key[0], key[1], key[2], c["n"], pct(c["exact"], c["n"]), pct(c["acceptable"], c["n"]),
            pct(c["second_exact"], c["n2"]), pct(c["shared"], c["size"])))
    return "\n".join(lines)


def flips(lead, new_variant="candidate"):
    """Mornings where the majority lead pick changed status between production and `new_variant`."""
    by = collections.defaultdict(list)
    for r in lead:
        by[(r["mode"], r["case_id"], r["variant"])].append(r)

    def status(rows):
        if not rows:
            return None
        ok = sum(r["acceptable"] for r in rows) * 2 > len(rows)
        top = collections.Counter(r["pick"] for r in rows).most_common(1)[0][0]
        return ok, top

    out = []
    for mode, case_id in sorted({(m, c) for m, c, _ in by}):
        old, new = status(by.get((mode, case_id, "production"))), status(by.get((mode, case_id, new_variant)))
        if old and new and old[0] != new[0]:
            out.append("{:<9} {:<46} {} (pick {} -> {})".format(mode, case_id[:46], "GAINED" if new[0] else "LOST", old[1], new[1]))
    return out


def leaderboard(lead, mode, exclude=()):
    """One row per variant, counted by MORNING: a morning counts when most of its
    repeats land in the labeled set. Repeats of one morning mostly agree, so
    mornings, not runs, are the honest sample size. `exclude` drops mornings
    (the few-shot variant quotes some as examples) so every row covers the same ones."""
    by = collections.defaultdict(list)
    for r in lead:
        if r["mode"] == mode and r["case_id"] not in exclude:
            by[(r["variant"], r["case_id"], r["set"])].append(r)
    rows = collections.defaultdict(lambda: collections.Counter())
    for (variant, _case, set_name), runs in by.items():
        ok = sum(r["acceptable"] for r in runs) * 2 > len(runs)
        exact = sum(r["exact"] for r in runs) * 2 > len(runs)
        for key in ("ALL", set_name):
            rows[variant]["n_" + key] += 1
            rows[variant]["ok_" + key] += int(ok)
            rows[variant]["exact_" + key] += int(exact)
    sets = ["ALL", "real", "invented", "invented_hard", "invented_hard2", "holdout1"]
    lines = ["{:<16}".format("variant") + "".join("{:>18}".format(x) for x in sets) + "   (mornings in set / exact lead)"]
    for variant in sorted(rows, key=lambda v: -rows[v]["ok_ALL"]):
        c = rows[variant]
        lines.append("{:<16}".format(variant) + "".join(
            "{:>18}".format("{}/{} | {}".format(c["ok_" + x], c["n_" + x], c["exact_" + x]) if c["n_" + x] else "-") for x in sets))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lead", nargs="+", required=True)
    ap.add_argument("--second", nargs="*", default=[])
    ap.add_argument("--modes", nargs="+", default=["standard", "heavy"])
    ap.add_argument("--union", action="store_true", help="score the lead against both labeling passes combined")
    ap.add_argument("--leaderboard", action="store_true", help="rank variants by mornings, lead slot only")
    ap.add_argument("--new", default="candidate", help="variant to compare against production in the flip list")
    ap.add_argument("--set", dest="set_name", default=None, help="only this set")
    args = ap.parse_args()
    lead, second = rescore(_load(args.lead), _load(args.second), set(args.modes), union=args.union)
    if args.set_name:
        lead = [r for r in lead if r["set"] == args.set_name]
        second = [r for r in second if r["set"] == args.set_name]
    if args.leaderboard:
        from benchmarks.editorial_decisions.candidate_prompts import FEWSHOT_CASES
        for mode in args.modes:
            print("== {}: all labeled mornings ==".format(mode))
            print(leaderboard(lead, mode))
            print("\n== {}: excluding the {} mornings the few-shot prompt quotes ==".format(mode, len(FEWSHOT_CASES)))
            print(leaderboard(lead, mode, exclude=set(FEWSHOT_CASES)))
        return
    print(table(lead, second))
    changed = flips(lead, args.new)
    print("\nlead-slot mornings whose majority pick moved in or out of the labeled set:")
    print("\n".join(changed) if changed else "  none")


if __name__ == "__main__":
    main()
