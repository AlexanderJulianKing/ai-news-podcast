"""How often does the labeler agree with himself? Makes no calls.

    python3 -m benchmarks.editorial_decisions.self_agreement

Compares the blind second pass (data/lead_pick_relabels.jsonl, made with
`label_server --relabel`) with the first labels, using the same scoring a model
gets: is the second-pass lead inside the first-pass labeled set, and how many of
the two stories are shared. A model cannot be expected to agree with the
labeler more often than he agrees with himself.
"""
import collections

from benchmarks.editorial_decisions.label_server import LABELS_PATH, RELABELS_PATH, load_labels
from benchmarks.editorial_decisions.run_lead_pick import labeled_cases, score_pick, story_key
from benchmarks.editorial_decisions.run_second_pick import score_pair


def main():
    first, second = load_labels(LABELS_PATH), load_labels(RELABELS_PATH)
    cases = {c["case_id"]: c for c, _ in labeled_cases()}
    rows = []
    for case_id, again in second.items():
        if case_id not in first or case_id not in cases:
            continue
        case, label = cases[case_id], first[case_id]
        row = {"confidence": label.get("confidence"), "set": case.get("set") or "real"}
        row.update(score_pick(again.get("lead_index"), case, label))
        if "second_index" in label and "second_index" in again:
            row.update(score_pair(again.get("lead_index"), again.get("second_index"), label, story_key(case)))
        rows.append(row)
    if not rows:
        print("No relabels yet. Run: python3 -m benchmarks.editorial_decisions.label_server --relabel --port 8918")
        return
    print("{} mornings relabeled".format(len(rows)))
    for name, key in (("all", None), ("by first-pass confidence", "confidence"), ("by set", "set")):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r[key] if key else "all"].append(r)
        print("\n" + name)
        for g, rs in sorted(groups.items()):
            two = [r for r in rs if "pair_size" in r]
            print("  {:<16} same lead {:>2}/{:<2}  lead in first-pass set {:>2}/{:<2}  two stories shared {}/{}".format(
                str(g), sum(r["exact"] for r in rs), len(rs), sum(r["acceptable"] for r in rs), len(rs),
                sum(r["pair_overlap"] for r in two), sum(r["pair_size"] for r in two)))


if __name__ == "__main__":
    main()
