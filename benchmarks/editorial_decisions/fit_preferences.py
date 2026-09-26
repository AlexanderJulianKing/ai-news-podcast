"""Learn how the labeler weighs news dimensions, and test whether it generalizes.

    python3 -m benchmarks.editorial_decisions.fit_preferences --ratings outputs/ratings_*.jsonl

Input: blind 0-4 ratings of every brief (rate_briefs.py) and the labels, both
passes when a relabel exists. Model: each brief gets a score = weights . features,
and the labeler's choice among a morning's briefs is a softmax over scores (a
conditional logit). Each labeling pass contributes two choices: the lead among
all briefs, then the second story among the rest (a rank-ordered, or "exploded",
logit). Weights are fit by gradient ascent with an L2 penalty.

Honesty check: every morning is predicted by a model fit WITHOUT that morning
(leave-one-morning-out). Those out-of-sample predictions are scored exactly as
an LLM's picks are scored, so the numbers compare directly with run_lead_pick.
"""
import argparse
import collections
import glob
import json

import numpy as np

from benchmarks.editorial_decisions.label_server import LABELS_PATH, RELABELS_PATH, load_labels
from benchmarks.editorial_decisions.rate_briefs import DIMENSIONS, all_cases
from benchmarks.editorial_decisions.run_lead_pick import score_pick, story_key
from benchmarks.editorial_decisions.run_second_pick import score_pair

EXTRA = ("outlets", "side_covered", "development")  # features read off the brief, not rated


def load_ratings(paths):
    """Mean rating per (case_id, brief index, dimension) across repeats."""
    sums = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") or "ratings" not in row:
                    continue
                for index, dims in row["ratings"].items():
                    for d, v in dims.items():
                        sums[(row["case_id"], int(index))][d].append(v)
    return {key: {d: float(np.mean(v)) for d, v in dims.items()} for key, dims in sums.items()}


def feature_names(interactions=True):
    names = list(DIMENSIONS) + list(EXTRA)
    if interactions:
        names += ["reach*tangible", "power*completed", "first*verified", "turn*development"]
    return names


def features(case, ratings, interactions=True):
    """Feature matrix for a morning's eligible briefs. Returns (indexes, X)."""
    indexes, rows = [], []
    for b in case["briefs"]:
        r = ratings.get((case["case_id"], b["index"]))
        if r is None or not b["eligible"]:
            continue
        tags = {t["tag"] for t in b["tags"]}
        x = [r[d] / 4.0 for d in DIMENSIONS]
        x += [min(len(b["reported_by"]), 5) / 5.0, float("SIDE-COVERED" in tags), float("DEVELOPMENT" in tags)]
        if interactions:
            x += [r["reach"] * r["tangible"] / 16.0, r["power"] * r["completed"] / 16.0,
                  r["first"] * r["verified"] / 16.0, r["turn"] / 4.0 * float("DEVELOPMENT" in tags)]
        indexes.append(b["index"])
        rows.append(x)
    return indexes, np.array(rows, dtype=float)


def choices_for(case, indexes, passes):
    """(candidate positions, chosen position) pairs: lead among all, then second among the rest."""
    out = []
    key = story_key(case)
    pos = {i: p for p, i in enumerate(indexes)}
    for label in passes:
        lead = label.get("lead_index")
        if lead not in pos:
            continue
        everyone = list(range(len(indexes)))
        out.append((everyone, pos[lead]))
        second = label.get("second_index")
        if second in pos and second != lead:
            rest = [p for p in everyone if key[indexes[p]] != key[lead]]  # drop the lead and its duplicates
            if pos[second] in rest:
                out.append((rest, pos[second]))
    return out


def fit(data, n_features, l2=1.0, steps=1500, lr=0.3):
    """data: list of (X, [(candidates, chosen), ...]). Returns the weight vector."""
    w = np.zeros(n_features)
    n_obs = float(sum(len(ch) for _, ch in data)) or 1.0
    for _ in range(steps):
        grad = -l2 * w / n_obs
        for X, chosen in data:
            for cand, pick in chosen:
                s = X[cand] @ w
                p = np.exp(s - s.max())
                p /= p.sum()
                grad += (X[pick] - p @ X[cand]) / n_obs
        w += lr * grad
    return w


def predict(X, indexes, case, w):
    """Top story, then the best story that is not the same event as the top one."""
    order = np.argsort(-(X @ w))
    key = story_key(case)
    lead = indexes[order[0]]
    second = next((indexes[p] for p in order[1:] if key[indexes[p]] != key[lead]), None)
    return lead, second


def evaluate(cases, ratings, first, relabels, l2, interactions, use_relabels=True):
    rows_in = []
    for case in cases:
        if case["case_id"] not in first:
            continue
        indexes, X = features(case, ratings, interactions)
        if len(indexes) < 2:
            continue
        passes = [first[case["case_id"]]] + ([relabels[case["case_id"]]] if use_relabels and case["case_id"] in relabels else [])
        rows_in.append((case, indexes, X, choices_for(case, indexes, passes)))
    n_features = rows_in[0][2].shape[1]
    results = []
    for held in range(len(rows_in)):
        train = [(X, ch) for j, (_c, _i, X, ch) in enumerate(rows_in) if j != held]
        w = fit(train, n_features, l2=l2)
        case, indexes, X, _ = rows_in[held]
        lead, second = predict(X, indexes, case, w)
        label = first[case["case_id"]]
        row = {"case_id": case["case_id"], "set": case.get("set") or "real", "lead": lead, "second": second}
        row.update(score_pick(lead, case, label))
        if "second_index" in label:
            row.update(score_pair(lead, second, label, story_key(case)))
        results.append(row)
    w_all = fit([(X, ch) for _c, _i, X, ch in rows_in], n_features, l2=l2)
    return results, w_all


def summarize(results):
    two = [r for r in results if "pair_size" in r]
    return {"n": len(results), "exact": sum(r["exact"] for r in results), "in_set": sum(r["acceptable"] for r in results),
            "shared": sum(r["pair_overlap"] for r in two), "size": sum(r["pair_size"] for r in two),
            "both": sum(1 for r in two if r["pair_overlap"] == r["pair_size"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratings", nargs="+", required=True)
    ap.add_argument("--l2", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--sets", nargs="+", default=["real", "invented", "invented_hard", "invented_hard2"])
    args = ap.parse_args()

    paths = [p for pattern in args.ratings for p in glob.glob(pattern)]
    ratings = load_ratings(paths)
    first, relabels = load_labels(LABELS_PATH), load_labels(RELABELS_PATH)
    cases = [c for c in all_cases() if (c.get("set") or "real") in args.sets]
    print("{} rated briefs from {} file(s); {} labeled mornings; {} relabeled".format(
        len(ratings), len(paths), sum(c["case_id"] in first for c in cases), sum(c["case_id"] in relabels for c in cases)))

    print("\nLeave-one-morning-out, scored against the first-pass labels like any model:")
    print("{:<34} {:>11} {:>13} {:>20} {:>12}".format("fit", "exact lead", "lead in set", "two stories covered", "both right"))
    best = None
    for interactions in (False, True):
        for l2 in args.l2:
            results, w = evaluate(cases, ratings, first, relabels, l2, interactions)
            s = summarize(results)
            name = "{} l2={}".format("with interactions" if interactions else "main effects only", l2)
            print("{:<34} {:>8}/{:<2} {:>10}/{:<2} {:>13}/{:<3} ({:.0f}%) {:>8}/{}".format(
                name, s["exact"], s["n"], s["in_set"], s["n"], s["shared"], s["size"], 100.0 * s["shared"] / s["size"], s["both"], s["n"]))
            if best is None or (s["shared"], s["in_set"]) > (best[0]["shared"], best[0]["in_set"]):
                best = (s, w, interactions, l2, results)

    s, w, interactions, l2, results = best
    print("\nWeights of the best row ({}, l2={}), fit on every morning. A weight is the pull of moving that".format(
        "with interactions" if interactions else "main effects only", l2))
    print("dimension from its lowest to its highest rating; positive pulls a story toward the top.")
    for name, value in sorted(zip(feature_names(interactions), w), key=lambda t: -abs(t[1])):
        print("  {:<18} {:+.2f}".format(name, value))
    by_set = collections.defaultdict(list)
    for r in results:
        by_set[r["set"]].append(r)
    print("\nby set (out-of-sample): " + " | ".join("{} {}/{} in set".format(k, sum(r["acceptable"] for r in v), len(v)) for k, v in sorted(by_set.items())))
    misses = [r["case_id"] for r in results if not r["acceptable"]]
    print("mornings still missed: " + ", ".join(m[5:40] for m in misses))


if __name__ == "__main__":
    main()
