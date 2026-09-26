"""Re-run only the rows that failed in saved lead-pick outputs, in place. Paid calls.

    python3 -m benchmarks.editorial_decisions.retry_errors outputs/lead_pick_X.jsonl [...]

A network outage leaves rows with an `error` and no essay. This re-asks exactly
those rows with the same tier and prompt variant, then rewrites each file with the
fixed rows in their original positions. Rows that fail again keep their error.
"""
import argparse
import json

from benchmarks.editorial_decisions.candidate_prompts import prompts_for
from benchmarks.editorial_decisions.run_lead_pick import LAST_USAGE, call_tier, capture_usage, labeled_cases, parse_pick, score_pick, unlabeled_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    import newscaster.config as config
    config.init()
    config.LLM_AUDIT_LOG_ENABLED = False
    capture_usage()
    cases = {c["case_id"]: (c, l) for c, l in labeled_cases() + unlabeled_cases()}

    for path in args.files:
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        fixed = still = 0
        for row in rows:
            if not row.get("error"):
                continue
            case, label = cases[row["case_id"]]
            LAST_USAGE.clear()
            try:
                essay = call_tier(row["mode"], case["user_prompt"], prompts_for(row.get("variant") or "production")[0])
            except Exception as e:
                row["error"] = "{}: {}".format(type(e).__name__, e)
                still += 1
                continue
            pick, answer, method = parse_pick(essay, case["briefs"])
            row.update(essay=essay, error=None, answer=answer, parse_method=method, pick_index=pick, usage=dict(LAST_USAGE), retried=True)
            if label is not None:
                row.update(score_pick(pick, case, label))
            fixed += 1
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("{}: fixed {}, still failing {}".format(path.split("/")[-1], fixed, still))


if __name__ == "__main__":
    main()
