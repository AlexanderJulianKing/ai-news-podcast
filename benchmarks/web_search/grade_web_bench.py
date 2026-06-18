#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.web_search.web_bench_lib import (
    DEFAULT_TASKS,
    html_report,
    iter_jsonl,
    load_json,
    score_result,
    summarize_scores,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a web benchmark run and generate a compact HTML report."
    )
    parser.add_argument("run", type=Path, help="JSONL file produced by run_web_bench.py.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, help="Scores JSON output path.")
    parser.add_argument("--html", type=Path, help="HTML report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_doc = load_json(args.tasks)
    tasks_by_id: Dict[str, Dict[str, Any]] = {
        task["id"]: task for task in tasks_doc.get("tasks", [])
    }
    result_records = list(iter_jsonl(args.run))
    score_records: List[Dict[str, Any]] = []

    for result in result_records:
        task_id = result.get("task_id")
        task = tasks_by_id.get(task_id)
        if not task:
            score_records.append(
                {
                    "task_id": task_id,
                    "model_id": result.get("model_id"),
                    "model_label": result.get("model_label"),
                    "score": 0,
                    "max_score": 0,
                    "percent": 0,
                    "failed": True,
                    "details": [],
                    "penalties": [],
                    "error": f"Unknown task id: {task_id}",
                }
            )
            continue

        score = score_result(task, result)
        score["usage"] = result.get("usage") or {}
        score["latency_seconds"] = result.get("latency_seconds") or 0
        score["task_category"] = task.get("category")
        score_records.append(score)

    summary = summarize_scores(score_records)
    score_doc = {
        "generated_at": utc_now_iso(),
        "run": str(args.run),
        "tasks": str(args.tasks),
        "summary": summary,
        "scores": score_records,
    }

    out_path = args.out or args.run.with_suffix(".scores.json")
    html_path = args.html or args.run.with_suffix(".report.html")
    write_json(out_path, score_doc)
    html_path.write_text(
        html_report(args.run, score_doc, result_records, score_records),
        encoding="utf-8",
    )

    print(f"Scores: {out_path}")
    print(f"Report: {html_path}")
    for model in summary.get("models", []):
        print(
            f"{model['model_label']}: {model['percent']:.2f}% "
            f"({model['score']:.2f}/{model['max_score']:.2f}), "
            f"failures={model['failures']}, cost=${model['cost']:.6f}"
        )


if __name__ == "__main__":
    main()
