#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.web_search.web_bench_lib import iter_jsonl, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize discover_then_fetch URL-discovery metrics.")
    parser.add_argument("run", type=Path, help="JSONL file produced by run_web_bench.py.")
    parser.add_argument("--out", type=Path, help="Discovery summary JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discoveries: Dict[str, Dict[str, Any]] = {}
    answer_cost = 0.0
    answer_records = 0

    for record in iter_jsonl(args.run):
        answer_records += 1
        answer_cost += float((record.get("usage") or {}).get("cost") or 0)
        discovery = record.get("discovery")
        if discovery and discovery.get("task_id") not in discoveries:
            discoveries[discovery["task_id"]] = discovery

    tasks = list(discoveries.values())
    total = len(tasks)
    discovery_cost = sum(float((task.get("usage") or {}).get("cost") or 0) for task in tasks)
    discovery_latency = sum(float(task.get("latency_seconds") or 0) for task in tasks)

    def rate(field: str) -> float:
        if not total:
            return 0.0
        return round(sum(1 for task in tasks if (task.get("discovery_score") or {}).get(field)) / total, 3)

    fetch_success_rates = [
        float((task.get("discovery_score") or {}).get("fetch_success_rate") or 0)
        for task in tasks
    ]
    summary = {
        "generated_at": utc_now_iso(),
        "run": str(args.run),
        "answer_records": answer_records,
        "discovery_tasks": total,
        "answer_cost": round(answer_cost, 6),
        "shared_discovery_cost": round(discovery_cost, 6),
        "combined_cost": round(answer_cost + discovery_cost, 6),
        "avg_discovery_latency_seconds": round(discovery_latency / total, 2) if total else 0.0,
        "exact_preferred_url_recall": rate("exact_preferred_url"),
        "fetched_preferred_url_recall": rate("fetched_preferred_url"),
        "preferred_host_recall": rate("preferred_host"),
        "fetched_preferred_host_recall": rate("fetched_preferred_host"),
        "source_domain_recall": rate("source_domain"),
        "fetched_source_domain_recall": rate("fetched_source_domain"),
        "avg_fetch_success_rate": round(sum(fetch_success_rates) / total, 3) if total else 0.0,
        "tasks": [
            {
                "task_id": task.get("task_id"),
                "candidate_count": (task.get("discovery_score") or {}).get("candidate_count"),
                "fetch_success_count": (task.get("discovery_score") or {}).get("fetch_success_count"),
                "fetch_success_rate": (task.get("discovery_score") or {}).get("fetch_success_rate"),
                "exact_preferred_url": (task.get("discovery_score") or {}).get("exact_preferred_url"),
                "preferred_host": (task.get("discovery_score") or {}).get("preferred_host"),
                "source_domain": (task.get("discovery_score") or {}).get("source_domain"),
                "cost": (task.get("usage") or {}).get("cost"),
                "latency_seconds": task.get("latency_seconds"),
                "candidate_urls": [
                    candidate.get("url") for candidate in task.get("candidates", [])
                ],
                "error": task.get("error"),
            }
            for task in tasks
        ],
    }

    out_path = args.out or args.run.with_suffix(".discovery.json")
    write_json(out_path, summary)
    print(f"Discovery summary: {out_path}")
    print(
        f"preferred_host={summary['preferred_host_recall']:.3f}, "
        f"source_domain={summary['source_domain_recall']:.3f}, "
        f"fetch_success={summary['avg_fetch_success_rate']:.3f}, "
        f"shared_discovery_cost=${summary['shared_discovery_cost']:.6f}, "
        f"combined_cost=${summary['combined_cost']:.6f}"
    )


if __name__ == "__main__":
    main()
