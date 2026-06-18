#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.web_search.web_bench_lib import (
    common_selection_args,
    append_jsonl,
    build_evidence_contract_payload,
    build_discovery_payload,
    build_controlled_payload,
    build_openrouter_payload,
    default_run_path,
    discovery_candidates,
    evidence_annotations,
    extract_openrouter_result,
    federal_reserve_statement_candidates,
    filter_validated_evidence,
    fetch_discovered_evidence,
    fetch_controlled_evidence,
    iter_jsonl,
    legistar_calendar_candidates,
    load_json,
    load_openrouter_key,
    nearby_source_candidates,
    nhc_advisory_archive_candidates,
    noaa_hurricane_outlook_candidates,
    parse_evidence_contract,
    score_discovery,
    select_models,
    select_tasks,
    source_hunter_extra_context,
    utc_now_iso,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a newsroom web-search benchmark through OpenRouter's web plugin."
    )
    common_selection_args(parser)
    parser.add_argument("--out", type=Path, default=None, help="JSONL output path.")
    parser.add_argument("--engine", default="parallel", choices=["exa", "parallel", "perplexity"])
    parser.add_argument(
        "--strategy",
        default="web_plugin",
        choices=["web_plugin", "controlled_fetch", "discover_then_fetch", "source_hunter"],
        help=(
            "web_plugin uses OpenRouter's web plugin; controlled_fetch fetches preferred sources first; "
            "discover_then_fetch discovers URLs with one shared web-plugin scout call, fetches them, then answers without web; "
            "source_hunter loops search/fetch/validate; source-specific resolvers are opt-in."
        ),
    )
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--max-source-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--discovery-model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--discovery-max-tokens", type=int, default=900)
    parser.add_argument(
        "--generate-evidence-contracts",
        action="store_true",
        help="Use a cheap model once per task to generate the source-validation evidence contract.",
    )
    parser.add_argument("--contract-model", default="google/gemma-4-31b-it")
    parser.add_argument("--contract-max-tokens", type=int, default=900)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--max-hunt-iterations", type=int, default=3)
    parser.add_argument("--nearby-source-limit", type=int, default=5)
    parser.add_argument("--nearby-source-depth", type=int, default=4)
    parser.add_argument(
        "--enable-source-resolvers",
        action="store_true",
        help="Opt in to source-system adapters such as Legistar, Federal Reserve, NOAA, and NHC resolvers.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds to sleep between calls.")
    parser.add_argument("--execute", action="store_true", help="Actually spend API calls.")
    parser.add_argument("--resume", action="store_true", help="Skip model/task pairs already present in --out.")
    parser.add_argument("--keys", type=Path, help="Optional keys.txt path.")
    parser.add_argument(
        "--no-reasoning-retry",
        action="store_true",
        help="Do not retry without reasoning if a model rejects reasoning_effort.",
    )
    return parser.parse_args()


def already_done(path: Optional[Path]) -> set[Tuple[str, str]]:
    if not path or not path.exists():
        return set()
    pairs = set()
    for record in iter_jsonl(path):
        if not record.get("error"):
            pairs.add((record.get("task_id"), record.get("model_id")))
    return pairs


def post_openrouter(
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
    retry_without_reasoning: bool,
) -> tuple[str, list[dict], dict, Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "newscaster-web-bench",
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    if response.ok:
        data = response.json()
        content, annotations, usage = extract_openrouter_result(data)
        return content, annotations, usage, {"reasoning_removed": False}

    body = response.text[:2000]
    if (
        retry_without_reasoning
        and response.status_code == 400
        and "reasoning" in payload
    ):
        retry_payload = dict(payload)
        retry_payload.pop("reasoning", None)
        retry_response = requests.post(OPENROUTER_URL, headers=headers, json=retry_payload, timeout=timeout)
        if retry_response.ok:
            data = retry_response.json()
            content, annotations, usage = extract_openrouter_result(data)
            return content, annotations, usage, {"reasoning_removed": True, "first_error": body}
        body = body + "\n\nRetry without reasoning failed:\n" + retry_response.text[:2000]
        raise RuntimeError(f"OpenRouter HTTP {retry_response.status_code}: {body}")

    raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {body}")


def main() -> None:
    args = parse_args()
    tasks_doc = load_json(args.tasks)
    models_doc = load_json(args.models)
    tasks = select_tasks(tasks_doc, args.task_ids, args.limit)
    models = select_models(models_doc, args.model_ids)
    out_path = args.out or default_run_path()
    pair_count = len(tasks) * len(models)

    print(f"Tasks: {len(tasks)}")
    print(f"Models: {len(models)}")
    print(f"Calls: {pair_count}")
    print(f"Strategy: {args.strategy}")
    if args.strategy == "web_plugin":
        print(f"Engine: {args.engine}, max_results={args.max_results}, max_tokens={args.max_tokens}")
    elif args.strategy == "controlled_fetch":
        print(f"Controlled fetch: max_source_chars={args.max_source_chars}, max_tokens={args.max_tokens}")
    elif args.strategy == "discover_then_fetch":
        print(
            f"Discovery: model={args.discovery_model}, engine={args.engine}, "
            f"max_results={args.max_results}, candidate_limit={args.candidate_limit}; "
            f"controlled answer max_source_chars={args.max_source_chars}, max_tokens={args.max_tokens}"
        )
    else:
        print(
            f"Source hunter: model={args.discovery_model}, engine={args.engine}, "
            f"max_results={args.max_results}, candidate_limit={args.candidate_limit}, "
            f"max_hunt_iterations={args.max_hunt_iterations}; "
            f"controlled answer max_source_chars={args.max_source_chars}, max_tokens={args.max_tokens}"
        )
    if args.generate_evidence_contracts:
        print(f"Evidence contracts: model={args.contract_model}, max_tokens={args.contract_max_tokens}")
    print(f"Output: {out_path}")
    for model in models:
        print(f"- {model['id']}: {model['model']} ({model.get('label', model['id'])})")

    if not args.execute:
        print("\nDry plan only. Add --execute to run paid API calls.")
        return

    api_key = load_openrouter_key(args.keys)
    done = already_done(out_path) if args.resume else set()
    run_started = utc_now_iso()
    evidence_by_task: Dict[str, Dict[str, Any]] = {}
    discovery_by_task: Dict[str, Dict[str, Any]] = {}
    source_hunt_by_task: Dict[str, Dict[str, Any]] = {}
    contract_by_task: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        active_task = task
        if args.generate_evidence_contracts and task["id"] not in contract_by_task:
            print(f"CONTRACT {task['id']}")
            contract_record: Dict[str, Any] = {
                "timestamp": utc_now_iso(),
                "task_id": task["id"],
                "model": args.contract_model,
            }
            try:
                contract_payload = build_evidence_contract_payload(
                    task,
                    tasks_doc.get("as_of", "unknown"),
                    args.contract_model,
                    args.contract_max_tokens,
                )
                contract_content, _, contract_usage, contract_metadata = post_openrouter(
                    api_key,
                    contract_payload,
                    args.timeout,
                    retry_without_reasoning=not args.no_reasoning_retry,
                )
                contract_record.update(
                    {
                        "content": contract_content,
                        "contract": parse_evidence_contract(contract_content),
                        "usage": contract_usage,
                        "metadata": contract_metadata,
                    }
                )
            except Exception as exc:
                contract_record.update(
                    {
                        "error": str(exc),
                        "contract": {"required_slots": [], "source_preferences": [], "reject_if": []},
                        "usage": {},
                        "metadata": {},
                    }
                )
                print(f"CONTRACT ERROR {task['id']}: {exc}")
            contract_by_task[task["id"]] = contract_record
        if task["id"] in contract_by_task:
            active_task = dict(task)
            active_task["evidence_contract"] = contract_by_task[task["id"]].get("contract") or {}

        if args.strategy == "controlled_fetch" and task["id"] not in evidence_by_task:
            print(f"FETCH {task['id']}")
            evidence_by_task[task["id"]] = fetch_controlled_evidence(
                active_task,
                max_source_chars=args.max_source_chars,
                timeout=args.timeout,
            )
        elif args.strategy == "discover_then_fetch" and task["id"] not in evidence_by_task:
            print(f"DISCOVER {task['id']}")
            discovery_payload = build_discovery_payload(
                active_task,
                tasks_doc.get("as_of", "unknown"),
                args.engine,
                args.max_results,
                args.discovery_max_tokens,
                args.discovery_model,
            )
            discovery_started = time.perf_counter()
            discovery_record: Dict[str, Any] = {
                "timestamp": utc_now_iso(),
                "task_id": task["id"],
                "model": args.discovery_model,
                "engine": args.engine,
                "max_results": args.max_results,
                "candidate_limit": args.candidate_limit,
            }
            try:
                discovery_content, discovery_annotations, discovery_usage, discovery_metadata = post_openrouter(
                    api_key,
                    discovery_payload,
                    args.timeout,
                    retry_without_reasoning=not args.no_reasoning_retry,
                )
                candidates = discovery_candidates(
                    discovery_content,
                    discovery_annotations,
                    limit=args.candidate_limit,
                )
                evidence = fetch_discovered_evidence(
                    active_task,
                    candidates,
                    max_source_chars=args.max_source_chars,
                    timeout=args.timeout,
                )
                discovery_record.update(
                    {
                        "content": discovery_content,
                        "annotations": discovery_annotations,
                        "usage": discovery_usage,
                        "metadata": discovery_metadata,
                        "candidates": candidates,
                    }
                )
                evidence_by_task[task["id"]] = evidence
            except Exception as exc:
                discovery_record.update(
                    {
                        "error": str(exc),
                        "annotations": [],
                        "usage": {},
                        "metadata": {},
                        "candidates": [],
                    }
                )
                evidence_by_task[task["id"]] = {"sources": []}
                print(f"DISCOVERY ERROR {task['id']}: {exc}")
            finally:
                discovery_record["latency_seconds"] = round(time.perf_counter() - discovery_started, 3)
                discovery_record["discovery_score"] = score_discovery(active_task, evidence_by_task[task["id"]])
                discovery_by_task[task["id"]] = discovery_record
        elif args.strategy == "source_hunter" and task["id"] not in evidence_by_task:
            print(f"HUNT {task['id']}")
            hunt_started = time.perf_counter()
            hunt_state: Dict[str, Any] = {
                "task_id": task["id"],
                "started_at": utc_now_iso(),
                "status": "running",
                "attempts": [],
                "validated_sources": [],
                "rejected_sources": [],
                "resolver_used": False,
                "usage": {"cost": 0.0},
            }
            seen_candidates = set()

            def _record_usage(usage: Dict[str, Any]) -> None:
                hunt_state["usage"]["cost"] += float((usage or {}).get("cost") or 0)

            for iteration in range(args.max_hunt_iterations):
                extra_context = source_hunter_extra_context(hunt_state, active_task)
                discovery_payload = build_discovery_payload(
                    active_task,
                    tasks_doc.get("as_of", "unknown"),
                    args.engine,
                    args.max_results,
                    args.discovery_max_tokens,
                    args.discovery_model,
                    extra_context=extra_context,
                )
                attempt: Dict[str, Any] = {
                    "iteration": iteration + 1,
                    "timestamp": utc_now_iso(),
                    "extra_context": extra_context,
                    "resolver_candidates": [],
                }
                try:
                    discovery_content, discovery_annotations, discovery_usage, discovery_metadata = post_openrouter(
                        api_key,
                        discovery_payload,
                        args.timeout,
                        retry_without_reasoning=not args.no_reasoning_retry,
                    )
                    _record_usage(discovery_usage)
                    candidates = discovery_candidates(
                        discovery_content,
                        discovery_annotations,
                        limit=args.candidate_limit,
                    )
                    attempt.update(
                        {
                            "content": discovery_content,
                            "annotations": discovery_annotations,
                            "usage": discovery_usage,
                            "metadata": discovery_metadata,
                            "candidates": candidates,
                        }
                    )
                except Exception as exc:
                    attempt.update(
                        {
                            "error": str(exc),
                            "annotations": [],
                            "usage": {},
                            "metadata": {},
                            "candidates": [],
                        }
                    )
                    candidates = []

                new_candidates = []
                for candidate in candidates:
                    key = candidate.get("canonical_url") or candidate.get("url")
                    if key in seen_candidates:
                        continue
                    seen_candidates.add(key)
                    new_candidates.append(candidate)

                if new_candidates:
                    raw_evidence = fetch_discovered_evidence(
                        active_task,
                        new_candidates,
                        max_source_chars=args.max_source_chars,
                        timeout=args.timeout,
                    )
                    validation = filter_validated_evidence(active_task, raw_evidence)
                    hunt_state["validated_sources"].extend(validation["sources"])
                    hunt_state["rejected_sources"].extend(validation["rejected_sources"])
                    attempt["fetched_evidence"] = raw_evidence
                    attempt["validated_count"] = len(validation["sources"])
                    attempt["rejected_count"] = len(validation["rejected_sources"])

                nearby_frontier = validation["rejected_sources"] if new_candidates else []
                attempt["nearby_expansions"] = []
                for nearby_depth in range(max(0, args.nearby_source_depth)):
                    if hunt_state["validated_sources"] or not nearby_frontier:
                        break
                    nearby_candidates = nearby_source_candidates(
                        active_task,
                        nearby_frontier,
                        seen_candidates,
                        limit=args.nearby_source_limit,
                    )
                    nearby_new = []
                    for candidate in nearby_candidates:
                        key = candidate.get("canonical_url") or candidate.get("url")
                        if key in seen_candidates:
                            continue
                        seen_candidates.add(key)
                        nearby_new.append(candidate)
                    nearby_record: Dict[str, Any] = {
                        "depth": nearby_depth + 1,
                        "candidates": nearby_new,
                    }
                    if nearby_new:
                        nearby_evidence = fetch_discovered_evidence(
                            active_task,
                            nearby_new,
                            max_source_chars=args.max_source_chars,
                            timeout=args.timeout,
                        )
                        nearby_validation = filter_validated_evidence(active_task, nearby_evidence)
                        hunt_state["validated_sources"].extend(nearby_validation["sources"])
                        hunt_state["rejected_sources"].extend(nearby_validation["rejected_sources"])
                        nearby_record["fetched_evidence"] = nearby_evidence
                        nearby_record["validated_count"] = len(nearby_validation["sources"])
                        nearby_record["rejected_count"] = len(nearby_validation["rejected_sources"])
                        nearby_frontier = nearby_validation["rejected_sources"]
                    else:
                        nearby_frontier = []
                    attempt["nearby_expansions"].append(nearby_record)

                if hunt_state["validated_sources"]:
                    hunt_state["status"] = "success"
                    hunt_state["attempts"].append(attempt)
                    break

                if args.enable_source_resolvers and not hunt_state["resolver_used"]:
                    try:
                        resolver_candidates = (
                            legistar_calendar_candidates(task, timeout=args.timeout)
                            + federal_reserve_statement_candidates(task, timeout=args.timeout)
                            + noaa_hurricane_outlook_candidates(task, timeout=args.timeout)
                            + nhc_advisory_archive_candidates(task, timeout=args.timeout)
                        )
                    except Exception as exc:
                        resolver_candidates = []
                        attempt["resolver_error"] = str(exc)
                    hunt_state["resolver_used"] = bool(resolver_candidates)
                    attempt["resolver_candidates"] = resolver_candidates
                    resolver_new = []
                    for candidate in resolver_candidates:
                        key = candidate.get("canonical_url") or candidate.get("url")
                        if key in seen_candidates:
                            continue
                        seen_candidates.add(key)
                        resolver_new.append(candidate)
                    if resolver_new:
                        raw_evidence = fetch_discovered_evidence(
                            active_task,
                            resolver_new,
                            max_source_chars=args.max_source_chars,
                            timeout=args.timeout,
                        )
                        validation = filter_validated_evidence(active_task, raw_evidence)
                        hunt_state["validated_sources"].extend(validation["sources"])
                        hunt_state["rejected_sources"].extend(validation["rejected_sources"])
                        attempt["resolver_evidence"] = raw_evidence
                        attempt["resolver_validated_count"] = len(validation["sources"])
                        attempt["resolver_rejected_count"] = len(validation["rejected_sources"])
                elif not args.enable_source_resolvers:
                    attempt["resolver_skipped"] = "disabled"

                hunt_state["attempts"].append(attempt)
                if hunt_state["validated_sources"]:
                    hunt_state["status"] = "success"
                    break

            if not hunt_state["validated_sources"]:
                hunt_state["status"] = "failed"
            hunt_state["finished_at"] = utc_now_iso()
            hunt_state["latency_seconds"] = round(time.perf_counter() - hunt_started, 3)
            hunt_state["usage"]["cost"] = round(hunt_state["usage"]["cost"], 6)
            evidence_by_task[task["id"]] = {"sources": hunt_state["validated_sources"]}
            source_hunt_by_task[task["id"]] = hunt_state

        for model in models:
            pair = (task["id"], model["id"])
            if pair in done:
                print(f"SKIP {model['id']} / {task['id']} (already present)")
                continue

            if args.strategy == "web_plugin":
                payload = build_openrouter_payload(
                    model,
                    active_task,
                    tasks_doc.get("as_of", "unknown"),
                    args.engine,
                    args.max_results,
                    args.max_tokens,
                )
                synthetic_annotations = None
                evidence = None
                discovery = None
                source_hunt = None
            elif args.strategy == "controlled_fetch":
                evidence = evidence_by_task[task["id"]]
                payload = build_controlled_payload(
                    model,
                    active_task,
                    evidence,
                    tasks_doc.get("as_of", "unknown"),
                    args.max_tokens,
                )
                synthetic_annotations = evidence_annotations(evidence)
                discovery = None
                source_hunt = None
            else:
                evidence = evidence_by_task[task["id"]]
                payload = build_controlled_payload(
                    model,
                    active_task,
                    evidence,
                    tasks_doc.get("as_of", "unknown"),
                    args.max_tokens,
                )
                synthetic_annotations = evidence_annotations(evidence)
                discovery = discovery_by_task.get(task["id"])
                source_hunt = source_hunt_by_task.get(task["id"])

            started = time.perf_counter()
            record: Dict[str, Any] = {
                "run_started": run_started,
                "timestamp": utc_now_iso(),
                "task_id": task["id"],
                "task_category": task.get("category"),
                "model_id": model["id"],
                "model_label": model.get("label", model["id"]),
                "model": model["model"],
                "strategy": args.strategy,
                "engine": args.engine,
                "max_results": args.max_results,
                "max_source_chars": args.max_source_chars,
                "max_tokens": args.max_tokens,
            }
            if evidence is not None:
                record["evidence"] = evidence
            if discovery is not None:
                record["discovery"] = discovery
            if source_hunt is not None:
                record["source_hunt"] = source_hunt
            if task["id"] in contract_by_task:
                record["evidence_contract"] = contract_by_task[task["id"]]
            try:
                if evidence is not None and not evidence.get("sources"):
                    print(f"NO_EVIDENCE {model['id']} / {task['id']}")
                    record.update(
                        {
                            "content": json.dumps(
                                {
                                    "answer": (
                                        "No accepted source evidence was available for this question, "
                                        "so the requested facts are not supported."
                                    ),
                                    "key_facts": [],
                                    "sources": [],
                                    "uncertainties": ["No source passed the source-hunter validation gate."],
                                },
                                sort_keys=True,
                            ),
                            "annotations": [],
                            "usage": {},
                            "metadata": {"skipped_reason": "no_accepted_source_evidence"},
                        }
                    )
                else:
                    print(f"RUN {model['id']} / {task['id']}")
                    content, annotations, usage, metadata = post_openrouter(
                        api_key,
                        payload,
                        args.timeout,
                        retry_without_reasoning=not args.no_reasoning_retry,
                    )
                    record.update(
                        {
                            "content": content,
                            "annotations": annotations or synthetic_annotations or [],
                            "usage": usage,
                            "metadata": metadata,
                        }
                    )
            except Exception as exc:
                record["error"] = str(exc)
                record["annotations"] = []
                record["usage"] = {}
                print(f"ERROR {model['id']} / {task['id']}: {exc}")
            finally:
                record["latency_seconds"] = round(time.perf_counter() - started, 3)
                append_jsonl(out_path, record)
                time.sleep(args.sleep)

    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    main()
