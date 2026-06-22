"""Controlled URL discovery, fetch, validation, and answer synthesis.

This is the production wrapper for the source-hunter workflow proven out in the
web-search benchmark. Discovery uses the existing Google CSE key. Answering uses
the normal router without grounding, constrained to locally fetched excerpts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import newscaster.config as _config
from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write, write_jsonl_log
from newscaster.search import search_web

from newscaster.source_hunter_primitives import (
    build_evidence_contract_payload,
    canonical_url,
    dedupe_url_candidates,
    fetch_discovered_evidence,
    filter_validated_evidence,
    nearby_source_candidates,
    parse_evidence_contract,
)


@dataclass
class SourceHunterResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    rejected_sources: list[dict[str, Any]] = field(default_factory=list)
    status: str = "failed"
    metadata: dict[str, Any] = field(default_factory=dict)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def _query_variants(question: str, topic: str | None, formatted_date: str | None) -> list[str]:
    question = (question or "").strip()
    topic = (topic or "").strip()
    date = (formatted_date or "").strip()
    # Lead with the specific question — the research agent and follow-up rounds ask pointed
    # questions (~10-30 words) that must drive the search. A long research *prompt* (the
    # 98-word Tier-2 brief instructions) is not query-like, so lead with the cleaner topic.
    # Either way the other term is searched within the first variants, so the iteration cap
    # (SOURCE_HUNTER_MAX_ITERATIONS) can't starve it.
    if question and question != topic and len(question.split()) <= 40:
        ordered = [question, topic]
    else:
        ordered = [topic, question]
    ordered = [term for term in dict.fromkeys(ordered) if term]
    variants = list(ordered)
    lead = ordered[0] if ordered else ""
    if lead and date:
        variants.append(f"{lead} {date}")
    if lead:
        variants.append(f"{lead} official source")
    return [variant for variant in dict.fromkeys(variants) if variant]


def _search_candidates(query: str, limit: int) -> list[dict[str, str]]:
    results = search_web(query, num_results=limit)
    candidates = []
    for result in results:
        url = result.get("url", "")
        if not url:
            continue
        candidates.append({
            "url": url,
            "canonical_url": canonical_url(url),
            "title": result.get("headline", ""),
            "reason": result.get("snippet", ""),
            "source": "google_cse",
        })
    return dedupe_url_candidates(candidates, limit=limit)


def _format_sources(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            f"SOURCE {index}\n"
            f"Title: {source.get('title') or source.get('candidate_title') or '(untitled)'}\n"
            f"URL: {source.get('final_url') or source.get('url')}\n"
            f"Content type: {source.get('content_type') or 'unknown'}\n"
            f"Relevant excerpt:\n{_clip(source.get('excerpt', ''), 5000)}"
        )
    return "\n\n".join(blocks)


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    validation = source.get("validation") or {}
    return {
        "title": source.get("title") or source.get("candidate_title") or "",
        "url": source.get("final_url") or source.get("url") or "",
        "canonical_url": source.get("canonical_url") or canonical_url(source.get("url", "")),
        "content_type": source.get("content_type") or "",
        "char_count": source.get("char_count") or 0,
        "validation_score": validation.get("score"),
        "validation_reason": validation.get("reason", ""),
        # Persist the fetched excerpt so downstream faithfulness checks (and forensics) can
        # verify the synthesis against the actual source text the pipeline read, instead of
        # re-fetching. Capped to bound research-sidecar / audit size. The RAG indexer ignores
        # source_hunter_sources, so this is invisible to the embeddings store.
        "excerpt": _clip(source.get("excerpt") or "", 4000),
    }


def _rejected_summary(source: dict[str, Any]) -> dict[str, Any]:
    validation = source.get("validation") or {}
    return {
        "title": source.get("title") or source.get("candidate_title") or "",
        "url": source.get("final_url") or source.get("url") or "",
        "canonical_url": source.get("canonical_url") or canonical_url(source.get("url", "")),
        "error": source.get("error", ""),
        "validation_reason": validation.get("reason", ""),
    }


def _synthesize_answer(question: str, sources: list[dict[str, Any]], formatted_date: str | None,
                       mode: str) -> str:
    system_prompt = (
        "You are a careful newsroom research assistant. Report only what the controlled "
        "source excerpts establish — never invent, infer, or fill gaps from outside "
        "knowledge. Prefer exact names, dates, amounts, vote counts, agencies, and official "
        "language.\n\n"
        "Respond in two labeled sections:\n"
        "FINDINGS: the facts from the excerpts that bear on the question. If the excerpts "
        "establish nothing relevant, write 'None'.\n"
        "GAPS: the parts of the question the excerpts do NOT answer, phrased precisely "
        "enough that a follow-up search could target them. If the excerpts fully answer "
        "the question, write 'None'.\n"
        "Then a short Sources section listing the URLs you relied on."
    )
    prompt = (
        f"Date context: {formatted_date or 'unknown'}\n"
        f"Question: {question}\n\n"
        f"Controlled source evidence:\n{_format_sources(sources)}"
    )
    return get_llm_response(prompt, system_prompt=system_prompt, mode=mode, grounding=False)


def _generate_evidence_contract(question: str, formatted_date: str | None) -> dict[str, Any]:
    """Generate an evidence contract (required slots, source preferences, reject_if).

    The contract drives source validation: which facts a source must support, which
    source types are preferred, and which traps (calendars, previews, announcements
    without results) to reject. Returns an empty contract on any failure so validation
    degrades to general date/entity/topic constraints rather than breaking the hunt.
    """
    try:
        payload = build_evidence_contract_payload(
            {"question": question, "category": "news_research"},
            as_of=formatted_date or "unknown",
            model="contract",
        )
        messages = payload["messages"]
        raw = get_llm_response(
            messages[1]["content"],
            system_prompt=messages[0]["content"],
            mode="standard",
        )
        return parse_evidence_contract(raw)
    except Exception as exc:
        print_and_write(f"Source hunter contract generation failed: {exc}; using general constraints")
        return {}


def answer_with_source_hunter(question: str, *, topic: str | None = None,
                              formatted_date: str | None = None,
                              mode: str = "standard",
                              max_iterations: int | None = None) -> SourceHunterResult:
    """Answer a research question from controlled fetched sources.

    Returns status ``success`` when at least one source validates against the
    question. On failure it returns ``no_evidence`` instead of guessing.
    """
    max_iterations = max_iterations or _config.SOURCE_HUNTER_MAX_ITERATIONS
    task = {
        "id": "production_source_hunter",
        "question": question,
        "category": "news_research",
        "evidence_contract": _generate_evidence_contract(question, formatted_date),
    }
    seen: set[str] = set()
    validated_sources: list[dict[str, Any]] = []
    rejected_sources: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    for iteration, query in enumerate(_query_variants(question, topic, formatted_date), start=1):
        if iteration > max_iterations or validated_sources:
            break
        attempt: dict[str, Any] = {"iteration": iteration, "query": query}
        try:
            candidates = _search_candidates(query, _config.SOURCE_HUNTER_CANDIDATE_LIMIT)
        except Exception as exc:
            attempt["search_error"] = str(exc)
            attempts.append(attempt)
            print_and_write(f"Source hunter search failed: {exc}")
            continue

        new_candidates = []
        for candidate in candidates:
            key = candidate.get("canonical_url") or candidate.get("url")
            if key in seen:
                continue
            seen.add(key)
            new_candidates.append(candidate)
        attempt["candidate_count"] = len(new_candidates)

        if new_candidates:
            evidence = fetch_discovered_evidence(
                task,
                new_candidates,
                max_source_chars=_config.SOURCE_HUNTER_MAX_SOURCE_CHARS,
            )
            validation = filter_validated_evidence(task, evidence)
            validated_sources.extend(validation["sources"])
            rejected_sources.extend(validation["rejected_sources"])
            attempt["validated_count"] = len(validation["sources"])
            attempt["rejected_count"] = len(validation["rejected_sources"])
            nearby_frontier = validation["rejected_sources"]
        else:
            nearby_frontier = []

        attempt["nearby_expansions"] = []
        for nearby_depth in range(max(0, _config.SOURCE_HUNTER_NEARBY_SOURCE_DEPTH)):
            if validated_sources or not nearby_frontier:
                break
            nearby_candidates = nearby_source_candidates(
                task,
                nearby_frontier,
                seen,
                limit=_config.SOURCE_HUNTER_NEARBY_SOURCE_LIMIT,
            )
            nearby_new = []
            for candidate in nearby_candidates:
                key = candidate.get("canonical_url") or candidate.get("url")
                if key in seen:
                    continue
                seen.add(key)
                nearby_new.append(candidate)

            nearby_record = {"depth": nearby_depth + 1, "candidate_count": len(nearby_new)}
            if nearby_new:
                nearby_evidence = fetch_discovered_evidence(
                    task,
                    nearby_new,
                    max_source_chars=_config.SOURCE_HUNTER_MAX_SOURCE_CHARS,
                )
                nearby_validation = filter_validated_evidence(task, nearby_evidence)
                validated_sources.extend(nearby_validation["sources"])
                rejected_sources.extend(nearby_validation["rejected_sources"])
                nearby_record["validated_count"] = len(nearby_validation["sources"])
                nearby_record["rejected_count"] = len(nearby_validation["rejected_sources"])
                nearby_frontier = nearby_validation["rejected_sources"]
            else:
                nearby_frontier = []
            attempt["nearby_expansions"].append(nearby_record)

        attempts.append(attempt)

    if not validated_sources:
        result = SourceHunterResult(
            answer="No accepted source evidence was found for this question.",
            sources=[],
            rejected_sources=[_rejected_summary(source) for source in rejected_sources],
            status="no_evidence",
            metadata={"attempts": attempts},
        )
        _audit_source_hunter(question, topic, formatted_date, result)
        return result

    try:
        answer = _synthesize_answer(question, validated_sources, formatted_date, mode)
    except Exception as exc:
        print_and_write(f"Source hunter answer synthesis failed: {exc}")
        result = SourceHunterResult(
            answer="Accepted source evidence was found, but answer synthesis failed.",
            sources=[_source_summary(source) for source in validated_sources],
            rejected_sources=[_rejected_summary(source) for source in rejected_sources],
            status="synthesis_failed",
            metadata={"attempts": attempts, "error": str(exc)},
        )
        _audit_source_hunter(question, topic, formatted_date, result)
        return result

    # Validated sources but a partial synthesis (FINDINGS with open GAPS) is still usable
    # research: callers — especially the research agent's Opus controller — get the grounded
    # facts plus an explicit gap to target next, rather than having everything discarded.
    result = SourceHunterResult(
        answer=answer,
        sources=[_source_summary(source) for source in validated_sources],
        rejected_sources=[_rejected_summary(source) for source in rejected_sources],
        status="success",
        metadata={"attempts": attempts},
    )
    _audit_source_hunter(question, topic, formatted_date, result)
    return result


def _audit_source_hunter(question: str, topic: str | None, formatted_date: str | None,
                         result: SourceHunterResult) -> None:
    # Coverage tracking: log every URL the grounded search could NOT capture to its own jsonl,
    # separate from (and not gated by) the full audit, so "what can't be captured" is trivially
    # queryable. A fetch failure is a rejected source carrying an `error` (the fetch exception);
    # validation rejections — fetched fine but didn't support the claim — carry only a
    # `validation_reason` and are excluded here.
    fetch_failures = [s for s in result.rejected_sources if s.get("error")]
    for failure in fetch_failures:
        write_jsonl_log("source_hunter_fetch_failures", {
            "event": "fetch_failure",
            "question": question,
            "topic": topic,
            "formatted_date": formatted_date,
            "url": failure.get("url", ""),
            "canonical_url": failure.get("canonical_url", ""),
            "error": failure.get("error", ""),
        })
    if fetch_failures:
        print_and_write(
            f"SOURCE-HUNTER: {len(fetch_failures)} URL(s) could not be captured "
            f"(fetch failed); logged to source_hunter_fetch_failures"
        )

    if not getattr(_config, "SOURCE_HUNTER_AUDIT_LOG_ENABLED", False):
        return
    write_jsonl_log("source_hunter_audit", {
        "event": "source_hunter",
        "question": question,
        "topic": topic,
        "formatted_date": formatted_date,
        "status": result.status,
        "answer": result.answer,
        "sources": result.sources,
        "rejected_sources": result.rejected_sources,
        "attempts": result.metadata.get("attempts", []),
        "metadata": result.metadata,
    })


def answer_with_escalation(question: str, *, topic: str | None = None,
                           formatted_date: str | None = None,
                           label: str = "Source hunter") -> SourceHunterResult:
    """Run the source hunter at ``standard``, escalating to ``advanced`` on non-success.

    Returns whichever ``SourceHunterResult`` we stop at: the standard result when it
    succeeds, otherwise the advanced result. Centralizes the standard->advanced escalation
    policy (and its log line) shared by the pipeline, the research agent, and the topic
    finder; callers read ``.answer`` / ``.status`` / ``.sources`` as they need.
    """
    result = answer_with_source_hunter(
        question, topic=topic, formatted_date=formatted_date, mode="standard",
    )
    if result.status == "success":
        return result
    print_and_write(f"{label} standard returned {result.status}; trying advanced research reader")
    return answer_with_source_hunter(
        question, topic=topic, formatted_date=formatted_date, mode="advanced",
    )
