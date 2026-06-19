"""Adaptive selected-story research loop using LangGraph.

The graph controls *what to ask next* for a selected main story. It deliberately
keeps the existing provider router, Google CSE wrapper, article scraper pipeline,
and RAG store instead of replacing them with LangChain abstractions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

import newscaster.config as _config
from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write
from newscaster.prompts import (
    RESEARCH_ADVERSARY_PROMPT,
    RESEARCH_CONTROLLER_PROMPT,
    RESEARCH_CONTROLLER_REPAIR_PROMPT,
    RESEARCH_MEMORY_PROMPT,
)
from newscaster.rag.retrieve import retrieve_prior_research
from newscaster.search import search_web
from newscaster.text_utils import extract_json
from newscaster.scrapers.topic_finder import result_piper
from newscaster.source_hunter import answer_with_source_hunter


_ALLOWED_QUESTION_TYPES = {
    "premise_challenge",
    "scale_check",
    "source_check",
    "timeline_check",
    "mechanism_check",
    "counterevidence_check",
    "freshness_check",
}


class ResearchState(TypedDict, total=False):
    topic: str
    topic_index: int
    formatted_date: str
    formatted_date2: str
    summary_prompt: str
    successful_summary_counter: int
    articles: list[dict[str, Any]]
    followups: list[dict[str, Any]]
    memory_note: str
    adversary_decision: dict[str, Any]
    adversary_ran: bool
    iterations: int
    consecutive_failures: int
    max_iterations: int
    min_iterations: int
    last_decision: dict[str, Any]
    done_reason: str


@dataclass
class AdaptiveResearchResult:
    summary_prompt: str
    successful_summary_counter: int
    articles: list[dict[str, Any]]
    followups: list[dict[str, Any]]
    memory_note: str = ""
    done_reason: str = ""


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _format_prior_hits(hits) -> str:
    sections = []
    for hit in hits:
        sections.append(
            "[Prior coverage - {outlet}, {date}, similarity {sim:.2f}]\n"
            "Headline: {headline}\n"
            "URL: {url}\n"
            "{text}".format(
                outlet=hit.outlet or "unknown",
                date=hit.date,
                sim=hit.similarity,
                headline=hit.headline or "(none)",
                url=hit.url or "(none)",
                text=hit.text,
            )
        )
    return "\n\n".join(sections)


def _extract_json_object(text: str) -> dict[str, Any]:
    return extract_json(text, want=dict)


def _normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    status = str(decision.get("status", "")).strip().lower()
    if status == "done":
        confidence = str(decision.get("confidence") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        return {
            "status": "done",
            "reason": str(decision.get("reason") or "controller said enough evidence exists").strip(),
            "confidence": confidence,
        }

    if status != "continue":
        raise ValueError(f"unsupported controller status: {status!r}")

    action = str(decision.get("action", "")).strip().lower()
    reason = str(decision.get("reason") or "controller requested more research").strip()
    if action == "grounded_search":
        question = str(decision.get("question") or "").strip()
        if not question:
            raise ValueError("grounded_search decision missing question")
        question_type = str(decision.get("question_type") or "source_check").strip().lower()
        if question_type not in _ALLOWED_QUESTION_TYPES:
            question_type = "source_check"
        return {
            "status": "continue",
            "action": "grounded_search",
            "question": question,
            "question_type": question_type,
            "reason": reason,
        }

    if action == "article_search":
        query = str(decision.get("query") or "").strip()
        if not query:
            raise ValueError("article_search decision missing query")
        return {
            "status": "continue",
            "action": "article_search",
            "query": query,
            "reason": reason,
        }

    raise ValueError(f"unsupported controller action: {action!r}")


def parse_controller_decision(raw_response: str, *, allow_repair: bool = False) -> dict[str, Any]:
    """Parse and validate the controller's JSON decision.

    When `allow_repair` is true, one light-model repair call is allowed after a
    malformed first response. Tests use this public helper directly.
    """
    try:
        return _normalize_decision(_extract_json_object(raw_response))
    except (ValueError, json.JSONDecodeError):
        if not allow_repair:
            raise
        repaired = get_llm_response(
            raw_response,
            system_prompt=RESEARCH_CONTROLLER_REPAIR_PROMPT,
            mode="light",
        )
        return _normalize_decision(_extract_json_object(repaired))


def build_research_memory_note(topic: str, formatted_date: str, formatted_date2: str,
                               summary_prompt: str) -> str:
    """Retrieve prior research and turn it into a compact dated memory note."""
    if not _config.RAG_RESEARCH_MEMORY_ENABLED:
        return ""

    query = (
        f"Topic: {topic}\n"
        f"Today: {formatted_date}\n\n"
        f"Current seed evidence:\n{_clip(summary_prompt, 12000)}"
    )
    try:
        hits = retrieve_prior_research(query, exclude_date=formatted_date2)
    except Exception as e:
        print_and_write(f"Research agent memory retrieval failed: {e}; continuing without memory")
        return ""

    if not hits:
        return ""

    user_prompt = (
        f"TODAY'S STORY:\nTopic: {topic}\nDate: {formatted_date}\n\n"
        f"CURRENT SEED EVIDENCE:\n{_clip(summary_prompt, 10000)}\n\n"
        f"RETRIEVED PRIOR COVERAGE:\n{_format_prior_hits(hits)}"
    )
    try:
        return get_llm_response(
            user_prompt,
            system_prompt=RESEARCH_MEMORY_PROMPT,
            mode="heavy",
        ).strip()
    except Exception as e:
        print_and_write(f"Research agent memory note failed: {e}; continuing without memory")
        return ""


def _recent_followups(followups: list[dict[str, Any]]) -> str:
    if not followups:
        return "(none)"
    compact = []
    for item in followups[-8:]:
        compact.append({
            "iteration": item.get("iteration"),
            "question_type": item.get("question_type"),
            "question": item.get("question"),
            "answer": _clip(item.get("answer", ""), 1200),
        })
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _controller_payload(state: ResearchState) -> str:
    remaining = max(0, state["max_iterations"] - state.get("iterations", 0))
    return (
        f"TOPIC: {state['topic']}\n"
        f"DATE: {state['formatted_date']}\n"
        f"ITERATIONS_COMPLETED: {state.get('iterations', 0)}\n"
        f"REMAINING_BUDGET: {remaining}\n"
        f"ARTICLE_COUNT: {len(state.get('articles', []))}\n"
        f"MIN_ITERATIONS: {state.get('min_iterations', 0)}\n\n"
        f"RAG MEMORY NOTE:\n{state.get('memory_note') or '(none)'}\n\n"
        f"SECOND-PERSPECTIVE ADVERSARIAL QUESTION:\n"
        f"{json.dumps(state.get('adversary_decision') or {}, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT COMPLETED Q&A:\n{_recent_followups(state.get('followups', []))}\n\n"
        f"CURRENT EVIDENCE:\n{_clip(state.get('summary_prompt', ''), 18000)}"
    )


def _adversary_payload(state: ResearchState) -> str:
    done_decision = state.get("last_decision") or {}
    return (
        f"TOPIC: {state['topic']}\n"
        f"DATE: {state['formatted_date']}\n\n"
        f"OPUS EDITOR JUST SAID THE STORY WAS READY:\n"
        f"{json.dumps(done_decision, ensure_ascii=False, indent=2)}\n\n"
        f"RAG MEMORY NOTE:\n{state.get('memory_note') or '(none)'}\n\n"
        f"CURRENT EVIDENCE:\n{_clip(state.get('summary_prompt', ''), 18000)}"
    )


def _normalize_adversary_decision(raw_response: str) -> dict[str, Any]:
    raw = _extract_json_object(raw_response)
    question = str(raw.get("question") or "").strip()
    if not question:
        raise ValueError("adversary response missing question")
    question_type = str(raw.get("question_type") or "counterevidence_check").strip().lower()
    if question_type not in _ALLOWED_QUESTION_TYPES:
        question_type = "counterevidence_check"
    return {
        "status": "continue",
        "action": "grounded_search",
        "question": question,
        "question_type": question_type,
        "reason": str(raw.get("reason") or "second-perspective adversarial check").strip(),
        "asker": "GPT-5.5 Adversary",
        "adversary_guided": True,
    }


def _minimum_iteration_decision(topic: str) -> dict[str, Any]:
    return {
        "status": "continue",
        "action": "grounded_search",
        "question": (
            "What changed today or in the past 48 hours about this story, what claims "
            f"about {topic} remain uncertain, and what should be verified before summarizing it?"
        ),
        "question_type": "freshness_check",
        "reason": "minimum research pass before stopping",
    }


def _memory_node(state: ResearchState) -> ResearchState:
    note = build_research_memory_note(
        state["topic"],
        state["formatted_date"],
        state["formatted_date2"],
        state["summary_prompt"],
    )
    if note:
        print_and_write("Research agent: retrieved prior-memory note")
    return {"memory_note": note}


def _adversary_node(state: ResearchState) -> ResearchState:
    if not _config.AGENTIC_RESEARCH_ADVERSARY_ENABLED:
        return {"adversary_decision": {}, "adversary_ran": True}

    try:
        raw = get_llm_response(
            _adversary_payload(state),
            system_prompt=RESEARCH_ADVERSARY_PROMPT,
            mode="adversary",
        )
        decision = _normalize_adversary_decision(raw)
        print_and_write("Research adversary decision:", decision)
        return {
            "adversary_decision": decision,
            "adversary_ran": True,
            "last_decision": decision,
        }
    except Exception as e:
        print_and_write(f"Research adversary failed: {e}; keeping Opus done decision")
        return {"adversary_decision": {}, "adversary_ran": True}


def _controller_node(state: ResearchState) -> ResearchState:
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", _config.AGENTIC_RESEARCH_MAX_ITERATIONS)
    if iterations >= max_iterations:
        return {
            "last_decision": {
                "status": "done",
                "reason": "research budget exhausted",
                "confidence": "medium",
            },
            "done_reason": "research budget exhausted",
        }

    try:
        raw = get_llm_response(
            _controller_payload(state),
            system_prompt=RESEARCH_CONTROLLER_PROMPT,
            mode="heavy",
        )
        decision = parse_controller_decision(raw, allow_repair=True)
        next_iteration = iterations + 1
        if decision["status"] == "done" and next_iteration < state.get("min_iterations", 0):
            decision = _minimum_iteration_decision(state["topic"])
        print_and_write("Research agent decision:", decision)
        return {
            "last_decision": decision,
            "iterations": next_iteration,
            "done_reason": decision.get("reason", "") if decision["status"] == "done" else "",
        }
    except Exception as e:
        failures = state.get("consecutive_failures", 0) + 1
        print_and_write(f"Research agent controller failed: {e}; failures={failures}")
        return {
            "iterations": iterations + 1,
            "consecutive_failures": failures,
            "last_decision": {},
            "done_reason": f"controller failed: {e}",
        }


def _grounded_search_node(state: ResearchState) -> ResearchState:
    decision = state.get("last_decision") or {}
    question = decision.get("question", "")
    try:
        source_hunter_status = ""
        source_hunter_sources: list[dict[str, Any]] = []
        if _config.SOURCE_HUNTER_ENABLED:
            source_result = answer_with_source_hunter(
                question,
                topic=state.get("topic"),
                formatted_date=state.get("formatted_date"),
                mode="standard",
            )
            source_hunter_status = source_result.status
            source_hunter_sources = source_result.sources
            if source_result.status == "success":
                answer = source_result.answer
            else:
                print_and_write(
                    f"Research agent source hunter returned {source_result.status}; "
                    "trying advanced research reader"
                )
                advanced_result = answer_with_source_hunter(
                    question,
                    topic=state.get("topic"),
                    formatted_date=state.get("formatted_date"),
                    mode="advanced",
                )
                source_hunter_status = advanced_result.status
                source_hunter_sources = advanced_result.sources
                if advanced_result.status != "success":
                    raise RuntimeError(
                        f"source hunter failed: standard={source_result.status}, "
                        f"advanced={advanced_result.status}"
                    )
                answer = advanced_result.answer
        else:
            raise RuntimeError("source hunter disabled for research agent grounded_search")
        followups = list(state.get("followups", []))
        question_type = decision.get("question_type", "source_check")
        action_label = "source_hunter"
        record = {
            "asker": decision.get("asker", "Research Agent"),
            "question": question,
            "answer": answer,
            "challenging": question_type in {"premise_challenge", "counterevidence_check"},
            "iteration": state.get("iterations", 0),
            "question_type": question_type,
            "action": action_label,
            "reason": decision.get("reason", ""),
            "memory_guided": bool((state.get("memory_note") or "").strip()),
            "adversary_guided": bool(decision.get("adversary_guided")),
            "source_hunter_status": source_hunter_status,
            "source_hunter_sources": source_hunter_sources,
        }
        followups.append(record)
        summary_prompt = (
            state.get("summary_prompt", "")
            + f"\n\nResearch agent asked ({question_type}): {question}\n"
            + f"Controlled source hunter reported:\n{answer}"
        )
        return {
            "summary_prompt": summary_prompt,
            "followups": followups,
            "consecutive_failures": 0,
        }
    except Exception as e:
        failures = state.get("consecutive_failures", 0) + 1
        print_and_write(f"Research agent grounded search failed: {e}; failures={failures}")
        return {"consecutive_failures": failures, "done_reason": f"grounded search failed: {e}"}


def _article_search_node(state: ResearchState) -> ResearchState:
    decision = state.get("last_decision") or {}
    query = decision.get("query", "")
    try:
        results = search_web(query, 9)
        summary_prompt = state.get("summary_prompt", "")
        counter = state.get("successful_summary_counter", 0)
        articles = list(state.get("articles", []))
        before_counter = counter
        for result in results:
            summary_prompt, counter = result_piper(
                summary_prompt,
                counter,
                state["topic"],
                result,
                state["topic_index"],
                state["formatted_date2"],
                articles=articles,
            )
            if counter > before_counter:
                break
        if counter == before_counter:
            raise RuntimeError("article_search found no additional relevant article")
        return {
            "summary_prompt": summary_prompt,
            "successful_summary_counter": counter,
            "articles": articles,
            "consecutive_failures": 0,
        }
    except Exception as e:
        failures = state.get("consecutive_failures", 0) + 1
        print_and_write(f"Research agent article search failed: {e}; failures={failures}")
        return {"consecutive_failures": failures, "done_reason": f"article search failed: {e}"}


def _route_after_controller(state: ResearchState) -> str:
    if state.get("consecutive_failures", 0) >= 2:
        return "end"
    decision = state.get("last_decision") or {}
    if (
        decision.get("status") == "done"
        and _config.AGENTIC_RESEARCH_ADVERSARY_ENABLED
        and not state.get("adversary_ran", False)
        and state.get("iterations", 0) < state.get("max_iterations", _config.AGENTIC_RESEARCH_MAX_ITERATIONS)
        and decision.get("reason") != "research budget exhausted"
    ):
        return "adversary"
    if state.get("iterations", 0) >= state.get("max_iterations", _config.AGENTIC_RESEARCH_MAX_ITERATIONS):
        if decision.get("status") != "continue":
            return "end"
    if decision.get("status") == "done":
        return "end"
    action = decision.get("action")
    if action == "grounded_search":
        return "grounded_search"
    if action == "article_search":
        return "article_search"
    return "controller"


def _route_after_adversary(state: ResearchState) -> str:
    decision = state.get("last_decision") or {}
    if decision.get("adversary_guided") and decision.get("action") == "grounded_search":
        return "grounded_search"
    return "end"


def _route_after_tool(state: ResearchState) -> str:
    if state.get("consecutive_failures", 0) >= 2:
        return "end"
    if state.get("iterations", 0) >= state.get("max_iterations", _config.AGENTIC_RESEARCH_MAX_ITERATIONS):
        return "end"
    return "controller"


def _build_graph():
    # Deferred import keeps the rest of the pipeline importable if the optional
    # dependency is missing; _gather_one_topic catches graph setup failures.
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ResearchState)
    builder.add_node("memory", _memory_node)
    builder.add_node("adversary", _adversary_node)
    builder.add_node("controller", _controller_node)
    builder.add_node("grounded_search", _grounded_search_node)
    builder.add_node("article_search", _article_search_node)
    builder.add_edge(START, "memory")
    builder.add_edge("memory", "controller")
    builder.add_conditional_edges(
        "controller",
        _route_after_controller,
        {
            "adversary": "adversary",
            "grounded_search": "grounded_search",
            "article_search": "article_search",
            "controller": "controller",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "adversary",
        _route_after_adversary,
        {"grounded_search": "grounded_search", "end": END},
    )
    builder.add_conditional_edges(
        "grounded_search",
        _route_after_tool,
        {"controller": "controller", "end": END},
    )
    builder.add_conditional_edges(
        "article_search",
        _route_after_tool,
        {"controller": "controller", "end": END},
    )
    return builder.compile()


def run_adaptive_research(topic: str, topic_index: int, formatted_date: str, formatted_date2: str,
                          summary_prompt: str, successful_summary_counter: int,
                          articles=None, followups=None) -> AdaptiveResearchResult:
    """Run adaptive research for a selected story and return the enriched prompt.

    `articles` and `followups` are synchronized back into the provided accumulator
    lists so the existing research sidecar write path remains unchanged.
    """
    graph = _build_graph()
    initial_articles = list(articles or [])
    initial_followups = list(followups or [])
    state: ResearchState = {
        "topic": topic,
        "topic_index": topic_index,
        "formatted_date": formatted_date,
        "formatted_date2": formatted_date2,
        "summary_prompt": summary_prompt,
        "successful_summary_counter": successful_summary_counter,
        "articles": initial_articles,
        "followups": initial_followups,
        "memory_note": "",
        "adversary_decision": {},
        "adversary_ran": False,
        "iterations": 0,
        "consecutive_failures": 0,
        "max_iterations": _config.AGENTIC_RESEARCH_MAX_ITERATIONS,
        "min_iterations": _config.AGENTIC_RESEARCH_MIN_ITERATIONS,
        "last_decision": {},
        "done_reason": "",
    }
    final = graph.invoke(state)

    final_articles = list(final.get("articles", initial_articles))
    final_followups = list(final.get("followups", initial_followups))
    if articles is not None:
        articles[:] = final_articles
    if followups is not None:
        followups[:] = final_followups

    return AdaptiveResearchResult(
        summary_prompt=final.get("summary_prompt", summary_prompt),
        successful_summary_counter=final.get(
            "successful_summary_counter", successful_summary_counter
        ),
        articles=final_articles,
        followups=final_followups,
        memory_note=final.get("memory_note", ""),
        done_reason=final.get("done_reason", ""),
    )
