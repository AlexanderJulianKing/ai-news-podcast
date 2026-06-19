"""Search provider abstraction.

Production callers should use ``search_web`` instead of importing the Google
CSE wrapper directly. Google remains the primary provider while it works;
OpenRouter web search is the emergency fallback.
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

import newscaster.config as _config
from newscaster.logging import print_and_write, write_jsonl_log
from newscaster.scrapers.google_search import google_official_search


def _normalize_result(headline: str, url: str, snippet: str = "") -> dict[str, str]:
    return {
        "headline": (headline or url or "Search result").strip(),
        "url": (url or "").strip(),
        "snippet": (snippet or "").strip(),
    }


def _dedupe_results(results: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for result in results:
        url = result.get("url", "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(_normalize_result(
            result.get("headline", ""),
            url,
            result.get("snippet", ""),
        ))
        if len(deduped) >= limit:
            break
    return deduped


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _results_from_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, str]]:
    results = []
    for annotation in annotations or []:
        citation = annotation.get("url_citation") or {}
        url = citation.get("url") or annotation.get("url") or ""
        if not url:
            continue
        results.append(_normalize_result(
            citation.get("title") or annotation.get("title") or url,
            url,
            citation.get("content") or annotation.get("content") or "",
        ))
    return results


def _results_from_content(content: str) -> list[dict[str, str]]:
    rows = []
    for item in _extract_json_array(content):
        if not isinstance(item, dict):
            continue
        rows.append(_normalize_result(
            str(item.get("headline") or item.get("title") or ""),
            str(item.get("url") or ""),
            str(item.get("snippet") or item.get("description") or ""),
        ))
    if rows:
        return rows

    fallback_rows = []
    for url in re.findall(r"https?://[^\s)\"'>\\]+", content or ""):
        fallback_rows.append(_normalize_result(url, url, ""))
    return fallback_rows


def openrouter_web_search(query: str, num_results: int = 8, days_prior: int = 1) -> list[dict[str, str]]:
    """Discover URLs through OpenRouter's web-search tool."""
    prompt = (
        "Search the current web for sources that would answer this newsroom query.\n"
        f"Query: {query}\n"
        f"Freshness preference: results from the last {days_prior} day(s), if available.\n\n"
        f"Return only a JSON array of up to {num_results} objects. Each object must have "
        '"headline", "url", and "snippet". Prefer primary or official sources when relevant.'
    )
    payload = {
        "model": _config.SEARCH_OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "plugins": [{
            "id": "web",
            "engine": _config.SEARCH_OPENROUTER_ENGINE,
            "max_results": num_results,
            "search_prompt": (
                "A web search was conducted for a newsroom source-discovery step. "
                "Prefer official or primary sources and return useful URLs."
            ),
        }],
        "reasoning": {"effort": "low"},
        "usage": {"include": True},
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {_config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Newscaster Search Fallback",
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=(10, 60),
    )
    response.raise_for_status()
    data = response.json()
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    annotations = message.get("annotations") or []
    results = _results_from_annotations(annotations) + _results_from_content(content)
    deduped = _dedupe_results(results, num_results)
    if not deduped:
        raise RuntimeError("OpenRouter web search returned no URL results")
    if getattr(_config, "SEARCH_AUDIT_LOG_ENABLED", False):
        write_jsonl_log("search_audit", {
            "event": "openrouter_web_search",
            "provider": "openrouter_web",
            "query": query,
            "num_results": num_results,
            "days_prior": days_prior,
            "model": _config.SEARCH_OPENROUTER_MODEL,
            "engine": _config.SEARCH_OPENROUTER_ENGINE,
            "results": deduped,
            "raw_content": content,
            "annotations": annotations,
        })
    return deduped


def openrouter_web_brief(question: str, *, model: str | None = None,
                         max_results: int = 5) -> str:
    """Answer a research question with a single web-grounded LLM call.

    Uses OpenRouter's web-search plugin to ground a cheap model (Gemma 4 by
    default) and returns the synthesized brief text. This is the selection-stage
    counterpart to the source hunter: one cheap call for the gist of a story so
    the editor can judge its importance, not a fully validated multi-source hunt.
    Raises on transport/HTTP failure or an empty completion so callers can fall
    back to an UNVERIFIED marker.
    """
    model = model or _config.STANDARD_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "plugins": [{
            "id": "web",
            "engine": _config.SEARCH_OPENROUTER_ENGINE,
            "max_results": max_results,
            "search_prompt": (
                "A web search was conducted for a newsroom background brief. "
                "Prefer recent, reputable, and primary sources."
            ),
        }],
        "usage": {"include": True},
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {_config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Newscaster Tier-2 Brief",
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=(10, 90),
    )
    response.raise_for_status()
    data = response.json()
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenRouter web brief returned an empty completion")
    if getattr(_config, "SEARCH_AUDIT_LOG_ENABLED", False):
        write_jsonl_log("search_audit", {
            "event": "openrouter_web_brief",
            "provider": "openrouter_web",
            "question": question,
            "model": model,
            "engine": _config.SEARCH_OPENROUTER_ENGINE,
            "max_results": max_results,
            "brief": content,
            "annotations": message.get("annotations") or [],
            "cost": (data.get("usage") or {}).get("cost"),
        })
    return content


def search_web(query: str, num_results: int = 8, days_prior: int = 1,
               provider: str | None = None) -> list[dict[str, str]]:
    """Search the web with provider fallback and normalized result shape."""
    primary = provider or _config.SEARCH_PROVIDER
    fallback = _config.SEARCH_FALLBACK_PROVIDER

    def _call(selected: str) -> list[dict[str, str]]:
        if selected == "google_cse":
            return google_official_search(query, num_results=num_results, days_prior=days_prior)
        if selected == "openrouter_web":
            return openrouter_web_search(query, num_results=num_results, days_prior=days_prior)
        raise ValueError(f"Unknown search provider: {selected}")

    try:
        results = _dedupe_results(_call(primary), num_results)
        if results or not _config.SEARCH_FALLBACK_ON_EMPTY or not fallback:
            if getattr(_config, "SEARCH_AUDIT_LOG_ENABLED", False):
                write_jsonl_log("search_audit", {
                    "event": "search_web",
                    "query": query,
                    "num_results": num_results,
                    "days_prior": days_prior,
                    "provider": primary,
                    "fallback_provider": fallback,
                    "fallback_used": False,
                    "results": results,
                })
            return results
        print_and_write(f"Search provider {primary} returned no results; trying {fallback}")
    except Exception as exc:
        if not fallback:
            raise
        print_and_write(f"Search provider {primary} failed: {exc}; trying {fallback}")

    if fallback == primary:
        return []
    fallback_results = _dedupe_results(_call(fallback), num_results)
    if getattr(_config, "SEARCH_AUDIT_LOG_ENABLED", False):
        write_jsonl_log("search_audit", {
            "event": "search_web",
            "query": query,
            "num_results": num_results,
            "days_prior": days_prior,
            "provider": primary,
            "fallback_provider": fallback,
            "fallback_used": True,
            "results": fallback_results,
        })
    return fallback_results
