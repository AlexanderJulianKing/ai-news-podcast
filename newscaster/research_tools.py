"""Tool-using researcher: the model searches, opens pages, searches inside them, follows
links, and stops when the question is answered, the way Claude Code researches.

Replaces the source hunter's fixed search-and-filter pipeline when
RESEARCH_TOOL_LOOP_ENABLED is on. Why: on the 32 real lookups of 2026-09-24/25, a blind
Opus comparison preferred this loop's answers 31-1 over the patched source hunter and
32-0 over what aired (benchmarks/agentic_search/). The fixed pipeline let code rules,
not the model, decide which pages counted, and showed the model one keyword-picked
excerpt per page.

Honesty guarantee: every fact must carry an exact quote, and code keeps a fact only when
that quote is found in the text of a page the loop actually fetched. A lookup with no
verified fact returns ``no_evidence``.

Returns a plain dict (status, answer, sources, metadata) that source_hunter wraps in its
SourceHunterResult, so callers and the fact-checker see the same shape as before. Each
source's ``excerpt`` is the page text around its verified quotes.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

import newscaster.config as _config
from newscaster.logging import print_and_write, write_jsonl_log
from newscaster.search import search_web
from newscaster.source_hunter_primitives import canonical_url, fetch_source_text

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PART_CHARS = 6000

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web. Returns up to 8 results with title, url and snippet.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A short search query, like you would type into Google."},
            "days": {"type": "integer", "description": "Only pages from the last N days (default 30)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "open_page",
        "description": f"Open a URL and read its text, {PART_CHARS} characters per part. Part 1 also lists the page's links.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "part": {"type": "integer", "description": "Which part to read (default 1)."}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "find_in_page",
        "description": "Search inside a page for a word or phrase. Returns up to 6 snippets around the matches.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "text": {"type": "string"}}, "required": ["url", "text"]}}},
]

SYSTEM = """You are a newsroom researcher. Today is {date}. Answer the question using the tools.
Work like a skilled reporter: search, read the most promising pages, search inside long pages, follow links to
primary sources (a bill's page, a court opinion, an agency's own release), and write sharper searches based on
what you learn. Keep going until the question is answered or it is clear the answer is not published yet.
For a broad question ("tell me about this story"), gather facts from at least three different outlets or primary
sources rather than reading one page in depth.
You have at most {steps} tool calls.
Report only what you read in a page you opened. Copy each quote exactly as it appears, without adding "..." or
changing punctuation. When you are done, reply with JSON only:
{{"answer": "a short direct answer",
  "facts": [{{"fact": "...", "quote": "exact words copied from the page", "url": "..."}}],
  "gaps": ["anything you could not find"]}}"""


def _norm(text: str) -> str:
    """Lowercase letters and digits only, so curly quotes, dashes and spacing never cause a false miss."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def quote_on_page(quote: str, page_text: str) -> bool:
    """True when the quote (or every 15+ character piece of it split at '...') appears in the page."""
    hay = _norm(page_text)
    pieces = [_norm(p) for p in re.split(r"\.\.\.|…", quote or "")]
    pieces = [p for p in pieces if len(p) >= 15] or [_norm(quote)]
    return all(p and p in hay for p in pieces)


def _excerpt_around(page_text: str, quotes: list[str], width: int = 700, limit: int = 4000) -> str:
    """Page text around each verified quote, so the fact-checker checks against the real page."""
    text = page_text or ""
    lowered = text.lower()
    windows = []
    for quote in quotes:
        probe = re.split(r"\.\.\.|…", quote)[0].strip()[:60].lower()
        at = lowered.find(probe) if probe else -1
        if at < 0:
            continue
        windows.append(text[max(0, at - width): at + len(quote) + width])
    excerpt = "\n...\n".join(dict.fromkeys(windows))
    return excerpt[:limit] if excerpt else text[:limit]


class _Pages:
    """Fetched pages for one lookup, fetched once each."""

    def __init__(self):
        self.pages: dict[str, dict[str, Any]] = {}

    def get(self, url: str) -> dict[str, Any]:
        if url not in self.pages:
            self.pages[url] = fetch_source_text(url, timeout=20)
        return self.pages[url]


def _run_tool(name: str, args: dict[str, Any], pages: _Pages) -> Any:
    try:
        if name == "web_search":
            results = search_web(args["query"], num_results=8, days_prior=int(args.get("days") or 30))
            return [{"title": r.get("headline"), "url": r.get("url"), "snippet": (r.get("snippet") or "")[:300]}
                    for r in results]
        if name == "open_page":
            page = pages.get(args["url"])
            text = page.get("text") or ""
            parts = max(1, -(-len(text) // PART_CHARS))
            part = min(max(1, int(args.get("part") or 1)), parts)
            out = {"title": page.get("title"), "published": page.get("published_date"), "part": part,
                   "parts": parts, "text": text[(part - 1) * PART_CHARS: part * PART_CHARS]}
            if part == 1:
                out["links"] = [{"text": (link.get("text") or "")[:80], "url": link.get("url")}
                                for link in (page.get("links") or [])[:60]]
            return out
        if name == "find_in_page":
            text = pages.get(args["url"]).get("text") or ""
            low, needle = text.lower(), (args.get("text") or "").lower()
            hits = [m.start() for m in re.finditer(re.escape(needle), low)][:6] if needle else []
            if not hits:
                words = re.findall(r"\w{4,}", needle)
                hits = sorted({m.start() for w in words for m in re.finditer(re.escape(w), low)})[:6]
            return {"matches": len(hits), "snippets": [text[max(0, h - 250): h + 350] for h in hits]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"error": f"unknown tool {name}"}


def _chat(payload: dict[str, Any]) -> dict[str, Any]:
    """One OpenRouter call with retries on rate limits, server errors and timeouts."""
    last = None
    for attempt in range(4):
        try:
            response = requests.post(OPENROUTER_URL, json=payload, timeout=180,
                                     headers={"Authorization": f"Bearer {_config.OPENROUTER_API_KEY}"})
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(f"OpenRouter HTTP {response.status_code}")
            else:
                response.raise_for_status()
                data = response.json()
                if data.get("choices"):
                    return data
                last = RuntimeError(f"OpenRouter returned no choices: {str(data)[:200]}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"research tool loop: model call failed: {last}")


def _audit(model: str, data: dict[str, Any]) -> None:
    if not getattr(_config, "LLM_AUDIT_LOG_ENABLED", False):
        return
    usage = data.get("usage") or {}
    write_jsonl_log("llm_audit", {
        "event": "success", "phase": "research_tools", "provider": "openrouter", "model": model,
        "name": "Research tool loop", "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
    })


def _parse_final(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def research_with_tools(question: str, formatted_date: str | None) -> dict[str, Any]:
    """Research one question with tools. Raises on model failure so the caller can fall back."""
    model = getattr(_config, "RESEARCH_TOOL_MODEL", "openai/gpt-6-luna")
    effort = getattr(_config, "RESEARCH_TOOL_EFFORT", "medium")
    max_steps = getattr(_config, "RESEARCH_TOOL_MAX_STEPS", 30)
    max_seconds = getattr(_config, "RESEARCH_TOOL_MAX_SECONDS", 300)
    pages = _Pages()
    messages = [{"role": "system", "content": SYSTEM.format(date=formatted_date or "unknown", steps=max_steps)},
                {"role": "user", "content": question}]
    steps: list[str] = []
    cost, started, warned = 0.0, time.time(), False
    final_text = ""
    for _ in range(max_steps + 4):
        out_of_budget = len(steps) >= max_steps or time.time() - started > max_seconds
        payload = {"model": model, "messages": messages, "tools": TOOLS, "usage": {"include": True},
                   "reasoning": {"effort": effort}}
        if out_of_budget:
            payload["tool_choice"] = "none"
            messages.append({"role": "user", "content": "Research limit reached. Write your final JSON answer now from what you have read."})
        elif not warned and len(steps) >= max_steps - 4:
            warned = True
            messages.append({"role": "user", "content": f"Only {max_steps - len(steps)} tool calls left. Finish the most important lookups, then answer."})
        data = _chat(payload)
        _audit(model, data)
        cost += float((data.get("usage") or {}).get("cost") or 0)
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls or out_of_budget:
            final_text = msg.get("content") or ""
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call.get("function", {}).get("name", "")
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(name, args, pages)
            steps.append(f"{name} {json.dumps(args, ensure_ascii=False)[:160]}")
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": json.dumps(result, ensure_ascii=False)[:20000]})

    parsed = _parse_final(final_text) or {}
    verified: dict[str, list[str]] = {}
    kept, dropped = [], 0
    for fact in parsed.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        url = fact.get("url") or ""
        page = pages.pages.get(url) or {}
        if fact.get("quote") and quote_on_page(fact["quote"], page.get("text") or ""):
            kept.append(fact)
            verified.setdefault(url, []).append(fact["quote"])
        else:
            dropped += 1
    gaps = [str(g) for g in parsed.get("gaps") or [] if str(g).strip()]
    metadata = {"engine": "tools", "model": model, "steps": steps, "cost": round(cost, 4),
                "seconds": round(time.time() - started), "facts_kept": len(kept), "facts_dropped": dropped}
    print_and_write(f"Research tool loop: {len(steps)} tool calls, {len(kept)} verified facts "
                    f"({dropped} dropped), {metadata['seconds']}s, ${cost:.4f}")
    if not kept:
        return {"status": "no_evidence", "sources": [], "metadata": metadata,
                "answer": "No accepted source evidence was found for this question."
                          + ("\n\nGAPS:\n" + "\n".join(f"- {g}" for g in gaps) if gaps else "")}

    sources = []
    for url, quotes in verified.items():
        page = pages.pages[url]
        sources.append({
            "title": page.get("title") or url, "url": page.get("url") or url, "canonical_url": canonical_url(url),
            "content_type": page.get("content_type") or "", "char_count": page.get("char_count") or 0,
            "validation_score": None, "validation_reasons": ["quote verified on page"],
            "excerpt": _excerpt_around(page.get("text") or "", quotes),
        })
    lines = ["FINDINGS:"]
    lines += [f"- {f.get('fact')} (\"{f.get('quote')}\") [{f.get('url')}]" for f in kept]
    if parsed.get("answer"):
        lines += ["", f"ANSWER: {parsed['answer']}"]
    lines += ["", "GAPS:"] + ([f"- {g}" for g in gaps] or ["None"])
    lines += ["", "Sources:"] + [f"- {s['title']} — {s['url']}" for s in sources]
    return {"status": "success", "answer": "\n".join(lines), "sources": sources, "metadata": metadata}
