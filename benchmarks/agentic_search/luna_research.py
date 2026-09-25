"""Let a model research one question the way Claude Code does: search, open pages,
search inside them, follow links, and stop when the question is answered.

Tools: web_search (Google CSE, as production), open_page (full text in parts, plus
the page's links), find_in_page (snippets around a phrase). The final answer must cite
every fact with an exact quote, and code checks each quote against the fetched text.

Run from the repo root:
    python3 -m benchmarks.agentic_search.luna_research --model openai/gpt-6-luna --effort medium
"""
import argparse
import json
import os
import re
import sys
import time

import requests

os.environ.setdefault("NEWSCASTER_LOG_DIR", "/tmp/nc_agentic_search_logs")
import newscaster.config as config  # noqa: E402

config.init()
config.LLM_AUDIT_LOG_ENABLED = False
config.SEARCH_AUDIT_LOG_ENABLED = False
from newscaster.search import search_web  # noqa: E402
from newscaster.source_hunter_primitives import fetch_source_text  # noqa: E402

PART_CHARS = 6000
MAX_STEPS = 30   # 16 in the first 32-lookup run: all 22 follow-ups hit it

QUESTIONS = [
    ("bills", "Which election-protection bills did California Gov. Gavin Newsom sign in mid-September 2026 after "
              "Riverside County Sheriff Chad Bianco's ballot seizure? Give each bill's number, author, what it "
              "does, and when it takes effect."),
    ("annihilate", "What exactly did President Trump say about annihilating Iran in his September 2026 speech at "
                   "the U.N. General Assembly, and on what date did he say it?"),
    ("australia", "When did the OpenAI agent breach the Australian government's Medicare statistics portal, and "
                  "when did OpenAI tell Services Australia about it?"),
]

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
Report only what you read in a page. Copy each quote exactly as it appears, without adding "..." or changing
punctuation. When you are done, reply with JSON only:
{{"answer": "a short direct answer",
  "facts": [{{"fact": "...", "quote": "exact words copied from the page", "url": "..."}}],
  "gaps": ["anything you could not find"]}}"""

_pages = {}   # default cache for single runs; batch runs pass their own dict per lookup


def _norm(text):
    """Lowercase letters and digits only, so curly quotes, dashes and spacing never cause a false miss."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def quote_on_page(quote, page_text):
    """True when the quote (or every 15+ character piece of it split at '...') appears in the page."""
    hay = _norm(page_text)
    pieces = [_norm(p) for p in re.split(r"\.\.\.|\u2026", quote or "")]
    pieces = [p for p in pieces if len(p) >= 15] or [_norm(quote)]
    return all(p and p in hay for p in pieces)


def _get_page(url, pages=None):
    pages = _pages if pages is None else pages
    if url not in pages:
        pages[url] = fetch_source_text(url)
    return pages[url]


def run_tool(name, args, pages=None):
    try:
        if name == "web_search":
            results = search_web(args["query"], num_results=8, days_prior=int(args.get("days") or 30))
            return [{"title": r.get("headline"), "url": r.get("url"), "snippet": (r.get("snippet") or "")[:300]}
                    for r in results]
        if name == "open_page":
            page = _get_page(args["url"], pages)
            text = page.get("text") or ""
            parts = max(1, -(-len(text) // PART_CHARS))
            part = min(max(1, int(args.get("part") or 1)), parts)
            out = {"title": page.get("title"), "published": page.get("published_date"), "part": part,
                   "parts": parts, "text": text[(part - 1) * PART_CHARS: part * PART_CHARS]}
            if part == 1:
                out["links"] = [{"text": (l.get("text") or "")[:80], "url": l.get("url")}
                                for l in (page.get("links") or [])[:60]]
            return out
        if name == "find_in_page":
            text = _get_page(args["url"], pages).get("text") or ""
            low, needle = text.lower(), args["text"].lower()
            hits = [m.start() for m in re.finditer(re.escape(needle), low)][:6]
            if not hits:   # fall back to any of the words
                words = [w for w in re.findall(r"\w{4,}", needle)]
                hits = sorted({m.start() for w in words for m in re.finditer(re.escape(w), low)})[:6]
            return {"matches": len(hits), "snippets": [text[max(0, h - 250): h + 350] for h in hits]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"error": f"unknown tool {name}"}


def research(question, model, effort, date, pages=None):
    messages = [{"role": "system", "content": SYSTEM.format(date=date, steps=MAX_STEPS)},
                {"role": "user", "content": question}]
    steps, cost, t0 = [], 0.0, time.time()
    for _ in range(MAX_STEPS + 2):
        payload = {"model": model, "messages": messages, "tools": TOOLS, "usage": {"include": True},
                   "reasoning": {"effort": effort}}
        if len(steps) >= MAX_STEPS:
            payload["tool_choice"] = "none"
            messages.append({"role": "user", "content": "Tool limit reached. Write your final JSON answer now from what you have read."})
        elif len(steps) >= MAX_STEPS - 4 and not any(str(m.get("content", "")).startswith("Only ") for m in messages if m["role"] == "user"):
            messages.append({"role": "user", "content": f"Only {MAX_STEPS - len(steps)} tool calls left. Finish the most important lookups, then answer."})
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=300,
                          headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}, json=payload)
        r.raise_for_status()
        data = r.json()
        cost += float((data.get("usage") or {}).get("cost") or 0)
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or "", steps, cost, time.time() - t0
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(name, args, pages)
            steps.append((name, args, result))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)[:20000]})
    return "", steps, cost, time.time() - t0


def summarize_step(name, args, result):
    if name == "web_search":
        top = ", ".join((x.get("url") or "")[:70] for x in (result if isinstance(result, list) else [])[:3])
        return f"search {args.get('query')!r} (days={args.get('days', 30)}) -> {top}"
    if name == "open_page":
        if "error" in result:
            return f"open {args.get('url')} -> {result['error'][:80]}"
        return f"open {args.get('url')} part {result.get('part')}/{result.get('parts')}"
    return f"find {args.get('text')!r} in {args.get('url')} -> {result.get('matches', result.get('error'))} matches"


def check_quotes(final, pages=None):
    pages = _pages if pages is None else pages
    try:
        data = json.loads(re.search(r"\{.*\}", final, re.S).group(0))
    except Exception:
        return None, []
    checks = []
    for f in data.get("facts", []):
        page = pages.get(f.get("url")) or {}
        ok = bool(f.get("quote")) and quote_on_page(f["quote"], page.get("text"))
        checks.append((ok, f))
    return data, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--date", default="September 25, 2026")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for key, question in QUESTIONS:
        if args.only and key not in args.only.split(","):
            continue
        _pages.clear()
        final, steps, cost, secs = research(question, args.model, args.effort, args.date)
        print(f"\n{'=' * 100}\n[{key}] {args.model} ({args.effort}) | {len(steps)} tool calls | ${cost:.3f} | {secs:.0f}s")
        print(f"Q: {question}")
        for i, (name, a, res) in enumerate(steps, 1):
            print(f"  {i:2}. {summarize_step(name, a, res)}")
        data, checks = check_quotes(final)
        if data is None:
            print("FINAL (not JSON):", final[:1500])
            continue
        print("ANSWER:", data.get("answer"))
        for ok, f in checks:
            print(f"  [{'quote found' if ok else 'QUOTE NOT ON PAGE'}] {f.get('fact')} <{(f.get('url') or '')[:90]}>")
        for g in data.get("gaps", []):
            print("  GAP:", g)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
