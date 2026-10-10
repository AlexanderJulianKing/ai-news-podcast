"""Claude Haiku 5.5 vs GPT-6 Luna as the tool-using researcher, on the 32 real lookups.

    python3 -m benchmarks.haiku_vs_luna.run_research ../agentic_search/outputs/replay_32_v2.jsonl outputs/research.jsonl

Luna runs the production loop (newscaster.research_tools.research_with_tools). Haiku runs
the same loop with the same tools, system prompt, step and time limits, and the same quote
check; only the model calls differ (Anthropic tool use instead of OpenRouter). Both run at
medium effort, now, so they search the same web.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("NEWSCASTER_LOG_DIR", "/tmp/nc_haiku_bench")
import anthropic  # noqa: E402
import httpx  # noqa: E402

import newscaster.config as config  # noqa: E402

config.init()
config.LLM_AUDIT_LOG_ENABLED = False
from newscaster import research_tools as rt  # noqa: E402
from newscaster.llm.claude import _create_message, _usage_to_dict  # noqa: E402

HAIKU = "claude-haiku-5-5"
TOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"],
          "input_schema": t["function"]["parameters"]} for t in rt.TOOLS]


# Round 3 (2026-10-10): both judges agreed Haiku's research answers were better, but Codex
# flagged its fact lines and answers for claiming more than their own quotes (17 vs Luna's 6;
# both spot-checked flags were real). This general rule is the fix under test.
QUOTE_DISCIPLINE = """
Keep every fact line as close to its quote as you can: do not add timing, causes, comparisons, or conclusions
that the quoted words do not state. Your answer must not claim more than your quoted facts support; when the
quotes leave something open, say it is not established."""


def _cache_last_block(messages):
    """Prompt caching: mark only the newest block, so each turn reads the conversation so far from cache."""
    for m in messages:
        if isinstance(m["content"], list):
            for block in m["content"]:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    last["content"][-1]["cache_control"] = {"type": "ephemeral"}


def haiku_research(question, formatted_date, effort="medium", extra_rules="", cache=False):
    """research_with_tools with Anthropic tool use. Returns (final_text, pages, steps, cost, seconds)."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=httpx.Timeout(300.0, connect=10.0))
    max_steps, max_seconds = config.RESEARCH_TOOL_MAX_STEPS, config.RESEARCH_TOOL_MAX_SECONDS
    pages = rt._Pages()
    system = rt.SYSTEM.format(date=formatted_date or "unknown", steps=max_steps) + extra_rules
    messages = [{"role": "user", "content": question}]
    steps, cost, started, warned, final_text = [], 0.0, time.time(), False, ""
    for _ in range(max_steps + 4):
        out_of_budget = len(steps) >= max_steps or time.time() - started > max_seconds
        kwargs = dict(model=HAIKU, max_tokens=16000, system=system, messages=messages, tools=TOOLS,
                      thinking={"type": "adaptive"}, extra_body={"output_config": {"effort": effort}})
        if out_of_budget:
            kwargs["tool_choice"] = {"type": "none"}
        if cache:
            _cache_last_block(messages)
        msg = None
        for attempt in range(4):
            try:
                msg = _create_message(client, **kwargs)
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError):
                time.sleep(5 * (attempt + 1))
        if msg is None:
            raise RuntimeError("Haiku call failed after retries")
        cost += (_usage_to_dict(msg.usage, HAIKU) or {}).get("estimated_cost_usd") or 0
        calls = [b for b in msg.content if b.type == "tool_use"]
        if not calls or out_of_budget:
            final_text = "".join(b.text for b in msg.content if b.type == "text")
            break
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for c in calls:
            result = rt._run_tool(c.name, dict(c.input or {}), pages)
            steps.append(f"{c.name} {json.dumps(c.input, ensure_ascii=False)[:160]}")
            results.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": json.dumps(result, ensure_ascii=False)[:20000]})
        # Same nudges as production, added to the tool-result turn (Anthropic needs alternating turns).
        if len(steps) >= max_steps or time.time() - started > max_seconds:
            results.append({"type": "text", "text": "Research limit reached. Write your final JSON answer now from what you have read."})
        elif not warned and len(steps) >= max_steps - 4:
            warned = True
            results.append({"type": "text", "text": f"Only {max_steps - len(steps)} tool calls left. Finish the most important lookups, then answer."})
        messages.append({"role": "user", "content": results})
    return final_text, pages, steps, cost, time.time() - started


def as_answer(final_text, pages):
    """The production loop's verification and answer text, applied to a final reply."""
    parsed = rt._parse_final(final_text) or {}
    kept, dropped = [], 0
    for fact in parsed.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        page = pages.pages.get(fact.get("url") or "") or {}
        if fact.get("quote") and rt.quote_on_page(fact["quote"], page.get("text") or ""):
            kept.append(fact)
        else:
            dropped += 1
    gaps = [str(g) for g in parsed.get("gaps") or [] if str(g).strip()]
    if not kept:
        return ("No accepted source evidence was found for this question."
                + ("\n\nGAPS:\n" + "\n".join(f"- {g}" for g in gaps) if gaps else "")), 0, dropped
    lines = ["FINDINGS:"] + [f"- {f.get('fact')} (\"{f.get('quote')}\") [{f.get('url')}]" for f in kept]
    if parsed.get("answer"):
        lines += ["", f"ANSWER: {parsed['answer']}"]
    lines += ["", "GAPS:"] + ([f"- {g}" for g in gaps] or ["None"])
    return "\n".join(lines), len(kept), dropped


def run(row):
    out = {"question": row["question"], "topic": row.get("topic"), "date": row["date"]}
    try:
        res = rt.research_with_tools(row["question"], row["date"])
        md = res["metadata"]
        out["luna"] = {"text": res["answer"], "kept": md["facts_kept"], "dropped": md["facts_dropped"],
                       "steps": len(md["steps"]), "cost": md["cost"], "seconds": md["seconds"]}
    except Exception as exc:
        out["luna"] = {"text": f"ERROR {exc!r}"[:300], "kept": 0, "dropped": 0, "steps": 0, "cost": 0, "seconds": 0}
    try:
        final, pages, steps, cost, secs = haiku_research(row["question"], row["date"])
        text, kept, dropped = as_answer(final, pages)
        out["haiku"] = {"text": text, "kept": kept, "dropped": dropped, "steps": len(steps),
                        "cost": round(cost, 4), "seconds": round(secs)}
    except Exception as exc:
        out["haiku"] = {"text": f"ERROR {exc!r}"[:300], "kept": 0, "dropped": 0, "steps": 0, "cost": 0, "seconds": 0}
    return out


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    with open(sys.argv[2], "w", encoding="utf-8") as f, ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(run, rows), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            L, H = r["luna"], r["haiku"]
            print(f"{i:2}/{len(rows)} luna {L['kept']} facts {L['steps']} steps {L['seconds']}s ${L['cost']:.4f} | "
                  f"haiku {H['kept']} facts ({H['dropped']} dropped) {H['steps']} steps {H['seconds']}s ${H['cost']:.4f}", flush=True)
