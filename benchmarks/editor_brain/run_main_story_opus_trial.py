#!/usr/bin/env python3
"""Run one selected-story gather and add an Opus editor read.

This is intentionally a benchmark harness, not production pipeline code. It lets
us inspect how a single headline flows through article collection, source-hunter
research, the Opus-controlled adaptive loop, standard synthesis, and a final
Opus editor interpretation.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import newscaster.config as config
from newscaster.dates import format_spoken_date
from newscaster.llm import get_llm_response
from newscaster.pipeline import _gather_one_topic
from newscaster.prompts import (
    CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE,
    FOLLOW_UP_PROMPT_TEMPLATE,
)
from newscaster.scrapers.dropsite import DROP_SITE_FEED_URL, _extract_items


OUTPUT_DIR = Path("benchmarks/editor_brain/outputs")


OPUS_EDITOR_PROMPT = """You are the show's senior editor reviewing one selected main story after the reporting pipeline has gathered evidence.

Your job is not to rewrite the whole story. Your job is to judge what the show now knows and what the final segment should emphasize.

Return a newsroom editor memo with these headings:
EDITORIAL THESIS:
WHAT HAPPENED:
WHY IT MATTERS:
BEST EVIDENCE:
UNCERTAINTIES OR WEAK SPOTS:
WHAT A LISTENER NEEDS TO UNDERSTAND:
SEGMENT ANGLE:

Be concrete. Attribute claims to sources when the evidence includes source names or URLs. Do not invent missing facts. If the evidence is thin, say so plainly.
"""


def _slug(text: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug or "headline")[:limit].strip("-")


def _default_headline() -> str:
    response = requests.get(
        DROP_SITE_FEED_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; EditorBrainTrial/1.0)"},
        timeout=(5, 20),
    )
    response.raise_for_status()
    items = _extract_items(response.content, now=datetime.now(timezone.utc), lookback_hours=72)
    if not items:
        raise RuntimeError("Drop Site feed did not contain a recent headline")
    return items[0]["title"]


def _format_json_block(value) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def _write_report(path: Path, payload: dict) -> None:
    articles = payload.get("articles", [])
    followups = payload.get("followups", [])
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Opus Main Story Trial</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f3ee; color: #171717; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ margin-top: 32px; border-top: 1px solid #d8d1c5; padding-top: 18px; }}
    .meta {{ color: #5f5a52; margin-bottom: 22px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    .panel {{ background: #fff; border: 1px solid #ddd6ca; border-radius: 8px; padding: 16px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid #ddd6ca; border-radius: 8px; padding: 16px; line-height: 1.45; }}
    .count {{ font-weight: 650; }}
    a {{ color: #0f5f8f; }}
  </style>
</head>
<body>
<main>
  <h1>Opus Main Story Trial</h1>
  <div class="meta">{html.escape(payload["created_at"])} | headline: {html.escape(payload["headline"])}</div>

  <div class="grid">
    <div class="panel"><div class="count">{len(articles)}</div><div>article records gathered</div></div>
    <div class="panel"><div class="count">{len(followups)}</div><div>research follow-ups</div></div>
    <div class="panel"><div class="count">{payload.get("summary_chars", 0)}</div><div>standard synthesis characters</div></div>
    <div class="panel"><div class="count">{payload.get("opus_editor_chars", 0)}</div><div>Opus editor memo characters</div></div>
  </div>

  <h2>Opus Editor Memo</h2>
  <pre>{html.escape(payload.get("opus_editor_memo", ""))}</pre>

  <h2>Standard Pipeline Synthesis</h2>
  <pre>{html.escape(payload.get("standard_summary", ""))}</pre>

  <h2>Articles</h2>
  <pre>{_format_json_block(articles)}</pre>

  <h2>Follow-Ups</h2>
  <pre>{_format_json_block(followups)}</pre>
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline", help="Headline/topic to run through the selected-story path.")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Temporarily override AGENTIC_RESEARCH_MAX_ITERATIONS for this run.")
    parser.add_argument("--min-iterations", type=int, default=None,
                        help="Temporarily override AGENTIC_RESEARCH_MIN_ITERATIONS for this run.")
    args = parser.parse_args()

    config.init()
    if args.max_iterations is not None:
        config.AGENTIC_RESEARCH_MAX_ITERATIONS = args.max_iterations
    if args.min_iterations is not None:
        config.AGENTIC_RESEARCH_MIN_ITERATIONS = args.min_iterations

    today = date.today()
    formatted_date = format_spoken_date(today)
    formatted_date2 = today.strftime("%Y_%m_%d")
    headline = args.headline or _default_headline()

    articles: list[dict] = []
    followups: list[dict] = []
    standard_summary = _gather_one_topic(
        headline,
        0,
        formatted_date,
        formatted_date2,
        FOLLOW_UP_PROMPT_TEMPLATE.format(date=formatted_date),
        CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE.format(date=formatted_date),
        articles=articles,
        followups=followups,
    )

    editor_input = {
        "headline": headline,
        "date": formatted_date,
        "standard_pipeline_synthesis": standard_summary,
        "article_records": articles,
        "followup_records": followups,
    }
    opus_editor_memo = get_llm_response(
        json.dumps(editor_input, ensure_ascii=False, indent=2),
        system_prompt=OPUS_EDITOR_PROMPT,
        mode="heavy",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_DIR / f"{stamp}_{_slug(headline)}"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "headline": headline,
        "formatted_date": formatted_date,
        "formatted_date2": formatted_date2,
        "max_iterations": config.AGENTIC_RESEARCH_MAX_ITERATIONS,
        "min_iterations": config.AGENTIC_RESEARCH_MIN_ITERATIONS,
        "articles": articles,
        "followups": followups,
        "standard_summary": standard_summary,
        "summary_chars": len(standard_summary),
        "opus_editor_memo": opus_editor_memo,
        "opus_editor_chars": len(opus_editor_memo),
    }
    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(html_path, payload)

    print(f"headline: {headline}")
    print(f"articles: {len(articles)}")
    print(f"followups: {len(followups)}")
    print(f"json: {json_path}")
    print(f"html: {html_path}")


if __name__ == "__main__":
    main()
