"""Source-hunter fetch, extraction, validation, and nearby-link primitives.

Single source of truth for the source-hunter toolkit. Self-contained (stdlib +
requests + BeautifulSoup only). The web-search benchmark
(``benchmarks/web_search/web_bench_lib.py``) re-exports from here, so the
primitives are maintained in exactly one place.
"""

import argparse
import html
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]  # repo root
# Benchmark resources (used only by the web-search benchmark harness, never by the
# production source hunter); located under the repo's benchmarks/ tree. These are
# Path objects only — production never reads them, so this adds no runtime dependency.
_BENCH_DIR = ROOT / "benchmarks" / "web_search"
DEFAULT_TASKS = _BENCH_DIR / "tasks.json"
DEFAULT_MODELS = _BENCH_DIR / "models.json"
DEFAULT_OUTPUT_DIR = _BENCH_DIR / "outputs"

STOPWORDS = {
    "about", "according", "after", "again", "against", "also", "and", "are", "because",
    "before", "being", "between", "both", "but", "can", "city", "county", "date",
    "did", "does", "each", "from", "give", "had", "has", "have", "how", "include",
    "for", "into", "its", "june", "list", "main", "must", "not", "now", "off", "one",
    "only", "other", "over", "say", "source", "state", "still", "that", "the",
    "their", "them", "then", "there", "this", "through", "using", "was", "were",
    "what", "when", "where", "which", "while", "who", "why", "will", "with",
}

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

READER_BASE_URL = "https://r.jina.ai/"
MAX_DIRECT_FETCH_BYTES = 8_000_000
MAX_READER_FETCH_BYTES = 2_000_000


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def select_tasks(tasks_doc: Dict[str, Any], task_ids: Optional[List[str]], limit: Optional[int]) -> List[Dict[str, Any]]:
    tasks = list(tasks_doc.get("tasks", []))
    if task_ids:
        wanted = set(task_ids)
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(sorted(missing))}")
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def select_models(models_doc: Dict[str, Any], model_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    models = list(models_doc.get("models", []))
    if model_ids:
        wanted = set(model_ids)
        models = [model for model in models if model["id"] in wanted]
        missing = wanted - {model["id"] for model in models}
        if missing:
            raise SystemExit(f"Unknown model id(s): {', '.join(sorted(missing))}")
    return models


def load_openrouter_key(keys_path: Optional[Path] = None) -> str:
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()

    path = keys_path or (ROOT / "keys.txt")
    if not path.exists():
        raise SystemExit(
            "OpenRouter key not found. Set OPENROUTER_API_KEY or add openrouter_api to keys.txt."
        )

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            name, value = line.split(":", 1)
            if name.strip() == "openrouter_api" and value.strip():
                return value.strip()

    raise SystemExit("OpenRouter key not found in keys.txt under openrouter_api.")


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def annotation_text(annotations: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for annotation in annotations or []:
        citation = annotation.get("url_citation") or annotation
        for key in ("title", "url", "content"):
            value = citation.get(key)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def combined_result_text(result: Dict[str, Any]) -> str:
    return "\n".join(
        [
            str(result.get("content") or ""),
            annotation_text(result.get("annotations") or []),
        ]
    )


def domain_matches(text: str, annotations: List[Dict[str, Any]], domain: str) -> bool:
    domain = domain.lower().lstrip(".")
    haystack = normalize_text(text)
    if domain in haystack:
        return True
    for annotation in annotations or []:
        citation = annotation.get("url_citation") or annotation
        url = citation.get("url") or ""
        host = urlparse(url).netloc.lower()
        if host == domain or host.endswith("." + domain):
            return True
    return False


def _term_present(text: str, term: str) -> bool:
    return normalize_text(term) in text


def score_result(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    text = normalize_text(combined_result_text(result))
    annotations = result.get("annotations") or []
    details: List[Dict[str, Any]] = []
    max_points = 0.0
    earned = 0.0

    failed = bool(result.get("error"))

    for check in task.get("checks", []):
        points = float(check.get("points", 0))
        max_points += points
        all_terms = check.get("all") or []
        any_terms = check.get("any") or []
        all_ok = all(_term_present(text, term) for term in all_terms)
        any_ok = True if not any_terms else any(_term_present(text, term) for term in any_terms)
        matched = all_ok and any_ok and not failed
        if matched:
            earned += points
        details.append(
            {
                "type": "fact",
                "name": check.get("name", "check"),
                "points": points,
                "earned": points if matched else 0.0,
                "matched": matched,
                "all": all_terms,
                "any": any_terms,
            }
        )

    for source in task.get("source_domains", []):
        points = float(source.get("points", 0))
        max_points += points
        domain = source.get("domain", "")
        matched = bool(domain and domain_matches(text, annotations, domain) and not failed)
        if matched:
            earned += points
        details.append(
            {
                "type": "source",
                "name": domain,
                "points": points,
                "earned": points if matched else 0.0,
                "matched": matched,
            }
        )

    penalties: List[Dict[str, Any]] = []
    penalty_points = 0.0
    for forbidden in task.get("forbidden", []):
        phrase = forbidden.get("phrase", "")
        points = float(forbidden.get("points", 0))
        if phrase and _term_present(text, phrase):
            penalty_points += points
            penalties.append({"phrase": phrase, "points": points})

    earned = max(0.0, earned - penalty_points)
    if failed:
        earned = 0.0

    percent = (earned / max_points * 100.0) if max_points else 0.0
    return {
        "task_id": task["id"],
        "model_id": result.get("model_id"),
        "model_label": result.get("model_label"),
        "score": round(earned, 3),
        "max_score": round(max_points, 3),
        "percent": round(percent, 2),
        "failed": failed,
        "details": details,
        "penalties": penalties,
    }


def summarize_scores(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, Any]] = {}
    for record in records:
        model_id = record.get("model_id") or "unknown"
        bucket = by_model.setdefault(
            model_id,
            {
                "model_id": model_id,
                "model_label": record.get("model_label") or model_id,
                "score": 0.0,
                "max_score": 0.0,
                "tasks": 0,
                "failures": 0,
                "cost": 0.0,
                "latency_seconds": 0.0,
            },
        )
        bucket["score"] += float(record.get("score") or 0)
        bucket["max_score"] += float(record.get("max_score") or 0)
        bucket["tasks"] += 1
        bucket["failures"] += 1 if record.get("failed") else 0
        bucket["cost"] += float(record.get("usage", {}).get("cost") or 0)
        bucket["latency_seconds"] += float(record.get("latency_seconds") or 0)

    models = []
    for bucket in by_model.values():
        max_score = bucket["max_score"]
        bucket["percent"] = round((bucket["score"] / max_score * 100.0) if max_score else 0.0, 2)
        bucket["score"] = round(bucket["score"], 3)
        bucket["max_score"] = round(max_score, 3)
        bucket["cost"] = round(bucket["cost"], 6)
        bucket["avg_latency_seconds"] = round(
            bucket["latency_seconds"] / bucket["tasks"] if bucket["tasks"] else 0.0,
            2,
        )
        del bucket["latency_seconds"]
        models.append(bucket)

    models.sort(key=lambda item: (-item["percent"], item["cost"]))
    return {"models": models}


def build_messages(task: Dict[str, Any], as_of: str) -> List[Dict[str, str]]:
    system = (
        "You are a careful newsroom research assistant. Use current web results, prefer official "
        "or primary sources, and do not use stale background as current fact. If sources conflict, "
        "say so. Answer compactly but include exact dates, numbers, and source URLs."
    )
    user = f"""Benchmark date: {as_of}
Task category: {task.get('category', 'unknown')}
Question: {task['question']}

Return a strict JSON object only, with these keys:
{{
  "answer": "one to three paragraphs",
  "key_facts": ["fact 1", "fact 2"],
  "sources": [{{"title": "source title", "url": "https://...", "supports": "what it supports"}}],
  "uncertainties": ["anything still unclear, or []"]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_openrouter_payload(
    model_config: Dict[str, Any],
    task: Dict[str, Any],
    as_of: str,
    engine: str,
    max_results: int,
    max_tokens: int,
) -> Dict[str, Any]:
    plugin: Dict[str, Any] = {
        "id": "web",
        "engine": engine,
        "max_results": max_results,
        "search_prompt": (
            f"A web search was conducted for a newsroom benchmark on {as_of}. "
            "Use these results as evidence. Prefer official or primary sources."
        ),
    }
    payload: Dict[str, Any] = {
        "model": model_config["model"],
        "messages": build_messages(task, as_of),
        "plugins": [plugin],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "stream": False,
    }
    if model_config.get("reasoning_effort"):
        payload["reasoning"] = {"effort": model_config["reasoning_effort"]}
    return payload


def build_discovery_payload(
    task: Dict[str, Any],
    as_of: str,
    engine: str,
    max_results: int,
    max_tokens: int,
    model: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    plugin: Dict[str, Any] = {
        "id": "web",
        "engine": engine,
        "max_results": max_results,
        "search_prompt": (
            f"Search the live web for official or primary sources as of {as_of}. "
            "Prioritize government, agency, docket, agenda, and PDF source pages. "
            "Return source URLs, not a prose answer."
        ),
    }
    system = (
        "You are a source-discovery scout for a newsroom pipeline. Find candidate URLs "
        "that a separate scraper can fetch. Prefer official primary sources over news "
        "summaries. Do not answer the question from memory."
    )
    user = f"""Benchmark date: {as_of}
Task category: {task.get('category', 'unknown')}
Question: {task['question']}

Additional source-hunting context:
{extra_context}

Return strict JSON only:
{{
  "search_queries_used": ["query 1", "query 2"],
  "candidate_sources": [
    {{"url": "https://...", "title": "source title", "reason": "why this should contain the primary evidence"}}
  ]
}}
"""
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "plugins": [plugin],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "stream": False,
    }


def _json_from_text(content: str) -> Optional[Dict[str, Any]]:
    json_text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", json_text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        json_text = fence.group(1).strip()
    try:
        parsed = json.loads(json_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_evidence_contract_payload(
    task: Dict[str, Any],
    as_of: str,
    model: str,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    system = (
        "You convert newsroom research questions into evidence contracts. "
        "Do not answer the question. Do not use outside knowledge. "
        "Only describe what a source must contain to answer the question."
    )
    user = f"""Benchmark date: {as_of}
Task category: {task.get('category', 'unknown')}
Question: {task['question']}

Return strict JSON only:
{{
  "required_slots": [
    {{
      "name": "short_snake_case_name",
      "label": "human-readable required fact",
      "evidence_type": "text|number|money|percent|date|time|range|location|name",
      "keywords": ["terms that should appear near the evidence"],
      "search_terms": ["terms useful for retrying source discovery"]
    }}
  ],
  "source_preferences": ["official domain or source type, if implied by the question"],
  "reject_if": ["generic source problems to reject, such as announcement without results"]
}}

Guidelines:
- Make slots generic and question-derived, not benchmark-derived.
- Split compound requests into separate slots.
- For "how many", counts, amounts, votes, ranges, wind, pressure, or percentages, use a numeric evidence_type.
- For "what did X say/decide/require", include the concrete decision or requirement as a text slot.
- Do not include expected answer values.
"""
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "stream": False,
    }


def parse_evidence_contract(content: str) -> Dict[str, Any]:
    parsed = _json_from_text(content)
    if not parsed:
        return {"required_slots": [], "source_preferences": [], "reject_if": []}
    return normalize_evidence_contract(parsed)


def canonical_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    host = parsed.netloc.lower()
    path = re.sub(r"/+", "/", parsed.path or "/")
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunparse((parsed.scheme.lower(), host, path.rstrip("/") or "/", "", query, ""))


def urls_from_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    candidates = []
    for annotation in annotations or []:
        citation = annotation.get("url_citation") or annotation
        url = citation.get("url")
        if url:
            candidates.append(
                {
                    "url": url,
                    "title": citation.get("title") or "",
                    "reason": (citation.get("content") or "")[:500],
                    "source": "annotation",
                }
            )
    return candidates


def urls_from_text(content: str) -> List[Dict[str, str]]:
    candidates = []
    for url in re.findall(r"https?://[^\s\"'<>),]+", content or ""):
        candidates.append({"url": url, "title": "", "reason": "", "source": "content"})
    parsed = _json_from_text(content)
    if isinstance(parsed, dict):
        for item in parsed.get("candidate_sources", []) or []:
            if isinstance(item, dict) and item.get("url"):
                candidates.append(
                    {
                        "url": item.get("url", ""),
                        "title": item.get("title") or "",
                        "reason": item.get("reason") or "",
                        "source": "json",
                    }
                )
    return candidates


def dedupe_url_candidates(candidates: List[Dict[str, str]], limit: int = 10) -> List[Dict[str, str]]:
    deduped = []
    seen = set()
    for candidate in candidates:
        canon = canonical_url(candidate.get("url", ""))
        if not canon or canon in seen:
            continue
        seen.add(canon)
        row = dict(candidate)
        row["canonical_url"] = canon
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def discovery_candidates(content: str, annotations: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, str]]:
    candidates = urls_from_annotations(annotations) + urls_from_text(content)
    return dedupe_url_candidates(candidates, limit=limit)


def _clean_lines(text: str) -> List[str]:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.replace("\xa0", " ")).strip()
        if line:
            lines.append(line)
    return lines


_PUBLISHED_META_ATTRS = (
    {"property": "article:published_time"},
    {"name": "article:published_time"},
    {"property": "og:published_time"},
    {"itemprop": "datePublished"},
    {"name": "datePublished"},
    {"name": "pubdate"},
    {"name": "publish-date"},
    {"name": "date"},
)
_JSONLD_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})')
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _first_iso_date(value: str) -> Optional[str]:
    """First real yyyy-mm-dd inside ``value``, or None."""
    if not value:
        return None
    match = _ISO_DATE_RE.search(value)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        datetime(year, month, day)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_published_date(content: bytes) -> Optional[str]:
    """Publication date (yyyy-mm-dd) from page metadata, or None.

    Must run on the RAW bytes: ``_html_to_text`` decomposes <script>, which would
    destroy the JSON-LD block many outlets use. Most news pages keep the publish
    date here rather than in the visible body text, so reading it is the only way
    to tell a story from today apart from one from five months ago.
    """
    try:
        soup = BeautifulSoup(content, "html.parser")
    except Exception:
        soup = None
    if soup is not None:
        for attrs in _PUBLISHED_META_ATTRS:
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                iso = _first_iso_date(str(tag.get("content")))
                if iso:
                    return iso
        time_tag = soup.find("time")
        if time_tag is not None:
            iso = _first_iso_date(
                str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True))
            )
            if iso:
                return iso
    try:
        raw = content.decode("utf-8", "ignore") if isinstance(content, bytes) else str(content)
    except Exception:
        return None
    match = _JSONLD_PUBLISHED_RE.search(raw)
    return match.group(1) if match else None


def _html_to_text(content: bytes, base_url: str = "") -> Tuple[str, str, List[Dict[str, str]]]:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    links = []
    for tag in soup.find_all("a", href=True):
        href = str(tag.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        url = urljoin(base_url, href) if base_url else href
        if not urlparse(url).scheme.startswith("http"):
            continue
        link_text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        links.append({"url": url, "text": link_text})
    text = "\n".join(_clean_lines(soup.get_text("\n")))
    return title, text, links[:250]


def _pdf_to_text(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        pdf_file.write(content)
        pdf_file.flush()
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_file.name, "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "pdftotext failed").strip())
    return "\n".join(_clean_lines(result.stdout))


def _response_looks_blocked(response: requests.Response) -> bool:
    if response.status_code in {401, 403, 406, 429, 503}:
        return True
    content_type = (response.headers.get("content-type") or "").lower()
    if "pdf" in content_type:
        return False
    encoding = response.encoding or "utf-8"
    sample = normalize_text(response.content[:2000].decode(encoding, errors="ignore"))
    return any(
        marker in sample
        for marker in (
            "just a moment",
            "checking your browser",
            "enable javascript and cookies",
            "attention required",
            "access denied",
        )
    )


def _timeout_tuple(timeout: int) -> Tuple[int, int]:
    connect_timeout = max(5, min(15, int(timeout)))
    read_timeout = max(10, min(30, int(timeout)))
    return connect_timeout, read_timeout


def _read_response_body(response: requests.Response, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        remaining = max_bytes - total
        if remaining <= 0:
            break
        chunks.append(chunk[:remaining])
        total += len(chunks[-1])
        if total >= max_bytes:
            break
    return b"".join(chunks)


def _reader_url(url: str) -> str:
    return READER_BASE_URL + url


def _reader_text_to_title_and_body(text: str, url: str) -> Tuple[str, str]:
    title = ""
    lines = _clean_lines(text)
    for line in lines[:8]:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            break
    marker = "Markdown Content:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return title or url, "\n".join(_clean_lines(text))


def fetch_source_text(url: str, timeout: int = 30) -> Dict[str, Any]:
    response = requests.get(
        url,
        timeout=_timeout_tuple(timeout),
        headers=FETCH_HEADERS,
        stream=True,
    )
    response._content = _read_response_body(response, MAX_DIRECT_FETCH_BYTES)
    fetch_mode = "direct"
    if _response_looks_blocked(response):
        reader_response = requests.get(
            _reader_url(url),
            timeout=_timeout_tuple(timeout),
            headers=FETCH_HEADERS,
            stream=True,
        )
        reader_response._content = _read_response_body(reader_response, MAX_READER_FETCH_BYTES)
        reader_response.raise_for_status()
        title, text = _reader_text_to_title_and_body(reader_response.text, url)
        return {
            "url": url,
            "title": title,
            "content_type": "text/plain; reader-fallback",
            "status_code": reader_response.status_code,
            "text": text,
            "char_count": len(text),
            "fetch_mode": "reader_fallback",
            "links": [],
        }
    response.raise_for_status()
    content_type = (response.headers.get("content-type") or "").lower()
    final_url = response.url
    title = ""
    if "pdf" in content_type or final_url.lower().endswith(".pdf"):
        title = Path(urlparse(final_url).path).name or "PDF source"
        text = _pdf_to_text(response.content)
        links: List[Dict[str, str]] = []
    else:
        title, text, links = _html_to_text(response.content, final_url)
    return {
        "url": final_url,
        "title": title or final_url,
        "content_type": content_type,
        "status_code": response.status_code,
        "text": text,
        "published_date": extract_published_date(response.content),
        "char_count": len(text),
        "fetch_mode": fetch_mode,
        "links": links,
    }


def question_terms(question: str) -> List[str]:
    normalized = normalize_text(question)
    terms = [
        term.strip(".,:;()[]{}") for term in re.findall(r"[a-z0-9][a-z0-9.,:;(){}\[\]-]{1,}", normalized)
        if len(term) >= 3 and term not in STOPWORDS
    ]
    for item in re.findall(r"\bitem\s+([0-9]+[a-z]?)\b", normalized):
        terms.append(item)
    for amount in re.findall(r"\$?\d[\d,]*(?:\.\d+)?", question):
        terms.append(amount.lower().replace("$", ""))
    # Keep order but remove duplicates.
    seen = set()
    ordered = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered


def _has_number(text: str) -> bool:
    return bool(re.search(r"(?:\$)?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b(?:\s*%)?", text))


def _number_near_term(normalized: str, term: str, window: int = 120) -> bool:
    terms = _keyword_aliases(term)
    for term in terms:
        term = normalize_text(term)
        if not term:
            continue
        for match in re.finditer(re.escape(term), normalized):
            start = max(0, match.start() - window)
            end = min(len(normalized), match.end() + window)
            if _has_number(normalized[start:end]):
                return True
    return False


def _keyword_aliases(term: str) -> List[str]:
    normalized = normalize_text(term)
    aliases = {
        "rfp": ["rfp", "request for proposal"],
        "motion": ["motion", "movement", "moving"],
        "wind": ["wind", "winds"],
        "pressure": ["pressure", "mb"],
        "first-time": ["first-time", "first time"],
        "reserve-policy": ["reserve-policy", "reserve policy", "reserve", "reserves"],
        "start": ["start", "commencing", "commence", "effective"],
        "start date": ["start date", "commencing", "commence", "effective"],
    }
    values = aliases.get(normalized, [normalized])
    if "-" in normalized:
        values.append(normalized.replace("-", " "))
    if "/" in normalized:
        values.extend(part for part in normalized.split("/") if part)
    return list(dict.fromkeys(values))


def _keyword_present(term: str, normalized: str) -> bool:
    return any(alias in normalized for alias in _keyword_aliases(term))


def _label_has(label: str, terms: Iterable[str]) -> bool:
    normalized = normalize_text(label)
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)


def _has_number_near_keywords(normalized: str, keywords: Iterable[str], window: int = 120) -> bool:
    for keyword in keywords:
        if _number_near_term(normalized, keyword, window=window):
            return True
    return False


def _slot_missing_is_hard(slot: Dict[str, Any], generated_contract: bool) -> bool:
    if slot.get("hard_veto") is True:
        return True
    if generated_contract:
        return False
    return slot.get("evidence_type") in {"number", "money", "percent", "date", "time", "range", "location"}


def _slot_by_name(slots: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {slot.get("name", ""): slot for slot in slots if slot.get("name")}


def _missing_slot_name(reason: str) -> str:
    return reason.split(":", 1)[1] if ":" in reason else ""


def _hard_evidence_missing(reason: str, slot_lookup: Dict[str, Dict[str, Any]], generated_contract: bool) -> bool:
    if not reason.startswith("evidence_missing:"):
        return False
    slot = slot_lookup.get(_missing_slot_name(reason), {})
    return _slot_missing_is_hard(slot, generated_contract)


def _number_anywhere_or_near(normalized: str, keywords: Iterable[str]) -> bool:
    return _has_number_near_keywords(normalized, keywords) or _has_number(normalized)


def _date_or_deadline_present(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
            normalized,
        )
        or re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", normalized)
        or re.search(r"\b\d+\s+(?:calendar\s+)?days?\b", normalized)
        or "two weeks" in normalized
    )


def _pattern_present(patterns: Iterable[str], normalized: str) -> bool:
    return any(re.search(pattern, normalized) for pattern in patterns)


def _nearby_number_present_for_slot(slot: Dict[str, Any], normalized: str) -> bool:
    keywords = slot.get("number_near_any") or slot.get("keywords") or []
    return _number_anywhere_or_near(normalized, keywords)


def _slot_type_shape_supported(slot: Dict[str, Any], normalized: str) -> bool:
    evidence_type = slot.get("evidence_type")
    patterns = slot.get("patterns_any", [])
    if evidence_type == "text" or evidence_type == "name":
        return True
    if evidence_type == "date":
        return _pattern_present(patterns, normalized) or _date_or_deadline_present(normalized)
    if evidence_type in {"number", "money", "percent", "range", "time", "location"}:
        return _pattern_present(patterns, normalized) or _nearby_number_present_for_slot(slot, normalized)
    return True


PRIMARY_SOURCE_HOST_MARKERS = (
    "legistar.com",
    "granicus.com",
    "iqm2.com",
    "civicclerk.com",
    "novusagenda.com",
    "municode.com",
    "docs.cpuc.ca.gov",
    "apps.cpuc.ca.gov",
)

SECONDARY_SOURCE_HOST_MARKERS = (
    "apnews.com",
    "canarymedia.com",
    "cnbc.com",
    "cnn.com",
    "forbes.com",
    "fullertonobserver.com",
    "kiplinger.com",
    "latimes.com",
    "legistorm.com",
    "nytimes.com",
    "politico.com",
    "reddit.com",
    "reuters.com",
    "stoel.com",
    "substack.com",
    "twitter.com",
    "x.com",
)

SOCIAL_SOURCE_HOST_MARKERS = (
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "threads.net",
    "tiktok.com",
    "youtube.com",
)

SOURCE_PREFERENCE_STOPWORDS = {
    "agency", "decision", "decisions", "domain", "filing", "filings", "gazette",
    "government", "meeting", "official", "page", "primary", "release", "releases",
    "site", "source", "sources", "statement", "website",
}


def _source_url(source: Dict[str, Any]) -> str:
    return source.get("final_url") or source.get("url") or ""


def _source_host(source: Dict[str, Any]) -> str:
    host = urlparse(_source_url(source)).netloc.lower()
    return re.sub(r"^www\.", "", host)


def _host_matches_domain(host: str, domain: str) -> bool:
    domain = re.sub(r"^www\.", "", domain.lower().strip(" .,/"))
    return bool(domain) and (host == domain or host.endswith("." + domain))


def _domain_like_terms(text: str) -> List[str]:
    return [
        term.lower().strip(".,;:()[]{}")
        for term in re.findall(r"\b[a-z0-9][a-z0-9.-]+\.[a-z]{2,}\b", text.lower())
    ]


def _source_blob(source: Dict[str, Any], text: str = "") -> str:
    return "\n".join(
        [
            source.get("title") or "",
            source.get("candidate_title") or "",
            _source_url(source),
            text or source.get("excerpt") or "",
        ]
    )


def _entity_appears_in_source(constraints: Dict[str, Any], source_blob: str, host: str) -> bool:
    if not constraints.get("entities"):
        return False
    normalized_blob = normalize_text(source_blob)
    normalized_host = normalize_text(host.replace(".", " "))
    for entity in constraints["entities"]:
        terms = [term for term in question_terms(entity) if len(term) > 2]
        if terms and all(term in normalized_blob or term in normalized_host for term in terms):
            return True
    return False


def _looks_primary_source(host: str, source_blob: str, constraints: Dict[str, Any]) -> bool:
    if host.endswith(".gov") or host.endswith(".mil") or ".gov." in host:
        return True
    if any(marker in host for marker in PRIMARY_SOURCE_HOST_MARKERS):
        return True
    # Some local governments use branded .com domains, such as "ocgov.com".
    if "gov" in host and _entity_appears_in_source(constraints, source_blob, host):
        return True
    normalized = normalize_text(source_blob)
    if (
        _entity_appears_in_source(constraints, source_blob, host)
        and any(term in normalized for term in ("official website", "official site", "city council", "board of supervisors"))
    ):
        return True
    return False


def _looks_secondary_source(host: str) -> bool:
    return any(marker in host for marker in SECONDARY_SOURCE_HOST_MARKERS + SOCIAL_SOURCE_HOST_MARKERS)


def _preference_matches_source(
    preference: str,
    source: Dict[str, Any],
    text: str,
    constraints: Dict[str, Any],
    is_primary: bool,
) -> bool:
    host = _source_host(source)
    source_blob = _source_blob(source, text)
    normalized_source = normalize_text(source_blob + "\n" + host.replace(".", " "))
    normalized_pref = normalize_text(preference)
    if not normalized_pref:
        return False

    domain_terms = _domain_like_terms(preference)
    if domain_terms:
        return any(_host_matches_domain(host, domain) for domain in domain_terms)

    primary_pref_terms = {"official", "primary", "government", "agency", "agenda", "minutes", "docket", "filing", "regulatory"}
    if any(term in normalized_pref for term in primary_pref_terms):
        if not is_primary:
            return False
        entity_terms = [
            term
            for entity in constraints.get("entities", [])
            for term in question_terms(entity)
            if len(term) > 2
        ]
        return not entity_terms or any(term in normalized_source for term in entity_terms)

    pref_terms = [
        term
        for term in question_terms(preference)
        if term not in SOURCE_PREFERENCE_STOPWORDS and len(term) > 2
    ]
    if pref_terms and all(term in normalized_source for term in pref_terms[:4]):
        return True
    return False


def _classify_source_for_question(
    task: Dict[str, Any],
    source: Dict[str, Any],
    text: str,
    constraints: Dict[str, Any],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    host = _source_host(source)
    source_blob = _source_blob(source, text)
    is_primary = _looks_primary_source(host, source_blob, constraints)
    is_secondary = _looks_secondary_source(host)
    preferences = [str(pref) for pref in contract.get("source_preferences") or [] if str(pref).strip()]
    preference_matches = [
        preference
        for preference in preferences
        if _preference_matches_source(preference, source, text, constraints, is_primary)
    ]
    return {
        "host": host,
        "is_primary": is_primary,
        "is_secondary": is_secondary,
        "source_preferences": preferences,
        "preference_match": bool(preference_matches) if preferences else True,
        "matched_preferences": preference_matches,
    }


def _fragment_is_too_broad(label: str) -> bool:
    words = question_terms(label)
    return len(words) <= 2 and any(word.isdigit() for word in words)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(value)).strip("_")
    return slug[:72] or "answer_slot"


def _split_list_fragment(fragment: str) -> List[str]:
    normalized = normalize_text(fragment)
    normalized = re.sub(r"\bvs\.?\b|\bversus\b", ", ", normalized)
    normalized = re.sub(r"\band\s+(?:what|which|who|how|whether)\b", ", ", normalized)
    normalized = normalized.replace(" and ", ", ")
    return [part.strip(" .,:;") for part in normalized.split(",") if part.strip(" .,:;")]


def _clean_slot_label(fragment: str) -> str:
    label = normalize_text(fragment)
    label = re.sub(
        r"^(?:also\s+)?(?:what|which|who|when|where|whether|how many|how much|how did|how does|how)\s+",
        "",
        label,
    )
    label = re.sub(r"^(?:include|identify|summarize|explain|compare|list|give|say)\s+", "", label)
    label = re.sub(r"\b(?:was|were|is|are|did|does|do|listed|mentioned|reported|described|involved)\b", "", label)
    label = re.sub(r"\b(?:the|a|an|its|their|this|that)\b", "", label)
    label = re.sub(r"\s+", " ", label).strip(" .,:;")
    return label


def _question_requirement_fragments(question: str) -> List[str]:
    normalized = normalize_text(question)
    normalized = normalized.replace("?", ".")
    fragments: List[str] = []
    for marker in (
        "include ",
        "identify ",
        "summarize ",
        "explain ",
        "compare ",
        "list ",
    ):
        start = normalized.find(marker)
        if start != -1:
            tail = normalized[start + len(marker):]
            tail = re.split(r"\.(?:\s|$)", tail, maxsplit=1)[0]
            fragments.extend(_split_list_fragment(tail))

    clause_text = re.sub(r"\band\s+(?=(?:what|which|who|when|where|whether|how)\b)", ", ", normalized)
    for clause in re.split(r"[,;]", clause_text):
        clause = clause.strip()
        if re.match(r"^(?:what|which|who|when|where|whether|how many|how much|how did|how does|how)\b", clause):
            fragments.extend(_split_list_fragment(clause))

    cleaned = []
    seen = set()
    for fragment in fragments:
        label = _clean_slot_label(fragment)
        if len(label) < 3:
            continue
        if _fragment_is_too_broad(label):
            continue
        if label in {
            "question",
            "source",
            "sources",
            "evidence",
            "it",
            "they",
            "one",
            "which one",
        }:
            continue
        if label not in seen:
            seen.add(label)
            cleaned.append(label)
    return cleaned


def _infer_evidence_type(label: str) -> str:
    normalized = normalize_text(label)
    if _label_has(normalized, ("amount", "salary", "cost", "dollar", "funding", "grant")):
        return "money"
    if _label_has(normalized, ("percent", "percentage", "probability", "chance", "change", "decrease", "increase")):
        return "percent"
    if _label_has(normalized, ("date", "deadline", "when", "effective", "start", "commencing")):
        return "date"
    if _label_has(normalized, ("time", "timestamp")) and "first-time" not in normalized and "first time" not in normalized:
        return "time"
    if _label_has(normalized, ("range", "between")):
        return "range"
    if _label_has(normalized, ("location", "where", "latitude", "longitude")):
        return "location"
    if _label_has(normalized, ("count", "number", "vote", "votes", "wind", "pressure", "motion", "rate")) or "how many" in normalized:
        return "number"
    if _label_has(normalized, ("who", "person", "vendor", "company", "agency", "named")):
        return "name"
    return "text"


def _slot_patterns_for_type(evidence_type: str) -> List[str]:
    if evidence_type == "money":
        return [r"\$\s?\d", r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s+dollars?\b"]
    if evidence_type == "percent":
        return [r"\b\d+(?:\.\d+)?\s*%"]
    if evidence_type == "date":
        return [
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d+\s+(?:calendar\s+)?days?\b",
            r"\btwo\s+weeks\b",
        ]
    if evidence_type == "time":
        return [r"\b\d{1,2}:?\d{0,2}\s*(?:am|pm|utc|edt|est|cdt|cst|pdt|pst)\b"]
    if evidence_type == "range":
        return [
            r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\s*(?:%|percent\b)?",
            r"\b\d+(?:[- ]\d/\d)?\s+to\s+\d+(?:[- ]\d/\d)?\s+percent\b",
        ]
    if evidence_type == "location":
        return [r"\b\d{1,2}(?:\.\d+)?[ns]\s+\d{1,3}(?:\.\d+)?[we]\b"]
    if evidence_type == "number":
        return [r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\b"]
    return []


def _slot_from_label(label: str) -> Dict[str, Any]:
    evidence_type = _infer_evidence_type(label)
    keywords = [
        term for term in question_terms(label)
        if term not in {"what", "which", "who", "how", "many", "much", "were", "was", "did"}
    ]
    if not keywords:
        keywords = [label]
    return {
        "name": _slug(label),
        "label": label,
        "evidence_type": evidence_type,
        "keywords": keywords,
        "min_keyword_hits": 0 if normalize_text(label) in {"amount"} else min(1, len(keywords)),
        "patterns_any": _slot_patterns_for_type(evidence_type),
        "number_near_any": [],
        "search_terms": keywords[:5] or [label],
    }


def normalize_evidence_contract(contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {"required_slots": [], "source_preferences": [], "reject_if": []}
    raw_slots = contract.get("required_slots") or contract.get("slots") or []
    slots = []
    for item in raw_slots:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("description") or item.get("name")
        if not label:
            continue
        slot = _slot_from_label(str(label))
        slot["name"] = _slug(str(item.get("name") or slot["name"]))
        slot["label"] = str(label)
        slot["evidence_type"] = str(item.get("evidence_type") or slot["evidence_type"]).lower()
        slot["keywords"] = [
            normalize_text(term)
            for term in (item.get("keywords") or slot["keywords"])
            if normalize_text(term)
        ]
        slot["search_terms"] = [
            normalize_text(term)
            for term in (item.get("search_terms") or item.get("keywords") or slot["search_terms"])
            if normalize_text(term)
        ]
        slot["patterns_any"] = item.get("patterns_any") or _slot_patterns_for_type(slot["evidence_type"])
        slot["hard_veto"] = bool(item.get("hard_veto"))
        slot["number_near_any"] = item.get("number_near_any") or (
            slot["keywords"] if slot["evidence_type"] in {"number", "money", "percent", "range"} else []
        )
        slot["min_keyword_hits"] = int(item.get("min_keyword_hits") or min(2, len(slot["keywords"]) or 1))
        slots.append(slot)
    return {
        "required_slots": slots,
        "source_preferences": contract.get("source_preferences") or [],
        "reject_if": contract.get("reject_if") or [],
    }


def answer_slot_requirements(question: str, contract: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    normalized_contract = normalize_evidence_contract(contract)
    if normalized_contract["required_slots"]:
        return normalized_contract["required_slots"]
    return [_slot_from_label(label) for label in _question_requirement_fragments(question)]


def _slot_supported(slot: Dict[str, Any], text: str) -> bool:
    normalized = normalize_text(text)
    if any(term and not _keyword_present(term, normalized) for term in slot.get("terms_all", [])):
        return False
    terms_any = slot.get("terms_any", [])
    if terms_any and not any(_keyword_present(term, normalized) for term in terms_any):
        return False
    keywords = [normalize_text(term) for term in slot.get("keywords", []) if normalize_text(term)]
    if keywords:
        keyword_hits = sum(1 for term in keywords if _keyword_present(term, normalized))
        min_hits = min(int(slot["min_keyword_hits"]) if "min_keyword_hits" in slot else 1, len(keywords))
        if keyword_hits < min_hits:
            return False
    if not _slot_type_shape_supported(slot, normalized):
        return False
    return True


def evaluate_answer_slots(question: str, text: str, contract: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    slots = answer_slot_requirements(question, contract)
    results = []
    missing = []
    for slot in slots:
        supported = _slot_supported(slot, text)
        row = {
            "name": slot["name"],
            "label": slot["label"],
            "evidence_type": slot.get("evidence_type", "text"),
            "supported": supported,
            "search_terms": slot.get("search_terms", []),
        }
        results.append(row)
        if not supported:
            missing.append(row)
    return {
        "slots": results,
        "missing": missing,
        "coverage": round((len(results) - len(missing)) / len(results), 3) if results else 1.0,
    }


def select_relevant_excerpts(
    text: str,
    question: str,
    max_chars: int,
    contract: Optional[Dict[str, Any]] = None,
) -> str:
    lines = _clean_lines(text)
    if not lines:
        return ""
    terms = question_terms(question)
    item_refs = set(re.findall(r"\bitem\s+([0-9]+[a-z]?)\b", normalize_text(question)))
    normalized_question = normalize_text(question)
    slots = answer_slot_requirements(question, contract)
    page_chrome_markers = (
        "main menu", "toggle", "search submit", "stay connected", "subscribe",
        "facebook", "instagram", "linkedin", "youtube", "official websites use",
        "skip to main content", "back to home",
    )
    assertion_markers = (
        "announced", "approved", "decided", "directed", "reported", "said",
        "stated", "voted", "will", "effective", "for release",
    )

    chunks: List[Tuple[int, int, int, str]] = []
    window = 28 if slots or any(term in normalized_question for term in ("advisory", "statement", "outlook")) else 14
    stride = 10 if window > 14 else 7
    for start in range(0, len(lines), stride):
        chunk_lines = lines[start:start + window]
        chunk = "\n".join(chunk_lines)
        hay = normalize_text(chunk)
        score = 0
        for term in terms:
            if term in hay:
                score += 1
        for slot in slots:
            if _slot_supported(slot, chunk):
                score += 8
            else:
                score += sum(2 for term in slot.get("search_terms", []) if normalize_text(term) in hay)
        for item in item_refs:
            if re.search(rf"(^|\s){re.escape(item)}(\s|$)", hay):
                score += 8
        if score and re.search(r"\b\d+(?:[./-]\d+)?(?:\s*(?:to|-|–)\s*\d+(?:[./-]\d+)?)?\s*(?:percent|%)\b", hay):
            score += 8
        elif score and re.search(r"\b\d[\d,]*(?:\.\d+)?\b", hay):
            score += 3
        if score and any(marker in hay for marker in assertion_markers):
            score += 3
        chrome_hits = sum(1 for marker in page_chrome_markers if marker in hay)
        if chrome_hits and not any(marker in hay for marker in assertion_markers):
            score -= chrome_hits * 4
        if slots and score < 4:
            continue
        if score:
            chunks.append((score, start, min(start + window, len(lines)), chunk))

    if not chunks:
        return "\n".join(lines)[:max_chars]

    chunks.sort(key=lambda row: (-row[0], row[1]))
    selected: List[Tuple[int, int, str]] = []
    selected_ranges: List[Tuple[int, int]] = []
    used_chars = 0
    for score, start, end, chunk in chunks:
        if any(not (end < used_start or start > used_end) for used_start, used_end in selected_ranges):
            continue
        chunk = chunk[:max_chars]
        if used_chars + len(chunk) > max_chars and selected:
            continue
        selected.append((start, score, chunk))
        selected_ranges.append((start, end))
        used_chars += len(chunk)
        if used_chars >= max_chars:
            break

    selected.sort(key=lambda row: row[0])
    joined = "\n\n---\n\n".join(f"[chunk score {score}]\n{chunk}" for _, score, chunk in selected)
    joined = joined[:max_chars]

    excerpt_evaluation = evaluate_answer_slots(question, joined, contract)
    full_evaluation = evaluate_answer_slots(question, text, contract)
    missing_names = {slot["name"] for slot in excerpt_evaluation.get("missing", [])}
    rescue_slots = [
        slot
        for slot in answer_slot_requirements(question, contract)
        if slot["name"] in missing_names
        and any(row["name"] == slot["name"] and row["supported"] for row in full_evaluation.get("slots", []))
    ]
    if not rescue_slots:
        return joined

    rescue_chunks: List[Tuple[int, int, str]] = []
    for slot in rescue_slots:
        slot_terms = [
            normalize_text(term)
            for term in (slot.get("search_terms") or slot.get("keywords") or [slot.get("label", "")])
            if normalize_text(term)
        ]
        best: Optional[Tuple[int, int, str]] = None
        for start in range(0, len(lines), 8):
            chunk_lines = lines[start:start + 18]
            chunk = "\n".join(chunk_lines)
            hay = normalize_text(chunk)
            if not _slot_supported(slot, chunk):
                continue
            score = 20 + sum(2 for term in slot_terms if term in hay)
            if best is None or score > best[0]:
                best = (score, start, chunk)
        if best:
            rescue_chunks.append(best)

    if not rescue_chunks:
        return joined

    rescue_chunks.sort(key=lambda row: (-row[0], row[1]))
    rescue_budget = min(3000, max(800, max_chars // 3))
    base_limit = max(800, max_chars - rescue_budget)
    if len(joined) > base_limit:
        joined = joined[:base_limit].rstrip()
    for score, _, chunk in rescue_chunks:
        block = f"\n\n---\n\n[slot rescue score {score}]\n{chunk}"
        if len(joined) + len(block) > max_chars:
            remaining = max_chars - len(joined)
            if remaining <= 80:
                break
            joined += block[:remaining]
            break
        joined += block
    return joined[:max_chars]


def fetch_controlled_evidence(
    task: Dict[str, Any],
    max_source_chars: int = 9000,
    timeout: int = 30,
) -> Dict[str, Any]:
    sources = []
    for url in task.get("preferred_sources", []):
        source: Dict[str, Any] = {"url": url}
        try:
            fetched = fetch_source_text(url, timeout=timeout)
            excerpt = select_relevant_excerpts(
                fetched["text"],
                task["question"],
                max_source_chars,
                task.get("evidence_contract"),
            )
            source.update(
                {
                    "ok": True,
                    "final_url": fetched["url"],
                    "title": fetched["title"],
                    "content_type": fetched["content_type"],
                    "char_count": fetched["char_count"],
                    "fetch_mode": fetched.get("fetch_mode", "direct"),
                    "published_date": fetched.get("published_date"),
                    "excerpt": excerpt,
                    "excerpt_chars": len(excerpt),
                    "links": fetched.get("links", []),
                }
            )
        except Exception as exc:
            source.update({"ok": False, "error": str(exc)})
        sources.append(source)
    return {"sources": sources}


def fetch_discovered_evidence(
    task: Dict[str, Any],
    candidates: List[Dict[str, str]],
    max_source_chars: int = 9000,
    timeout: int = 30,
) -> Dict[str, Any]:
    sources = []
    for candidate in candidates:
        url = candidate.get("url", "")
        source: Dict[str, Any] = {
            "url": url,
            "canonical_url": candidate.get("canonical_url") or canonical_url(url),
            "candidate_title": candidate.get("title", ""),
            "candidate_reason": candidate.get("reason", ""),
            "candidate_source": candidate.get("source", ""),
        }
        try:
            fetched = fetch_source_text(url, timeout=timeout)
            excerpt = select_relevant_excerpts(
                fetched["text"],
                task["question"],
                max_source_chars,
                task.get("evidence_contract"),
            )
            source.update(
                {
                    "ok": True,
                    "final_url": fetched["url"],
                    "title": fetched["title"],
                    "content_type": fetched["content_type"],
                    "char_count": fetched["char_count"],
                    "fetch_mode": fetched.get("fetch_mode", "direct"),
                    "published_date": fetched.get("published_date"),
                    "excerpt": excerpt,
                    "excerpt_chars": len(excerpt),
                    "links": fetched.get("links", []),
                }
            )
        except Exception as exc:
            source.update({"ok": False, "error": str(exc)})
        sources.append(source)
    return {"sources": sources}


def _url_has_unfetchable_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    return bool(re.search(r"\.(?:css|gif|ico|jpg|jpeg|js|json|png|svg|webp|xml|zip)$", path))


def _constraint_time_terms(constraints: Dict[str, Any]) -> List[str]:
    terms = []
    offsets = {
        "utc": 0,
        "edt": 4,
        "est": 5,
        "cdt": 5,
        "cst": 6,
        "pdt": 7,
        "pst": 8,
        "et": 4,
        "ct": 5,
        "pt": 7,
    }
    for raw_time in constraints.get("times", []):
        normalized = normalize_text(raw_time).replace(".", "")
        terms.append(normalized)
        match = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\s*([a-z]{2,3})?\b", normalized)
        if not match:
            continue
        hour = int(match.group(1))
        minute = int(match.group(2))
        meridiem = match.group(3) or ""
        tz = match.group(4) or ""
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        terms.extend([
            f"{hour:02d}{minute:02d}",
            f"{hour}{minute:02d}",
            f"{(hour % 12) or 12}{minute:02d}",
        ])
        if tz in offsets:
            utc_hour = (hour + offsets[tz]) % 24
            terms.extend([f"{utc_hour:02d}{minute:02d}", f"{utc_hour:02d}{minute:02d} utc"])
    return list(dict.fromkeys(term for term in terms if term))


def _constraint_date_terms(constraints: Dict[str, Any]) -> List[str]:
    terms = []
    for date_info in constraints.get("dates", []):
        terms.extend(_date_variants(date_info["month"], date_info["day"], date_info["year"]))
        terms.append(f"{date_info['month']:02d}{date_info['day']:02d}")
        terms.append(f"{date_info['day']:02d}")
        terms.append(str(date_info["year"]))
    for year in constraints.get("target_years", []):
        terms.append(str(year))
    return list(dict.fromkeys(normalize_text(term) for term in terms if normalize_text(term)))


def _nearby_link_score(task: Dict[str, Any], link: Dict[str, str], constraints: Dict[str, Any]) -> int:
    url = link.get("url", "")
    hay = normalize_text(" ".join([link.get("text", ""), urlparse(url).path, urlparse(url).query]))
    if not hay:
        return 0

    score = 0
    for term in question_terms(task["question"]):
        if term in hay:
            score += 2

    for slot in answer_slot_requirements(task["question"], task.get("evidence_contract")):
        for term in slot.get("search_terms", []) + slot.get("keywords", []):
            if normalize_text(term) in hay:
                score += 3

    for term in _constraint_date_terms(constraints):
        if term and term in hay:
            score += 3

    for term in _constraint_time_terms(constraints):
        if term and term in hay:
            score += 5

    doc_words = set(constraints.get("document_types", []))
    doc_words.update({"archive", "pdf", "document", "attachment"})
    for word in doc_words:
        if word in hay:
            score += 2

    if any(word in hay for word in ("public", "advisory", "statement", "agenda", "minutes", "decision", "release", "report")):
        score += 2
    if "archive" in hay:
        score += 8
    if any(word in hay for word in ("index", "contents", "published", "document", "attachment")):
        score += 3

    question_hay = normalize_text(task["question"])
    if "map" not in question_hay and "graphic" not in question_hay:
        if any(word in hay for word in ("graphic", "graphics", "cone", "map", "swath", "radii", "probabilities", "probability")):
            score -= 8
    if "local" not in question_hay and "product" not in question_hay:
        if any(word in hay for word in ("local product", "local products")):
            score -= 8
    if any(word in hay for word in ("facebook", "instagram", "youtube", "privacy", "subscribe", "contact", "login")):
        score -= 4
    return score


def nearby_source_candidates(
    task: Dict[str, Any],
    sources: List[Dict[str, Any]],
    seen_urls: Iterable[str],
    limit: int = 5,
) -> List[Dict[str, str]]:
    constraints = extract_constraints(task["question"])
    seen = set(seen_urls)
    candidates: List[Tuple[int, Dict[str, str]]] = []
    for source in sources:
        if not source.get("ok") or not source.get("links"):
            continue
        validation = source.get("validation") or {}
        source_quality = (validation.get("matched") or {}).get("source_quality") or {}
        if not source_quality.get("is_primary"):
            continue
        source_host = _source_host(source)
        if not source_host:
            continue
        for link in source.get("links", []):
            url = link.get("url", "")
            canon = canonical_url(url)
            if not canon or canon in seen or _url_has_unfetchable_extension(url):
                continue
            link_host = re.sub(r"^www\.", "", urlparse(url).netloc.lower())
            if link_host != source_host:
                continue
            score = _nearby_link_score(task, link, constraints)
            if score <= 0:
                continue
            candidates.append(
                (
                    score,
                    {
                        "url": url,
                        "canonical_url": canon,
                        "title": link.get("text", ""),
                        "reason": f"nearby official link from {source.get('final_url') or source.get('url')} score={score}",
                        "source": "nearby_link",
                    },
                )
            )
    candidates.sort(key=lambda row: (-row[0], row[1]["url"]))
    return dedupe_url_candidates([candidate for _, candidate in candidates], limit=limit)


def _date_variants(month: int, day: int, year: int) -> List[str]:
    month_name = [name for name, value in MONTHS.items() if value == month][0]
    month_cap = month_name.title()
    month_abbr = month_cap[:3]
    return [
        f"{month_name} {day}, {year}",
        f"{month_cap} {day}, {year}",
        f"{month_abbr} {day}, {year}",
        f"{month_abbr} {day} {year}",
        f"{month}/{day}/{year}",
        f"{month:02d}/{day:02d}/{year}",
        f"{year}-{month:02d}-{day:02d}",
    ]


ENTITY_SUFFIXES = (
    "Agency", "Authority", "Board", "Center", "Commission", "Committee", "Council",
    "Court", "Department", "District", "Office", "Reserve", "University", "County",
)


def _clean_entity_phrase(phrase: str) -> str:
    phrase = re.sub(r"^(?:according to|for|from|using|as of|the|a|an)\s+", "", phrase.strip(), flags=re.IGNORECASE)
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.:;?'\"")
    return phrase


def _extract_named_entities(question: str) -> List[str]:
    candidates: List[str] = []
    suffix_pattern = "|".join(re.escape(suffix) for suffix in ENTITY_SUFFIXES)
    phrase_pattern = re.compile(
        rf"\b(?:[A-Z][A-Za-z0-9&.-]*|[A-Z]{{2,}})"
        rf"(?:\s+(?:of|the|and|for|in|[A-Z][A-Za-z0-9&.-]*|[A-Z]{{2,}})){{0,6}}"
        rf"\s+(?:{suffix_pattern})\b"
    )
    for match in phrase_pattern.finditer(question):
        candidates.append(_clean_entity_phrase(match.group(0)))

    possessive_pattern = re.compile(
        r"\b((?:[A-Z][A-Za-z0-9&.-]*|[A-Z]{2,})(?:\s+(?:of|the|and|for|in|[A-Z][A-Za-z0-9&.-]*|[A-Z]{2,})){0,5})'s\b"
    )
    for match in possessive_pattern.finditer(question):
        candidates.append(_clean_entity_phrase(match.group(1)))

    entities = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_text(candidate)
        terms = question_terms(normalized)
        if not normalized or normalized in MONTHS or normalized in {"according", "using", "current"}:
            continue
        if len(terms) == 1 and len(terms[0]) < 3:
            continue
        if normalized not in seen:
            seen.add(normalized)
            entities.append(normalized)
    return entities


def extract_constraints(question: str) -> Dict[str, Any]:
    normalized = normalize_text(question)
    dates = []
    for match in re.finditer(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),\s+(\d{4})\b",
        normalized,
    ):
        before = normalized[max(0, match.start() - 12):match.start()]
        dates.append(
            {
                "month": MONTHS[match.group(1)],
                "day": int(match.group(2)),
                "year": int(match.group(3)),
                "soft": "as of" in before,
                "text": match.group(0),
            }
        )
    for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", normalized):
        dates.append(
            {
                "month": int(match.group(1)),
                "day": int(match.group(2)),
                "year": int(match.group(3)),
                "soft": False,
                "text": match.group(0),
            }
        )

    times = re.findall(
        r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:cdt|cst|edt|est|pdt|pst|utc|et|ct|pt)?\b",
        normalized,
    )
    item_refs = re.findall(r"\bitem\s+([0-9]+[a-z]?)\b", normalized)
    target_years = []
    seen_years = set()
    for pattern in (
        r"\b(?:in|for|during)\s+(20\d{2})\b",
        r"\b(20\d{2})\s+(?:[a-z][a-z-]*\s+){0,4}(?:advisory|cases?|count|outlook|release|report|results?|rule|season|statement)\b",
    ):
        for match in re.finditer(pattern, normalized):
            year = match.group(1)
            if year not in seen_years:
                seen_years.add(year)
                target_years.append(year)

    entities = _extract_named_entities(question)

    document_types = []
    for term in ("agenda", "statement", "advisory", "release", "report", "pdf", "count", "outlook"):
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            document_types.append(term)

    return {
        "dates": dates,
        "times": times,
        "item_refs": item_refs,
        "target_years": target_years,
        "entities": entities,
        "document_types": document_types,
        "terms": question_terms(question),
    }


def _date_present(text: str, date_info: Dict[str, Any]) -> bool:
    hay = normalize_text(text)
    return any(normalize_text(variant) in hay for variant in _date_variants(
        date_info["month"], date_info["day"], date_info["year"]
    ))


# --- as-of date gating -------------------------------------------------------
# A source whose fetched text never states a date cannot be shown to be stale.
# News pages routinely carry the publication date in metadata, a <time> element,
# or as a relative string ("2 hours ago"), none of which survive text extraction.
# Gating those on the question's as-of date rejected 78% of the coverage the
# hunter had already fetched and read, which is what drove the overview's side
# stories to `no_evidence`. So the date gate applies only to sources that state a
# date, with a few days of slack for time zones and for stories the show reaches
# a day or two late.
DATE_WINDOW_BACK_DAYS = 3
DATE_WINDOW_FORWARD_DAYS = 1
# Reject sources whose publication date cannot be established at all, from either
# page metadata or the body text. Recency is the whole point of a news source, so
# "we cannot tell how old this is" is treated as a failure rather than a pass.
REQUIRE_KNOWN_PUBLISH_DATE = True

_MONTH_WORDS = sorted(
    set(list(MONTHS) + [name[:3] for name in MONTHS]), key=len, reverse=True
)
_ANY_DATE_RE = re.compile(
    r"\b(?:" + "|".join(_MONTH_WORDS) + r")\.?\s+\d{1,2}\b"
    r"|\b\d{4}-\d{2}-\d{2}"  # no trailing \b: ISO stamps run straight into "T09:00"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE,
)


def _text_states_a_date(text: str) -> bool:
    """True when the text names a date we could actually compare against."""
    return bool(_ANY_DATE_RE.search(text or ""))


def _date_present_in_window(text: str, date_info: Dict[str, Any],
                            back_days: int = DATE_WINDOW_BACK_DAYS,
                            forward_days: int = DATE_WINDOW_FORWARD_DAYS) -> bool:
    """True when the text names any date inside the recency window around ``date_info``."""
    try:
        base = datetime(date_info["year"], date_info["month"], date_info["day"])
    except (KeyError, TypeError, ValueError):
        return _date_present(text, date_info)
    for offset in range(-forward_days, back_days + 1):
        shifted = base - timedelta(days=offset)
        if _date_present(
            text, {"month": shifted.month, "day": shifted.day, "year": shifted.year}
        ):
            return True
    return False


def _as_of_hard_dates(as_of: Optional[str]) -> List[Dict[str, Any]]:
    """Hard-date constraints parsed from a task's as-of date string, or []."""
    if not as_of:
        return []
    try:
        return [d for d in extract_constraints(str(as_of))["dates"] if not d.get("soft")]
    except Exception:
        return []


def _iso_within_window(published_iso: str,
                       hard_dates: List[Dict[str, Any]],
                       back_days: int = DATE_WINDOW_BACK_DAYS,
                       forward_days: int = DATE_WINDOW_FORWARD_DAYS) -> bool:
    """True when a yyyy-mm-dd publish date sits inside the window of any hard date."""
    try:
        year, month, day = (int(part) for part in published_iso.split("-")[:3])
        published = datetime(year, month, day)
    except (ValueError, AttributeError):
        return True  # unparseable: fall back to the other checks rather than veto
    for info in hard_dates:
        try:
            base = datetime(info["year"], info["month"], info["day"])
        except (KeyError, TypeError, ValueError):
            continue
        delta = (base - published).days
        if -forward_days <= delta <= back_days:
            return True
    return False


def _time_present(text: str, wanted_time: str) -> bool:
    hay = normalize_text(text)
    wanted = normalize_text(wanted_time).replace(".", "")
    loose = wanted.replace(" ", "")
    compact_hay = re.sub(r"[\s:.]", "", hay)
    compact_wanted = re.sub(r"[\s:.]", "", wanted)
    return (
        wanted in hay.replace(".", "")
        or loose in hay.replace(" ", "").replace(".", "")
        or compact_wanted in compact_hay
    )


def _question_term_present(term: str, text: str) -> bool:
    normalized = normalize_text(text)
    if term in normalized:
        return True
    aliases = {
        "rfp": ["request for proposal"],
        "amount": ["$", "dollar", "cost"],
        "work": ["analysis", "design", "service"],
        "approved": ["approve", "approval"],
        "advisory": ["bulletin", "forecast advisory", "public advisory"],
    }
    return any(alias in normalized for alias in aliases.get(term, []))


def _contract_reject_reasons(
    contract: Dict[str, Any],
    text: str,
    slot_evaluation: Dict[str, Any],
    source_quality: Optional[Dict[str, Any]] = None,
) -> List[str]:
    normalized = normalize_text(text)
    missing_slots = slot_evaluation.get("missing", [])
    if not missing_slots:
        return []

    trap_markers = (
        "preview", "webinar",
        "registration", "register", "will release", "will host", "to release",
    )
    secondary_trap_markers = ("summary", "copy", "recap", "general", "statistics", "stats", "commentary")
    missing_markers = (
        "without", "does not provide", "do not provide", "lacks", "missing",
        "not provide", "no specific", "no numeric", "without the actual", "other than",
    )
    source_quality = source_quality or {}
    is_primary = bool(source_quality.get("is_primary"))
    reasons = []
    for reject in contract.get("reject_if") or []:
        reject_text = normalize_text(reject)
        if not reject_text:
            continue
        if "other than" in reject_text and any(term in reject_text for term in ("date", "year", "time")):
            continue
        if is_primary and any(marker in reject_text for marker in ("summary", "commentary", "news", "secondary", "copy")):
            continue
        names_missing = any(marker in reject_text for marker in missing_markers)
        active_trap_markers = trap_markers if is_primary else trap_markers + secondary_trap_markers
        trap_on_page = any(marker in normalized for marker in active_trap_markers)
        trap_in_rule = any(marker in reject_text for marker in active_trap_markers)
        if names_missing and (trap_on_page or trap_in_rule):
            reasons.append(f"contract_reject:{_slug(reject_text)}")
    return reasons


def validate_source_for_question(task: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    constraints = extract_constraints(task["question"])
    title = source.get("title") or source.get("candidate_title") or ""
    url = source.get("final_url") or source.get("url") or ""
    excerpt = source.get("excerpt") or ""
    text = "\n".join([title, url, excerpt])
    normalized = normalize_text(text)
    reasons = []
    matched: Dict[str, Any] = {}

    if not source.get("ok"):
        return {
            "passed": False,
            "score": 0,
            "matched": matched,
            "reasons": ["fetch_failed"],
        }

    hard_dates = [date for date in constraints["dates"] if not date.get("soft")]
    # Always append the task's as-of date as an extra anchor. On the research-agent
    # path the question text has no literal date, so without this the whole date
    # block is skipped and main-story sources get no recency validation at all. On
    # the overview path it duplicates the question's own date (harmless). Appending
    # rather than replacing also fixes an event-anchor trap: a question about an
    # Aug 22 event asked on Aug 27 must not reject fresh Aug 27 coverage.
    for extra in _as_of_hard_dates(task.get("as_of")):
        if not any(
            h.get("year") == extra["year"]
            and h.get("month") == extra["month"]
            and h.get("day") == extra["day"]
            for h in hard_dates
        ):
            hard_dates.append(extra)
    if hard_dates:
        # Exact matches still drive `score` below, so ranking is unchanged.
        matched_dates = [_date_present(text, date) for date in hard_dates]
        matched["dates"] = matched_dates
        if not any(matched_dates):
            published = source.get("published_date")
            if published:
                matched["published_date"] = published
                if not _iso_within_window(published, hard_dates):
                    reasons.append("date_mismatch")
            elif _text_states_a_date(text):
                if not any(
                    _date_present_in_window(text, date) for date in hard_dates
                ):
                    reasons.append("date_mismatch")
            elif REQUIRE_KNOWN_PUBLISH_DATE:
                # Nothing in the metadata and nothing in the body says when this
                # was written. A headline like "California fire" matches a story
                # from yesterday and one from six years ago equally well, so an
                # undateable source is not evidence that something happened
                # today. Flip this constant to relax the rule.
                reasons.append("date_unknown")

    if constraints["times"]:
        matched_times = [_time_present(text, time_value) for time_value in constraints["times"]]
        matched["times"] = matched_times
        if not any(matched_times):
            reasons.append("time_mismatch")

    if constraints["target_years"]:
        matched_years = [year in normalized for year in constraints["target_years"]]
        matched["target_years"] = matched_years
        if not any(matched_years):
            reasons.append("year_mismatch")

    if constraints["item_refs"]:
        if "calendar.aspx" in normalize_text(url):
            reasons.append("document_mismatch:calendar_listing")
        item_matches = []
        for item in constraints["item_refs"]:
            item_matches.append(bool(re.search(rf"(^|\s|#){re.escape(item)}(\s|$)", normalized)))
        matched["items"] = item_matches
        if not any(item_matches):
            reasons.append("item_mismatch")

    entity_matches = {}
    for entity in constraints["entities"]:
        required = question_terms(entity)
        ok = all(term in normalized for term in required)
        entity_matches[entity] = ok
        if not ok:
            reasons.append(f"entity_mismatch:{entity}")
    if entity_matches:
        matched["entities"] = entity_matches

    entity_terms = {
        term
        for entity in constraints["entities"]
        for term in question_terms(entity)
    }
    terms = [
        term for term in constraints["terms"]
        if term not in {
            "agenda", "item", "june", "2026",
            "identify", "what", "from", "according", "current", "source", "date",
            "meeting", "city", "official", "using", "include", "for", "described",
        }
        and term not in entity_terms
        and not term.isdigit()
    ]
    term_hits = [term for term in terms if _question_term_present(term, text)]
    matched["topic_terms"] = term_hits
    if constraints["item_refs"] and len(term_hits) < 2:
        reasons.append("topic_mismatch")
    elif terms and not term_hits:
        reasons.append("topic_mismatch")

    normalized_contract = normalize_evidence_contract(task.get("evidence_contract"))
    source_quality = _classify_source_for_question(task, source, text, constraints, normalized_contract)
    matched["source_quality"] = source_quality
    if normalized_contract.get("source_preferences") and not source_quality["preference_match"]:
        reasons.append("source_preference_mismatch")

    generated_contract = bool(normalized_contract["required_slots"])
    slot_evaluation = evaluate_answer_slots(task["question"], text, normalized_contract)
    slot_lookup = _slot_by_name(slot_evaluation.get("slots", []))
    if slot_evaluation["slots"]:
        matched["evidence_slots"] = slot_evaluation
        for slot in slot_evaluation["missing"]:
            reasons.append(f"evidence_missing:{slot['name']}")
    reasons.extend(_contract_reject_reasons(normalized_contract, text, slot_evaluation, source_quality))

    score = 0
    score += 2 if matched.get("dates") and any(matched["dates"]) else 0
    score += 2 if matched.get("items") and any(matched["items"]) else 0
    score += 2 if matched.get("entities") and all(matched["entities"].values()) else 0
    score += min(3, len(term_hits))
    score += sum(1 for slot in slot_evaluation.get("slots", []) if slot.get("supported"))
    score += 1 if source.get("ok") else 0

    # Base constraints (date / year / time / item / document / entity) always gate — a
    # source that fails these is genuinely off-target for the question.
    base_hard = {
        reason for reason in reasons
        if reason.startswith("date_mismatch")
        or reason.startswith("date_unknown")
        or reason.startswith("year_mismatch")
        or reason.startswith("time_mismatch")
        or reason.startswith("item_mismatch")
        or reason.startswith("document_mismatch")
        or reason.startswith("entity_mismatch")
    }
    # Contract-derived rejections come from the LLM-generated evidence contract
    # (source-preference rules, reject_if traps, required-slot coverage). That contract is
    # well-calibrated for precise factual tasks (Fed rate, council votes) but miscalibrated
    # for breaking news, where it invents slots a confirmed story may not surface, narrows
    # source preferences past major outlets, and trips "speculative" traps on real coverage —
    # collapsing recall to zero (a UK-PM resignation confirmed by NBC/CBS/NPR was rejected on
    # every source). For news_research the contract is therefore ADVISORY: it still shapes
    # `score` (ranking) but does not veto a source that clears the base checks. For every
    # other category it remains a hard gate.
    contract_hard = {
        reason for reason in reasons
        if reason.startswith("source_preference_mismatch")
        or reason.startswith("contract_reject")
        or _hard_evidence_missing(reason, slot_lookup, generated_contract)
    }
    contract_advisory = task.get("category") == "news_research"
    hard_fail_reasons = base_hard if contract_advisory else base_hard | contract_hard
    passed = not hard_fail_reasons and "topic_mismatch" not in reasons
    return {
        "passed": passed,
        "score": score,
        "matched": matched,
        "reasons": reasons,
    }


def filter_validated_evidence(task: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    validated = []
    rejected = []
    for source in evidence.get("sources", []):
        result = validate_source_for_question(task, source)
        row = dict(source)
        row["validation"] = result
        if result["passed"]:
            validated.append(row)
        else:
            rejected.append(row)
    validated.sort(
        key=lambda source: (
            bool(((source.get("validation") or {}).get("matched") or {}).get("source_quality", {}).get("preference_match")),
            bool(((source.get("validation") or {}).get("matched") or {}).get("source_quality", {}).get("is_primary")),
            (source.get("validation") or {}).get("score") or 0,
        ),
        reverse=True,
    )
    return {
        "sources": validated,
        "rejected_sources": rejected,
    }


def _legistar_calendar_url(task: Dict[str, Any]) -> Optional[str]:
    constraints = extract_constraints(task["question"])
    if "riverside city council" not in constraints["entities"]:
        return None
    dates = [date for date in constraints["dates"] if not date.get("soft")]
    if not dates:
        return "https://riversideca.legistar.com/Calendar.aspx"
    date = dates[0]
    return (
        "https://riversideca.legistar.com/Calendar.aspx?"
        f"From={date['year']}-{date['month']:02d}-01&To={date['year']}-{date['month']:02d}-31"
    )


def legistar_calendar_candidates(task: Dict[str, Any], timeout: int = 30) -> List[Dict[str, str]]:
    calendar_url = _legistar_calendar_url(task)
    if not calendar_url:
        return []
    constraints = extract_constraints(task["question"])
    dates = [date for date in constraints["dates"] if not date.get("soft")]
    date = dates[0] if dates else None
    response = requests.get(
        calendar_url,
        timeout=timeout,
        headers=FETCH_HEADERS,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    candidates = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "MeetingDetail.aspx" not in href:
            continue
        row = link.find_parent("tr")
        row_text = row.get_text(" ", strip=True) if row else ""
        row_norm = normalize_text(row_text)
        if "city council" not in row_norm:
            continue
        if date and not _date_present(row_text, date):
            continue
        url = requests.compat.urljoin(calendar_url, href)
        candidates.append(
            {
                "url": url,
                "title": row_text[:180],
                "reason": "legistar_calendar_match",
                "source": "legistar_resolver",
            }
        )
    return dedupe_url_candidates(candidates, limit=5)


def federal_reserve_statement_candidates(task: Dict[str, Any], timeout: int = 30) -> List[Dict[str, str]]:
    constraints = extract_constraints(task["question"])
    terms = set(constraints["terms"])
    asks_fed_statement = (
        "statement" in constraints["document_types"]
        and (
            "fomc" in terms
            or "federal reserve" in " ".join(constraints["entities"])
            or {"federal", "reserve"} <= terms
        )
    )
    if not asks_fed_statement:
        return []
    dates = [date for date in constraints["dates"] if not date.get("soft")]
    if not dates:
        return []
    date = dates[0]
    stamp = f"{date['year']}{date['month']:02d}{date['day']:02d}"
    calendar_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    response = requests.get(
        calendar_url,
        timeout=timeout,
        headers=FETCH_HEADERS,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    candidates = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if f"monetary{stamp}a.htm" not in href:
            continue
        url = requests.compat.urljoin(calendar_url, href)
        candidates.append(
            {
                "url": url,
                "title": link.get_text(" ", strip=True) or f"FOMC statement {stamp}",
                "reason": "federal_reserve_calendar_statement_match",
                "source": "federal_reserve_resolver",
            }
        )
    return dedupe_url_candidates(candidates, limit=3)


def noaa_hurricane_outlook_candidates(task: Dict[str, Any], timeout: int = 30) -> List[Dict[str, str]]:
    constraints = extract_constraints(task["question"])
    normalized = normalize_text(task["question"])
    terms = set(constraints["terms"])
    if "noaa" not in terms or "outlook" not in constraints["document_types"]:
        return []
    if "atlantic" not in normalized or "hurricane" not in normalized:
        return []
    return [
        {
            "url": "https://www.cpc.ncep.noaa.gov/products/outlooks/hurricane.shtml",
            "title": "NOAA Climate Prediction Center Atlantic Hurricane Outlook",
            "reason": "noaa_cpc_hurricane_outlook_resolver",
            "source": "noaa_outlook_resolver",
        }
    ]


def _storm_name_from_question(question: str) -> Optional[str]:
    normalized = normalize_text(question)
    match = re.search(
        r"\b(?:tropical storm|hurricane|tropical depression|potential tropical cyclone)\s+([a-z]+)\b",
        normalized,
    )
    if match:
        return match.group(1).upper()
    return None


def _utc_hhmm_for_time(time_value: str) -> Optional[str]:
    normalized = normalize_text(time_value).replace(".", "")
    match = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\s*(cdt|cst|edt|est|pdt|pst|utc|et|ct|pt)?", normalized)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3)
    zone = match.group(4) or "utc"
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    offsets = {
        "utc": 0,
        "cdt": -5,
        "cst": -6,
        "ct": -5,
        "edt": -4,
        "est": -5,
        "et": -4,
        "pdt": -7,
        "pst": -8,
        "pt": -7,
    }
    utc_hour = (hour - offsets.get(zone, 0)) % 24
    return f"{utc_hour:02d}{minute:02d}"


def nhc_advisory_archive_candidates(task: Dict[str, Any], timeout: int = 30) -> List[Dict[str, str]]:
    constraints = extract_constraints(task["question"])
    terms = set(constraints["terms"])
    if "nhc" not in terms or "advisory" not in constraints["document_types"]:
        return []
    storm_name = _storm_name_from_question(task["question"])
    dates = [date for date in constraints["dates"] if not date.get("soft")]
    if not storm_name or not dates:
        return []
    date = dates[0]
    archive_url = f"https://www.nhc.noaa.gov/archive/{date['year']}/{storm_name}.shtml"
    response = requests.get(
        archive_url,
        timeout=timeout,
        headers=FETCH_HEADERS,
    )
    response.raise_for_status()
    html_text = response.text
    wanted_hhmm = _utc_hhmm_for_time(constraints["times"][0]) if constraints["times"] else None
    stamp_prefix = f"{date['year']}{date['month']:02d}{date['day']:02d}"
    candidates = []
    pattern = re.compile(
        r"<!--\s*(\d{8})\s+(\d{4})\s*-->\s*<a\s+href=\"([^\"]*public(?:_a)?\.[^\"]+\.shtml)\"[^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html_text):
        day_stamp, hhmm, href, link_text = match.groups()
        if day_stamp != stamp_prefix:
            continue
        if wanted_hhmm and hhmm != wanted_hhmm:
            continue
        url = requests.compat.urljoin(archive_url, html.unescape(href))
        title = BeautifulSoup(link_text, "html.parser").get_text(" ", strip=True)
        candidates.append(
            {
                "url": url,
                "title": f"NHC {storm_name.title()} public advisory {title} UTC",
                "reason": "nhc_archive_public_advisory_match",
                "source": "nhc_archive_resolver",
            }
        )
    return dedupe_url_candidates(candidates, limit=5)


def _evidence_checklist(question: str, contract: Optional[Dict[str, Any]] = None) -> List[str]:
    return [slot["label"] for slot in answer_slot_requirements(question, contract)]


def source_hunter_extra_context(state: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> str:
    rejected = state.get("rejected_sources", [])[-8:]
    checklist = _evidence_checklist(task["question"], task.get("evidence_contract")) if task else []
    contract = normalize_evidence_contract(task.get("evidence_contract")) if task else {"source_preferences": []}
    source_preferences = contract.get("source_preferences") or []
    if not rejected and not checklist and not source_preferences:
        return ""
    lines = []
    if source_preferences:
        lines.append("Prefer sources matching this generated source preference:")
        for item in source_preferences:
            lines.append(f"- {item}")
    if checklist:
        lines.append("The accepted source must contain evidence for these requested answer fields:")
        for item in checklist:
            lines.append(f"- {item}")
    if rejected:
        lines.append("Previous fetched URLs were rejected. Avoid repeating them:")
    for source in rejected:
        validation = source.get("validation") or {}
        missing = [
            slot.get("label") or slot.get("name")
            for slot in ((validation.get("matched") or {}).get("evidence_slots") or {}).get("missing", [])
        ]
        missing_text = f"; missing={', '.join(missing)}" if missing else ""
        lines.append(
            f"- {source.get('final_url') or source.get('url')} rejected for "
            f"{', '.join(validation.get('reasons') or ['unknown'])}; "
            f"title={source.get('title') or source.get('candidate_title')}"
            f"{missing_text}"
        )
    lines.append(
        "Search for a page matching the exact date/entity/item/time constraints and the missing answer fields. "
        "Do not return event announcements, calendars, previews, recaps, or secondary copies unless they contain the requested facts."
    )
    return "\n".join(lines)


def score_discovery(task: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    preferred = [canonical_url(url) for url in task.get("preferred_sources", [])]
    preferred = [url for url in preferred if url]
    preferred_hosts = {urlparse(url).netloc.lower() for url in preferred}
    source_domains = {source.get("domain", "").lower() for source in task.get("source_domains", [])}
    source_domains = {domain for domain in source_domains if domain}

    candidates = evidence.get("sources", [])
    fetched = [source for source in candidates if source.get("ok")]
    candidate_canons = [source.get("canonical_url") or canonical_url(source.get("url", "")) for source in candidates]
    fetched_canons = [
        canonical_url(source.get("final_url") or source.get("url", ""))
        for source in fetched
    ]
    candidate_hosts = {urlparse(url).netloc.lower() for url in candidate_canons if url}
    fetched_hosts = {urlparse(url).netloc.lower() for url in fetched_canons if url}

    exact_preferred = any(url in candidate_canons for url in preferred)
    fetched_preferred = any(url in fetched_canons for url in preferred)
    host_preferred = bool(preferred_hosts & candidate_hosts)
    fetched_host_preferred = bool(preferred_hosts & fetched_hosts)
    source_domain_hit = any(
        host == domain or host.endswith("." + domain)
        for host in candidate_hosts
        for domain in source_domains
    )
    fetched_source_domain_hit = any(
        host == domain or host.endswith("." + domain)
        for host in fetched_hosts
        for domain in source_domains
    )
    return {
        "candidate_count": len(candidates),
        "fetch_success_count": len(fetched),
        "fetch_success_rate": round(len(fetched) / len(candidates), 3) if candidates else 0.0,
        "exact_preferred_url": exact_preferred,
        "fetched_preferred_url": fetched_preferred,
        "preferred_host": host_preferred,
        "fetched_preferred_host": fetched_host_preferred,
        "source_domain": source_domain_hit,
        "fetched_source_domain": fetched_source_domain_hit,
    }


def evidence_annotations(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotations = []
    for source in evidence.get("sources", []):
        if not source.get("ok"):
            continue
        annotations.append(
            {
                "type": "url_citation",
                "url_citation": {
                    "url": source.get("final_url") or source.get("url"),
                    "title": source.get("title") or source.get("url"),
                    "content": (source.get("excerpt") or "")[:1200],
                    "start_index": 0,
                    "end_index": 0,
                },
            }
        )
    return annotations


def build_controlled_payload(
    model_config: Dict[str, Any],
    task: Dict[str, Any],
    evidence: Dict[str, Any],
    as_of: str,
    max_tokens: int,
) -> Dict[str, Any]:
    system = (
        "You are a careful newsroom research assistant. Answer only from the controlled "
        "source excerpts supplied by the pipeline. Prefer exact names, dates, amounts, "
        "vote counts, and official language. If the supplied evidence does not support "
        "a requested fact, say that it is not supported instead of guessing."
    )
    source_blocks = []
    for index, source in enumerate(evidence.get("sources", []), start=1):
        if source.get("ok"):
            source_blocks.append(
                f"SOURCE {index}\n"
                f"Title: {source.get('title')}\n"
                f"URL: {source.get('final_url') or source.get('url')}\n"
                f"Fetched characters: {source.get('char_count')}\n"
                f"Relevant excerpt:\n{source.get('excerpt')}"
            )
        else:
            source_blocks.append(
                f"SOURCE {index}\n"
                f"URL: {source.get('url')}\n"
                f"FETCH ERROR: {source.get('error')}"
            )
    checklist = _evidence_checklist(task["question"], task.get("evidence_contract"))
    checklist_block = (
        "\n".join(f"- {item}" for item in checklist)
        if checklist else
        "- No structured checklist inferred; answer the question directly from the source excerpts."
    )

    user = f"""Benchmark date: {as_of}
Task category: {task.get('category', 'unknown')}
Question: {task['question']}

Evidence checklist:
{checklist_block}

Controlled source evidence:
{chr(10).join(source_blocks)}

Return a strict JSON object only, with these keys:
{{
  "answer": "one to three paragraphs",
  "key_facts": ["fact 1", "fact 2"],
  "sources": [{{"title": "source title", "url": "https://...", "supports": "what it supports"}}],
  "uncertainties": ["anything still unclear, or []"]
}}
"""
    payload: Dict[str, Any] = {
        "model": model_config["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "stream": False,
    }
    if model_config.get("reasoning_effort"):
        payload["reasoning"] = {"effort": model_config["reasoning_effort"]}
    return payload


def extract_openrouter_result(data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not str(content).strip():
        raise ValueError("OpenRouter response has empty message content")
    annotations = message.get("annotations") or []
    usage = data.get("usage") or {}
    return str(content).strip(), annotations, usage


def default_run_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"run_{stamp}.jsonl"


def html_report(
    run_path: Path,
    score_doc: Dict[str, Any],
    result_records: List[Dict[str, Any]],
    score_records: List[Dict[str, Any]],
) -> str:
    scores_by_pair = {
        (record.get("task_id"), record.get("model_id")): record for record in score_records
    }
    rows = []
    for model in score_doc.get("summary", {}).get("models", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(model['model_label'])}</td>"
            f"<td>{model['percent']:.2f}%</td>"
            f"<td>{model['score']:.2f}/{model['max_score']:.2f}</td>"
            f"<td>{model['failures']}</td>"
            f"<td>${model['cost']:.6f}</td>"
            f"<td>{model['avg_latency_seconds']:.2f}s</td>"
            "</tr>"
        )

    details = []
    for result in result_records:
        score = scores_by_pair.get((result.get("task_id"), result.get("model_id")), {})
        content = result.get("content") or result.get("error") or ""
        citations = []
        for annotation in result.get("annotations") or []:
            citation = annotation.get("url_citation") or annotation
            url = citation.get("url")
            title = citation.get("title") or url
            if url:
                citations.append(
                    f'<li><a href="{html.escape(url)}">{html.escape(title or url)}</a></li>'
                )
        citations_html = "<ul>" + "".join(citations) + "</ul>" if citations else "<p>No annotations captured.</p>"
        details.append(
            "<section>"
            f"<h3>{html.escape(result.get('model_label', 'unknown'))} / {html.escape(result.get('task_id', 'unknown'))}</h3>"
            f"<p><b>Score:</b> {score.get('score', 0):.2f}/{score.get('max_score', 0):.2f} "
            f"({score.get('percent', 0):.2f}%) | <b>Latency:</b> {result.get('latency_seconds', 0):.2f}s | "
            f"<b>Cost:</b> ${float(result.get('usage', {}).get('cost') or 0):.6f}</p>"
            f"<pre>{html.escape(content)}</pre>"
            f"<details><summary>Citations</summary>{citations_html}</details>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Web Search Benchmark Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; background: #f8fafc; }}
    h1, h2, h3 {{ color: #101828; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    section {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 6px; }}
    a {{ color: #0b5cad; }}
  </style>
</head>
<body>
  <h1>Web Search Benchmark Report</h1>
  <p><b>Run:</b> {html.escape(str(run_path))}</p>
  <p><b>Generated:</b> {html.escape(utc_now_iso())}</p>
  <h2>Model Summary</h2>
  <table>
    <thead><tr><th>Model</th><th>Score</th><th>Points</th><th>Failures</th><th>Cost</th><th>Average Latency</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Raw Answers</h2>
  {''.join(details)}
</body>
</html>
"""


def common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--task", action="append", dest="task_ids", help="Run only this task id; can repeat.")
    parser.add_argument("--model", action="append", dest="model_ids", help="Run only this model id; can repeat.")
    parser.add_argument("--limit", type=int, help="Limit task count after filtering.")
