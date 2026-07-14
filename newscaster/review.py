"""Pre-TTS editorial gate — quote grounding check.

Runs between ``write_scripts()`` and ``generate_audio()`` in ``pipeline.main()``.
It pulls the directly-attributed quotes the script marks with the spoken
"... said, quote, <words>, endquote" convention (see ``prompts.py``) and checks
each against the *source* text, flagging any quote that appears in no source.

Ground truth is the source-hunter excerpts persisted during gather, NOT the on-disk
segment summaries: the summaries are LLM-written and already contain any invented
quote, so matching against them would rubber-stamp the fabrication. The source hunter
now records each validated source's excerpt in its audit, so the gate reads that — the
actual text the pipeline fetched — with no re-download.

This pass is deliberately conservative and mechanical. It is reliable for
distinctive multi-word quotes (e.g. the unverifiable NORML "vindication of
personal freedom" line in the 2026-06-19 episode would have been flagged), and it
intentionally *skips* very short quotes (e.g. "not enough") that are too common to
verify by string match and are really an *attribution* problem for the companion
LLM plausibility pass to own.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from newscaster.editor_agent import revise_script
from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write, write_jsonl_log
from newscaster.search import openrouter_web_brief

# prompts.py instructs the script writer: "Person A said, quote, yadda yadda, endquote".
_QUOTE_RE = re.compile(r"\bquote\b[\s,:]*(.+?)[\s,:]*\bendquote\b", re.IGNORECASE | re.DOTALL)
_MIN_VERIFY_CHARS = 12  # shorter quotes are too common to verify by bare string match


def extract_quotes(script_text: str) -> list[str]:
    """Return the directly-attributed quotes marked with the quote/endquote convention."""
    quotes = []
    for match in _QUOTE_RE.finditer(script_text or ""):
        quote = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:—-")
        if quote:
            quotes.append(quote)
    return quotes


def _normalize(text: str) -> str:
    text = (text or "").lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class QuoteVerdict:
    quote: str
    grounded: bool
    reason: str


def verify_quotes(script_text: str, corpus_text: str) -> list[QuoteVerdict]:
    """Check each marked quote in the script against the source corpus."""
    corpus = _normalize(corpus_text)
    verdicts = []
    for quote in extract_quotes(script_text):
        normalized = _normalize(quote)
        if len(normalized) < _MIN_VERIFY_CHARS:
            verdicts.append(QuoteVerdict(quote, True, "too short to verify by string match — skipped"))
        elif normalized in corpus:
            verdicts.append(QuoteVerdict(quote, True, "found in sources"))
        else:
            verdicts.append(QuoteVerdict(quote, False, "not found in any source — possible fabrication"))
    return verdicts


# --- the gate (sketch): hooks into pipeline.main() between write_scripts and generate_audio ---

_AUDIT_PATH = "logs/source_hunter_audit.jsonl"
_EDITOR_SNIPPET_CHARS = 24000
_SNIPPET_CHUNK_CHARS = 3500
_SNIPPET_CHUNK_OVERLAP = 350
_SNIPPET_MAX_CHUNKS = 8
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.IGNORECASE)
_STOPWORDS = {
    "about", "after", "again", "against", "also", "another", "because", "been", "being",
    "between", "before", "could", "from", "have", "into", "just", "more", "only", "over",
    "said", "says", "source", "sources", "that", "their", "there", "these", "they", "this",
    "those", "through", "under", "were", "what", "when", "where", "which", "while", "with",
    "without", "would",
}


def _source_hunter_excerpts(date2: str) -> list[str]:
    """Fallback: source-hunter's validated raw excerpts for the day from the global audit.

    This audit mixes candidate-story and selected-story research, so review code should prefer
    the selected segment research artifacts when they exist.
    """
    if not os.path.exists(_AUDIT_PATH):
        return []
    iso_day = date2.replace("_", "-")  # 2026_06_19 -> audit timestamp prefix 2026-06-19
    out = []
    with open(_AUDIT_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (record.get("timestamp") or "").startswith(iso_day):
                continue
            for source in record.get("sources", []):
                excerpt = source.get("excerpt")
                if excerpt:
                    out.append(excerpt)
    return out


def _dedupe_texts(texts: list[str]) -> list[str]:
    seen = set()
    out = []
    for text in texts:
        normalized = (text or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _segment_research_path(date2: str, idx: int) -> Path:
    return Path(f"segment_summaries/{date2}_segment{idx}_research.json")


def _segment_research(date2: str, idx: int) -> dict:
    try:
        return json.loads(_segment_research_path(date2, idx).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _segment_research_excerpts(date2: str, idx: int) -> list[str]:
    """Source-hunter excerpts used by the selected-story research agent for one segment."""
    research = _segment_research(date2, idx)
    excerpts = []
    for followup in research.get("followups") or []:
        if not isinstance(followup, dict):
            continue
        for source in followup.get("source_hunter_sources") or []:
            if not isinstance(source, dict):
                continue
            excerpt = source.get("excerpt")
            if excerpt:
                excerpts.append(excerpt)
    return _dedupe_texts(excerpts)


def _selected_segment_indices(date2: str) -> list[int]:
    indices = []
    pattern = f"segment_summaries/{date2}_segment*_research.json"
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"_segment(\d+)_research\.json$", path)
        if match:
            indices.append(int(match.group(1)))
    return indices


def _selected_research_excerpts(date2: str) -> list[str]:
    excerpts = []
    for idx in _selected_segment_indices(date2):
        excerpts.extend(_segment_research_excerpts(date2, idx))
    return _dedupe_texts(excerpts)


def _article_sources(pattern: str) -> list[str]:
    """Raw scraped article text (the writer's pages, never the LLM summaries) for matching files."""
    out = []
    for path in sorted(glob.glob(pattern)):
        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        if content.strip():
            out.append(content)
    return out


def _gather_manifest(date2: str) -> dict:
    path = Path(f"segment_summaries/{date2}_GATHER_MANIFEST.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _side_story_briefs(date2: str) -> list[tuple[str, str]]:
    manifest = _gather_manifest(date2)
    briefs = []
    for item in manifest.get("side_story_briefs") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            headline, brief = item[0], item[1]
        elif isinstance(item, dict):
            headline = item.get("headline") or item.get("title") or ""
            brief = item.get("brief") or item.get("summary") or item.get("text") or ""
        else:
            continue
        headline = str(headline or "").strip()
        brief = str(brief or "").strip()
        if headline or brief:
            briefs.append((headline, brief))
    return briefs


def _matched_source_hunter_excerpts(date2: str, headlines: list[str]) -> list[str]:
    """Source-hunter excerpts whose audit topic matches selected side-story headlines."""
    if not os.path.exists(_AUDIT_PATH):
        return []
    keys = [_normalize(headline) for headline in headlines if _normalize(headline)]
    if not keys:
        return []
    iso_day = date2.replace("_", "-")
    excerpts = []
    with open(_AUDIT_PATH, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (record.get("timestamp") or "").startswith(iso_day):
                continue
            topic_key = _normalize(record.get("topic") or "")
            if not topic_key:
                continue
            if not any(key in topic_key or topic_key in key for key in keys):
                continue
            for source in record.get("sources", []):
                if not isinstance(source, dict):
                    continue
                excerpt = source.get("excerpt")
                if excerpt:
                    title = str(source.get("title") or record.get("topic") or "source").strip()
                    excerpts.append(f"SOURCE: {title}\n{excerpt}")
    return _dedupe_texts(excerpts)


def build_overview_corpus(date2: str) -> str:
    """Ground the overview in the side-story briefs the overview writer actually used."""
    briefs = _side_story_briefs(date2)
    chunks = []
    chunks.extend(_matched_source_hunter_excerpts(date2, [headline for headline, _ in briefs]))
    for headline, brief in briefs:
        if headline or brief:
            chunks.append(f"SIDE STORY: {headline}\n{brief}".strip())
    return "\n\n".join(_dedupe_texts(chunks))


def build_source_corpus(date2: str) -> str:
    """Selected-story ground truth = raw source text for stories that made the episode.

    This intentionally excludes source-hunter evidence gathered for candidate stories that were
    never selected. Those candidates live in the global audit and are useful diagnostically, but
    they add large, irrelevant context to the pre-TTS review prompts.
    """
    selected = (_article_sources(f"segment_summaries/{date2}_segment*_article*_source.txt")
                + _selected_research_excerpts(date2))
    if selected:
        return "\n\n".join(_dedupe_texts(selected))
    return "\n\n".join(_article_sources(f"segment_summaries/{date2}_segment*_article*_source.txt")
                       + _source_hunter_excerpts(date2))


def build_segment_corpus(date2: str, idx: int) -> str:
    """Per-segment ground truth: just THIS segment's scraped articles + research-agent excerpts.

    Scoping out the other segments' articles and unrelated candidate-story source-hunter excerpts
    improves the faithfulness pass's recall on subtle contradictions — a long, noisy corpus drowns
    them (the model can spot a reversal in two sentences but misses it inside 80k of text).
    """
    scoped = (_article_sources(f"segment_summaries/{date2}_segment{idx}_article*_source.txt")
              + _segment_research_excerpts(date2, idx))
    if scoped:
        return "\n\n".join(_dedupe_texts(scoped))
    return "\n\n".join(_article_sources(f"segment_summaries/{date2}_segment{idx}_article*_source.txt"))


_SEGMENT_RE = re.compile(r"_segment_(\d+)\.txt$")


def _corpus_for_script(date2: str, name: str, full_corpus: str) -> str:
    """Return source corpus for scripts that actually carry source-grounded news copy."""
    m = _SEGMENT_RE.search(name)
    if m:
        scoped = build_segment_corpus(date2, int(m.group(1)))
        if scoped.strip():
            return scoped
    if name.endswith("_overview.txt"):
        return build_overview_corpus(date2)
    if name.endswith("_intro1.txt") or name.endswith("_intro2.txt") or name.endswith("_outro.txt"):
        return ""
    return full_corpus


def _content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in _TOKEN_RE.findall((text or "").lower()):
        token = token.strip("'_-")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _source_chunks(corpus: str, *, chunk_chars: int = _SNIPPET_CHUNK_CHARS,
                   overlap: int = _SNIPPET_CHUNK_OVERLAP) -> list[str]:
    text = (corpus or "").strip()
    if not text:
        return []
    sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]
    chunks = []
    for section in sections:
        if len(section) <= chunk_chars:
            chunks.append(section)
            continue
        step = max(1, chunk_chars - overlap)
        for start in range(0, len(section), step):
            chunk = section[start:start + chunk_chars].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_chars >= len(section):
                break
    return chunks


def _snippet_corpus_for_report(report: str, corpus: str, *, max_chars: int = _EDITOR_SNIPPET_CHARS) -> str:
    """Return the source chunks most relevant to a fact-finder report.

    The editor only needs evidence for the flagged claims, not the entire source corpus. This is
    intentionally lexical and deterministic: it cannot invent support, and if it finds nothing we
    leave the caller to decide whether to skip or fall back.
    """
    query_tokens = _content_tokens(report)
    chunks = _source_chunks(corpus)
    if not query_tokens or not chunks:
        return ""

    chunk_token_sets = [_content_tokens(chunk) for chunk in chunks]

    def score_tokens(tokens: set[str], chunk_tokens: set[str]) -> int:
        overlap = tokens & chunk_tokens
        if not overlap:
            return 0
        return (len(overlap) * 10) + sum(1 for token in overlap if len(token) >= 7)

    selected_indices = []
    seen_indices = set()
    for line in (ln.strip() for ln in (report or "").splitlines() if ln.strip()):
        line_tokens = _content_tokens(line)
        if not line_tokens:
            continue
        best = None
        for index, chunk_tokens in enumerate(chunk_token_sets):
            if index in seen_indices:
                continue
            score = score_tokens(line_tokens, chunk_tokens)
            if score and (best is None or score > best[0]):
                best = (score, index)
        if best is not None:
            seen_indices.add(best[1])
            selected_indices.append(best[1])
            if len(selected_indices) >= _SNIPPET_MAX_CHUNKS:
                break

    scored = []
    for index, chunk_tokens in enumerate(chunk_token_sets):
        if index in seen_indices:
            continue
        score = score_tokens(query_tokens, chunk_tokens)
        if score:
            scored.append((score, index))

    if not selected_indices and not scored:
        return ""
    selected = [chunks[index] for index in selected_indices]
    for _, index in sorted(scored, key=lambda item: (-item[0], item[1])):
        if len(selected) >= _SNIPPET_MAX_CHUNKS:
            break
        selected.append(chunks[index])

    out = []
    total = 0
    for chunk in selected:
        if total >= max_chars:
            break
        remaining = max_chars - total
        trimmed = chunk[:remaining].strip()
        if trimmed:
            out.append(trimmed)
            total += len(trimmed) + 2
    return "\n\n".join(_dedupe_texts(out))


_FAITHFULNESS_SYSTEM = (
    "You are a fact-checking editor. You are given SOURCE MATERIAL (raw text from the articles the "
    "newsroom actually read) and a SCRIPT written from it. Flag any SCRIPT statement that CONTRADICTS "
    "the source — a different action, status, location, direction, time, number, or person — or that "
    "ATTRIBUTES a statement, quote, or position to the wrong person. Watch especially for CONFLATIONS "
    "(the script merges two separate source facts into one false claim) and REVERSALS (the source says "
    "someone DEPARTED but the script says they ARRIVED). Also flag specific facts or quotes with no "
    "support in the source. Use ONLY the source material; do NOT use outside knowledge, and do NOT flag "
    "mere rephrasing that preserves the meaning. Ignore the broadcast's own current date and the weather "
    "— the system supplies those, and the news sources will not contain them. Output one line per problem "
    "as 'FLAG: <script phrase> — <what the source actually says>'. If everything matches the source, "
    "output exactly 'NONE'."
)


def faithfulness_flags(script_text: str, corpus_text: str, mode: str = "advanced") -> list[str]:
    """LLM faithfulness pass: flag script statements that contradict or aren't supported by the sources.

    Targets the contradictions/conflations/misattributions the mechanical quote check can't — e.g. the
    overview anchor rewording a real quote ("war of aggression") into an invented one ("fuel for the
    fire"), or saying a VP "arrived" when the source says he "left for". Source-grounded only: it must
    never use world knowledge, since the news postdates any model's cutoff. Runs on the *advanced* model
    (GLM): a controlled test showed Gemma reliably misses subtle conflations in a long corpus (0/5 even
    over a 5x ensemble) while GLM catches them, so model capability — not run count — is what matters.
    Pair it with a per-segment scoped corpus (build_segment_corpus) to keep the context tight. Fails open
    (returns []) on a missing corpus or any error — the gate must never block the run.
    """
    corpus = (corpus_text or "").strip()
    script = (script_text or "").strip()
    if not corpus or not script:
        return []
    prompt = (
        f"SOURCE MATERIAL:\n{corpus[:160000]}\n\n---\n\nSCRIPT:\n{script}\n\n---\n\n"
        "List contradictions, conflations, misattributions, and unsupported claims as 'FLAG: ...' lines, or 'NONE'."
    )
    try:
        out = get_llm_response(prompt, system_prompt=_FAITHFULNESS_SYSTEM, mode=mode)
    except Exception as exc:
        print_and_write(f"FAITHFULNESS pass error (non-blocking): {exc}")
        return []
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip().upper().startswith("FLAG:")]


_STABLE_FACT_SYSTEM = (
    "You verify ONLY well-established background facts in a news script — the names, current job "
    "titles, roles, and affiliations of well-known people and organizations that you know with high "
    "confidence from longstanding knowledge (for example, who currently leads a major company). Do "
    "NOT flag anything tied to current events: recent dates, votes, poll numbers, dollar amounts, "
    "court rulings, or breaking news — you cannot verify those and they postdate your training. Flag "
    "ONLY a clearly wrong stable fact, such as an outdated or incorrect current title for a well-known "
    "person. Output one line per problem as 'FLAG: <script phrase> — <correction>'. If none, output 'NONE'."
)


def stable_fact_flags(script_text: str, mode: str = "standard") -> list[str]:
    """World-knowledge pass: flag stable-fact errors (names/titles/roles/affiliations).

    Catches the source-error class the source-grounded passes cannot — e.g. a source's
    "Google CEO Eric Schmidt" when Pichai has led Google for years. Unlike the faithfulness pass
    it deliberately USES the model's world knowledge, but scoped strictly to STABLE facts, never
    current events: the news postdates any model's training cutoff, so judging *current* facts by
    training knowledge rejects real news as fake. It carries some noise from the model's own
    imperfect knowledge, so it is advisory only. Flag-only and fail-open.
    """
    script = (script_text or "").strip()
    if not script:
        return []
    try:
        out = get_llm_response(script, system_prompt=_STABLE_FACT_SYSTEM, mode=mode)
    except Exception as exc:
        print_and_write(f"STABLE-FACT pass error (non-blocking): {exc}")
        return []
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip().upper().startswith("FLAG:")]


def _search_confirms_error(claim: str) -> str | None:
    """Web-grounded verification of a suspected stable-fact error. Returns the corrected fact when
    live sources confirm the claim is wrong, else None (correct, or unverifiable — don't flag)."""
    question = (
        f'A news script states: "{claim}". Using current, authoritative web sources, is that claim '
        "factually correct as of now? Reply 'WRONG: <the correct current fact>' if it is incorrect, "
        "or 'CORRECT' if it is accurate."
    )
    try:
        answer = (openrouter_web_brief(question) or "").strip()
    except Exception as exc:
        print_and_write(f"STABLE-FACT search-verify error (non-blocking): {exc}")
        return None
    return answer if answer.upper().startswith("WRONG") else None


def verified_stable_fact_flags(script_text: str) -> list[str]:
    """Stable-fact pass with search verification.

    The model proposes suspect title/role errors from memory (stable_fact_flags), then each is
    confirmed against the live web before it's flagged. This turns advisory memory-guesses into
    verified flags and filters the model's own mistakes — in testing it cleared Doerr's real
    "chairman" title and Bernie Sanders while confirming the Schmidt error. Bounded: only the
    suspects are searched (~1/episode), so it doesn't reintroduce broad search cost. Because the
    flags are now web-confirmed, this pass is a candidate to graduate from flag-only to blocking.
    """
    confirmed = []
    for suspect in stable_fact_flags(script_text):          # memory: "FLAG: <phrase> — <guess>"
        claim = suspect.split("—")[0].split(":", 1)[-1].strip()
        if not claim:
            continue
        correction = _search_confirms_error(claim)
        if correction:
            confirmed.append(f"FLAG: {claim} — {correction}")
    return confirmed


def review_scripts(date2: str) -> list[tuple[str, str, str]]:
    """Flag (don't block) ungrounded quotes, unsupported claims, and stale facts before TTS.

    Runs over every script for the day — the overview AND the segments — because fabrications
    appear in both (the overview anchor mangled a quote in the 2026-06-19 episode). Three passes:
      1. mechanical (source-grounded): marked "quote, ..., endquote" spans must appear verbatim;
      2. LLM faithfulness (source-grounded): claims/paraphrases the sources don't support;
      3. stable-fact (world knowledge): wrong stable names/titles/roles a source got wrong.
    Ground truth for 1-2 is the persisted source-hunter excerpts (build_source_corpus), not the LLM
    summaries. The stable-fact pass needs no corpus, so it runs even before persistence kicks in.
    Flag-only for now; later escalate to softening the line or blocking the run.

    Wire into pipeline.main(), between write_scripts() and generate_audio().
    """
    full_corpus = build_source_corpus(date2)
    if not full_corpus.strip():
        print_and_write("QUOTE-CHECK: no persisted source excerpts yet; running stable-fact pass only.")
    flags: list[tuple[str, str, str]] = []
    for script_path in sorted(glob.glob(f"output_scripts/{date2}_*.txt")):
        name = Path(script_path).name
        text = Path(script_path).read_text(encoding="utf-8")
        corpus = _corpus_for_script(date2, name, full_corpus)  # segments check against their own sources
        if corpus.strip():
            for verdict in verify_quotes(text, corpus):
                if not verdict.grounded:
                    flags.append((name, "quote", verdict.quote))
                    print_and_write(f'QUOTE-CHECK [{name}] ungrounded quote: "{verdict.quote}" — {verdict.reason}')
            for flag in faithfulness_flags(text, corpus):
                flags.append((name, "faithfulness", flag))
                print_and_write(f"FAITHFULNESS [{name}] {flag}")
        for flag in verified_stable_fact_flags(text):
            flags.append((name, "stable-fact", flag))
            print_and_write(f"STABLE-FACT [{name}] {flag}")
    if not flags:
        print_and_write("QUOTE-CHECK: no quote / faithfulness / stable-fact issues found.")
    return flags


# --- agentic editor: turn confirmed factual-error flags into in-place fixes before TTS ---

def _atomic_write_text(path: str, content: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tmp, path)


def _format_report(script_flags: list[tuple[str, str]]) -> str:
    """Render one script's flags as a report the editor can read."""
    lines = []
    for ftype, text in script_flags:
        if ftype == "quote":
            lines.append(f'FLAG (quote not found verbatim in any source): "{text}"')
        else:
            lines.append(text)  # faithfulness / stable-fact are already "FLAG: <phrase> — <why/correction>"
    return "\n".join(lines)


def _absence_only_report(report: str) -> bool:
    lines = [ln.strip().lower() for ln in (report or "").splitlines() if ln.strip()]
    if not lines:
        return False
    absence_markers = (
        "no support",
        "not found",
        "unsupported",
        "absent",
        "not in any source",
        "not in the source",
        "not in sources",
    )
    contradiction_markers = (
        "source says",
        "source consistently says",
        "wrong:",
        "should be",
    )
    for line in lines:
        if not any(marker in line for marker in absence_markers):
            return False
        if "source only says" in line or any(marker in line for marker in contradiction_markers):
            return False
    return True


def _audit_edits(date2: str, name: str, outcome) -> None:
    write_jsonl_log("fact_finder_edits", {
        "event": "fact_finder_edit",
        "date": date2,
        "script": name,
        "applied": [{"find": e.find, "replace": e.replace, "wrong": e.wrong,
                     "correct": e.correct, "basis": e.basis} for e in outcome.applied],
        "rejected": outcome.rejected,
    })


def review_and_revise_scripts(date2: str) -> dict:
    """Run the fact-finder, then auto-fix confirmed factual discrepancies in place before TTS.

    The three-pass review still logs every flag. For each script with a CORPUS-GROUNDED flag (quote
    or faithfulness), an Opus editor — vetted by a GPT-5.5 adversary (see editor_agent) — proposes
    minimal find/replace fixes ONLY for genuine factual discrepancies with a known correct value;
    verified edits are written back to output_scripts/ so generate_audio() voices the corrected script.
    Stable-fact (world-knowledge) flags stay ADVISORY and are never auto-edited: their search-verify is
    defeated by post-cutoff bias, so acting on them can corrupt a correct fact (it tried to "fix" the
    real "Field Marshal Asim Munir" to "General"). Anything the editor can't justify stays flag-only.
    Fail-open throughout.
    """
    full_corpus = build_source_corpus(date2)
    flags = review_scripts(date2)
    if not flags:
        return {"flags": 0, "edited_scripts": 0, "edits": 0}

    by_script: dict[str, list[tuple[str, str]]] = {}
    for name, ftype, text in flags:
        by_script.setdefault(name, []).append((ftype, text))

    edited_scripts = 0
    total_edits = 0
    for name, script_flags in by_script.items():
        # The editor acts ONLY on corpus-grounded flags (quote + faithfulness): the source is a
        # reliable ground truth there. Stable-fact flags are world-knowledge and stay advisory —
        # their search-verify is defeated by post-cutoff bias (it "confirmed" the correct
        # "Field Marshal Asim Munir" was wrong because the promotion postdates the model's cutoff),
        # so auto-editing them would corrupt true facts. They are still logged by review_scripts.
        editable = [(ft, tx) for ft, tx in script_flags if ft != "stable-fact"]
        if not editable:
            print_and_write(f"FACT-FINDER EDITOR [{name}]: only advisory stable-fact flag(s); not auto-editing")
            continue
        path = f"output_scripts/{name}"
        try:
            original = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print_and_write(f"FACT-FINDER EDITOR: cannot read {path}: {exc}; skipping")
            continue
        report = _format_report(editable)
        corpus = _corpus_for_script(date2, name, full_corpus)  # segment editor checks its own sources
        snippet_corpus = _snippet_corpus_for_report(report, corpus)
        if snippet_corpus:
            print_and_write(
                f"FACT-FINDER EDITOR [{name}]: using {len(snippet_corpus)} chars of relevant source snippets "
                f"from {len(corpus)} chars"
            )
        elif _absence_only_report(report):
            print_and_write(
                f"FACT-FINDER EDITOR [{name}]: only unsupported/absent-source flag(s); "
                "skipping auto-edit"
            )
            continue
        else:
            print_and_write(
                f"FACT-FINDER EDITOR [{name}]: no relevant snippets found; "
                "running editor with report-only evidence"
            )
        try:
            outcome = revise_script(original, report, snippet_corpus, label=name)
        except Exception as exc:  # belt-and-suspenders; revise_script is already fail-open
            print_and_write(f"FACT-FINDER EDITOR crashed on {name}: {exc}; leaving script unchanged")
            continue
        if not outcome.changed:
            print_and_write(
                f"FACT-FINDER EDITOR [{name}]: no factual-discrepancy edits applied "
                f"({len(script_flags)} flag(s); {len(outcome.rejected)} proposal(s) declined)"
            )
            continue
        try:
            _atomic_write_text(path, outcome.text)
        except OSError as exc:
            print_and_write(f"FACT-FINDER EDITOR: failed to write {path}: {exc}; leaving original")
            continue
        edited_scripts += 1
        total_edits += len(outcome.applied)
        for e in outcome.applied:
            print_and_write(
                f'FACT-FINDER EDIT [{name}] "{e.find}" -> "{e.replace}"  '
                f'({e.correct or e.wrong}; basis: {e.basis})'
            )
        _audit_edits(date2, name, outcome)

    print_and_write(f"FACT-FINDER EDITOR: applied {total_edits} fix(es) across {edited_scripts} script(s).")
    return {"flags": len(flags), "edited_scripts": edited_scripts, "edits": total_edits}
