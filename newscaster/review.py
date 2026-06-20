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


def build_source_corpus(date2: str) -> str:
    """Ground truth = the RAW source text the pipeline actually fetched — never the LLM summaries.

    The summaries are LLM-written and already contain any invented claim, so matching against them
    would rubber-stamp the fabrication. The corpus is raw page text from two persisted places:
      1. the scraped articles the script writer read
         (``segment_summaries/{date}_segment*_article*_source.txt``, written at gather time);
      2. the source-hunter's validated excerpts (``logs/source_hunter_audit.jsonl``), scoped to the
         day via the audit timestamp.
    Together these cover what the writer used, so the faithfulness/quote passes stop false-flagging
    real, well-sourced claims as "not in source material".
    """
    texts = []

    # 1. Raw scraped article text — the bulk of what the writer drew on.
    for path in sorted(glob.glob(f"segment_summaries/{date2}_segment*_article*_source.txt")):
        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        if content.strip():
            texts.append(content)

    # 2. Source-hunter validated excerpts for the day.
    if os.path.exists(_AUDIT_PATH):
        iso_day = date2.replace("_", "-")  # 2026_06_19 -> audit timestamp prefix 2026-06-19
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
                        texts.append(excerpt)

    return "\n\n".join(texts)


_FAITHFULNESS_SYSTEM = (
    "You are a fact-checking editor. You are given SOURCE MATERIAL (excerpts from the articles "
    "the newsroom actually read) and a SCRIPT written from them. Find statements in the SCRIPT "
    "that the SOURCE MATERIAL does not support — especially direct quotes or paraphrased "
    "statements attributed to a person that the sources do not show them saying, and specific "
    "facts or numbers absent from the sources. Use ONLY the source material; do NOT use outside "
    "knowledge, and do NOT flag a claim merely for being phrased differently. Output one line per "
    "problem as 'FLAG: <script phrase> — <why>'. If everything is supported, output exactly 'NONE'."
)


def faithfulness_flags(script_text: str, corpus_text: str, mode: str = "standard") -> list[str]:
    """LLM faithfulness pass: flag script claims/quotes the source excerpts don't support.

    Catches the paraphrase/attribution fabrications the mechanical quote check can't — e.g. the
    overview anchor rewording a real quote ("war of aggression") into an invented one ("fuel for
    the fire"). Source-grounded only: it must never use world knowledge, since the news postdates
    any model's training cutoff (a world-knowledge review rejects real current events as fake).
    Fails open (returns []) on a missing corpus or any error — the gate must never block the run.
    """
    corpus = (corpus_text or "").strip()
    script = (script_text or "").strip()
    if not corpus or not script:
        return []
    prompt = (
        f"SOURCE MATERIAL:\n{corpus[:160000]}\n\n---\n\nSCRIPT:\n{script}\n\n---\n\n"
        "List unsupported claims/quotes as 'FLAG: ...' lines, or 'NONE'."
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
    corpus = build_source_corpus(date2)
    have_corpus = bool(corpus.strip())
    if not have_corpus:
        print_and_write("QUOTE-CHECK: no persisted source excerpts yet; running stable-fact pass only.")
    flags: list[tuple[str, str, str]] = []
    for script_path in sorted(glob.glob(f"output_scripts/{date2}_*.txt")):
        name = Path(script_path).name
        text = Path(script_path).read_text(encoding="utf-8")
        if have_corpus:
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

    The three-pass review still logs every flag. For each script that drew flags, an Opus editor
    — vetted by a GPT-5.5 adversary (see editor_agent) — proposes minimal find/replace fixes ONLY
    for the flags that are genuine factual errors with a known correct value; verified edits are
    written back to output_scripts/ so generate_audio() voices the corrected script. Everything the
    editor can't justify (unsupported, stylistic, uncertain) stays flag-only. Fail-open throughout.
    """
    corpus = build_source_corpus(date2)
    flags = review_scripts(date2)
    if not flags:
        return {"flags": 0, "edited_scripts": 0, "edits": 0}

    by_script: dict[str, list[tuple[str, str]]] = {}
    for name, ftype, text in flags:
        by_script.setdefault(name, []).append((ftype, text))

    edited_scripts = 0
    total_edits = 0
    for name, script_flags in by_script.items():
        path = f"output_scripts/{name}"
        try:
            original = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print_and_write(f"FACT-FINDER EDITOR: cannot read {path}: {exc}; skipping")
            continue
        report = _format_report(script_flags)
        try:
            outcome = revise_script(original, report, corpus, label=name)
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
