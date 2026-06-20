"""Agentic fact-finder editor — fix confirmed factual errors in a script before TTS.

This revives the research-agent pattern (Opus controller + GPT-5.5 adversary, bounded
loop) as an *editor*. After the three-pass fact-finder produces its flags, this hands
Opus the full report, the script, and its ground-truth sources and lets it propose
MINIMAL find/replace edits — but ONLY for confirmed FACTUAL DISCREPANCIES: a checkable
fact in the script (a name, current title/role, affiliation, number, date, place,
who-did-what, or the substance of an attributed quote) that the ground truth
*contradicts* with a DIFFERENT value. "Unsupported"/absent-from-sources and
wording/style differences are NOT discrepancies and are left flag-only.

Safety, because the edited script is broadcast:
- Deterministic application. Opus never rewrites the script; it proposes exact
  find/replace pairs that plain Python applies, and ONLY when the span is unambiguous
  (the find occurs exactly once) — otherwise the case is left flagged for a human.
- Verify-then-apply. Every proposed edit is vetted by an independent adversary
  (GPT-5.5) *before* it is applied, so there is never a revert and nothing unverified
  ever lands.
- Fail-open. Any error (propose, adversary, apply) leaves the script untouched.
- Length guard. If the net result is implausibly shorter than the original, the whole
  edit set is discarded.
- Audited + gated by FACT_FINDER_AUTOEDIT_ENABLED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import newscaster.config as _config
from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write
from newscaster.text_utils import extract_json

_MAX_CORPUS = 140000
_MIN_KEEP_RATIO = 0.6  # discard the whole edit set if the result is shorter than this fraction


@dataclass
class Edit:
    find: str        # exact span copied from the script
    replace: str     # corrected span
    wrong: str = ""  # the wrong fact, in plain words
    correct: str = ""  # the correct fact, in plain words
    basis: str = ""  # the source / verified correction that proves it


@dataclass
class EditOutcome:
    text: str
    applied: list[Edit] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {"find": ..., "reason": ...}
    changed: bool = False


_PROPOSE_SYSTEM = (
    "You are a meticulous broadcast copy editor correcting a news script just before it is read "
    "aloud. You are given the SCRIPT, a fact-finder REPORT (flags), and the SOURCE MATERIAL the "
    "newsroom actually read.\n\n"
    "Your ONLY job is to correct confirmed FACTUAL DISCREPANCIES: a checkable fact in the script — a "
    "name, a current job title/role, an affiliation, a number, a date, a place, who-did-what, or the "
    "substance of an attributed quote — that the ground truth CONTRADICTS with a DIFFERENT value. "
    "Ground truth is the SOURCE MATERIAL, or a verified correction already in the report (a line of the "
    "form 'FLAG: <claim> — WRONG: <correct fact>').\n\n"
    "STRICT RULES:\n"
    "- Fix a fact ONLY when it is actually WRONG and you have the CORRECT value from the source material "
    "or a verified correction. If a flag only says a claim is 'unsupported' or 'not in the sources', that "
    "means ABSENT, not wrong — DO NOT touch it. Wording, tone, paraphrasing, emphasis and style are NOT "
    "discrepancies — DO NOT touch them.\n"
    "- Make the SMALLEST change that fixes the fact: replace only the wrong span, keeping the surrounding "
    "spoken phrasing intact. Never rewrite a sentence, never add commentary, never introduce a new fact.\n"
    "- 'find' MUST be copied VERBATIM from the SCRIPT (exact characters, including punctuation and the "
    "spoken 'quote/endquote' words if present) so it can be located by exact string match.\n"
    "- A value that merely DIFFERS from the source is not automatically wrong. Many facts have more than "
    "one correct form — a person is often affiliated with both a hospital and its medical school, or holds "
    "more than one valid title. Do NOT change such a value just to match the source's wording; only change "
    "a value that is genuinely FALSE.\n"
    "- If you are not certain a fact is wrong AND you have its correct value from the provided material, "
    "leave it out.\n\n"
    "Return ONLY JSON: {\"edits\": [{\"find\": \"<verbatim span>\", \"replace\": \"<corrected span>\", "
    "\"wrong\": \"<the wrong fact>\", \"correct\": \"<the correct fact>\", \"basis\": \"<the source text or "
    "verified correction that proves it>\"}]}. If there are no confirmable factual discrepancies, return "
    "{\"edits\": []}."
)

_ADVERSARY_SYSTEM = (
    "You are a skeptical fact-checking editor reviewing PROPOSED corrections to a script that is about to "
    "be broadcast. You are given the SOURCE MATERIAL, the fact-finder report, the script, and a list of "
    "proposed find/replace edits.\n\n"
    "Approve an edit ONLY if ALL of these hold:\n"
    "1. the original 'find' span states a CHECKABLE fact that is genuinely WRONG;\n"
    "2. the 'replace' span states the CORRECT fact, supported by the SOURCE MATERIAL or a verified "
    "correction in the report — never by outside guessing or memory;\n"
    "3. the change is MINIMAL and does not alter meaning beyond fixing that fact.\n\n"
    "REJECT anything that is stylistic, a paraphrase, merely 'unsupported'/absent (not actually "
    "contradicted), an original value that could itself be valid (e.g. an alternative correct affiliation, "
    "title, or institution that merely differs from the source's wording), uncertain, or not grounded in "
    "the provided material. When in doubt, REJECT — leaving a flagged line unchanged is safe; broadcasting "
    "a wrong correction is not.\n\n"
    "Return ONLY JSON: {\"verdicts\": [{\"index\": <the edit's 0-based position>, \"approve\": <true|false>, "
    "\"reason\": \"<one line>\"}]}."
)


def _propose_edits(script: str, report: str, corpus: str, already: list[Edit]) -> list[Edit]:
    """Ask Opus for find/replace edits that fix factual discrepancies. Raises on LLM/parse error."""
    already_note = ""
    if already:
        done = "\n".join(f'- "{e.find}" -> "{e.replace}"' for e in already)
        already_note = (
            "\n\nALREADY APPLIED THIS RUN (do not repeat these, and note the script now reflects them):\n"
            + done
        )
    prompt = (
        f"SOURCE MATERIAL:\n{(corpus or '(none provided)')[:_MAX_CORPUS]}\n\n---\n\n"
        f"FACT-FINDER REPORT (flags):\n{report}\n\n---\n\n"
        f"SCRIPT:\n{script}{already_note}\n\n---\n\n"
        "Return the JSON of factual-discrepancy edits, or {\"edits\": []}."
    )
    raw = get_llm_response(prompt, system_prompt=_PROPOSE_SYSTEM, mode="heavy")
    data = extract_json(raw, want=dict)
    edits: list[Edit] = []
    for item in data.get("edits") or []:
        if not isinstance(item, dict):
            continue
        find = str(item.get("find") or "")
        if not find:
            continue
        edits.append(Edit(
            find=find,
            replace=str(item.get("replace") if item.get("replace") is not None else ""),
            wrong=str(item.get("wrong") or ""),
            correct=str(item.get("correct") or ""),
            basis=str(item.get("basis") or ""),
        ))
    return edits


def _verify_edits(edits: list[Edit], script: str, report: str, corpus: str) -> tuple[list[Edit], list[dict]]:
    """Adversary (GPT-5.5) vets each proposed edit. Returns (approved, rejected). Raises on LLM/parse error."""
    payload = {"edits": [
        {"index": i, "find": e.find, "replace": e.replace,
         "claimed_wrong": e.wrong, "claimed_correct": e.correct, "basis": e.basis}
        for i, e in enumerate(edits)
    ]}
    prompt = (
        f"SOURCE MATERIAL:\n{(corpus or '(none provided)')[:_MAX_CORPUS]}\n\n---\n\n"
        f"FACT-FINDER REPORT:\n{report}\n\n---\n\n"
        f"SCRIPT:\n{script[:60000]}\n\n---\n\n"
        f"PROPOSED EDITS:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return the JSON verdicts."
    )
    raw = get_llm_response(prompt, system_prompt=_ADVERSARY_SYSTEM, mode="adversary")
    data = extract_json(raw, want=dict)
    reasons: dict[int, str] = {}
    approved_idx: set[int] = set()
    for v in data.get("verdicts") or []:
        if not isinstance(v, dict) or "index" not in v:
            continue
        try:
            idx = int(v["index"])
        except (TypeError, ValueError):
            continue
        reasons[idx] = str(v.get("reason") or "")
        if v.get("approve") is True:
            approved_idx.add(idx)
    approved = [e for i, e in enumerate(edits) if i in approved_idx]
    rejected = [{"find": e.find, "reason": reasons.get(i) or "adversary did not approve"}
                for i, e in enumerate(edits) if i not in approved_idx]
    return approved, rejected


def _apply_edits(text: str, edits: list[Edit]) -> tuple[str, list[Edit], list[dict]]:
    """Deterministically apply verbatim find/replace edits. No LLM.

    An edit lands ONLY when its `find` occurs EXACTLY ONCE in the current text. A `find` that is
    missing (0x) or ambiguous (>1x) is skipped and logged rather than risk editing the wrong span:
    for a broadcast script, leaving an ambiguous fix flagged for a human is safer than guessing
    which occurrence to change.
    """
    applied: list[Edit] = []
    skipped: list[dict] = []
    for e in edits:
        if not e.find:
            skipped.append({"find": e.find, "reason": "empty find"})
            continue
        if e.find == e.replace:
            skipped.append({"find": e.find, "reason": "no-op (find == replace)"})
            continue
        count = text.count(e.find)
        if count == 0:
            skipped.append({"find": e.find, "reason": "find not present (may overlap an earlier edit)"})
            continue
        if count > 1:
            skipped.append({"find": e.find, "reason": f"find is ambiguous (appears {count}x); left flagged for review"})
            continue
        text = text.replace(e.find, e.replace)  # exactly one occurrence
        applied.append(e)
    return text, applied, skipped


def revise_script(script_text: str, report: str, corpus: str, *, label: str = "script") -> EditOutcome:
    """Fix confirmed factual discrepancies in `script_text` in a bounded propose/verify/apply loop.

    Fail-open: returns the original text with changed=False when auto-edit is disabled, on any
    error, or when nothing is confidently fixable. Nothing unverified is ever applied.
    """
    if not getattr(_config, "FACT_FINDER_AUTOEDIT_ENABLED", False):
        return EditOutcome(text=script_text, changed=False)
    if not (script_text or "").strip() or not (report or "").strip():
        return EditOutcome(text=script_text, changed=False)

    max_rounds = max(1, int(getattr(_config, "FACT_FINDER_AUTOEDIT_MAX_ROUNDS", 3)))
    current = script_text
    all_applied: list[Edit] = []
    all_rejected: list[dict] = []

    for round_i in range(max_rounds):
        try:
            proposed = _propose_edits(current, report, corpus or "", all_applied)
        except Exception as exc:
            print_and_write(f"FACT-FINDER EDITOR propose error [{label}] round {round_i}: {exc}")
            break
        if not proposed:
            break
        try:
            approved, rejected = _verify_edits(proposed, current, report, corpus or "")
        except Exception as exc:
            # Never apply edits the adversary did not clear.
            print_and_write(f"FACT-FINDER EDITOR adversary error [{label}] round {round_i}: {exc}; applying none")
            break
        all_rejected.extend(rejected)
        if not approved:
            break
        new_text, applied, skipped = _apply_edits(current, approved)
        all_rejected.extend(skipped)
        if not applied or new_text == current:
            break
        current = new_text
        all_applied.extend(applied)

    # Catastrophe guard: a minimal-edit pass should never shrink the script much.
    if all_applied and len(current) < _MIN_KEEP_RATIO * len(script_text):
        print_and_write(
            f"FACT-FINDER EDITOR [{label}]: result implausibly short "
            f"({len(current)} vs {len(script_text)} chars); discarding all edits"
        )
        return EditOutcome(text=script_text, applied=[], rejected=all_rejected, changed=False)

    return EditOutcome(
        text=current,
        applied=all_applied,
        rejected=all_rejected,
        changed=bool(all_applied) and current != script_text,
    )
