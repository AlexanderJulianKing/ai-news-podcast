"""Sunday opinion section (2026-10-23): one opinion piece replaces the two main stories.

Alex's specs: it takes a side, can be about anything, runs between one and two main stories
long, is read by its own voice, and must not sound AI-written. A once-in-years story keeps
the normal show (breaking_override); otherwise the main picks join the roundup.

Stages, each saved to segment_summaries/<key>_opinion_state.json so a retry doesn't repay:
  1. pitch: Opus proposes three pieces against the week's coverage and picks one, with questions;
  2. research: the tool-using researcher answers them (quotes checked on the page);
  3. gaps: Opus revises the claim to fit the evidence and asks follow-ups, which are researched;
  4. draft: Opus writes the piece from that evidence alone; banned AI-isms are rewritten;
  5. self-check: the faithfulness pass reads the piece against those quotes, and Opus fixes or keeps each flag.
The evidence is saved as slot 0's research record, so the usual fact check and editor run on it.
"""
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from newscaster import config as _config
from newscaster import edition
from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write
from newscaster.prompts import (OPINION_GAPS_PROMPT as GAPS_PROMPT, OPINION_OVERRIDE_PROMPT as OVERRIDE_PROMPT,
                                OPINION_PITCH_PROMPT as PITCH_PROMPT, OPINION_REVISE_PROMPT as REVISE_PROMPT,
                                OPINION_SOURCING_PROMPT as SOURCING_PROMPT, OPINION_WRITER_PROMPT as WRITER_PROMPT)
from newscaster.rag.indexer import build_research_record
from newscaster.research_tools import research_with_tools
from newscaster.review import faithfulness_flags

SLOT = 0
_SENT = re.compile(r"(?<=[.!?])\s+")

# Phrases from the 2026-10-22 samples that real opinion writers never use (opinion/BRIEF_v2.md).
_BANNED = [
    r"\bone writer's (?:argument|view|opinion)\b", r"\bnot (?:the news desk|reporting from)\b",
    r"\bthis is (?:the|an|my) opinion\b",
    r"\bhere(?: is|'s) (?:my|the) (?:argument|logic|point|case|thing|answer)\b",
    r"\bhere(?: is|'s) where I land\b", r"\blet me (?:start|begin|be clear|walk you)\b", r"\blet's (?:start|begin)\b",
    r"\bstrongest (?:objection|counterargument|argument against)\b", r"\bdeserves? a (?:real |serious )?answer\b",
    r"\bworth taking seriously\b", r"\bchange my mind\b",
    r"\b(?:to be fair|I'll be honest|to be honest|I want to be fair)\b",
    r"\bmy argument is simple\b", r"\bthat line is the whole problem\b", r"\bthink about what that means\b",
    r"\bmake no mistake\b", r"\bit's worth noting\b", r"\bin conclusion\b",
    r"\byou (?:might|may) (?:wonder|fairly ask|be thinking|ask)\b", r"\bsome will say\b",
    r"\bend ?quote\b",
]


def is_opinion_day(day=None, which=None):
    """Sunday mornings, while the section is switched on."""
    day = day or date.today()
    return (bool(getattr(_config, "OPINION_SUNDAY_ENABLED", False)) and day.weekday() == 6
            and not edition.is_evening(which))


def breaking_override(top_headline, brief, top_headlines):
    """True when today's top story is too big to skip as the lead. On any failure, True:
    missing an opinion piece is cheaper than not leading with a war."""
    prompt = (f"TODAY'S TOP STORY: {top_headline}\n\nRESEARCH BRIEF:\n{(brief or '')[:6000]}\n\n"
              "THE DAY'S OTHER TOP HEADLINES:\n" + "\n".join(f"- {h}" for h in top_headlines[:10]))
    try:
        reply = get_llm_response(prompt, system_prompt=OVERRIDE_PROMPT, mode="routine")
    except Exception as exc:
        print_and_write(f"OPINION: override check failed ({exc!r:.200}); keeping the normal show")
        return True
    m = re.search(r"OVERRIDE:\s*(YES|NO)", reply or "", re.I)
    if not m:
        print_and_write(f"OPINION: override reply unreadable ({(reply or '')[:200]!r}); keeping the normal show")
        return True
    print_and_write(f"OPINION: override check: {(reply or '').strip()[:300]}")
    return m.group(1).upper() == "YES"


def roundup_with_mains(main_headlines, candidates):
    """On an opinion Sunday the day's two main picks lead the roundup instead."""
    return [h for h in main_headlines if h] + [c for c in candidates if c not in main_headlines]


def banned_phrases(text):
    found = []
    for pattern in _BANNED:
        found += [m.group(0) for m in re.finditer(pattern, text or "", re.I)]
    return found


def _json(reply):
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {(reply or '')[:200]!r}")
    return json.loads(m.group(0))


def _week_digest(today, days=7, max_chars=24000):
    """What the show aired over the last week: each main story's opening and each roundup item's first sentence."""
    lines = []
    for back in range(days, 0, -1):
        day = (today - timedelta(days=back)).strftime("%Y_%m_%d")
        for key in (day, day + edition.EVENING_SUFFIX):
            items = []
            for path in sorted(glob.glob(f"output_scripts/{key}_segment_*.txt")):
                if not edition.belongs_to(path, key):
                    continue
                first = open(path, encoding="utf-8").read().split("\n")[0].split(":", 1)[-1].strip()
                items.append("MAIN: " + " ".join(_SENT.split(first)[:2]))
            path = f"output_scripts/{key}_overview.txt"
            if os.path.exists(path):
                for para in open(path, encoding="utf-8").read().split("\n"):
                    if len(para) > 80:
                        items.append("ROUNDUP: " + _SENT.split(para.strip())[0])
            if items:
                lines.append(f"## {key}\n" + "\n".join(items))
    return "\n\n".join(lines)[-max_chars:]


def _history():
    try:
        with open(_config.OPINION_HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _research(questions, formatted_date):
    def one(question):
        try:
            r = research_with_tools(question, formatted_date)
        except Exception as exc:
            print_and_write(f"OPINION research failed for {question[:80]!r}: {exc!r:.200}")
            return {"asker": "Opinion research", "question": question, "answer": "Research failed.",
                    "source_hunter_sources": []}
        return {"asker": "Opinion research", "question": question, "answer": r.get("answer", ""),
                "source_hunter_sources": r.get("sources") or []}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, [q for q in questions if str(q).strip()]))


def _evidence(followups):
    return "\n\n".join(f"Q: {f['question']}\n{f['answer']}" for f in followups if f.get("source_hunter_sources"))


def _parse_piece(reply):
    m = re.search(r"TITLE:\s*(.+?)\s*\n\s*TEASER:\s*(.+?)\s*\n\s*SCRIPT:\s*\n(.*)", reply or "", re.S)
    if not m:
        raise ValueError(f"writer reply not in TITLE/TEASER/SCRIPT form: {(reply or '')[:200]!r}")
    return m.group(1).strip().strip('"'), m.group(2).strip(), m.group(3).strip()


def _paragraphs(script):
    out = []
    for para in re.split(r"\n\s*\n", script.replace("*", "")):
        para = re.sub(r"^\s*(?:Opinion|Narrator)\s*:\s*", "", " ".join(para.split()))
        if para:
            out.append(para)
    return out


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9$%]+", (text or "").lower().replace("\u2019", "'")))


def self_check(script, evidence):
    """Run the faithfulness pass against the writer's own evidence; Opus rules on each flag.
    A "supported" verdict must quote the evidence, and code checks the quote is really there;
    a "fix" is an exact find/replace that code applies. Returns (script, audit)."""
    # The checker finds different things on each run (9 and 15 flags on one script, 2026-10-23), so
    # two runs are merged. Most flags are false alarms; the ruling below sorts them out.
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(lambda _: faithfulness_flags(script, evidence), range(2)))
    flags, seen = [], set()
    for flag in runs[0] + runs[1]:
        span = _norm(flag)[:80]
        if span not in seen:
            seen.add(span)
            flags.append(flag)
    audit = {"flags": flags, "verdicts": [], "applied": 0, "skipped": [], "unbacked": []}
    if not flags:
        return script, audit
    numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(flags, 1))
    try:
        verdicts = _json(get_llm_response(f"FLAGS:\n{numbered}\n\nEVIDENCE:\n{evidence}\n\nSCRIPT:\n{script}",
                                          system_prompt=SOURCING_PROMPT, mode="routine")).get("verdicts") or []
    except Exception as exc:
        print_and_write(f"OPINION self-check: ruling failed ({exc!r:.200}); keeping the draft")
        return script, audit
    audit["verdicts"] = verdicts
    ev = _norm(evidence)
    for v in verdicts:
        kind = str(v.get("verdict", "")).lower()
        if kind == "fix" and v.get("find") and v["find"] in script:
            find = v["find"]
            start = script.index(find)
            end = start + len(find)
            while end < len(script) and script[end] in '.!?"\u201d\u2019\'':   # take the sentence's own punctuation
                end += 1
            if not re.search(r'[.!?]["\u201d\u2019\']*$', script[start:end]):
                audit["skipped"].append(v)            # not a whole sentence: replacing it could leave a fragment
                continue
            replace = (v.get("replace") or "").strip()
            if replace and not re.search(r'[.!?]["\u201d\u2019\']*$', replace):
                replace += "."
            script = script[:start] + replace + script[end:]
            audit["applied"] += 1
        elif kind == "supported" and _norm(v.get("quote"))[:200] not in ev:
            audit["unbacked"].append(v)
    script = re.sub(r"[ \t]{2,}", " ", script)
    print_and_write(f"OPINION self-check: {len(flags)} flag(s): {audit['applied']} fixed, {len(audit['skipped'])} fix(es) skipped, "
                    f"{sum(str(v.get('verdict')).lower() == 'supported' for v in verdicts)} supported, "
                    f"{sum(str(v.get('verdict')).lower() == 'opinion' for v in verdicts)} opinion, "
                    f"{len(audit['unbacked'])} 'supported' without a matching quote")
    return script, audit


def write_opinion(key, formatted_date):
    """Write the opinion segment (slot 0), its research record, the episode title and the history entry."""
    state_path = f"segment_summaries/{key}_opinion_state.json"
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        state = {}

    def save():
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    today = edition.key_date(key) or date.today()
    if "pitch" not in state:
        recent = "\n".join(f"- {h.get('key')}: {h.get('topic')} — {h.get('claim')}" for h in _history()[-12:]) or "None yet."
        user = (f"TODAY: {formatted_date}. The piece airs this morning.\n\nTHIS WEEK'S COVERAGE ON THE SHOW:\n"
                f"{_week_digest(today) or 'Unavailable.'}\n\nRECENT OPINION PIECES (don't repeat):\n{recent}")
        state["pitch"] = _json(get_llm_response(user, system_prompt=PITCH_PROMPT, mode="heavy"))
        save()
    pitch = state["pitch"]
    chosen = pitch["candidates"][int(pitch.get("chosen", 0))]
    print_and_write(f"OPINION pitch: {chosen.get('style')} | {chosen.get('topic')} | {chosen.get('claim')}")

    if "followups" not in state:
        state["followups"] = _research(pitch.get("questions", [])[:10], formatted_date)
        save()
    if "claim" not in state:
        user = (f"TOPIC: {chosen.get('topic')}\nCLAIM: {chosen.get('claim')}\n\nEVIDENCE:\n"
                f"{_evidence(state['followups']) or 'None found.'}")
        gaps = _json(get_llm_response(user, system_prompt=GAPS_PROMPT, mode="routine"))
        state["followups"] += _research(gaps.get("questions", [])[:6], formatted_date)
        state["claim"] = gaps.get("claim") or chosen.get("claim")
        save()
    followups = state["followups"]
    evidence = _evidence(followups)
    if not evidence:
        raise RuntimeError("opinion research found no usable evidence")

    user = (f"TODAY: {formatted_date}\nTOPIC: {chosen.get('topic')}\nCLAIM: {state['claim']}\n"
            f"WHY IT ISN'T THE USUAL TAKE: {chosen.get('why_new')}\n\nEVIDENCE (direct quotes with their pages):\n{evidence}")
    title, teaser, script = _parse_piece(get_llm_response(user, system_prompt=WRITER_PROMPT, mode="heavy"))
    words = len(script.split())
    if not 1400 <= words <= 3200:
        print_and_write(f"OPINION draft is {words} words; asking for 1,700 to 2,600")
        title, teaser, script = _parse_piece(get_llm_response(
            user + f"\n\nYour last draft was {words} words. Write 1,700 to 2,600.", system_prompt=WRITER_PROMPT, mode="heavy"))
    found = banned_phrases(script)
    if found:
        print_and_write(f"OPINION draft uses banned phrases {found}; rewriting those sentences")
        revised = get_llm_response("BANNED PHRASES FOUND: " + "; ".join(found) + "\n\nSCRIPT:\n" + script,
                                   system_prompt=REVISE_PROMPT, mode="routine")
        if revised and len(revised.split()) > 0.8 * len(script.split()):
            script = revised.strip()
        left = banned_phrases(script)
        if left:
            print_and_write(f"OPINION: banned phrases remain after the rewrite: {left}")

    # Check the piece against the quotes its writer saw (the later fact check reads far more text, and
    # its editor fixes contradictions but never cuts an unsupported claim or a wrong attribution).
    script, state["self_check"] = self_check(script, evidence)
    save()

    paragraphs = _paragraphs(script)
    record = build_research_record(key, SLOT, chosen.get("topic", ""), None, [], followups)
    with open(f"segment_summaries/{key}_segment{SLOT}_research.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    handoff = ("Grace: It's Sunday, so in place of our main stories we turn to our opinion section. Today's piece was "
               "written by an AI model from sourced research, and it doesn't necessarily reflect the views "
               f"of Alex or of this program. {teaser}")
    with open(f"output_scripts/{key}_segment_{SLOT}.txt", "w", encoding="utf-8") as f:
        f.write(handoff + "\n" + "\n".join(f"Opinion: {p}" for p in paragraphs) + "\n")
    title = re.sub(r"\s*,\s*", " — ", title)       # the uploader splits this file on commas
    with open(f"episode_titles/{key}.txt", "w", encoding="utf-8") as f:
        f.write(f"Opinion: {title}")
    history = _history() + [{"key": key, "topic": chosen.get("topic"), "claim": state["claim"], "title": title}]
    with open(_config.OPINION_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print_and_write(f"OPINION written: {title!r}, {sum(len(p.split()) for p in paragraphs)} words, "
                    f"{len(paragraphs)} paragraphs, {sum(bool(f.get('source_hunter_sources')) for f in followups)} "
                    f"of {len(followups)} lookups with evidence")
    return {"title": title, "teaser": teaser}
