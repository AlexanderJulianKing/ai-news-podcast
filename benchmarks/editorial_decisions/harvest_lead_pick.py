"""Harvest lead-pick cases from the LLM audit log.

A case is one morning's Tier 3 "most important story" call: the exact system
prompt and research document Opus saw. The audit log records prompts but not
responses, so the production pick is recovered from the daily log when one is
available. Read-only over the inputs; writes data/lead_pick_cases.jsonl.

    python3 -m benchmarks.editorial_decisions.harvest_lead_pick \
        --audit logs/llm_audit.jsonl --logs-dir logs
"""
import argparse
import glob
import hashlib
import json
import os
import re

TIER3_PREFIX = "Given the following research briefs on today's top candidate stories, select the single most important story"
EXTRACT_PREFIX = "The text below explains the selection of a story."  # prompts.HEADLINE_EXTRACTION_PROMPT
_BRIEF_SPLIT = re.compile(r"^--- Brief (\d+) ---\s*$", re.M)
_TAG = re.compile(r"^\s*\[([A-Z][A-Z -]*?)(?::\s*([^\]]+))?\]\s*")
# Tags the Tier 3 prompt bars from leading. [SIDE-COVERED], [DEVELOPMENT] and
# [MAJOR ESCALATION] stay eligible.
INELIGIBLE_TAGS = {"UPDATE"}

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "data", "lead_pick_cases.jsonl")


def split_tags(headline):
    """Return (clean_headline, [(tag, slug_or_None), ...]) for a tagged headline."""
    tags = []
    rest = headline
    while True:
        m = _TAG.match(rest)
        if not m:
            break
        tags.append((m.group(1).strip(), (m.group(2) or "").strip() or None))
        rest = rest[m.end():]
    return rest.strip(), tags


def parse_briefs(research_document):
    """Split a Tier 3 research document into briefs, plus any text around them.

    Returns (briefs, preamble, trailer). The trailer is the coverage-notes block
    appended after the last brief, when present.
    """
    parts = _BRIEF_SPLIT.split(research_document)
    preamble = parts[0].strip()
    briefs = []
    trailer = ""
    for i in range(1, len(parts), 2):
        index = int(parts[i])
        body = parts[i + 1]
        is_last = i + 2 >= len(parts)
        if is_last:
            cut = body.find("=== Coverage notes")  # prompts.COVERAGE_NOTES_HEADER
            if cut != -1:
                trailer = body[cut:].strip()
                body = body[:cut]
        m = re.search(r"^Headline: (.*)$", body, re.M)
        raw_headline = m.group(1).strip() if m else ""
        headline, tags = split_tags(raw_headline)
        rb = re.search(r"^Reported by: (.*)$", body, re.M)
        text = body
        if m:
            text = text.replace(m.group(0), "", 1)
        if rb:
            text = text.replace(rb.group(0), "", 1)
        briefs.append({
            "index": index,
            "raw_headline": raw_headline,
            "headline": headline,
            "tags": [{"tag": t, "slug": s} for t, s in tags],
            "eligible": not any(t in INELIGIBLE_TAGS for t, _ in tags),
            "reported_by": [s.strip() for s in rb.group(1).split(",")] if rb else [],
            "brief": text.strip(),
        })
    return briefs, preamble, trailer


def _content_key(briefs):
    """Identify a morning by its tag-stripped headlines, so a replay of the same
    briefs under different eligibility tags groups with the original."""
    joined = "\n".join(sorted(b["headline"].lower() for b in briefs))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_STOP = set("a an the of in on at to for and or as by with from than that this is are was were be it its".split())


def _content_words(text):
    return {w for w in _norm(text).split() if w not in _STOP}


def same_headline(a, b):
    """True when two headlines are one story: equal once normalized, or one is
    the opening of the other (the same item scraped at two lengths)."""
    a, b = _norm(a), _norm(b)
    short, long_ = sorted((a, b), key=len)
    return a == b or (len(short) >= 30 and long_.startswith(short))


def _headline_core(headline):
    """The short title before the first colon ('Rising Winter Heating Costs: ...'),
    when there is one of three or more words; else the whole headline."""
    title = headline.split(":", 1)[0]
    return title if ":" in headline and len(_norm(title).split()) >= 3 else headline


def match_pick(answer, briefs):
    """Map an 'Answer:' line back to a brief index, or None when nothing matches.

    Exact or contained headlines win first, then word overlap. The last stage
    handles a paraphrased answer: the share of the headline's core words that
    the answer repeats. It must clear 0.6 and beat every different headline by
    0.15, so a vague answer matches nothing. Identical headlines (duplicate
    briefs of one story) tie, and the lower index is returned.
    """
    a = _norm(answer)
    if not a:
        return None
    best, best_overlap = None, 0.0
    a_words = set(a.split())
    for b in briefs:
        h = _norm(b["headline"])
        if not h:
            continue
        if h in a or a in h:
            return b["index"]
        h_words = set(h.split())
        overlap = len(a_words & h_words) / float(len(a_words | h_words))
        if overlap > best_overlap:
            best, best_overlap = b["index"], overlap
    if best_overlap >= 0.6:
        return best

    a_content = _content_words(answer)
    scored = []
    for b in briefs:
        core = _content_words(_headline_core(b["headline"]))
        if core:
            shared = len(a_content & core)
            # fewer than three shared words is too little to call a match
            scored.append((shared / float(len(core)) if shared >= 3 else 0.0, b["index"], b["headline"]))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    top_score, top_index, top_headline = scored[0]
    rivals = [sc for sc, _i, h in scored[1:] if not same_headline(h, top_headline)]
    if top_score >= 0.6 and (not rivals or top_score - rivals[0] >= 0.15):
        return top_index

    # Last resort: the answer restates the story in its own words (Opus does
    # this). Score the share of the answer's words found anywhere in each brief.
    if len(a_content) < 5:
        return None
    body = []
    for b in briefs:
        text = _content_words(b["headline"] + " " + (b.get("brief") or ""))
        body.append((len(a_content & text) / float(len(a_content)), b["index"], b["headline"]))
    body.sort(key=lambda t: (-t[0], t[1]))
    top_score, top_index, top_headline = body[0]
    rivals = [sc for sc, _i, h in body[1:] if not same_headline(h, top_headline)]
    if top_score >= 0.6 and (not rivals or top_score - rivals[0] >= 0.2):
        return top_index
    return None


def production_picks(logs_dir):
    """Read daily logs for the first 'Answer:' after 'TIER 3: Selecting stories'.

    Returns {YYYY-MM-DD: answer_text}. The first Answer after the Tier 3 marker
    is the important-story pick; the everyman pick comes second.
    """
    picks = {}
    for path in sorted(glob.glob(os.path.join(logs_dir, "log_*.txt"))):
        m = re.search(r"log_(\d\d)_(\d\d)_(\d\d)\.txt$", path)
        if not m:
            continue
        date = "20{}-{}-{}".format(*m.groups())
        seen_marker = False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "TIER 3: Selecting stories" in line:
                    seen_marker = True
                    continue
                if seen_marker:
                    am = re.search(r"Answer:\s*(.+)", line)
                    if am:
                        picks[date] = am.group(1).strip().strip("*").strip()
                        break
    return picks


def essay_answer(essay):
    """Pull the chosen headline out of a selection essay.

    Opus writes either 'Answer: <headline>' or a markdown heading '# Answer'
    with the headline on the next non-empty line. The last match wins.
    """
    found = None
    lines = (essay or "").replace("*", "").split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^\s*#*\s*(?:final\s+)?answer\s*:?\s*(.*)$", line, re.I)
        if not m:
            continue
        rest = m.group(1).strip()
        if not rest:
            rest = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        if rest:
            found = rest
    return found


def _load_success_rows(audit_paths):
    rows = []
    for path in audit_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # phase "benchmark" marks replays made by this benchmark, not mornings
                if r.get("event") == "success" and r.get("phase") != "benchmark":
                    rows.append(r)
    return sorted(rows, key=lambda r: r["timestamp"])


def harvest(audit_paths, logs_dir=None):
    """Build one case per morning, with every logged run of it kept in `runs`.

    The audit log has no responses, but the pipeline feeds Opus's selection
    essay straight into headline_extractor, so the next logged call carries the
    essay as its user prompt. That recovers the pick for each run. The daily
    log is the fallback for the latest run only.
    """
    rows = _load_success_rows(audit_paths)
    picks = production_picks(logs_dir) if logs_dir else {}
    groups = {}
    for i, r in enumerate(rows):
        if not (r.get("system_prompt") or "").startswith(TIER3_PREFIX):
            continue
        briefs, preamble, trailer = parse_briefs(r["user_prompt"])
        if len(briefs) < 2:
            continue

        essay = None
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt and (nxt.get("system_prompt") or "").startswith(EXTRACT_PREFIX):
            essay = nxt.get("user_prompt")
        answer = essay_answer(essay)
        tag_counts = {}
        for b in briefs:
            for t in b["tags"]:
                tag_counts[t["tag"]] = tag_counts.get(t["tag"], 0) + 1
        run = {
            "timestamp": r["timestamp"],
            "model": r.get("model"),
            "system_prompt_sha": hashlib.sha1(r["system_prompt"].encode("utf-8")).hexdigest()[:10],
            "tag_counts": tag_counts,
            "eligible_indexes": [b["index"] for b in briefs if b["eligible"]],
            "answer": answer,
            "pick_index": match_pick(answer, briefs) if answer else None,
            "essay": essay,
        }

        date = r["timestamp"][:10]
        key = (date, _content_key(briefs))
        prior = groups.get(key)
        runs = (prior["runs"] if prior else []) + [run]
        case = {
            "case_id": "{}_{}".format(date, key[1]),
            "date": date,
            "timestamp": r["timestamp"],
            "model": r.get("model"),
            "system_prompt": r["system_prompt"],
            "user_prompt": r["user_prompt"],
            "preamble": preamble,
            "coverage_notes": trailer,
            "briefs": briefs,
            "runs": runs,
            "n_variants": len(runs),
        }
        if answer is None and picks.get(date):
            answer = picks[date]
        case["production_answer"] = answer
        case["production_pick_index"] = match_pick(answer, briefs) if answer else None
        groups[key] = case  # the latest run of the same briefs supplies the shown tags
    return sorted(groups.values(), key=lambda c: c["timestamp"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", nargs="+", required=True, help="llm_audit*.jsonl files")
    ap.add_argument("--logs-dir", default=None, help="daily log_YY_MM_DD.txt directory, for the production pick")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    cases = harvest(args.audit, args.logs_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with_pick = sum(1 for c in cases if c["production_pick_index"] is not None)
    print("{} cases across {} dates; production pick matched for {}. Wrote {}".format(
        len(cases), len({c["date"] for c in cases}), with_pick, args.out))


if __name__ == "__main__":
    main()
