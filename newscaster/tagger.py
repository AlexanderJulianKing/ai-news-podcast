"""Repetition tagger: the model decides, the code edits.

The old tagger asked a model to retype the whole headline pool (about 100 lines)
with tags added and repeats deleted. A retyped pool can silently lose lines or
whole sections, and a lost line looks the same as a deliberate deletion. In a
2026-09-23 test on real mornings both Gemma and GPT-6 Luna dropped whole outlet
sections.

Here every headline line gets a number. The model returns one decision per number
(new / update / major / same, plus the story slug), in batches. The code applies
the tags to the original lines, so a line leaves the pool only on an explicit
"same" verdict. Numbers the model skips are asked again; anything still missing
stays in the pool untagged, and slugs are checked against the tracked stories.
"""
import re

from newscaster.logging import print_and_write
from newscaster.text_utils import extract_json

VERDICTS = ("new", "update", "major", "same")
_ALIASES = {"new story": "new", "major escalation": "major", "escalation": "major",
            "same story": "same", "same story, no new info": "same", "no new info": "same", "remove": "same"}
_TAG = {"update": "UPDATE", "major": "MAJOR ESCALATION"}
_OUTPUT_OLD = "Return the modified text and nothing else."
_LEAD = re.compile(r"^(\s*(?:[-*•]|\d+[.)])?\s*)")


def is_headline_line(line):
    """Same rule as the pool-line count: section headers and blank lines are short."""
    return len(line.strip()) > 20


def structured_prompt(template_prompt, ledger_mode):
    """Keep the existing category definitions; replace only the output instruction."""
    if _OUTPUT_OLD not in template_prompt:
        raise ValueError("tagger prompt no longer ends with the expected output instruction")
    fmt = (
        "OUTPUT: Do not rewrite or repeat any headline. The headlines are numbered. Where the definitions above "
        "say REMOVE, KEEP, or prepend a tag, give a verdict instead: \"same\" (SAME STORY, NO NEW INFO), "
        "\"update\", \"major\" (MAJOR ESCALATION), or \"new\" (NEW STORY). "
        + ("For \"update\" and \"major\", set \"arc\" to the arc_slug of the matching [ARC: ...] entry, copied "
           "exactly; otherwise set it to null. " if ledger_mode else "Set \"arc\" to null. ")
        + "Return JSON only, with exactly one decision for every numbered headline: "
        "{\"decisions\": [{\"n\": 1, \"verdict\": \"new\", \"arc\": null}]}"
    )
    return template_prompt.replace(_OUTPUT_OLD, fmt)


def number_lines(lines, indexes):
    return "\n".join("{}. {}".format(n, lines[i].strip()) for n, i in indexes)


def parse_decisions(reply, wanted, valid_slugs, ledger_mode):
    """Return ({n: (verdict, arc)}, problems). Unknown numbers and verdicts are dropped;
    an update/major whose slug is not a tracked story is kept as an untagged new story."""
    data = extract_json(reply, dict)
    decisions, problems = {}, []
    for item in data.get("decisions") or []:
        try:
            n = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        verdict = _ALIASES.get(verdict, verdict)
        if n not in wanted or verdict not in VERDICTS:
            continue
        arc = item.get("arc")
        arc = str(arc).strip() if arc not in (None, "", "null") else None
        if ledger_mode and verdict in _TAG:
            if arc not in valid_slugs:
                problems.append("line {}: '{}' is not a tracked story; kept untagged".format(n, arc))
                verdict, arc = "new", None
        else:
            arc = None if not ledger_mode else arc
        decisions[n] = (verdict, arc)
    return decisions, problems


def apply_decisions(lines, numbered, decisions, ledger_mode):
    """Rebuild the pool from the original lines. Returns (text, counts)."""
    by_index = {i: n for n, i in numbered}
    out, counts = [], {v: 0 for v in VERDICTS}
    counts["undecided"] = 0
    for i, line in enumerate(lines):
        n = by_index.get(i)
        if n is None:
            out.append(line)
            continue
        verdict, arc = decisions.get(n, (None, None))
        if verdict is None:
            counts["undecided"] += 1
            out.append(line)
            continue
        counts[verdict] += 1
        if verdict == "same":
            continue
        if verdict in _TAG:
            tag = "[{}: {}]".format(_TAG[verdict], arc) if ledger_mode else "[{}]".format(_TAG[verdict])
            lead = _LEAD.match(line).group(1)
            out.append(lead + tag + " " + line[len(lead):].lstrip())
        else:
            out.append(line)
    return "\n".join(out), counts


def tag_pool(all_headlines, template_prompt, ask, *, ledger_mode, valid_slugs=(), batch_size=40, label="tagger"):
    """Tag the pool. `ask(user_prompt, system_prompt)` returns the model's reply text.

    Batches of `batch_size` headlines; each batch gets one retry on unparseable JSON and
    one follow-up call for any numbers it skipped. Returns the rebuilt pool text.
    """
    system_prompt = structured_prompt(template_prompt, ledger_mode)
    lines = (all_headlines or "").split("\n")
    numbered = [(n, i) for n, i in enumerate((i for i, l in enumerate(lines) if is_headline_line(l)), start=1)]
    valid = set(valid_slugs or ())
    decisions, problems = {}, []
    for start in range(0, len(numbered), batch_size):
        batch = numbered[start:start + batch_size]
        pending = dict(batch)
        for attempt in (1, 2, 3):
            if not pending:
                break
            todo = sorted(pending.items())
            try:
                reply = ask(number_lines(lines, todo), system_prompt)
                got, bad = parse_decisions(reply, set(pending), valid, ledger_mode)
            except Exception as exc:  # unparseable reply or call failure: try again
                print_and_write("{}: batch at headline {} failed (attempt {}/3): {}".format(label, todo[0][0], attempt, exc))
                continue
            decisions.update(got)
            problems += bad
            pending = {n: i for n, i in pending.items() if n not in got}
    text, counts = apply_decisions(lines, numbered, decisions, ledger_mode)
    print_and_write("{}: {} headlines -> {}".format(label, len(numbered), ", ".join("{} {}".format(v, counts[v]) for v in counts if counts[v])))
    if counts["undecided"]:
        print_and_write("{} WARNING: {} headline(s) got no decision and were kept untagged".format(label, counts["undecided"]))
    for p in problems[:10]:
        print_and_write("{}: {}".format(label, p))
    return text
