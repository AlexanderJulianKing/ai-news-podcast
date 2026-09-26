"""Replay labeled lead-pick mornings through a model tier and score the picks.

    python3 -m benchmarks.editorial_decisions.run_lead_pick --modes heavy advanced --dry-run
    python3 -m benchmarks.editorial_decisions.run_lead_pick --modes heavy advanced standard

Each mode is a pipeline tier (light, standard, advanced, adversary, heavy), so
the model behind it is whatever newscaster/config.py maps it to. Calls use
the router's retry path with the GPT-5.5 fallback switched off, and cost real money; --dry-run prints the plan and an
input-token estimate without calling anything.

The pick is read from the essay with the harvester's parser, not with the
pipeline's LLM extractor, so scoring adds no calls and is repeatable.
Real and invented mornings are always reported apart.
"""
import argparse
import json
import os
import re
from datetime import datetime

from benchmarks.editorial_decisions.harvest_lead_pick import essay_answer, match_pick, same_headline  # noqa: F401
from benchmarks.editorial_decisions.hypotheticals import load_hypotheticals
from benchmarks.editorial_decisions.label_server import CASES_PATH, _read_jsonl, load_labels

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODES = ("light", "standard", "advanced", "adversary", "heavy")
# Models that are not a pipeline tier, addressable by name. Anthropic models go
# through the pipeline's own adapter (adaptive thinking, effort high). No
# refusal fallback is configured on purpose: a fallback would score another
# model under this one's name, so a refusal is recorded as an error row.
EXTRA_TIERS = {
    "fable": {"provider": "anthropic", "model": "claude-fable-5-1"},
    "opus5": {"provider": "anthropic", "model": "claude-opus-5"},
    "opus55": {"provider": "anthropic", "model": "claude-opus-5-5"},
    "gpt6sol": {"provider": "openrouter", "model": "openai/gpt-6-sol", "name": "GPT-6 Sol", "reasoning": "high"},
    "gpt6luna": {"provider": "openrouter", "model": "openai/gpt-6-luna", "name": "GPT-6 Luna", "reasoning": "high"},
}
# USD per million tokens (input, output), for models the pipeline's adapter has no price for.
EXTRA_PRICES = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5": (5.0, 25.0),
                "claude-opus-5-5": (4.0, 20.0)}  # Opus 5.5: OpenRouter list price, 2026-09-22


def tier_spec(mode):
    if mode in EXTRA_TIERS:
        return dict(EXTRA_TIERS[mode])
    from newscaster.llm.router import _select_primary
    return _select_primary(mode, False, False)


def usage_cost(usage):
    """Dollar cost of one call from its captured usage, or 0.0 when unknown."""
    usage = usage or {}
    known = usage.get("cost") or usage.get("estimated_cost_usd")
    if known:
        return known
    price = EXTRA_PRICES.get(usage.get("model"))
    if not price:
        return 0.0
    return (usage.get("total_input_tokens") or usage.get("input_tokens") or 0) * price[0] / 1e6 + (usage.get("output_tokens") or 0) * price[1] / 1e6


def labeled_cases(include_real=True, include_synthetic=True):
    labels = load_labels()
    cases = []
    if include_real:
        cases += _read_jsonl(CASES_PATH)
    if include_synthetic:
        cases += load_hypotheticals()
    return [(c, labels[c["case_id"]]) for c in cases if c["case_id"] in labels]


GROUPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "story_groups.json")


def story_key(case):
    """Map each brief index to a canonical index for its story.

    Real mornings can carry the same story twice. Briefs whose headlines are
    identical, or where one opens the other, are merged automatically; story_groups.json adds hand-checked
    groups for same-event briefs worded differently. Picking either copy of a
    story is the same editorial decision, so scoring compares canonical indexes.
    """
    canon, roots = {}, []
    for b in case["briefs"]:
        root = next((i for i, h in roots if same_headline(h, b["headline"])), None)
        if root is None:
            roots.append((b["index"], b["headline"]))
            root = b["index"]
        canon[b["index"]] = root
    manual = {}
    if os.path.exists(GROUPS_PATH):
        with open(GROUPS_PATH, "r", encoding="utf-8") as f:
            manual = json.load(f).get("groups", {})
    for group in manual.get(case.get("case_id"), []):
        root = min(canon[i] for i in group)
        for i, c in list(canon.items()):
            if c in {canon[j] for j in group}:
                canon[i] = root
    return canon


def unlabeled_cases(set_name=None):
    """Mornings with no label yet. Running a frozen prompt on them BEFORE they are
    labeled locks the predictions in, so the labels cannot influence them."""
    labels = load_labels()
    cases = _read_jsonl(CASES_PATH) + load_hypotheticals()
    return [(c, None) for c in cases if c["case_id"] not in labels and (not set_name or (c.get("set") or "real") == set_name)]


def score_pick(pick_index, case, label):
    """Classify one pick against one label.

    exact: the labeler's lead. acceptable: the lead or an 'also fine' brief.
    barred: the model chose a brief its own input tagged ineligible.
    unparsed: no brief could be matched to the essay's answer.
    A 'none deserved' label has no lead, so every pick scores as a miss.
    """
    eligible = {b["index"] for b in case["briefs"] if b["eligible"]}
    key = story_key(case)
    ok = {key[i] for i in (label.get("acceptable_indexes") or [])}
    lead = key.get(label.get("lead_index"))
    if lead is not None:
        ok.add(lead)
    pick = key.get(pick_index)
    return {
        "unparsed": pick_index is None,
        "exact": pick is not None and (pick == lead or pick in {key.get(i) for i in label.get("lead_alternates", [])}),
        "acceptable": pick is not None and pick in ok,
        "barred": pick_index is not None and pick_index not in eligible,
    }


def parse_pick(essay, briefs):
    """Return (pick_index, answer_text, method) for a selection essay.

    'answer': an Answer marker was found, as the prompt asks. 'opening': no
    marker, but one of the first three non-empty lines is a brief's headline,
    which is how the smaller models reply ("## Selected Story", then the
    headline). Only the opening is tried; scanning the whole essay would match
    stories the model discussed and rejected.
    """
    answer = essay_answer(essay)
    if answer:
        pick = match_pick(answer, briefs)
        if pick is not None:
            return pick, answer, "answer"
    opening = [l.strip() for l in (essay or "").replace("*", "").split("\n") if l.strip()][:3]
    for line in opening:
        line = re.sub(r"^(selected\s+)?(headline|story)\s*:\s*", "", line, flags=re.I)
        pick = match_pick(line, briefs)
        if pick is not None:
            return pick, line, "opening"
    # "The most important story is Brief 5." Only a line that announces the pick
    # counts; a line that merely discusses Brief 5 must not.
    named = re.compile(r"(?:most important story|lead story|i select|i choose|selected story|my pick|answer)[^.\n]{0,80}?\bbrief\s+(\d+)", re.I)
    valid = {b["index"] for b in briefs}
    for line in ([answer] if answer else []) + opening:
        m = named.search(line or "")
        if m and int(m.group(1)) in valid:
            return int(m.group(1)), line, "brief_number"
    return None, None, "none"


def summarize(rows):
    """Aggregate per (mode, real/invented). Returns a list of dict rows."""
    groups = {}
    for r in rows:
        key = (r["mode"], r.get("set") or ("invented" if r["synthetic"] else "real"))
        g = groups.setdefault(key, {"mode": key[0], "set": key[1], "model": r.get("model"), "n": 0, "exact": 0, "acceptable": 0, "barred": 0, "unparsed": 0})
        g["n"] += 1
        for k in ("exact", "acceptable", "barred", "unparsed"):
            g[k] += int(r[k])
    return [groups[k] for k in sorted(groups)]


def format_summary(summary):
    lines = ["{:<10} {:<14} {:>3} {:>7} {:>11} {:>7} {:>9}  {}".format("mode", "set", "n", "exact", "acceptable", "barred", "unparsed", "model")]
    for g in summary:
        lines.append("{:<10} {:<14} {:>3} {:>7} {:>11} {:>7} {:>9}  {}".format(
            g["mode"], g["set"], g["n"], g["exact"], g["acceptable"], g["barred"], g["unparsed"], g.get("model") or ""))
    return "\n".join(lines)


def _mode_model(mode):
    return tier_spec(mode).get("model")


LAST_USAGE = {}


def capture_usage():
    """The router hands token usage to its audit hook and nowhere else. Swap the
    hook for one that keeps the latest usage in LAST_USAGE, so spend is reportable."""
    import newscaster.llm.router as router

    def _capture(event, *a, usage=None, **k):
        if event == "success":
            LAST_USAGE.clear()
            LAST_USAGE.update(usage or {})
    router._audit_llm_event = _capture


def call_tier(mode, user_prompt, system_prompt):
    """Call the tier's own model, with the router's retries but no fallback.

    get_llm_response falls back to GPT-5.5 when a tier fails. In a benchmark that
    would score GPT-5.5's pick under another model's name, so a failure here
    raises and is recorded as an error row.
    """
    import uuid

    from newscaster.llm.router import _call_with_retry

    spec = tier_spec(mode)
    return _call_with_retry(spec, user_prompt, system_prompt, call_id=str(uuid.uuid4()), phase="benchmark")


def run(modes, pairs, system_prompt, repeats=1, out_path=None, variant="production"):
    rows = []
    for mode in modes:
        model = _mode_model(mode)
        for case, label in pairs:
            for rep in range(repeats):
                LAST_USAGE.clear()
                try:
                    essay = call_tier(mode, case["user_prompt"], system_prompt)
                    error = None
                except Exception as e:  # a failed call is a result, not a crash
                    essay, error = "", "{}: {}".format(type(e).__name__, e)
                pick, answer, method = parse_pick(essay, case["briefs"])
                row = {"mode": mode, "model": model, "case_id": case["case_id"], "synthetic": bool(case.get("synthetic")),
                       "set": case.get("set") or "real",
                       "repeat": rep, "variant": variant, "answer": answer, "parse_method": method, "pick_index": pick,
                       "lead_index": (label or {}).get("lead_index"), "acceptable_indexes": (label or {}).get("acceptable_indexes"),
                       "essay": essay, "error": error, "usage": dict(LAST_USAGE)}
                if label is not None:
                    row.update(score_pick(pick, case, label))
                else:  # predicted before any label exists; compare_variants scores it later
                    row.update({"unparsed": pick is None, "exact": False, "acceptable": False, "barred": False, "prelabel": True})
                rows.append(row)
                if out_path:  # append as we go, so a killed run keeps what it paid for
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if label is None:  # never print a pick for an unlabeled morning
                    print("{:<10} {:<42} {}".format(mode, case["case_id"][:42], "ERROR" if error else ("unparsed" if pick is None else "recorded")))
                else:
                    print("{:<10} {:<42} pick={} lead={} {}".format(
                        mode, case["case_id"][:42], pick, label.get("lead_index"),
                        "EXACT" if row["exact"] else ("ok" if row["acceptable"] else ("ERROR" if error else "miss"))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", nargs="+", default=["heavy"], choices=MODES + tuple(EXTRA_TIERS))
    ap.add_argument("--only", choices=("real", "invented"), default=None)
    ap.add_argument("--set", dest="set_name", default=None, help="only this set: real, invented, or invented_hard")
    ap.add_argument("--repeats", type=int, default=1, help="calls per case, to see run-to-run spread")
    ap.add_argument("--variant", default="production", help="which Tier 3 prompts to test; candidate_prompts.py holds the revisions")
    ap.add_argument("--unlabeled", action="store_true", help="run only mornings that have NO label yet (predictions locked before labeling)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from benchmarks.editorial_decisions.candidate_prompts import prompts_for
    system_prompt = prompts_for(args.variant)[0]

    pairs = labeled_cases(include_real=args.only != "invented", include_synthetic=args.only != "real")
    if args.unlabeled:
        pairs = unlabeled_cases(args.set_name)
    elif args.set_name:
        pairs = [(c, l) for c, l in pairs if (c.get("set") or "real") == args.set_name]
    from benchmarks.editorial_decisions.candidate_prompts import FEWSHOT_EXCLUDE
    skip = FEWSHOT_EXCLUDE.get(args.variant, set())  # mornings the prompt itself quotes as examples
    pairs = [(c, l) for c, l in pairs if c["case_id"] not in skip]
    n_calls = len(pairs) * len(args.modes) * args.repeats
    est_in = sum((len(c["user_prompt"]) + len(system_prompt)) // 4 for c, _ in pairs) * args.repeats
    print("{} labeled cases x {} modes x {} repeats = {} calls; about {:,} input tokens per mode (chars/4 estimate)".format(
        len(pairs), len(args.modes), args.repeats, n_calls, est_in))
    for mode in args.modes:
        print("  {:<10} -> {}".format(mode, _mode_model(mode)))
    if args.dry_run:
        return

    import newscaster.config as config
    config.init()  # loads keys.txt; the key constants are None until this runs
    # Keep benchmark calls out of the pipeline's audit log. The harvester reads
    # that log, and a replayed Tier 3 prompt would look like a real morning.
    config.LLM_AUDIT_LOG_ENABLED = False
    capture_usage()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "lead_pick_{}_{}_{}.jsonl".format(args.variant, "-".join(args.modes), datetime.now().strftime("%Y%m%d_%H%M%S")))
    rows = run(args.modes, pairs, system_prompt, repeats=args.repeats, out_path=out, variant=args.variant)
    if not args.unlabeled:
        print("\n" + format_summary(summarize(rows)))
    print("\nWrote " + out)


if __name__ == "__main__":
    main()
