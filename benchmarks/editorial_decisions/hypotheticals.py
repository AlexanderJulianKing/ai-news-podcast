"""Load the invented lead-pick mornings in the same shape as harvested cases.

Each hypothetical is rendered into the production research-document format and
parsed back with the harvester's own parser, so a replay sees the same layout a
real morning has. Hypotheticals carry `synthetic: True` and must be reported
apart from real mornings.
"""
import json
import os

from benchmarks.editorial_decisions.harvest_lead_pick import parse_briefs

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_HERE, "hypothetical_cases.json")
# (file, set name). The hard set is reported apart from the first, easier one.
SETS = (
    (DEFAULT_PATH, "invented"),
    (os.path.join(_HERE, "hypothetical_cases_hard.json"), "invented_hard"),
    (os.path.join(_HERE, "hypothetical_cases_hard2.json"), "invented_hard2"),  # written by an Opus subagent, blind to the labels
    (os.path.join(_HERE, "hypothetical_cases_hold1.json"), "holdout1"),  # Opus subagent, blind; score once per frozen prompt
)
COVERAGE_NOTES_HEADER = "=== Coverage notes: stories previously mentioned only in the side-story roundup ==="  # prompts.COVERAGE_NOTES_HEADER


def render_research_document(briefs):
    """Mirror topic_finder._format_research_briefs."""
    sections = []
    for i, b in enumerate(briefs, 1):
        header = "--- Brief {} ---\nHeadline: {}\n".format(i, b["headline"])
        if b.get("reported_by"):
            header += "Reported by: {}\n".format(", ".join(b["reported_by"]))
        sections.append("{}\n{}\n".format(header, b["brief"]))
    return "\n".join(sections)


def load_hypotheticals(path=None):
    """All invented cases, or only those in `path` when one is given."""
    sets = ((path, "invented"),) if path else SETS
    cases = []
    for set_path, set_name in sets:
        with open(set_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cases += [_build(c, set_name) for c in raw["cases"]]
    return cases


def _build(c, set_name):
    document = render_research_document(c["briefs"])
    if c.get("coverage_notes"):  # same trailer topic_finder appends for Tier 3
        document += "\n\n" + COVERAGE_NOTES_HEADER + "\n" + "\n".join(c["coverage_notes"])
    briefs, preamble, trailer = parse_briefs(document)
    return {
        "case_id": c["case_id"],
        "date": "hypothetical",
        "synthetic": True,
        "set": set_name,
        "axis": c["axis"],
        "probe": c["probe"],
        "system_prompt": None,  # a replay supplies the Tier 3 prompt under test
        "user_prompt": document,
        "preamble": preamble,
        "coverage_notes": trailer,
        "briefs": briefs,
        "runs": [],
        "production_answer": None,
        "production_pick_index": None,
    }
