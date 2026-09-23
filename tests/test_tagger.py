"""The structured tagger: the model decides, code edits, nothing is lost by accident."""
import json

import pytest

from newscaster import tagger
from newscaster.dedup import build_headline_arc_map
from newscaster.prompts import LEDGER_REPETITION_REMOVER_TEMPLATE, REPETITION_REMOVER_TEMPLATE

POOL = """NPR:
US strikes three Iranian oil tankers in the Gulf of Oman overnight
Fed holds rates steady and signals one cut before the year ends
- Senate passes a six-week stopgap bill and averts a shutdown
AP:
Wildfire in San Bernardino County forces 30,000 people to evacuate
Retailer reports a data breach affecting sixty million customer accounts"""


def _reply(decisions):
    return json.dumps({"decisions": [{"n": n, "verdict": v, "arc": a} for n, v, a in decisions]})


def _ledger_prompt():
    return LEDGER_REPETITION_REMOVER_TEMPLATE.format(arc_summaries="[ARC: us_iran_escalation] ...")


def test_prompt_keeps_definitions_and_replaces_only_the_output_instruction():
    p = tagger.structured_prompt(_ledger_prompt(), ledger_mode=True)
    assert "Return the modified text and nothing else." not in p
    assert "MAJOR ESCALATION" in p and '"decisions"' in p and "arc_slug" in p
    assert "Set \"arc\" to null" in tagger.structured_prompt(REPETITION_REMOVER_TEMPLATE.format(recent_stories="x"), ledger_mode=False)


def test_code_applies_tags_and_only_same_verdicts_remove_lines():
    def ask(user, system):
        assert "1. US strikes three Iranian" in user and "5. Retailer reports" in user
        return _reply([(1, "update", "us_iran_escalation"), (2, "same", None), (3, "major", "us_iran_escalation"),
                       (4, "new", None), (5, "new", None)])
    out = tagger.tag_pool(POOL, _ledger_prompt(), ask, ledger_mode=True, valid_slugs=["us_iran_escalation"])
    lines = out.split("\n")
    assert "[UPDATE: us_iran_escalation] US strikes three Iranian oil tankers in the Gulf of Oman overnight" in lines
    assert not any("Fed holds rates" in l for l in lines)                       # the one explicit "same"
    assert "- [MAJOR ESCALATION: us_iran_escalation] Senate passes a six-week stopgap bill and averts a shutdown" in lines
    assert "NPR:" in lines and "AP:" in lines                                   # headers untouched
    assert "Wildfire in San Bernardino County forces 30,000 people to evacuate" in lines
    m = build_headline_arc_map(out)
    assert ("UPDATE", "us_iran_escalation") in m.values() and ("MAJOR ESCALATION", "us_iran_escalation") in m.values()


def test_skipped_numbers_are_asked_again_and_never_dropped():
    calls = []

    def ask(user, system):
        calls.append(user)
        if len(calls) == 1:
            return _reply([(1, "new", None), (2, "new", None)])   # skips 3, 4, 5
        return _reply([])                                         # never answers them
    out = tagger.tag_pool(POOL, _ledger_prompt(), ask, ledger_mode=True, valid_slugs=[])
    asked_again = [l.split(".")[0] for l in calls[1].split("\n")]
    assert len(calls) == 3 and asked_again == ["3", "4", "5"]
    for line in POOL.split("\n"):
        assert line in out.split("\n")                            # every original line survives


def test_unknown_slug_is_kept_untagged_and_bad_json_is_retried():
    replies = iter(["not json at all", _reply([(n, "update", "made_up_slug") for n in range(1, 6)])])
    out = tagger.tag_pool(POOL, _ledger_prompt(), lambda u, s: next(replies), ledger_mode=True, valid_slugs=["us_iran_escalation"])
    assert "[UPDATE" not in out
    assert out.split("\n") == POOL.split("\n")


def test_history_mode_tags_without_slugs_and_batches():
    seen = []

    def ask(user, system):
        nums = [int(l.split(".")[0]) for l in user.split("\n")]
        seen.append(nums)
        return _reply([(n, "update", "anything") for n in nums])
    out = tagger.tag_pool(POOL, REPETITION_REMOVER_TEMPLATE.format(recent_stories="x"), ask, ledger_mode=False, batch_size=2)
    assert seen == [[1, 2], [3, 4], [5]]
    assert "[UPDATE] US strikes" in out and "[UPDATE: " not in out


def test_changed_prompt_template_fails_loudly():
    with pytest.raises(ValueError):
        tagger.structured_prompt("no output instruction here", ledger_mode=True)
