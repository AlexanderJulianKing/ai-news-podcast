"""Stories left out of the roundup (UNVERIFIED) must not become audience memory."""
import json
from unittest.mock import patch

import newscaster.pipeline as pipeline
from newscaster.scrapers.topic_finder import TopicFinderResult


def test_unverified_side_story_is_not_recorded_as_learned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = {"arcs": {
        "aired_arc": {"episodes": [{"date": "2026_09_24", "coverage": "side", "coverage_slot": 1}]},
    }}
    tf = TopicFinderResult(topics=[], overview="", follow_up_prompt_text="", challenging_follow_up_prompt_text="",
                           arc_context=[], ledger=ledger, side_story_briefs=[
                               ("Trump threatens to annihilate Iran", "UNVERIFIED: Source-hunter research did not return accepted current evidence."),
                               ("Aired story", "FINDINGS: something real happened."),
                           ])
    prompts, updates = [], []
    def fake_llm(prompt, **kw):
        prompts.append(prompt)
        return json.dumps([{"learned": ["something real happened"]}])
    with patch.object(pipeline, "get_llm_response", side_effect=fake_llm), \
         patch.object(pipeline, "update_audience_learned", side_effect=lambda *a: updates.append(a)), \
         patch.object(pipeline, "save_ledger"):
        pipeline._extract_audience_learned("2026_09_24", tf)
    assert prompts and "annihilate" not in prompts[-1] and "Aired story" in prompts[-1]
    assert updates == [(ledger, "aired_arc", "2026_09_24", 1, ["something real happened"], "something real happened")]
