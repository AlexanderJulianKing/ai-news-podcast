"""2026-11-02: the lead (an update on the AI-safety thread) retold the July Hugging Face breach as news,
though it aired on Sept 24, 27 and 30. The writer saw only the thread's one-paragraph summary, which had
dropped Hugging Face. Earlier facts that name the same things as today's brief now go to the writer, dated.
"""
import os
from unittest.mock import patch

from newscaster.script import segments


def _arc():
    return {"slug": "ai_safety_cyberattacks",
            "audience_state": "AI developers have disclosed agents acting beyond their instructions.",
            "episodes": [
                {"date": "2026_09_27", "coverage": "main", "audience_learned": [
                    "METR found that about 1,200 agents coordinated the attack on Hugging Face.",
                    "OpenAI halted development of its latest models."]},
                {"date": "2026_10_11", "coverage": "main", "audience_learned": [
                    "Anthropic disabled live internet access for internal evaluations."]},
                {"date": "2026_11_02", "coverage": "main", "audience_learned": []}]}


def test_an_update_tells_the_writer_which_earlier_facts_it_already_aired(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts")
    brief = "OpenAI agents attacked RubyGems in May, two months before the Hugging Face breach that METR studied."
    script = "Grace: Good morning.\nEthan: Thanks, Grace. It happened today, according to RubyGems.\nGrace: Thank you."
    with patch.object(segments, "get_llm_response", return_value=script) as llm:
        segments.segments_writer({0: brief}, "2026_11_02", ["Ethan"], "November 2, 2026", arc_context=[_arc()])
    prompt = llm.call_args.kwargs["system_prompt"]
    assert "September 27" in prompt and "1,200 agents coordinated the attack on Hugging Face" in prompt
    assert "Anthropic disabled live internet access" not in prompt     # names nothing in today's brief
