"""gather_news writes per-slot research sidecars and calls index_day."""
import json
import os
from unittest.mock import patch

from newscaster.scrapers.topic_finder import TopicFinderResult
import newscaster.pipeline as pipeline


def _tf(topics, arc_context):
    return TopicFinderResult(
        topics=topics, overview="ov", follow_up_prompt_text="f",
        challenging_follow_up_prompt_text="c", arc_context=arc_context,
        ledger={"arcs": {}}, side_story_briefs=[],
    )


def test_gather_writes_sidecar_with_arc_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("segment_summaries", "output_scripts", "logs"):
        os.makedirs(d)

    tf_result = _tf(["topic-zero"], arc_context=[{"slug": "arc-zero"}])

    def fake_one_topic(topic, idx, *a, articles=None, followups=None, **k):
        if articles is not None:
            articles.append({"chunk_id": f"2026_11_05_seg{idx}_art0", "url": "u",
                             "outlet": "NPR", "original_headline": "h",
                             "published_date": None, "retrieved_date": "2026_11_05",
                             "surfacing_topic": topic, "summary": "captured summary"})
        return f"summary for {topic}"

    with patch("newscaster.pipeline.topic_finder", return_value=tf_result), \
         patch("newscaster.pipeline._gather_one_topic", side_effect=fake_one_topic), \
         patch("newscaster.pipeline.index_day", return_value=1) as mock_index:
        pipeline.gather_news("November 5, 2026", "2026_11_05")

    sidecar = "segment_summaries/2026_11_05_segment0_research.json"
    assert os.path.exists(sidecar)
    with open(sidecar) as f:
        rec = json.load(f)
    assert rec["arc_slug"] == "arc-zero"
    assert rec["articles"][0]["summary"] == "captured summary"
    mock_index.assert_called_once_with("2026_11_05")


def test_gather_index_failure_does_not_break_gather(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("segment_summaries", "output_scripts", "logs"):
        os.makedirs(d)
    tf_result = _tf(["t"], arc_context=[None])
    with patch("newscaster.pipeline.topic_finder", return_value=tf_result), \
         patch("newscaster.pipeline._gather_one_topic", side_effect=lambda topic, idx, *a, **k: f"s {topic}"), \
         patch("newscaster.pipeline.index_day", side_effect=RuntimeError("index boom")):
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")
    assert os.path.exists("segment_summaries/2026_11_05_segment0_summary.txt")
    assert os.path.exists("segment_summaries/2026_11_05_GATHER_COMPLETE.flag")
    assert result is tf_result


def test_gather_reindexes_on_marker_present_rerun(tmp_path, monkeypatch):
    """A completed day must still (re)index on rerun, so a prior indexing failure self-heals."""
    monkeypatch.chdir(tmp_path)
    for d in ("segment_summaries", "output_scripts", "logs"):
        os.makedirs(d)
    tf_result = _tf(["t"], arc_context=[None])
    pipeline._save_manifest(tf_result, "2026_11_05")
    with open("segment_summaries/2026_11_05_GATHER_COMPLETE.flag", "w") as f:
        f.write("2026_11_05")
    with patch("newscaster.pipeline.index_day", return_value=0) as mock_index:
        pipeline.gather_news("November 5, 2026", "2026_11_05")
    mock_index.assert_called_once_with("2026_11_05")
