"""Verify gather_news isolates per-topic failures: a failing topic leaves an
empty slot rather than shifting later topics down."""
import os
from unittest.mock import patch

import pytest

from newscaster.scrapers.topic_finder import TopicFinderResult
from newscaster.llm.errors import LLMRetriesExhaustedError, LLMTimeoutError
import newscaster.pipeline as pipeline


def _make_tf_result(topics):
    return TopicFinderResult(
        topics=topics,
        overview="Test overview text",
        follow_up_prompt_text="follow up",
        challenging_follow_up_prompt_text="challenging",
        arc_context=[None] * len(topics),
        ledger={"arcs": {}},
        side_story_briefs=[],
    )


def test_failed_topic_leaves_empty_slot(tmp_path, monkeypatch):
    """Topic 0 fails; topic 1 succeeds. segment0_summary.txt must NOT exist;
    segment1_summary.txt must contain topic 1's text."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["topic-zero", "topic-one"])

    def fake_one_topic(topic, topic_index, *_args, **_kwargs):
        if topic_index == 0:
            raise LLMRetriesExhaustedError("simulated topic-0 failure")
        return f"summary for {topic}"

    with patch('newscaster.pipeline.topic_finder', return_value=tf_result), \
         patch('newscaster.pipeline._gather_one_topic', side_effect=fake_one_topic):
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")

    assert result is tf_result
    assert not os.path.exists("segment_summaries/2026_11_05_segment0_summary.txt"), \
        "Failed slot 0 must not produce a summary file"
    assert os.path.exists("segment_summaries/2026_11_05_segment1_summary.txt"), \
        "Succeeded slot 1 must produce its summary file at slot 1, not slot 0"
    with open("segment_summaries/2026_11_05_segment1_summary.txt") as f:
        assert "topic-one" in f.read()


def test_all_topics_failing_leaves_no_summary_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["a", "b"])

    with patch('newscaster.pipeline.topic_finder', return_value=tf_result), \
         patch('newscaster.pipeline._gather_one_topic', side_effect=LLMTimeoutError("nope")):
        pipeline.gather_news("November 5, 2026", "2026_11_05")

    assert not os.path.exists("segment_summaries/2026_11_05_segment0_summary.txt")
    assert not os.path.exists("segment_summaries/2026_11_05_segment1_summary.txt")
    # Overview is still written even when all stories fail.
    assert os.path.exists("output_scripts/2026_11_05_overview.txt")
    # CRITICAL: GATHER_COMPLETE marker must NOT exist — otherwise next run sees
    # the marker, returns the manifest, and the script stage loops forever
    # against zero summaries.
    assert not os.path.exists("segment_summaries/2026_11_05_GATHER_COMPLETE.flag"), \
        "GATHER_COMPLETE must not be written when zero topics succeeded"


def test_corrupt_manifest_with_existing_summaries_raises_loudly(tmp_path, monkeypatch):
    """If the manifest file is unreadable (corrupt JSON) AND slot summaries already
    exist on disk, gather_news must raise rather than silently re-run topic_finder
    and orphan the existing summaries against newly-picked topics."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    # Existing slot summary from a prior partial run.
    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("preserved slot 0 content")
    # Corrupt manifest.
    with open(pipeline._manifest_path("2026_11_05"), "w") as f:
        f.write("this is not valid json {{{")

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder:
        with pytest.raises(pipeline._ManifestCorruptError):
            pipeline.gather_news("November 5, 2026", "2026_11_05")

    # topic_finder must NOT have been called — refusing to silently re-pick is the whole point.
    mock_topic_finder.assert_not_called()


def test_manifest_with_wrong_top_level_type_with_summaries_raises(tmp_path, monkeypatch):
    """A valid-JSON-but-wrong-type manifest (e.g. a naked list) must not slip
    through .get() with an AttributeError — it must fail closed when summaries
    are present."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("preserved slot 0")
    with open(pipeline._manifest_path("2026_11_05"), "w") as f:
        f.write('["valid", "json", "but", "wrong", "type"]')

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder:
        with pytest.raises(pipeline._ManifestCorruptError):
            pipeline.gather_news("November 5, 2026", "2026_11_05")
    mock_topic_finder.assert_not_called()


def test_corrupt_manifest_without_summaries_falls_through(tmp_path, monkeypatch):
    """If the manifest is corrupt but no slot summaries exist (genuine fresh run after
    a manifest write crashed), gather_news treats it as absent and re-runs topic_finder."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    with open(pipeline._manifest_path("2026_11_05"), "w") as f:
        f.write("garbage json {{{")

    tf_result = _make_tf_result(["fresh-a", "fresh-b"])

    with patch('newscaster.pipeline.topic_finder', return_value=tf_result), \
         patch('newscaster.pipeline._gather_one_topic',
               side_effect=lambda topic, idx, *a, **k: f"summary {topic}"):
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")

    assert result is tf_result
    assert os.path.exists("segment_summaries/2026_11_05_segment0_summary.txt")


def test_gather_skips_when_completion_marker_exists(tmp_path, monkeypatch):
    """gather_news short-circuits if the GATHER_COMPLETE marker is on disk —
    NOT just because segment0 happens to exist (which would block partial-rerun recovery)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("logs")
    with open("segment_summaries/2026_11_05_GATHER_COMPLETE.flag", "w") as f:
        f.write("2026_11_05")

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder:
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")

    assert result is None
    mock_topic_finder.assert_not_called()


def test_gather_writes_completion_marker_after_success(tmp_path, monkeypatch):
    """A successful gather (even with some failed slots) must write the marker."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["a", "b"])
    with patch('newscaster.pipeline.topic_finder', return_value=tf_result), \
         patch('newscaster.pipeline._gather_one_topic',
               side_effect=lambda topic, idx, *a, **k: f"summary {topic}" if idx == 1 else (_ for _ in ()).throw(LLMTimeoutError("nope"))):
        pipeline.gather_news("November 5, 2026", "2026_11_05")

    assert os.path.exists("segment_summaries/2026_11_05_GATHER_COMPLETE.flag"), \
        "Completion marker must be written even when some slots failed"


def test_marker_present_loads_manifest_and_returns_tf_result(tmp_path, monkeypatch):
    """When the GATHER_COMPLETE marker exists, gather_news must load and return the
    persisted TopicFinderResult — not None — so write_scripts and audience_learned
    extraction still get arc_context / ledger / side_story_briefs."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["topic-a", "topic-b"])
    tf_result.arc_context = [{"slug": "arc-a"}, {"slug": "arc-b"}]
    tf_result.ledger = {"arcs": {"arc-a": {"episodes": []}}}
    tf_result.side_story_briefs = [("h1", "b1"), ("h2", "b2")]

    # Simulate a completed prior run by writing manifest + marker directly.
    pipeline._save_manifest(tf_result, "2026_11_05")
    with open("segment_summaries/2026_11_05_GATHER_COMPLETE.flag", "w") as f:
        f.write("2026_11_05")

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder:
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")

    mock_topic_finder.assert_not_called()
    assert result is not None
    assert result.topics == ["topic-a", "topic-b"]
    assert result.arc_context == [{"slug": "arc-a"}, {"slug": "arc-b"}]
    assert result.ledger == {"arcs": {"arc-a": {"episodes": []}}}
    # Tuples in side_story_briefs survive the JSON round-trip.
    assert result.side_story_briefs == [("h1", "b1"), ("h2", "b2")]


def test_partial_rerun_reuses_manifest_topics_not_re_run_topic_finder(tmp_path, monkeypatch):
    """When marker is missing but manifest is present, gather_news must NOT call
    topic_finder again — that would re-pick non-deterministically and pair stale slot
    summaries with new topic metadata."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["original-a", "original-b"])
    pipeline._save_manifest(tf_result, "2026_11_05")
    # Existing slot 0 summary (from a prior partial run).
    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("preserved slot 0")

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder, \
         patch('newscaster.pipeline._gather_one_topic',
               side_effect=lambda topic, idx, *a, **k: f"freshly gathered {topic}"):
        result = pipeline.gather_news("November 5, 2026", "2026_11_05")

    mock_topic_finder.assert_not_called()
    assert result.topics == ["original-a", "original-b"]
    # Slot 0 preserved; slot 1 freshly gathered with the original topic name.
    with open("segment_summaries/2026_11_05_segment1_summary.txt") as f:
        assert "original-b" in f.read()


def test_gather_reuses_existing_slot_summary_on_partial_rerun(tmp_path, monkeypatch):
    """If marker is missing but slot 0's summary already exists (alongside its manifest),
    _gather_one_topic must NOT be called for slot 0 — the existing file gets reused."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["topic-a", "topic-b"])
    # Manifest was written by the prior run (immediately after topic selection),
    # so partial-rerun reuse requires it to be present.
    pipeline._save_manifest(tf_result, "2026_11_05")
    # Slot 0 was successfully gathered last time; slot 1 was missing.
    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("preserved slot 0 content")

    calls = []

    def fake_one_topic(topic, topic_index, *a, **k):
        calls.append(topic_index)
        return f"freshly gathered {topic}"

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder, \
         patch('newscaster.pipeline._gather_one_topic', side_effect=fake_one_topic):
        pipeline.gather_news("November 5, 2026", "2026_11_05")

    # topic_finder must NOT run — the manifest provides the topic selection.
    mock_topic_finder.assert_not_called()
    assert calls == [1], f"Expected slot 0 to be reused and only slot 1 to be gathered; got calls for {calls}"
    with open("segment_summaries/2026_11_05_segment0_summary.txt") as f:
        assert f.read() == "preserved slot 0 content"
    with open("segment_summaries/2026_11_05_segment1_summary.txt") as f:
        assert "freshly gathered" in f.read()


def test_manifest_absent_with_existing_summaries_raises(tmp_path, monkeypatch):
    """Symmetric to corrupt-manifest case: if manifest is missing but slot
    summaries exist, gather_news must raise rather than re-run topic_finder."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")
    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("preserved slot 0")

    with patch('newscaster.pipeline.topic_finder') as mock_topic_finder:
        with pytest.raises(pipeline._ManifestCorruptError):
            pipeline.gather_news("November 5, 2026", "2026_11_05")
    mock_topic_finder.assert_not_called()


def test_empty_existing_summary_file_is_ignored(tmp_path, monkeypatch):
    """A zero-byte / whitespace-only summary file from a botched prior write must
    be treated as absent — re-gathered, not blindly reused (which would let
    GATHER_COMPLETE be written against an empty story)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("output_scripts")
    os.makedirs("logs")

    tf_result = _make_tf_result(["topic-a"])
    pipeline._save_manifest(tf_result, "2026_11_05")
    # Empty existing summary on disk.
    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("   \n  \t  ")

    with patch('newscaster.pipeline._gather_one_topic',
               return_value="real freshly gathered content"):
        pipeline.gather_news("November 5, 2026", "2026_11_05")

    # Re-gathered content replaces the empty placeholder.
    with open("segment_summaries/2026_11_05_segment0_summary.txt") as f:
        assert "real freshly gathered" in f.read()
    # Marker written because we now have a real summary.
    assert os.path.exists("segment_summaries/2026_11_05_GATHER_COMPLETE.flag")
