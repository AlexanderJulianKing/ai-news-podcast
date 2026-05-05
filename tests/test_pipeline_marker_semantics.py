"""Tests for marker semantics: SCRIPTS_COMPLETE only when at least one segment
was produced; otherwise the next run must retry."""
import os
from unittest.mock import patch, MagicMock

import pytest

import newscaster.pipeline as pipeline


def test_scripts_marker_not_written_when_zero_segments_succeed(tmp_path, monkeypatch):
    """If segments_writer fails for every slot, write_scripts must NOT write
    SCRIPTS_COMPLETE — otherwise an all-failed transient becomes permanently sticky."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts")
    os.makedirs("episode_titles")
    os.makedirs("logs")
    os.makedirs("segment_summaries")
    # Pretend gather produced summaries for two slots.
    for i in range(2):
        with open(f"segment_summaries/2026_11_05_segment{i}_summary.txt", "w") as f:
            f.write(f"slot {i} content")

    # Patch heavy collaborators: weather, intro_writer, segments_writer (which writes nothing),
    # and audience_learned extraction.
    with patch('newscaster.pipeline.get_daily_temp', return_value="60/40"), \
         patch('newscaster.pipeline.intro_writer', return_value=("intro1", "intro2")), \
         patch('newscaster.pipeline.headline_maker', return_value={0: "h0", 1: "h1"}), \
         patch('newscaster.pipeline.segments_writer'), \
         patch('newscaster.pipeline._extract_audience_learned'):
        pipeline.write_scripts("November 5, 2026", "2026_11_05", "spoken-date", ['Ethan', 'Chloe'], tf_result=None)

    # Scripts marker must NOT exist because no segment files were produced.
    assert not os.path.exists("output_scripts/2026_11_05_SCRIPTS_COMPLETE.flag"), \
        "SCRIPTS_COMPLETE must not be written when zero segments succeeded"


def test_scripts_marker_ignores_orphan_segment_files_from_prior_runs(tmp_path, monkeypatch):
    """A wildcard glob would count e.g. 2026_11_05_segment_legacy.txt or stale slot
    files from a prior date-recycled run. The check must be slot-explicit."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts")
    os.makedirs("episode_titles")
    os.makedirs("logs")
    os.makedirs("segment_summaries")
    # Today's gather produced one summary at slot 0.
    with open("segment_summaries/2026_11_05_segment0_summary.txt", "w") as f:
        f.write("slot 0 content")
    # An orphan file from a manual edit / legacy run that matches segment_*.txt:
    with open("output_scripts/2026_11_05_segment_legacy.txt", "w") as f:
        f.write("not a real segment")

    # segments_writer is patched to NOT produce any segment_{slot}.txt for the
    # slots in stories. The orphan file must NOT count as success.
    with patch('newscaster.pipeline.get_daily_temp', return_value="60/40"), \
         patch('newscaster.pipeline.intro_writer', return_value=("intro1", "intro2")), \
         patch('newscaster.pipeline.headline_maker', return_value={0: "h0"}), \
         patch('newscaster.pipeline.segments_writer'), \
         patch('newscaster.pipeline._extract_audience_learned'):
        pipeline.write_scripts("November 5, 2026", "2026_11_05", "spoken-date", ['Ethan'], tf_result=None)

    assert not os.path.exists("output_scripts/2026_11_05_SCRIPTS_COMPLETE.flag"), \
        "SCRIPTS_COMPLETE must not be written when no slot in stories has its segment_{slot}.txt"


def test_scripts_marker_written_when_at_least_one_segment_succeeds(tmp_path, monkeypatch):
    """At least one produced segment script means the run got something; mark complete."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts")
    os.makedirs("episode_titles")
    os.makedirs("logs")
    os.makedirs("segment_summaries")
    for i in range(2):
        with open(f"segment_summaries/2026_11_05_segment{i}_summary.txt", "w") as f:
            f.write(f"slot {i} content")

    def fake_segments_writer(stories, formatted_date2, *a, **k):
        # Simulate one segment succeeding.
        with open(f'output_scripts/{formatted_date2}_segment_1.txt', 'w') as f:
            f.write("Grace: hi\nChloe: hello there\n")

    with patch('newscaster.pipeline.get_daily_temp', return_value="60/40"), \
         patch('newscaster.pipeline.intro_writer', return_value=("intro1", "intro2")), \
         patch('newscaster.pipeline.headline_maker', return_value={0: "h0", 1: "h1"}), \
         patch('newscaster.pipeline.segments_writer', side_effect=fake_segments_writer), \
         patch('newscaster.pipeline._extract_audience_learned'):
        pipeline.write_scripts("November 5, 2026", "2026_11_05", "spoken-date", ['Ethan', 'Chloe'], tf_result=None)

    assert os.path.exists("output_scripts/2026_11_05_SCRIPTS_COMPLETE.flag")
