"""Downstream stages (story_gatherer, headline_maker, intro_writer,
segments_writer, assemble_podcast) all use slot-keyed data and skip missing
slots consistently."""
import os
from unittest.mock import patch

import pytest

from newscaster.script.headlines import story_gatherer, headline_maker
from newscaster.script.intro import intro_writer
from newscaster.script.segments import segments_writer
from newscaster.audio.assembly import assemble_podcast


def test_story_gatherer_returns_dict_keyed_by_slot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_summaries")
    os.makedirs("logs", exist_ok=True)
    # Slot 0 missing on purpose; slots 1 and 2 present.
    with open("segment_summaries/2026_11_05_segment1_summary.txt", "w") as f:
        f.write("slot-one content")
    with open("segment_summaries/2026_11_05_segment2_summary.txt", "w") as f:
        f.write("slot-two content")

    stories = story_gatherer("2026_11_05")
    assert isinstance(stories, dict)
    assert set(stories.keys()) == {1, 2}
    assert stories[1] == "slot-one content"
    assert stories[2] == "slot-two content"


def test_headline_maker_preserves_slot_keys():
    stories = {1: "story-one", 3: "story-three"}
    with patch('newscaster.script.headlines.get_llm_response',
               side_effect=lambda story, **_: f"headline:{story}"):
        headlines = headline_maker(stories)
    assert isinstance(headlines, dict)
    assert set(headlines.keys()) == {1, 3}
    assert headlines[1] == "headline:story-one"
    assert headlines[3] == "headline:story-three"


def test_segments_writer_uses_slot_in_filename(tmp_path, monkeypatch):
    """A sparse stories dict {1: ...} must produce segment_1.txt, not segment_0.txt."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    stories = {1: "story content for slot 1"}
    voices_list = ['Ethan', 'Chloe', 'Ethan', 'Chloe', 'Grace']

    valid_script = "Grace: hello there\nChloe: I'm reporting from somewhere according to the wire\n" * 30

    with patch('newscaster.script.segments.get_llm_response', return_value=valid_script):
        segments_writer(stories, "2026_11_05", voices_list, "November 5, 2026", arc_context=[None, None])

    assert os.path.exists("output_scripts/2026_11_05_segment_1.txt")
    assert not os.path.exists("output_scripts/2026_11_05_segment_0.txt")


def test_intro_writer_iterates_dict_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("episode_titles", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    stories = {1: "story-one body", 2: "story-two body"}
    topics = {1: "topic-one", 2: "topic-two"}

    with patch('newscaster.script.intro.get_llm_response',
               side_effect=lambda body, **_: f"title:{body[:10]}") as mock_title, \
         patch('newscaster.script.intro.call_with_default',
               return_value="intro2 narration"):
        intro1, intro2 = intro_writer("November 5, 2026", "60/40 sunny", topics, "2026_11_05", stories)

    # one title call per story (more if titles too long, but our mock returns short)
    assert mock_title.call_count >= 2
    assert "intro2 narration" in intro2

    with open("episode_titles/2026_11_05.txt") as f:
        content = f.read()
        assert "title:" in content
        assert "," in content  # multiple titles joined


def test_assemble_podcast_skips_missing_slot(tmp_path, monkeypatch):
    """Slot 0 audio missing, slot 1 audio present → podcast assembles using only slot 1."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("segment_audio", exist_ok=True)
    os.makedirs("output_audio", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # We can't easily synth real mp3s in a unit test; mock pydub.AudioSegment.
    from unittest.mock import MagicMock
    mock_audio = MagicMock()
    mock_audio.__add__.return_value = mock_audio
    mock_audio.__iadd__.return_value = mock_audio

    # Pretend slot 1's mp3 exists; slot 0, 2, 3 don't.
    with open("segment_audio/2026_11_05_segment_1.mp3", "wb") as f:
        f.write(b"fake mp3 bytes")
    with open("segment_audio/2026_11_05_intro.mp3", "wb") as f:
        f.write(b"fake intro")
    with open("segment_audio/2026_11_05_outro.wav", "wb") as f:
        f.write(b"fake outro")

    with patch('newscaster.audio.assembly.AudioSegment') as MockAudioSegment:
        MockAudioSegment.from_mp3.return_value = mock_audio
        MockAudioSegment.silent.return_value = mock_audio
        assemble_podcast("2026_11_05")

    # Check the calls included segment_1.mp3 and not segment_0.mp3
    loaded_paths = [
        call.args[0] for call in MockAudioSegment.from_mp3.call_args_list
    ]
    assert any("segment_1.mp3" in p for p in loaded_paths)
    assert not any("segment_0.mp3" in p for p in loaded_paths)
