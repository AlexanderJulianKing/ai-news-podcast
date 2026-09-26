from types import SimpleNamespace
from unittest.mock import patch

from newscaster.upload import fit_title_to_limit, resumable_upload, write_upload_marker


def test_short_title_is_left_unchanged():
    t = "June 24, 2026 - A short, fine title"
    with patch("newscaster.upload.get_llm_response") as m:
        assert fit_title_to_limit(t) == t
    m.assert_not_called()  # no LLM call when already within the limit


def test_long_title_is_regenerated_under_limit():
    long_title = "June 24, 2026 - " + "x" * 120  # 136 chars, well over 100
    replacement = "June 24, 2026 - Supreme Court expands deportation power over green card holders"
    with patch("newscaster.upload.get_llm_response", return_value=replacement) as m:
        out = fit_title_to_limit(long_title)
    assert out == replacement
    assert len(out) <= 100
    m.assert_called_once()
    # The too-long title is shown to the model as the example to beat.
    assert long_title in m.call_args.args[0]


def test_strips_quotes_and_extra_lines_from_model_output():
    long_title = "June 24, 2026 - " + "y" * 120
    with patch("newscaster.upload.get_llm_response", return_value='  "A tidy short title"\nsome trailing note  '):
        out = fit_title_to_limit(long_title)
    assert out == "A tidy short title"


def test_truncates_when_model_keeps_returning_too_long():
    long_title = "June 24, 2026 - " + "word " * 40  # ~216 chars
    with patch("newscaster.upload.get_llm_response", return_value="z" * 150):  # always too long
        out = fit_title_to_limit(long_title, max_attempts=2)
    assert 0 < len(out) <= 100  # truncated as a last resort, never empty


def test_truncates_when_llm_errors():
    long_title = "June 24, 2026 - " + "alpha beta gamma delta epsilon " * 6
    with patch("newscaster.upload.get_llm_response", side_effect=RuntimeError("llm down")):
        out = fit_title_to_limit(long_title)
    assert 0 < len(out) <= 100
    assert " " in out  # truncated at a word boundary, not mid-word


def test_resumable_upload_returns_video_id():
    request = SimpleNamespace(
        uri="https://example.com/upload",
        headers={},
        body="{}",
        next_chunk=lambda: (None, {"id": "video-123"}),
    )
    assert resumable_upload(request) == "video-123"


def test_write_upload_marker_is_nonempty(tmp_path):
    write_upload_marker("2026_07_22", "video-123", str(tmp_path))
    assert tmp_path.joinpath("2026_07_22_UPLOAD_COMPLETE.flag").read_text() == "video-123\n"
