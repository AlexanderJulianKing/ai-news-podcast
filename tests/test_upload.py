from unittest.mock import patch

from newscaster.upload import fit_title_to_limit


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
