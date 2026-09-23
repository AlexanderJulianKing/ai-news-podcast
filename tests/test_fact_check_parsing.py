"""Flag replies must never be misread as a clean script."""
from unittest.mock import patch

from newscaster import review


def test_parse_flag_reply_accepts_markdown_shapes():
    out = "Here is what I found:\n- FLAG: a — b\n**FLAG:** c — d\n1. FLAG: e\nFLAG: f"
    flags, ok = review.parse_flag_reply(out)
    assert ok
    assert flags == ["FLAG: a — b", "FLAG: c — d", "FLAG: e", "FLAG: f"]


def test_parse_flag_reply_none_and_malformed():
    assert review.parse_flag_reply("NONE") == ([], True)
    assert review.parse_flag_reply("**NONE.**") == ([], True)
    assert review.parse_flag_reply("The script looks accurate to me.") == ([], False)
    assert review.parse_flag_reply("") == ([], False)


def test_malformed_reply_is_retried_then_accepted():
    replies = iter(["I checked it and it seems fine overall.", "FLAG: wrong date — the vote was Tuesday"])
    with patch.object(review, "get_llm_response", side_effect=lambda *a, **k: next(replies)) as mock:
        flags = review.faithfulness_flags("script text", "source text")
    assert mock.call_count == 2
    assert flags == ["FLAG: wrong date — the vote was Tuesday"]


def test_malformed_twice_fails_open_and_is_logged():
    with patch.object(review, "get_llm_response", return_value="Looks good."), \
         patch.object(review, "write_jsonl_log") as log:
        assert review.stable_fact_flags("script text") == []
    assert log.call_args.args[0] == "fact_check_malformed"


def test_search_verify_reads_markdown_wrong():
    with patch.object(review, "openrouter_web_brief", return_value="**WRONG:** she is the foreign minister"):
        assert review._search_confirms_error("x")
    with patch.object(review, "openrouter_web_brief", return_value="CORRECT"):
        assert review._search_confirms_error("x") is None
