"""Tests for the router's retry policy."""
from unittest.mock import patch

import pytest

import newscaster.config as _config
from newscaster.llm.router import get_llm_response, _backoff_delay
from newscaster.llm.errors import (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthError,
    LLMBadRequestError,
    LLMRetriesExhaustedError,
)


def _patch_no_sleep():
    return patch('newscaster.llm.router.time.sleep', return_value=None)


def test_succeeds_on_first_attempt_no_retry():
    with patch('newscaster.llm.router.gemini', return_value='ok') as mock_g:
        assert get_llm_response('p', mode='light') == 'ok'
        assert mock_g.call_count == 1


def test_retries_until_success():
    side_effects = [LLMTimeoutError("t1"), LLMTimeoutError("t2"), 'ok']
    with patch('newscaster.llm.router.gemini', side_effect=side_effects) as mock_g, _patch_no_sleep():
        assert get_llm_response('p', mode='light') == 'ok'
        assert mock_g.call_count == 3


def test_exhausts_retries_then_falls_back_to_openrouter():
    """After MAX_RETRIES retryable failures on primary, router falls through to OpenRouter."""
    with patch('newscaster.llm.router.gemini', side_effect=LLMTimeoutError("nope")) as mock_g, \
         patch('newscaster.llm.router.get_openrouter_response', return_value='fallback_ok') as mock_or, \
         _patch_no_sleep():
        result = get_llm_response('p', mode='light')
        assert result == 'fallback_ok'
        assert mock_g.call_count == _config.MAX_RETRIES
        assert mock_or.call_count == 1


def test_auth_error_falls_back_immediately_without_retrying():
    """An auth failure on the primary should jump straight to fallback (don't waste retries)."""
    with patch('newscaster.llm.router.gemini', side_effect=LLMAuthError("bad key")) as mock_g, \
         patch('newscaster.llm.router.get_openrouter_response', return_value='fallback_ok') as mock_or, \
         _patch_no_sleep():
        result = get_llm_response('p', mode='light')
        assert result == 'fallback_ok'
        assert mock_g.call_count == 1
        assert mock_or.call_count == 1


def test_bad_request_does_not_retry_or_fall_back():
    """A 400/bad-request on the primary indicates our payload is wrong; no retry, no fallback."""
    with patch('newscaster.llm.router.gemini', side_effect=LLMBadRequestError("malformed")) as mock_g, \
         patch('newscaster.llm.router.get_openrouter_response') as mock_or, \
         _patch_no_sleep():
        with pytest.raises(LLMBadRequestError):
            get_llm_response('p', mode='light')
        assert mock_g.call_count == 1
        mock_or.assert_not_called()


def test_both_primary_and_fallback_exhausted_raises_exhausted():
    with patch('newscaster.llm.router.gemini', side_effect=LLMTimeoutError("p-fail")), \
         patch('newscaster.llm.router.get_openrouter_response', side_effect=LLMTimeoutError("or-fail")), \
         _patch_no_sleep():
        with pytest.raises(LLMRetriesExhaustedError):
            get_llm_response('p', mode='light')


def test_backoff_uses_retry_after_for_rate_limit():
    e = LLMRateLimitError("slow down", retry_after=42.0)
    delay = _backoff_delay(0, e)
    assert delay == 42.0


def test_backoff_caps_retry_after_at_120():
    e = LLMRateLimitError("slow down", retry_after=600.0)
    assert _backoff_delay(0, e) == 120.0


def test_backoff_grows_with_attempt():
    """Without retry_after, backoff increases monotonically up to a cap."""
    e = LLMTimeoutError("nope")
    delays = [_backoff_delay(i, e) for i in range(5)]
    # Each delay is base + jitter [0,1]; bases are 2,4,8,16,32 capped at 60
    assert delays[0] < delays[2] < delays[4]
    for d in delays:
        assert d <= 61.0  # cap is 60 + 1 jitter
