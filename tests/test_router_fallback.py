"""Tests for cross-provider fallback (Gemini/Claude → OpenRouter with web tools)."""
from unittest.mock import patch

import pytest

import newscaster.config as _config
from newscaster.llm.router import get_llm_response
from newscaster.llm.errors import (
    LLMTimeoutError,
    LLMServerError,
    LLMRetriesExhaustedError,
    LLMMalformedResponseError,
)


def _patch_no_sleep():
    return patch('newscaster.llm.router.time.sleep', return_value=None)


def test_grounding_fallback_attaches_web_search_tool():
    """When a grounded Gemini call exhausts, the OpenRouter fallback gets openrouter:web_search."""
    with patch('newscaster.llm.router.gemini', side_effect=LLMTimeoutError("nope")), \
         patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_or, \
         _patch_no_sleep():
        result = get_llm_response('p', mode='light', grounding=True)
        assert result == 'ok'
        kwargs = mock_or.call_args.kwargs
        assert kwargs['tools'] == [{"type": "openrouter:web_search"}]
        assert mock_or.call_args.args[0] == 'p'
        assert mock_or.call_args.args[1] == _config.FALLBACK_MODEL


def test_url_context_fallback_attaches_web_fetch_tool():
    """When a url_context Gemini call exhausts, the fallback gets openrouter:web_fetch."""
    with patch('newscaster.llm.router.gemini', side_effect=LLMTimeoutError("nope")), \
         patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_or, \
         _patch_no_sleep():
        result = get_llm_response('p', mode='standard', url_context=True)
        assert result == 'ok'
        assert mock_or.call_args.kwargs['tools'] == [{"type": "openrouter:web_fetch"}]


def test_grounding_and_url_context_attaches_both_tools():
    with patch('newscaster.llm.router.gemini', side_effect=LLMTimeoutError("nope")), \
         patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_or, \
         _patch_no_sleep():
        get_llm_response('p', mode='light', grounding=True, url_context=True)
        tools = mock_or.call_args.kwargs['tools']
        assert {"type": "openrouter:web_search"} in tools
        assert {"type": "openrouter:web_fetch"} in tools


def test_no_tools_fallback_passes_no_tools():
    """Heavy mode with no grounding falls back without any tools attached."""
    with patch('newscaster.llm.router.claude', side_effect=LLMServerError("boom")), \
         patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_or, \
         _patch_no_sleep():
        get_llm_response('p', mode='heavy')
        assert mock_or.call_args.kwargs['tools'] is None


def test_heavy_falls_back_from_claude_to_openrouter():
    with patch('newscaster.llm.router.claude', side_effect=LLMServerError("anthropic down")), \
         patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_or, \
         patch('newscaster.llm.router.gemini') as mock_g, \
         _patch_no_sleep():
        result = get_llm_response('p', mode='heavy')
        assert result == 'ok'
        assert mock_or.call_count == 1
        mock_g.assert_not_called()


def test_malformed_response_is_treated_as_retryable():
    """Gemini text=None case: provider raises LLMMalformedResponseError, router retries."""
    side_effects = [LLMMalformedResponseError("text=None")] * 3 + ['ok']
    with patch('newscaster.llm.router.gemini', side_effect=side_effects) as mock_g, \
         _patch_no_sleep():
        assert get_llm_response('p', mode='light') == 'ok'
        assert mock_g.call_count == 4


def test_fallback_uses_configured_model():
    with patch('newscaster.llm.router.gemini', side_effect=LLMTimeoutError("nope")), \
         patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_or, \
         _patch_no_sleep():
        get_llm_response('p', mode='light')
        assert mock_or.call_args.args[1] == _config.FALLBACK_MODEL
