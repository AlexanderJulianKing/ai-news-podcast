"""Tests for newscaster.llm.router mode-based LLM routing."""
from unittest.mock import patch
import pytest

from newscaster.llm.router import get_llm_response


# Table of (mode, grounding, url_context) -> expected primary provider
ROUTING_CASES = [
    # (mode, grounding, url_context, expected_provider)
    ('light', False, False, 'google'),
    ('light', True, False, 'google'),
    ('light', False, True, 'google'),
    ('standard', True, False, 'google'),
    ('standard', False, True, 'google'),
    ('standard', False, False, 'openrouter'),
    ('advanced', False, False, 'openrouter'),
    ('advanced', True, False, 'google'),
    ('advanced', False, True, 'google'),
    ('adversary', False, False, 'openrouter'),
    ('adversary', True, False, 'google'),
    ('adversary', False, True, 'google'),
    ('plus', False, False, 'google'),
    ('plus', True, False, 'google'),
    ('heavy', True, False, 'google'),
    ('heavy', False, True, 'google'),
    ('heavy', False, False, 'anthropic'),
]


@pytest.mark.parametrize("mode,grounding,url_context,expected_provider", ROUTING_CASES)
def test_routing(mode, grounding, url_context, expected_provider):
    """Each (mode, grounding, url_context) combo dispatches to the expected primary provider."""
    with patch('newscaster.llm.router.gemini', return_value='gemini_response') as mock_gemini, \
         patch('newscaster.llm.router.claude', return_value='claude_response') as mock_claude, \
         patch('newscaster.llm.router.get_openrouter_response', return_value='openrouter_response') as mock_openrouter:

        result = get_llm_response('test prompt', mode=mode, grounding=grounding, url_context=url_context)

        if expected_provider == 'google':
            mock_gemini.assert_called_once()
            mock_claude.assert_not_called()
            mock_openrouter.assert_not_called()
            assert result == 'gemini_response'
        elif expected_provider == 'anthropic':
            mock_claude.assert_called_once()
            assert mock_claude.call_args[0][1] == 'claude-opus-4-8'
            mock_gemini.assert_not_called()
            mock_openrouter.assert_not_called()
            assert result == 'claude_response'
        elif expected_provider == 'openrouter':
            mock_openrouter.assert_called_once()
            mock_gemini.assert_not_called()
            mock_claude.assert_not_called()
            assert result == 'openrouter_response'


def test_standard_routes_to_gemma():
    with patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_openrouter:
        assert get_llm_response('test prompt', mode='standard') == 'ok'
    assert mock_openrouter.call_args.args[1] == 'google/gemma-4-31b-it'
    assert mock_openrouter.call_args.args[2] == 'Gemma 4 31B'
    assert mock_openrouter.call_args.args[3] is False


def test_advanced_routes_to_glm_medium():
    with patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_openrouter:
        assert get_llm_response('test prompt', mode='advanced') == 'ok'
    assert mock_openrouter.call_args.args[1] == 'z-ai/glm-5.2'
    assert mock_openrouter.call_args.args[2] == 'GLM 5.2 Medium'
    assert mock_openrouter.call_args.args[3] == 'medium'


def test_adversary_routes_to_gpt55_high_reasoning():
    with patch('newscaster.llm.router.get_openrouter_response', return_value='ok') as mock_openrouter:
        assert get_llm_response('test prompt', mode='adversary') == 'ok'
    assert mock_openrouter.call_args.args[1] == 'openai/gpt-5.5'
    assert mock_openrouter.call_args.args[2] == 'GPT-5.5 Adversary'
    assert mock_openrouter.call_args.args[3] == 'high'
