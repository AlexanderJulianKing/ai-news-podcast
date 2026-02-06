"""Tests for newscaster.llm.router mode-based LLM routing."""
from unittest.mock import patch, MagicMock
import pytest

from newscaster.llm.router import get_llm_response


# Table of (mode, grounding, url_context) -> expected provider function
ROUTING_CASES = [
    # (mode, grounding, url_context, expected_provider)
    ('light', False, False, 'openrouter'),
    ('light', True, False, 'google'),
    ('light', False, True, 'google'),
    ('standard', True, False, 'google'),
    ('standard', False, True, 'google'),
    ('standard', False, False, 'openrouter'),
    ('heavy', True, False, 'google'),
    ('heavy', False, True, 'google'),
    ('heavy', False, False, 'google'),
]


@pytest.mark.parametrize("mode,grounding,url_context,expected_provider", ROUTING_CASES)
def test_routing(mode, grounding, url_context, expected_provider):
    """Each (mode, grounding, url_context) combo routes to the expected provider."""
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
            mock_gemini.assert_not_called()
            mock_openrouter.assert_not_called()
            assert result == 'claude_response'
        elif expected_provider == 'openrouter':
            mock_openrouter.assert_called_once()
            mock_gemini.assert_not_called()
            mock_claude.assert_not_called()
            assert result == 'openrouter_response'
