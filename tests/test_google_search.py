"""Tests for newscaster.scrapers.google_search."""
from unittest.mock import patch, MagicMock
import pytest

from newscaster.scrapers.google_search import google_official_search


def _mock_cse_response(items):
    """Build a mock Google CSE service that returns the given items."""
    mock_service = MagicMock()
    mock_service.cse().list().execute.return_value = {"items": items}
    return mock_service


@patch('newscaster.scrapers.google_search.GOOGLE_CSE_ID', 'fake_cse_id')
@patch('newscaster.scrapers.google_search.GOOGLE_SEARCH_API_KEY', 'fake_key')
class TestGoogleOfficialSearch:

    @patch('newscaster.scrapers.google_search.get_llm_response', return_value='rephrased query')
    def test_zero_results_calls_llm_to_rephrase(self, mock_llm):
        """When search returns zero results, should call get_llm_response to rephrase."""
        with patch('newscaster.scrapers.google_search.time.sleep'):
            mock_service = MagicMock()
            # Always return empty results
            mock_service.cse().list().execute.return_value = {"items": []}

            with patch('googleapiclient.discovery.build', return_value=mock_service):
                result = google_official_search('test query', num_results=3)

            # LLM should have been called to rephrase (up to 4 times since 5 iterations)
            assert mock_llm.call_count >= 1
            # With no results ever found, should return empty list
            assert result == []

    def test_successful_search_returns_list_of_dicts(self):
        """Successful search should return a list of dicts with headline/url/snippet."""
        items = [
            {"title": "Test Headline", "link": "https://example.com", "snippet": "A snippet"},
            {"title": "Another", "link": "https://example2.com", "snippet": "Another snippet"},
        ]
        mock_service = _mock_cse_response(items)

        with patch('googleapiclient.discovery.build', return_value=mock_service):
            result = google_official_search('test query', num_results=2)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['headline'] == "Test Headline"
        assert result[0]['url'] == "https://example.com"
        assert result[1]['headline'] == "Another"

    def test_http_error_raises_runtime_error(self):
        """HttpError should be re-raised as RuntimeError."""
        from googleapiclient.errors import HttpError
        import json

        mock_service = MagicMock()
        error_content = json.dumps({"error": {"code": 403, "message": "invalid API key"}}).encode()
        mock_service.cse().list().execute.side_effect = HttpError(
            resp=MagicMock(status=403), content=error_content
        )

        with patch('googleapiclient.discovery.build', return_value=mock_service):
            with pytest.raises(RuntimeError, match="Google search HTTP error"):
                google_official_search('test query')
