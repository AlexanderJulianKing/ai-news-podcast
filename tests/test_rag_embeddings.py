"""Tests for the Gemini embeddings adapter (client mocked)."""
from unittest.mock import patch, MagicMock
import pytest

from newscaster.rag import embeddings
from newscaster.llm.errors import LLMError


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


def _fake_response(vectors):
    resp = MagicMock()
    resp.embeddings = [_FakeEmbedding(v) for v in vectors]
    return resp


def test_embed_texts_returns_vectors():
    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = _fake_response([[0.1, 0.2], [0.3, 0.4]])
    with patch("newscaster.rag.embeddings.genai.Client", return_value=fake_client):
        out = embeddings.embed_texts(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    _, kwargs = fake_client.models.embed_content.call_args
    assert kwargs["contents"] == ["a", "b"]


def test_embed_texts_empty_input_skips_api():
    with patch("newscaster.rag.embeddings.genai.Client") as ctor:
        assert embeddings.embed_texts([]) == []
    ctor.assert_not_called()


def test_embed_texts_maps_api_error_to_llmerror():
    fake_client = MagicMock()
    fake_client.models.embed_content.side_effect = RuntimeError("boom")
    with patch("newscaster.rag.embeddings.genai.Client", return_value=fake_client):
        with pytest.raises(LLMError):
            embeddings.embed_texts(["a"])
