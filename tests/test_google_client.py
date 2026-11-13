"""Gemini goes through Vertex AI on the Cloud project (billed with Chirp), not the AI Studio key (2026-11-12)."""
import os
from unittest.mock import patch

import newscaster.config as _config
from newscaster.llm import google_client


def test_vertex_when_configured(monkeypatch):
    monkeypatch.setattr(_config, "GEMINI_VIA_VERTEX", True)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    kw = google_client.client_kwargs()
    assert kw == {"vertexai": True, "project": _config.VERTEX_PROJECT, "location": _config.VERTEX_LOCATION}
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == _config.GOOGLE_CLOUD_CREDENTIALS


def test_api_key_when_vertex_is_off(monkeypatch):
    monkeypatch.setattr(_config, "GEMINI_VIA_VERTEX", False)
    monkeypatch.setattr(_config, "GOOGLE_GENAI_API_KEY", "k")
    assert google_client.client_kwargs() == {"api_key": "k"}


def test_every_gemini_caller_uses_it(monkeypatch):
    monkeypatch.setattr(_config, "GEMINI_VIA_VERTEX", True)
    from newscaster.rag import embeddings
    with patch("newscaster.rag.embeddings.genai.Client") as ctor:
        try:
            embeddings.embed_texts(["a"])
        except Exception:
            pass
    assert ctor.call_args.kwargs.get("vertexai") is True
