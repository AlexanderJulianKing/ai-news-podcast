"""Where Gemini calls go: Vertex AI on the Cloud project, or the AI Studio API key.

2026-11-12: AI Studio moved the API key to prepaid credit, which ran out on ~2026-10-26 and stopped
every Gemini call with 402s. Alex wants no separate prepaid balance, so Gemini now goes through
Vertex AI on the Cloud project that already bills Chirp (service account file, same as tts.py).
"""
import os

import newscaster.config as _config


def client_kwargs():
    """Arguments for genai.Client(...)."""
    if getattr(_config, "GEMINI_VIA_VERTEX", False):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", _config.GOOGLE_CLOUD_CREDENTIALS)
        return {"vertexai": True, "project": _config.VERTEX_PROJECT, "location": _config.VERTEX_LOCATION}
    return {"api_key": _config.GOOGLE_GENAI_API_KEY}
