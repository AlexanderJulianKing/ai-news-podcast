"""Gemini embeddings adapter.

Mirrors the error discipline of newscaster/llm/gemini.py: SDK exceptions are
classified into the typed LLMError hierarchy so callers can decide fallback.
"""
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import newscaster.config as _config
from newscaster.llm.errors import LLMMalformedResponseError, classify


def embed_texts(texts, *, task_type="RETRIEVAL_DOCUMENT", model=None, dimension=None):
    """Embed a list of strings. Returns list[list[float]] parallel to `texts`.

    Empty input returns [] without an API call. Raises a typed LLMError on failure.
    """
    if not texts:
        return []
    model = model or _config.EMBED_MODEL
    dimension = dimension or _config.EMBED_DIM
    try:
        client = genai.Client(api_key=_config.GOOGLE_GENAI_API_KEY)
        response = client.models.embed_content(
            model=model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                output_dimensionality=dimension,
                task_type=task_type,
            ),
        )
    except genai_errors.APIError as e:
        status_code = getattr(e, "code", None)
        cls = classify(e, status_code=status_code)
        raise cls(str(e), provider="google", model=model, status_code=status_code) from e
    except Exception as e:
        cls = classify(e)
        raise cls(str(e), provider="google", model=model) from e

    out = getattr(response, "embeddings", None)
    if not out:
        raise LLMMalformedResponseError(
            "embed_content returned no embeddings", provider="google", model=model
        )
    return [list(e.values) for e in out]
