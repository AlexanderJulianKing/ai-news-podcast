from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.genai.types import Tool, GenerateContentConfig

import newscaster.config as _config
from newscaster.llm.errors import (
    LLMMalformedResponseError,
    LLMTransportError,
    classify,
)


def gemini(user_prompt, system_prompt='You are an intelligent assistant.', model="gemini-2.5-flash-",
           grounding=False, url_context=False, thinking_budget=8000):
    """One logical attempt against the Google Gen AI API.

    Raises a typed LLMError on failure; the router decides whether to retry or fall back.
    """
    tools = []
    if grounding:
        tools.append(Tool(google_search=types.GoogleSearch))
    if url_context:
        tools.append(Tool(url_context=types.UrlContext))

    try:
        client = genai.Client(api_key=_config.GOOGLE_GENAI_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=GenerateContentConfig(
                tools=tools,
                response_modalities=["TEXT"],
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                )
            )
        )
    except genai_errors.APIError as e:
        status_code = getattr(e, 'code', None)
        cls = classify(e, status_code=status_code)
        raise cls(str(e), provider='google', model=model, status_code=status_code) from e
    except Exception as e:
        cls = classify(e)
        if cls is LLMTransportError:
            raise LLMTransportError(str(e), provider='google', model=model) from e
        raise cls(str(e), provider='google', model=model) from e

    if response.text is None:
        raise LLMMalformedResponseError(
            'Gemini returned response with text=None (often safety/recitation block or empty completion)',
            provider='google', model=model,
        )

    response_text = response.text

    if not response_text.strip():
        raise LLMMalformedResponseError(
            'Gemini returned empty/whitespace text',
            provider='google', model=model,
        )

    citations_list = []
    if grounding and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            if hasattr(candidate.grounding_metadata, 'grounding_chunks') and candidate.grounding_metadata.grounding_chunks:
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                        citations_list.append(chunk.web.title if hasattr(chunk.web, 'title') else "N/A")

    if len(citations_list) > 0:
        response_text = response_text + '\nSources:\n'
        for citation in citations_list:
            response_text = response_text + citation + '\n'
    return response_text
