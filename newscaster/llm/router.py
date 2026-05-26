"""Router: mode → provider dispatch + central retry policy + cross-provider fallback.

Provider adapters (gemini, claude, openrouter) make exactly one API call and
raise typed LLMError on failure. The router decides whether to retry, fall
back to OpenRouter (with web_search / web_fetch tools as applicable), or
re-raise. Callers never see sentinel strings — every result is either a
non-empty string or a raised LLMError.
"""

import random
import time

import newscaster.config as _config
from newscaster.logging import print_and_write
from newscaster.llm.gemini import gemini
from newscaster.llm.claude import claude
from newscaster.llm.openrouter import get_openrouter_response
from newscaster.llm.errors import (
    LLMError,
    LLMRetryableError,
    LLMNonRetryableError,
    LLMAuthError,
    LLMRateLimitError,
    LLMRetriesExhaustedError,
)


def _select_primary(mode, grounding, url_context):
    """Translate mode + flags into a primary-provider spec."""
    needs_tools = grounding or url_context

    if mode == 'heavy' and not needs_tools:
        return {'provider': 'anthropic', 'model': 'claude-opus-4-7'}

    # Everything else goes to Google Gemini.
    if mode == 'light' and not needs_tools:
        model = 'gemini-3.1-flash-lite'
    elif mode in ('light', 'standard') or (mode == 'standard' and not needs_tools):
        model = 'gemini-3-flash-preview'
    elif mode in ('plus', 'heavy'):
        model = 'gemini-3.1-pro-preview'
    else:
        model = 'gemini-3.1-flash-lite'

    return {
        'provider': 'google',
        'model': model,
        'grounding': grounding,
        'url_context': url_context,
    }


def _dispatch(spec, user_prompt, system_prompt):
    """Single-shot call to one provider. Raises typed LLMError on failure."""
    provider = spec['provider']
    model = spec['model']

    if provider == 'google':
        return gemini(
            user_prompt,
            system_prompt,
            model,
            spec.get('grounding', False),
            spec.get('url_context', False),
        )

    if provider == 'anthropic':
        return claude(user_prompt, model, system_prompt)

    if provider == 'openrouter':
        return get_openrouter_response(
            user_prompt,
            model,
            spec.get('name', model),
            spec.get('reasoning', False),
            system_prompt=system_prompt,
            tools=spec.get('tools'),
        )

    raise ValueError(f"Unknown provider: {provider}")


def _backoff_delay(attempt, exc):
    """Exponential backoff with jitter; honors Retry-After on rate limits (cap 120s)."""
    if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
        return min(exc.retry_after, 120.0)
    return min(2 * (2 ** attempt), 60) + random.uniform(0, 1)


def _call_with_retry(spec, user_prompt, system_prompt):
    """Retry the spec up to MAX_RETRIES on retryable errors. Re-raise non-retryable.

    On exhaustion raises LLMRetriesExhaustedError with the last cause.
    """
    last_exc = None
    label = f"{spec['provider']}/{spec['model']}"

    for attempt in range(_config.MAX_RETRIES):
        try:
            return _dispatch(spec, user_prompt, system_prompt)
        except LLMNonRetryableError:
            raise
        except LLMRetryableError as e:
            last_exc = e
            if attempt == _config.MAX_RETRIES - 1:
                break
            delay = _backoff_delay(attempt, e)
            print_and_write(
                f"LLM-RETRY [{label}] attempt {attempt + 1}/{_config.MAX_RETRIES}: {e}; waiting {delay:.1f}s"
            )
            time.sleep(delay)

    exhausted = LLMRetriesExhaustedError(
        f"All {_config.MAX_RETRIES} attempts failed",
        provider=spec['provider'],
        model=spec['model'],
        attempts=_config.MAX_RETRIES,
    )
    raise exhausted from last_exc


def _call_fallback(user_prompt, system_prompt, grounding, url_context):
    """Try the OpenRouter fallback (with web tools attached when applicable)."""
    tools = []
    if grounding:
        tools.append({"type": "openrouter:web_search"})
    if url_context:
        tools.append({"type": "openrouter:web_fetch"})

    fallback = {
        'provider': 'openrouter',
        'model': _config.FALLBACK_MODEL,
        'name': 'GPT-5.5 (low)',
        'reasoning': False,
        'tools': tools if tools else None,
    }
    return _call_with_retry(fallback, user_prompt, system_prompt)


def get_llm_response(user_prompt, system_prompt='You are an intelligent assistant.',
                     mode="light", grounding=False, url_context=False):
    primary = _select_primary(mode, grounding, url_context)

    try:
        return _call_with_retry(primary, user_prompt, system_prompt)
    except (LLMRetriesExhaustedError, LLMAuthError) as e:
        print_and_write(
            f"LLM-FALLBACK [{primary['provider']}/{primary['model']} → openrouter/{_config.FALLBACK_MODEL}]: {e}"
        )
        try:
            return _call_fallback(user_prompt, system_prompt, grounding, url_context)
        except LLMError as fallback_exc:
            print_and_write(
                f"LLM-FALLBACK-FAILED [openrouter/{_config.FALLBACK_MODEL}]: {fallback_exc}"
            )
            raise
