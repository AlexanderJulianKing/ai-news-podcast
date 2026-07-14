"""Router: mode → provider dispatch + central retry policy + cross-provider fallback.

Provider adapters (gemini, claude, openrouter) make exactly one API call and
raise typed LLMError on failure. The router decides whether to retry, fall
back to OpenRouter (with web_search / web_fetch tools as applicable), or
re-raise. Callers never see sentinel strings — every result is either a
non-empty string or a raised LLMError.
"""

import random
import time
import uuid

import newscaster.config as _config
from newscaster.logging import print_and_write, write_jsonl_log
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


def _audit_llm_event(event, spec, user_prompt, system_prompt, *, call_id, phase,
                     attempt=None, latency_seconds=None, response=None, error=None,
                     usage=None):
    if not getattr(_config, "LLM_AUDIT_LOG_ENABLED", False):
        return
    payload = {
        "event": event,
        "call_id": call_id,
        "phase": phase,
        "attempt": attempt,
        "latency_seconds": round(latency_seconds, 3) if latency_seconds is not None else None,
        "provider": spec.get("provider"),
        "model": spec.get("model"),
        "name": spec.get("name"),
        "grounding": bool(spec.get("grounding", False)),
        "url_context": bool(spec.get("url_context", False)),
        "tools": spec.get("tools"),
    }
    if getattr(_config, "LLM_AUDIT_LOG_PROMPTS", False):
        payload["system_prompt"] = system_prompt
        payload["user_prompt"] = user_prompt
    if response is not None and getattr(_config, "LLM_AUDIT_LOG_RESPONSES", False):
        payload["response"] = response
    if usage is not None:
        payload["usage"] = usage
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "total_input_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "cost",
        ):
            if key in usage:
                payload[key] = usage[key]
    if error is not None:
        payload["error"] = str(error)
        payload["error_type"] = type(error).__name__
    write_jsonl_log("llm_audit", payload)


def _select_primary(mode, grounding, url_context):
    """Translate mode + flags into a primary-provider spec."""
    needs_tools = grounding or url_context

    if mode == 'heavy' and not needs_tools:
        return {'provider': 'anthropic', 'model': _config.HEAVY_MODEL}

    if mode == 'standard' and not needs_tools:
        return {
            'provider': 'openrouter',
            'model': _config.STANDARD_MODEL,
            'name': 'Gemma 4 31B',
            'reasoning': False,
        }

    if mode == 'advanced' and not needs_tools:
        return {
            'provider': 'openrouter',
            'model': _config.ADVANCED_MODEL,
            'name': 'GLM 5.2 Medium',
            'reasoning': _config.ADVANCED_REASONING_EFFORT,
        }

    if mode == 'adversary' and not needs_tools:
        return {
            'provider': 'openrouter',
            'model': _config.ADVERSARY_MODEL,
            'name': 'GPT-5.5 Adversary',
            'reasoning': _config.ADVERSARY_REASONING_EFFORT,
        }

    # Native grounding/url_context remains on Gemini because those are provider tools.
    if mode == 'light' and not needs_tools:
        model = _config.LIGHT_MODEL
    elif mode in ('light', 'standard', 'advanced'):
        model = _config.TOOL_LIGHT_STANDARD_MODEL
    elif mode in ('plus', 'heavy'):
        model = _config.TOOL_PLUS_HEAVY_MODEL
    else:
        model = _config.LIGHT_MODEL

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
        return claude(user_prompt, model, system_prompt, include_usage=True)

    if provider == 'openrouter':
        return get_openrouter_response(
            user_prompt,
            model,
            spec.get('name', model),
            spec.get('reasoning', False),
            system_prompt=system_prompt,
            include_usage=True,
            tools=spec.get('tools'),
        )

    raise ValueError(f"Unknown provider: {provider}")


def _backoff_delay(attempt, exc):
    """Exponential backoff with jitter; honors Retry-After on rate limits (cap 120s)."""
    if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
        return min(exc.retry_after, 120.0)
    return min(2 * (2 ** attempt), 60) + random.uniform(0, 1)


def _call_with_retry(spec, user_prompt, system_prompt, *, call_id=None, phase="primary"):
    """Retry the spec up to MAX_RETRIES on retryable errors. Re-raise non-retryable.

    On exhaustion raises LLMRetriesExhaustedError with the last cause.
    """
    last_exc = None
    label = f"{spec['provider']}/{spec['model']}"
    call_id = call_id or str(uuid.uuid4())

    for attempt in range(_config.MAX_RETRIES):
        started = time.perf_counter()
        try:
            dispatch_result = _dispatch(spec, user_prompt, system_prompt)
            usage = None
            if isinstance(dispatch_result, tuple) and len(dispatch_result) == 2:
                response, usage = dispatch_result
            else:
                response = dispatch_result
            _audit_llm_event(
                "success",
                spec,
                user_prompt,
                system_prompt,
                call_id=call_id,
                phase=phase,
                attempt=attempt + 1,
                latency_seconds=time.perf_counter() - started,
                response=response,
                usage=usage,
            )
            return response
        except LLMNonRetryableError as e:
            _audit_llm_event(
                "non_retryable_error",
                spec,
                user_prompt,
                system_prompt,
                call_id=call_id,
                phase=phase,
                attempt=attempt + 1,
                latency_seconds=time.perf_counter() - started,
                error=e,
            )
            raise
        except LLMRetryableError as e:
            last_exc = e
            _audit_llm_event(
                "retryable_error",
                spec,
                user_prompt,
                system_prompt,
                call_id=call_id,
                phase=phase,
                attempt=attempt + 1,
                latency_seconds=time.perf_counter() - started,
                error=e,
            )
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
    _audit_llm_event(
        "retries_exhausted",
        spec,
        user_prompt,
        system_prompt,
        call_id=call_id,
        phase=phase,
        attempt=_config.MAX_RETRIES,
        error=exhausted,
    )
    raise exhausted from last_exc


def _call_fallback(user_prompt, system_prompt, grounding, url_context, *, call_id=None):
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
    return _call_with_retry(fallback, user_prompt, system_prompt, call_id=call_id, phase="fallback")


def get_llm_response(user_prompt, system_prompt='You are an intelligent assistant.',
                     mode="light", grounding=False, url_context=False):
    primary = _select_primary(mode, grounding, url_context)
    call_id = str(uuid.uuid4())

    try:
        return _call_with_retry(primary, user_prompt, system_prompt, call_id=call_id, phase="primary")
    except (LLMRetriesExhaustedError, LLMAuthError) as e:
        print_and_write(
            f"LLM-FALLBACK [{primary['provider']}/{primary['model']} → openrouter/{_config.FALLBACK_MODEL}]: {e}"
        )
        try:
            return _call_fallback(user_prompt, system_prompt, grounding, url_context, call_id=call_id)
        except LLMError as fallback_exc:
            print_and_write(
                f"LLM-FALLBACK-FAILED [openrouter/{_config.FALLBACK_MODEL}]: {fallback_exc}"
            )
            raise
