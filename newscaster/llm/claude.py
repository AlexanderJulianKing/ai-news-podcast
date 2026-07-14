import anthropic
import httpx

import newscaster.config as _config
from newscaster.llm.errors import (
    LLMMalformedResponseError,
    classify,
)


_ANTHROPIC_PRICING_PER_MTOK = {
    "claude-opus-4-8": {
        "input": 5.00,
        "output": 25.00,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
        "cache_read": 0.50,
    },
}


def _usage_to_dict(usage, model_to_use):
    """Normalize Anthropic usage into JSON-safe audit fields."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = dict(usage)
    else:
        raw = {
            name: getattr(usage, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "cache_creation",
            )
            if hasattr(usage, name)
        }

    cache_creation = raw.get("cache_creation") or {}
    if hasattr(cache_creation, "model_dump"):
        cache_creation = cache_creation.model_dump()
    if not isinstance(cache_creation, dict):
        cache_creation = {}

    input_tokens = raw.get("input_tokens") or 0
    output_tokens = raw.get("output_tokens") or 0
    cache_read_tokens = raw.get("cache_read_input_tokens") or 0
    cache_creation_tokens = raw.get("cache_creation_input_tokens") or 0
    cache_write_5m_tokens = cache_creation.get("ephemeral_5m_input_tokens")
    cache_write_1h_tokens = cache_creation.get("ephemeral_1h_input_tokens")

    # Older SDKs may only expose the aggregate cache_creation_input_tokens.
    if cache_write_5m_tokens is None and cache_write_1h_tokens is None:
        cache_write_5m_tokens = cache_creation_tokens
        cache_write_1h_tokens = 0

    total_input_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
    pricing = _ANTHROPIC_PRICING_PER_MTOK.get(model_to_use)
    estimated_cost = None
    if pricing:
        estimated_cost = (
            (input_tokens * pricing["input"])
            + (output_tokens * pricing["output"])
            + ((cache_write_5m_tokens or 0) * pricing["cache_write_5m"])
            + ((cache_write_1h_tokens or 0) * pricing["cache_write_1h"])
            + (cache_read_tokens * pricing["cache_read"])
        ) / 1_000_000

    return {
        "provider": "anthropic",
        "model": model_to_use,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation": cache_creation,
        "total_input_tokens": total_input_tokens,
        "total_tokens": total_input_tokens + output_tokens,
        "estimated_cost_usd": round(estimated_cost, 8) if estimated_cost is not None else None,
        "raw": raw,
    }


def claude(user_prompt, model_to_use="claude-sonnet-4-20250514", system_prompt='You are an intelligent assistant.',
           include_usage=False):
    """One logical attempt against the Anthropic API.

    Raises a typed LLMError on failure; the router decides whether to retry or fall back.
    """
    max_output_tokens = 16000
    thinking = True

    try:
        client = anthropic.Anthropic(api_key=_config.ANTHROPIC_API_KEY, timeout=httpx.Timeout(300.0, connect=5.0))
        if thinking == False:
            message = client.messages.create(
                model=model_to_use,
                max_tokens=max_output_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            )
        else:
            message = client.messages.create(
                model=model_to_use,
                max_tokens=max_output_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                thinking={"type": "adaptive"},
                extra_body={"output_config": {"effort": "high"}},
            )
    except anthropic.APIStatusError as e:
        status_code = getattr(e, 'status_code', None)
        cls = classify(e, status_code=status_code)
        raise cls(str(e), provider='anthropic', model=model_to_use, status_code=status_code) from e
    except Exception as e:
        cls = classify(e)
        raise cls(str(e), provider='anthropic', model=model_to_use) from e

    for block in message.content:
        if block.type == 'text':
            text = block.text
            if not text or not text.strip():
                raise LLMMalformedResponseError(
                    'Claude returned empty/whitespace text block',
                    provider='anthropic', model=model_to_use,
                )
            if include_usage:
                return text, _usage_to_dict(getattr(message, "usage", None), model_to_use)
            return text

    raise LLMMalformedResponseError(
        f'Claude response had no text block: {[b.type for b in message.content]}',
        provider='anthropic', model=model_to_use,
    )
