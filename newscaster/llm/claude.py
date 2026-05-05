import anthropic
import httpx

import newscaster.config as _config
from newscaster.llm.errors import (
    LLMMalformedResponseError,
    classify,
)


def claude(user_prompt, model_to_use="claude-sonnet-4-20250514", system_prompt='You are an intelligent assistant.'):
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
            return text

    raise LLMMalformedResponseError(
        f'Claude response had no text block: {[b.type for b in message.content]}',
        provider='anthropic', model=model_to_use,
    )
