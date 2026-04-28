import anthropic
import httpx

import newscaster.config as _config
from newscaster.logging import print_and_write


def claude(user_prompt, model_to_use="claude-sonnet-4-20250514", system_prompt='You are an intelligent assistant.'):
    client = anthropic.Anthropic(api_key=_config.ANTHROPIC_API_KEY, timeout=httpx.Timeout(300.0, connect=5.0))
    max_output_tokens = 16000
    thinking = True
    try:
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
    except Exception as e:
        print_and_write('claude failure', e)
        print_and_write('falling back to GPT-5.5 (low)')
        try:
            # Lazy import to avoid circular dependency
            from newscaster.llm.openrouter import get_openrouter_response
            return get_openrouter_response(
                user_prompt,
                'openai/gpt-5.5',
                'GPT-5.5 (low)',
                False,
                system_prompt=system_prompt
            )
        except Exception as fallback_error:
            print_and_write('GPT-5.5 fallback failure', fallback_error)
            raise

    print_and_write('claude message received')
    for block in message.content:
        if block.type == 'text':
            return block.text
    raise RuntimeError(f'claude response had no text block: {[b.type for b in message.content]}')
