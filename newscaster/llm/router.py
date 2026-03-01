from newscaster.llm.gemini import gemini
from newscaster.llm.claude import claude
from newscaster.llm.openrouter import get_openrouter_response


def get_llm_response(user_prompt, system_prompt='You are an intelligent assistant.',
                     mode="light", grounding=False, url_context=False):
    if mode == 'light' and grounding == False and url_context == False:
        model = 'x-ai/grok-4-fast'
        name = 'grok-4-fast'
        provider = 'openrouter'
    elif mode == 'light' and (grounding == True or url_context == True):
        model = 'gemini-3-flash-preview'
        provider = 'google'

    elif mode == 'standard' and (grounding == True or url_context == True):
        model = 'gemini-3-flash-preview'
        provider = 'google'
    elif mode == 'standard' and grounding == False and url_context == False:
        model = 'anthropic/claude-opus-4.6'
        name = 'Claude Opus 4.6'
        provider = 'openrouter'

    elif mode == 'heavy' and (grounding == True or url_context == True):
        model = 'gemini-3.1-pro-preview'
        provider = 'google'
    elif mode == 'heavy' and grounding == False and url_context == False:
        model = 'gemini-3.1-pro-preview'
        provider = 'google'
    else:
        model = 'x-ai/grok-4-fast'
        name = 'grok-4-fast'
        provider = 'openrouter'

    if provider == 'google':
        return gemini(user_prompt, system_prompt, model, grounding, url_context)

    elif provider == 'anthropic':
        return claude(user_prompt, model, system_prompt)

    elif provider == 'openrouter':
        model_response = get_openrouter_response(
            user_prompt, model, name, True, include_usage=False, system_prompt=system_prompt
        )
        return model_response
