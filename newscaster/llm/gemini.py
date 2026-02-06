import time

from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

import newscaster.config as _config
from newscaster.logging import print_and_write


def gemini(user_prompt, system_prompt='You are an intelligent assistant.', model="gemini-2.5-flash-",
           grounding=False, url_context=False, thinking_budget=8000):
    completed = False
    i = 0
    tools = []
    if grounding:
        tools.append(Tool(google_search=types.GoogleSearch))
    if url_context:
        tools.append(Tool(url_context=types.UrlContext))

    client = genai.Client(api_key=_config.GOOGLE_GENAI_API_KEY)

    while completed == False:
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
            completed = True
            if response.text == None:
                print_and_write('gemini returned None')
                print_and_write('system:', system_prompt)
                print_and_write('user', user_prompt)
                print_and_write('grounding', grounding)
                print_and_write('url context', url_context)
                i += 1
                time.sleep(30 * i)
                if i == 5:
                    return 'no response'

        except Exception as e:
            if i == 5:
                print_and_write(f'Gemini failed after 5 retries: {e}')
                print_and_write('system:', system_prompt)
                print_and_write('user', user_prompt)
                print_and_write('grounding', grounding)
                print_and_write('url context', url_context)
                return 'None'
            i += 1
            time.sleep(30 * i)
            print_and_write('failure, wating', str(30 * i), 'seconds')
    response_text = response.text
    if response_text == None:
        response_text = 'broken'

    citations_list = []
    if grounding and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            if hasattr(candidate.grounding_metadata, 'grounding_chunks') and candidate.grounding_metadata.grounding_chunks:
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                        citations_list.append(chunk.web.title if hasattr(chunk.web, 'title') else "N/A")
            elif hasattr(candidate.grounding_metadata, 'search_entry_point') and candidate.grounding_metadata.search_entry_point:
                pass

    if len(citations_list) > 0:
        response_text = response_text + '\nSources:\n'
        for citation in citations_list:
            response_text = response_text + citation + '\n'
    return response_text
