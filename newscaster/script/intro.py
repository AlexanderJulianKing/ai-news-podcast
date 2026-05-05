from newscaster.llm import get_llm_response, call_with_default
from newscaster.logging import print_and_write
from newscaster.prompts import TITLE_PROMPT, INTRO_PROMPT_TEMPLATE


def intro_writer(formatted_date, weather_string, topics, formatted_date2, stories):
    """topics and stories are dicts keyed by slot. Failed slots are absent."""
    headlines_by_slot = {}
    episode_title_parts = []
    for slot, story in stories.items():
        title = get_llm_response(story, system_prompt=TITLE_PROMPT, mode="standard")

        retry_count = 0
        while len(title) > 80 and retry_count < 5:
            title = get_llm_response(story, system_prompt=TITLE_PROMPT, mode="standard")
            retry_count += 1

        episode_title_parts.append(title)
        headlines_by_slot[slot] = title

    episode_title = ', '.join(episode_title_parts)
    print_and_write('episode_title', episode_title)
    outfile = open('episode_titles/{}.txt'.format(formatted_date2), 'w', encoding='utf-8')
    outfile.write(episode_title)
    outfile.close()

    intro1 = "Good Morning Alex and friends. I'm Grace. Today is {}, and you're listening to Alex's News.".format(formatted_date)
    intro_prompt = INTRO_PROMPT_TEMPLATE.format(
        intro1=intro1,
        date=formatted_date,
        weather=weather_string
    )

    headline_string = '\n'.join(headlines_by_slot[slot] for slot in sorted(headlines_by_slot)) + '\n'

    intro2 = call_with_default(
        f"On the program today, we have several stories. Stay with us. {headline_string}",
        headline_string,
        system_prompt=intro_prompt,
        _log_label='intro2-narration',
    )
    return intro1, intro2
