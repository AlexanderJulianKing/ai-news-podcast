from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write
from newscaster.prompts import TITLE_PROMPT, INTRO_PROMPT_TEMPLATE


def intro_writer(formatted_date, weather_string, topics, formatted_date2, stories):
    headlines = []
    episode_title = ''
    for story in stories:
        title = get_llm_response(story, system_prompt=TITLE_PROMPT, mode="standard")

        while len(title) > 80:
            title = get_llm_response(story, system_prompt=TITLE_PROMPT, mode="standard")

        episode_title = episode_title + title + ', '
        headlines.append(title)
    episode_title = episode_title[:-2]
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

    headline_string = ''
    for headline in headlines:
        headline_string = headline_string + headline + '\n'

    intro2 = get_llm_response(headline_string, system_prompt=intro_prompt)
    return intro1, intro2
