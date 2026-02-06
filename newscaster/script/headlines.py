from newscaster.llm import get_llm_response
from newscaster.prompts import HEADLINE_MAKER_PROMPT


def story_gatherer(formatted_date2):
    stories = []
    for i in range(4):
        try:
            with open('segment_summaries/{}_segment{}_summary.txt'.format(formatted_date2, i), encoding='utf-8') as storyfile:
                stories.append(storyfile.read())
        except:
            pass
    return stories


def headline_maker(stories):
    headlines = []
    for story in stories:
        headline = get_llm_response(story, system_prompt=HEADLINE_MAKER_PROMPT)
        headline = headline.strip("\"")
        headlines.append(headline)
    return headlines
