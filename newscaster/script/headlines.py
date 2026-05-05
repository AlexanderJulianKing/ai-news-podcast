from newscaster.llm import get_llm_response
from newscaster.logging import print_and_write
from newscaster.prompts import HEADLINE_MAKER_PROMPT


def story_gatherer(formatted_date2):
    """Load segment summaries from disk, keyed by slot index. Missing slots
    (failed during gather) are simply absent from the returned dict — downstream
    stages see a sparse mapping and skip those slots."""
    stories = {}
    for i in range(4):
        try:
            with open('segment_summaries/{}_segment{}_summary.txt'.format(formatted_date2, i), encoding='utf-8') as storyfile:
                stories[i] = storyfile.read()
        except FileNotFoundError:
            print_and_write(f'Slot {i} has no summary on disk; skipping')
        except Exception as e:
            print_and_write(f'Missing segment summary {i}: {e}')
    return stories


def headline_maker(stories):
    """stories: dict[int, str] (slot → summary). Returns dict[int, str] (slot → headline)."""
    headlines = {}
    for slot, story in stories.items():
        headline = get_llm_response(story, system_prompt=HEADLINE_MAKER_PROMPT)
        headlines[slot] = headline.strip("\"")
    return headlines
