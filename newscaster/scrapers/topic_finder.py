import json
from datetime import date

from newscaster.config import date_str
from newscaster.logging import print_and_write
from newscaster.text_utils import _grounded_response_needs_retry
from newscaster.prompts import (
    FOLLOW_UP_PROMPT_TEMPLATE,
    CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE,
    OVERVIEW_SYSTEM_PROMPT_TEMPLATE,
    MAIN_STORY_PROMPT,
    EVERYMAN_STORY_PROMPT,
    OVERVIEW_PICK_PROMPT,
    HEADLINE_EXTRACTION_PROMPT,
    OVERVIEW_ANCHOR_PROMPT,
    REPETITION_REMOVER_TEMPLATE,
)
from newscaster.llm import get_llm_response
from newscaster.dedup import load_recent_story_descriptions, summarize_story_for_archive
from newscaster.scrapers.calmatters import calmatters_scraper
from newscaster.scrapers.web import scrape_text


def headline_extractor(response):
    extraction_prompt = HEADLINE_EXTRACTION_PROMPT + response
    headline = get_llm_response(response, system_prompt=extraction_prompt)
    return headline


def determine_relevance(topic, result):
    prompt = "Given the search engine headline of '{}' and the search result snippet of '{}', do you think that the given website is a news article AND might be relevant to the topic of '{}'? . Give a yes or no answer.".format(
        result['headline'], result['snippet'], topic, date_str)

    response = get_llm_response(prompt, mode='light')

    if 'yes' in response.lower():
        return True
    else:
        return False


def summarize_text(text, article):
    text_length = len(text)
    print_and_write(f"Text length: {text_length} characters")
    prompt = 'Summarize this news article, but be sure to include where the article is from and as many important details as you can.\n\n'

    completion = get_llm_response(prompt + text)
    return completion


def result_piper(summary_prompt, successful_summary_counter, topic, result, i, formatted_date2):
    print_and_write('HEADLINE:', result['headline'], '\n')
    relevant = determine_relevance(topic, result)
    print_and_write(relevant)
    if relevant == True:

        print_and_write('ARTICLE SEEMS RELEVANT'), '\n'
        url = result['url']
        text = scrape_text(result['url'])
        if len(text) > 20000:
            print_and_write('ARTICLE IS UNREADABLE')
            return summary_prompt, successful_summary_counter
        summary = summarize_text(text, 'article')
        print_and_write('\nSUMMARY:', summary, '\n')
        relevance_prompt = "Given the topic of '" + topic + "', is there any relevant information in the summary of a news article below? Answer as a yes or no.\n" + summary
        completion = get_llm_response(relevance_prompt, mode='light')
        response = completion.replace('-', '')

        if 'yes' in response.lower():
            print_and_write('ARTICLE IS RELEVANT', '\n')
            news_source_prompt = 'What is the news outlet this url is associated with? Answer after writing "SOURCE:"\n' + url

            news_source_response = get_llm_response(news_source_prompt, mode='light')

            summary_prompt = summary_prompt + '\n'
            summary_prompt = summary_prompt + 'source:' + news_source_response + '\n'
            summary_prompt = summary_prompt + summary
            summary_prompt = summary_prompt + '\n'

            outfile = open('segment_summaries/{}_segment{}_article{}_summary.txt'.format(formatted_date2, i, successful_summary_counter), 'w', encoding='utf-8')
            outfile.write(summary)
            outfile.close()

            successful_summary_counter += 1

        else:
            print_and_write('ARTICLE IS NOT RELEVANT', '\n')
    return summary_prompt, successful_summary_counter


def summarize_headline_with_grounding(headline: str) -> str:
    headline_clean = (headline or '').strip()
    if not headline_clean:
        return 'UNVERIFIED: Empty headline received. Please provide a valid headline to summarize.'

    system_prompt = OVERVIEW_SYSTEM_PROMPT_TEMPLATE.format(date=date_str)
    base_prompt = (
        f"Headline: {headline_clean}\n"
        f"Date: {date_str}\n"
        "Instructions:\n"
        "- Use GoogleSearch to pull multiple reputable sources published today or within the past 48 hours.\n"
        "- Summarize the key facts in 3-5 sentences, attributing details to named outlets or officials.\n"
        "- Highlight why the development matters (policy impact, stakeholders, timeline).\n"
        "- End with 'Sources:' followed by one line per outlet (Outlet \u2014 brief descriptor).\n"
        "- If you still cannot verify after thorough searching, respond with 'UNVERIFIED:' plus the queries you tried.\n"
    )

    retry_prompt = (
        base_prompt +
        "\nYour first attempt did not return a verifiable summary. Expand your search to include official "
        "press releases, primary government domains, and credible national outlets. Do not say the "
        "headline did not happen; if sources conflict, explain the disagreement instead of denying the story."
    )

    for prompt in (base_prompt, retry_prompt):
        story = get_llm_response(prompt, system_prompt=system_prompt, grounding=True, mode='standard')
        if not _grounded_response_needs_retry(story):
            return story

    fallback_prompt = (
        base_prompt +
        "\nGrounded search attempts failed to verify the headline. Produce a brief note beginning with 'UNVERIFIED:' "
        "that lists the search queries you attempted and suggests what the editor should manually check next."
    )
    return get_llm_response(fallback_prompt, system_prompt=system_prompt, grounding=False, mode='standard')


def overview_process(overview):
    story_overviews = ''
    for i in range(5):
        number = str(i + 1)
        headline_finder_prompt = f'Find story number {number}. Only give the headline of that story.'

        headline_n = get_llm_response(overview, system_prompt=headline_finder_prompt, mode='light')

        story_finder_prompt = "Tell me more about the story behind this headline from today's paper. Include as many details as possible :\n" + headline_n
        try:
            print_and_write(story_finder_prompt)
            story = summarize_headline_with_grounding(headline_n)
            print_and_write(story)
            story_overviews = story_overviews + '\n' + story
        except Exception as e:
            print_and_write(f'Grounding failed in overview: {e}')

    return story_overviews


def topic_finder(formatted_date):
    today = date.today()
    recent_story_descriptions, history_found = load_recent_story_descriptions(window_days=7)
    if history_found:
        print_and_write("Loaded recent story summaries for deduping.")
    else:
        print_and_write("No recent story summaries found; using full headline set.")

    follow_up_prompt_text = FOLLOW_UP_PROMPT_TEMPLATE.format(date=formatted_date)
    challenging_follow_up_prompt_text = CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE.format(date=formatted_date)

    base_scraper_prompt = 'What are the latest headlines here released today, {}? If there are none released today, then say that there are none released from the news source today. And mention the news source.\n'.format(formatted_date)

    npr_specific_prompt = "What are the main headlines for NPR's morning news brief here released today, {}? If there are none released today, then say that there are none released from the news source today. And mention the news source. You can be descriptive when talking about the main headlines. \n".format(formatted_date)

    print_and_write('scraping NPR')
    npr_headlines = get_llm_response(npr_specific_prompt, grounding=True) + '\n'
    print_and_write('scraping AP')
    ap_prompt = (
        base_scraper_prompt
        + "If the page uses relative timestamps like 'Now' or 'minutes ago', assume they refer to today ({}) unless the text explicitly says otherwise. "
        + "Explicit date stamps that fall within the last 24 hours should also be treated as today's headlines. "
        + "Ignore sections that are clearly labeled as historical retrospectives such as 'Today in History'. "
        + "https://apnews.com"
    ).format(formatted_date)
    ap_headlines = get_llm_response(ap_prompt, url_context=True) + '\n'
    print_and_write('scraping DN')
    dn_headlines = get_llm_response(base_scraper_prompt + 'https://www.democracynow.org', grounding=True) + '\n'
    print_and_write('scraping PP')
    pp_headlines = get_llm_response(base_scraper_prompt + 'https://www.propublica.org', url_context=True) + '\n'
    print_and_write('scraping CM')
    calmatters_headlines = calmatters_scraper() + '\n'
    city_of_riverside_headlines = get_llm_response(
        'What are the latest headlines here released in the past two days? Today is {}. If there are none released today, then say that there are none released from the news source today or yesterday. And mention the news source. Do not give anything else. https://www.riversideca.gov/media'.format(formatted_date),
        mode='standard', url_context=True)

    all_headlines = 'NPR:\n' + npr_headlines + '\n\nThe Associated Press:\n' + ap_headlines + '\n\nDemocracy Now:\n' + dn_headlines + '\n\nProPublica\n' + pp_headlines + '\n\nCalMatters\n' + calmatters_headlines + '\n\nThe City of Riverside\n' + city_of_riverside_headlines

    print_and_write('all headlines')
    print_and_write(all_headlines)

    if history_found:
        repetition_remover_system_prompt = REPETITION_REMOVER_TEMPLATE.format(recent_stories=recent_story_descriptions)
        all_headlines = get_llm_response(all_headlines, system_prompt=repetition_remover_system_prompt, mode='standard')

    try:
        year_page_summary = get_llm_response(
            "Assume I am an LLM who does not know much about current events in United States and I'm trying to get enough context about the United States in order to make a news program for it for this evening. My context window ends December 31, 2025, so I need as much information as possible about the state of the world since then. Give me as much background information as I need about current events in the US. https://en.wikipedia.org/wiki/2026_in_the_United_States#",
            mode="standard", url_context=True)
    except Exception as e:
        year_page_summary = "Context unavailable, proceed with headlines only"
        print_and_write(f'Wikipedia context fetch failed: {e}')

    main_story_prompt = MAIN_STORY_PROMPT + " For background knowledge to assist your decision, here is a brief summary of current events\":\n" + year_page_summary

    print_and_write('prompt: ' + main_story_prompt + '\n\n' + all_headlines)

    important_response = get_llm_response(all_headlines, system_prompt=main_story_prompt, mode='heavy')

    important_response = important_response.replace('*', '')
    print_and_write()
    print_and_write(important_response)

    important_headline = headline_extractor(important_response)
    print_and_write(important_headline)

    everyman_string_prompt = EVERYMAN_STORY_PROMPT + '\n' + "also do not pick '{}' or any story that sounds like it.".format(important_headline)

    everyman_topic_response = get_llm_response(all_headlines, system_prompt=everyman_string_prompt, mode='standard')
    print_and_write('\nimportant topic for average person and why:', everyman_topic_response)
    everyman_headline = headline_extractor(everyman_topic_response)

    overview_string_prompt = OVERVIEW_PICK_PROMPT + '\n' + "Also, do not pick the major stories of the day, which are: '{}' or '{}'. Do not pick any stories related to the major stories. For example, if a story is about how the US is involved in some sort of conflict, do not pick another story about that same conflict. Also, do not pick inconsequenntial sensational stories like \"Lucy Letby, a former nurse convicted of murdering seven babies and attempting to murder six others, has lost her appeal bid in England\" or \"the bodies of two women from Kansas, missing since March 2023, were found in a buried freezer in rural Texas County, Oklahoma\" or \"the community in Homer, Alaska, is mourning the death of 70-year-old Dale Chorman, who was fatally attacked by a cow moose while photographing her calves\" or \"Details about the wrongful conviction of a Missouri man, who has served 33 years in prison, continue to be examined as the hearing proceeds to verify the facts of his case\". Also do not pick stuff like \"Sunday Puzzle: Supermarket Brands\", because that is clearly not a news story, but instead some sort of game that is mixed in with the headlines".format(important_headline, everyman_headline)

    overview_string_prompt = overview_string_prompt + '\n' + 'Also, for background knowledge to assist your judgement, here a brief summary of current events:' + year_page_summary

    print_and_write('overview_string_prompt', overview_string_prompt)

    overview = get_llm_response(all_headlines, system_prompt=overview_string_prompt, mode='standard')
    print_and_write('\noverview1:', overview)

    overview = overview_process(overview)
    print_and_write('\noverview2:', overview)

    overview = get_llm_response(overview, system_prompt=OVERVIEW_ANCHOR_PROMPT, mode='standard')

    overview = overview.replace("*", "")
    print_and_write('\noverview3:', overview)

    important_clean = important_headline.strip('\"').strip('\'').strip()
    everyman_clean = everyman_headline.strip('\"').strip('\'').strip()
    topics = [important_clean, everyman_clean]
    print_and_write()
    print_and_write(topics)

    today = date.today()
    formatted_date2 = today.strftime("%Y_%m_%d")
    filename = "stories_chosen/{}_stories_chosen.txt".format(formatted_date2)
    outstring = important_clean + ', ' + everyman_clean
    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(outstring)

    story_summaries = []
    try:
        important_summary = summarize_story_for_archive(important_clean, important_response)
    except Exception as error:
        print_and_write('Failed to summarize important story', error)
        important_summary = important_response.strip() if isinstance(important_response, str) else important_clean
    story_summaries.append({
        "headline": important_clean,
        "summary": important_summary,
        "type": "important",
    })
    try:
        everyman_summary = summarize_story_for_archive(everyman_clean, everyman_topic_response)
    except Exception as error:
        print_and_write('Failed to summarize everyman story', error)
        everyman_summary = everyman_topic_response.strip() if isinstance(everyman_topic_response, str) else everyman_clean
    story_summaries.append({
        "headline": everyman_clean,
        "summary": everyman_summary,
        "type": "everyman",
    })
    summary_filename = "stories_chosen/{}_story_summaries.json".format(formatted_date2)
    try:
        with open(summary_filename, 'w', encoding='utf-8') as summary_file:
            json.dump({
                "date": today.isoformat(),
                "stories": story_summaries,
            }, summary_file, ensure_ascii=False, indent=2)
        print_and_write('Wrote story summaries to', summary_filename)
    except IOError as error:
        print_and_write('Failed to write story summaries', error)
    return topics, overview, follow_up_prompt_text, challenging_follow_up_prompt_text
