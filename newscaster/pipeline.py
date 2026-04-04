import json
import os
import time
from datetime import date

from newscaster.logging import print_and_write
from newscaster.dates import format_spoken_date
from newscaster.text_utils import find_quoted_strings
from newscaster.prompts import (
    FOLLOW_UP_PROMPT_TEMPLATE,
    CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE,
    AUDIENCE_LEARNED_EXTRACTION_PROMPT,
    OVERVIEW_AUDIENCE_LEARNED_PROMPT,
    OUTRO_TEMPLATE,
)
from newscaster.llm import get_llm_response
from newscaster.scrapers.topic_finder import topic_finder, TopicFinderResult
from newscaster.scrapers.google_search import google_official_search
from newscaster.scrapers.topic_finder import result_piper
from newscaster.dedup import update_audience_learned, save_ledger
from newscaster.weather import get_daily_temp
from newscaster.script.headlines import story_gatherer, headline_maker
from newscaster.script.intro import intro_writer
from newscaster.script.segments import segments_writer
from newscaster.audio.tts import text2speech
from newscaster.audio.intro_music import fun_intromaker
from newscaster.audio.assembly import assemble_podcast


def _run_follow_up_rounds(summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text):
    """Run multiple follow-up question rounds to enrich a story summary.

    Each round: generate a follow-up question, search for the answer via grounding,
    and append both to the summary. Runs through light, standard, and heavy modes
    with both regular and challenging follow-up prompts.
    """
    rounds = [
        (follow_up_prompt_text, 'light', 'Gemini Flash Lite'),
        (challenging_follow_up_prompt_text, 'light', 'Gemini Flash Lite'),
        (follow_up_prompt_text, 'standard', 'Gemini Flash'),
        (challenging_follow_up_prompt_text, 'standard', 'Gemini Flash'),
        (follow_up_prompt_text, 'plus', 'Gemini Pro 3.1'),
        (challenging_follow_up_prompt_text, 'plus', 'Gemini Pro 3.1'),
        (follow_up_prompt_text, 'heavy', 'Claude Opus 4.6'),
        (challenging_follow_up_prompt_text, 'heavy', 'Claude Opus 4.6'),
    ]

    for prompt_template, mode, asker_name in rounds:
        follow_up_creator_response = get_llm_response(summary_prompt, system_prompt=prompt_template, mode=mode)

        if mode == 'heavy':
            print_and_write('heavy follow_up_creator_response', follow_up_creator_response)

        if "\"" in follow_up_creator_response or "\'" in follow_up_creator_response:
            follow_up_question = find_quoted_strings(follow_up_creator_response)[0]
        else:
            follow_up_question = follow_up_creator_response

        is_challenging = (prompt_template == challenging_follow_up_prompt_text)
        label = 'challenging follow_up_question' if is_challenging else 'follow_up_question'
        print_and_write(label, follow_up_question)
        response = get_llm_response(follow_up_question, mode='light', grounding=True)
        print_and_write(response)

        summary_prompt = summary_prompt + '\n' + f'{asker_name} asked:' + follow_up_question + '\nNews sources found by Gemini Flash reported:\n' + response

    return summary_prompt


def gather_news(formatted_date, formatted_date2):
    """Stage 1: Scrape news, search for details, and write segment summaries."""
    filename = "segment_summaries/{}_segment0_summary.txt".format(formatted_date2)
    if os.path.exists(filename):
        print_and_write(f"The file '{filename}' exists.")
        return None

    print_and_write(f"The file '{filename}' does not exist.")

    tf_result = topic_finder(formatted_date)
    # Support both TopicFinderResult and legacy tuple
    if isinstance(tf_result, TopicFinderResult):
        topics = tf_result.topics
        overview = tf_result.overview
        follow_up_prompt_text = tf_result.follow_up_prompt_text
        challenging_follow_up_prompt_text = tf_result.challenging_follow_up_prompt_text
    else:
        topics, overview, follow_up_prompt_text, challenging_follow_up_prompt_text = tf_result
        tf_result = None

    stories = []

    for topic in topics:
        topic_OG = topic
        topic_index = topics.index(topic)
        successful_summary_counter = 0
        waiting_time = [0, 5, 60, 60, 300, 600]
        for waiting_duration in waiting_time:
            time.sleep(waiting_duration)
            try:
                search_results = google_official_search(topic, 9)
                break
            except Exception as e:
                print_and_write(f'Google search failed: {e}')

        print_and_write('\n\n')
        print_and_write('TOPIC:', topic, '\n')
        summary_prompt = 'Today is {}.'.format(formatted_date)
        summary_prompt_alpha = "Given the topic of '" + topic + "', and given the summaries below, please create a long synthesis text that holds as much information about the topic as possible. Do not include information irrelevant  to the topic, like side stories or topics that do not have anything to do with the main story. For example, if the main story is about a fire, if the page also includes a mention of abortion, you should remove it. That being said, your job is to synthesize a cohesive text that includes as much information as possible. I want DEPTH. In addition, it is of the utmost importance that you include the news outlets the stories come from, like CNN, the New York Times, ABC News, NPR, etc. The text below includes article summaries as well as follow-up questions and answers that provide additional context and nuance — be sure to incorporate those insights into the synthesis. Also note that today is " + formatted_date + '.\n\n """'

        summary_prompt2 = "Given the summary below, please refine the information by eliminating any sentences that aren't relevant to the core narrative. However, it's crucial to keep the majority of the information, including all key aspects, background context, and important details tied to the story. Remove any redundant elements but retain all relevant facts, figures, and potential implications. Aim for a summary that is concise yet comprehensive in preserving the essence of the original text." + '\n\n """'
        irrelevance_prompt = 'Is there any information irrelevant to the main story in the below text, like information about upcoming programming or advertisements? For example, if the main story is about a fire, if the page the story was on also includes a mention of abortion, you should remove it. Give a yes or no answer and explain why: \n\n'
        for result in search_results:
            print_and_write(result)
            summary_prompt, successful_summary_counter = result_piper(summary_prompt, successful_summary_counter, topic, result, topic_index, formatted_date2)

            if successful_summary_counter == 3:
                break

        if successful_summary_counter < 1:
            for waiting_duration in waiting_time:
                time.sleep(waiting_duration)
                try:
                    search_results = google_official_search(topic, 9, days_prior=3)
                    break
                except Exception as e:
                    print_and_write(f'Google search failed: {e}')
            for result in search_results:
                print_and_write(result)
                try:
                    summary_prompt, successful_summary_counter = result_piper(summary_prompt, successful_summary_counter, topic, result, topic_index, formatted_date2)
                except Exception as e:
                    print_and_write(f'result_piper failed for topic {topic}: {e}')
                    pass
                if successful_summary_counter == 3:
                    break

        if successful_summary_counter < 1:
            topic_OG = topic

            topic = get_llm_response(topic, system_prompt='Please change the following query into a rephrased google search query that can get more relevant results. Please only give the new query and do not give quotation marks.')
            topic = topic.strip("\"").strip('\'').strip('\n')
            print_and_write('new topic:', topic)
            for waiting_duration in waiting_time:
                time.sleep(waiting_duration)
                try:
                    search_results = google_official_search(topic, 9)
                    break
                except Exception as e:
                    print_and_write(f'Google search failed: {e}')
            for result in search_results:
                print_and_write(result)
                try:
                    summary_prompt, successful_summary_counter = result_piper(summary_prompt, successful_summary_counter, topic, result, topic_index, formatted_date2)
                except Exception as e:
                    print_and_write(f'result_piper failed for topic {topic}: {e}')
                    pass
                if successful_summary_counter == 3:
                    break

        if successful_summary_counter > 0:
            topic = topic_OG
            perplexity_prompt = 'Tell me about this story with as much detail as possible:\n' + topic
            summary_prompt = summary_prompt + get_llm_response(perplexity_prompt, grounding=True, mode='light')
            successful_summary_counter += 1

        if successful_summary_counter >= 1:
            summary_prompt = _run_follow_up_rounds(
                summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text
            )

            print_and_write('SUPER SUMMARY LENGTH:', len(summary_prompt))

            super_summary = get_llm_response(summary_prompt, system_prompt=summary_prompt_alpha, mode='standard')
            print_and_write('SUPER SUMMARY ITERATION 1:', super_summary, '\n')
            irrelevance_prompt = irrelevance_prompt + super_summary
            irrelevance_answer = get_llm_response(irrelevance_prompt)
            print_and_write('IRRELEVANT INFORMATION?:' + irrelevance_answer)
            if 'yes' in irrelevance_answer.lower():
                summary_prompt2 = summary_prompt2 + super_summary
                super_summary = get_llm_response(summary_prompt2, mode='standard')
                print_and_write('SUPER SUMMARY ITERATION 2:', super_summary, '\n')
            stories.append(super_summary)

    for i in range(len(stories)):
        outfile = open("segment_summaries/{}_segment{}_summary.txt".format(formatted_date2, i), 'w', encoding='utf-8')
        outfile.write(stories[i])
        outfile.close()

    outfile = open("output_scripts/{}_overview.txt".format(formatted_date2), 'w', encoding='utf-8')
    outfile.write(overview)
    outfile.close()

    return tf_result


def _extract_audience_learned(formatted_date2, tf_result):
    """Extract audience_learned from segment summaries and side story briefs, update ledger."""
    if not tf_result or not isinstance(tf_result, TopicFinderResult):
        return
    ledger = tf_result.ledger
    if not ledger or not ledger.get("arcs"):
        return

    # Main stories: read segment summaries (arc_context is parallel to topics by index)
    for i in range(len(tf_result.topics)):
        summary_path = f"segment_summaries/{formatted_date2}_segment{i}_summary.txt"
        if not os.path.exists(summary_path):
            print_and_write(f"Segment summary not found for audience_learned: {summary_path}")
            continue

        arc_data = tf_result.arc_context[i] if i < len(tf_result.arc_context) else None
        if not arc_data:
            print_and_write(f"No arc found for story slot {i}")
            continue

        slug = arc_data["slug"]
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()
        except IOError:
            continue

        prompt = AUDIENCE_LEARNED_EXTRACTION_PROMPT.format(
            arc_topic=arc_data.get("topic", slug),
            audience_state=arc_data.get("audience_state", "(first coverage)"),
            summary_text=summary_text,
        )
        try:
            response = get_llm_response(prompt, mode="light")
            parsed = json.loads(response)
            learned = parsed.get("learned", [])
            state = parsed.get("state", "")
            update_audience_learned(ledger, slug, formatted_date2, i, learned, state)
            print_and_write(f"Updated audience_learned for arc '{slug}': {len(learned)} facts")
        except (json.JSONDecodeError, Exception) as e:
            print_and_write(f"Failed to extract audience_learned for '{slug}': {e}")

    # Side stories: use overview briefs
    # The LLM response items are ordered to match the input briefs, so we match by position
    if tf_result.side_story_briefs:
        briefs_text = ""
        for j, (oh_headline, oh_brief) in enumerate(tf_result.side_story_briefs):
            briefs_text += f"\n--- Story {j + 1} ---\nHeadline: {oh_headline}\n{oh_brief}\n"
        prompt = OVERVIEW_AUDIENCE_LEARNED_PROMPT.format(side_story_briefs=briefs_text)
        try:
            response = get_llm_response(prompt, mode="light")
            parsed = json.loads(response)
            # Build slot→slug lookup for today's side stories
            side_slot_to_slug = {}
            for slug, arc in ledger["arcs"].items():
                for ep in arc.get("episodes", []):
                    if ep["date"] == formatted_date2 and ep["coverage"] == "side":
                        side_slot_to_slug[ep["coverage_slot"]] = slug
            for j, item in enumerate(parsed):
                item_learned = item.get("learned", [])
                slug = side_slot_to_slug.get(j)
                if slug:
                    update_audience_learned(
                        ledger, slug, formatted_date2, j,
                        item_learned, "; ".join(item_learned)
                    )
                    print_and_write(f"Updated audience_learned for side arc '{slug}'")
        except (json.JSONDecodeError, Exception) as e:
            print_and_write(f"Failed to extract side story audience_learned: {e}")

    save_ledger(ledger)
    print_and_write("Saved ledger after audience_learned extraction")


def write_scripts(formatted_date, formatted_date2, formatted_date3, voices_list, tf_result=None):
    """Stage 2: Generate dialogue scripts from segment summaries."""
    filename = "output_scripts/{}_segment_0.txt".format(formatted_date2)
    if os.path.exists(filename):
        print_and_write(f"The file '{filename}' exists.")
        return

    print_and_write(f"The file '{filename}' does not exist.")

    print_and_write('gathering stories')
    stories = story_gatherer(formatted_date2)
    successful_topics = headline_maker(stories)

    print_and_write('writing intro segment')
    weather_string = get_daily_temp()
    print_and_write(weather_string)

    intro1, intro2 = intro_writer(formatted_date3, weather_string, successful_topics, formatted_date2, stories)
    outfile = open('output_scripts/{}_intro1.txt'.format(formatted_date2), 'w', encoding='utf-8')
    outfile.write(intro1)
    outfile.close()
    outfile = open('output_scripts/{}_intro2.txt'.format(formatted_date2), 'w', encoding='utf-8')
    outfile.write(intro2)
    outfile.close()
    print_and_write(intro1)
    print_and_write(intro2)

    # arc_context is already a list parallel to topics (slot 0, slot 1, ...)
    arc_context_list = None
    if tf_result and isinstance(tf_result, TopicFinderResult) and tf_result.arc_context:
        arc_context_list = tf_result.arc_context

    print_and_write('writing segments')
    segments_writer(stories, formatted_date2, voices_list, formatted_date, arc_context=arc_context_list)

    # Extract audience_learned after scripts are written (uses segment summaries)
    _extract_audience_learned(formatted_date2, tf_result)

    outro = OUTRO_TEMPLATE
    outfile = open('output_scripts/{}_outro.txt'.format(formatted_date2), 'w', encoding='utf-8')
    outfile.write(outro)
    outfile.close()


def generate_audio(formatted_date2, voices_list):
    """Stage 3: Synthesize speech, add intro music, and assemble final podcast."""
    text2speech(formatted_date2, voices_list)
    fun_intromaker(formatted_date2)
    assemble_podcast(formatted_date2)


def _ensure_output_dirs():
    """Create output directories if they don't exist."""
    for d in [
        "segment_summaries", "output_scripts", "output_audio",
        "segment_audio", "stories_chosen", "episode_titles", "logs",
    ]:
        os.makedirs(d, exist_ok=True)


def main():
    _ensure_output_dirs()

    today = date.today()
    formatted_date = today.strftime("%B %d, %Y")
    formatted_date2 = today.strftime("%Y_%m_%d")
    formatted_date3 = format_spoken_date(today)
    voices_list = ['Ethan', 'Chloe', 'Ethan', 'Chloe', 'Grace']

    print_and_write(formatted_date3)

    tf_result = gather_news(formatted_date, formatted_date2)
    write_scripts(formatted_date, formatted_date2, formatted_date3, voices_list, tf_result=tf_result)
    generate_audio(formatted_date2, voices_list)
