import json
from dataclasses import dataclass, field
from datetime import date, datetime

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
    LEDGER_REPETITION_REMOVER_TEMPLATE,
    TIER1_TRIAGE_PROMPT,
    TIER2_RESEARCH_PROMPT,
    TIER3_IMPORTANT_STORY_PROMPT,
    TIER3_EVERYMAN_STORY_PROMPT,
    TIER3_OVERVIEW_PICK_PROMPT,
)
from newscaster.llm import get_llm_response, call_with_default, LLMError
from newscaster.dedup import (
    load_recent_story_descriptions,
    summarize_story_for_archive,
    load_ledger,
    save_ledger,
    prune_ledger,
    format_arcs_for_dedup,
    create_arc,
    update_arc,
    find_matching_arc,
    strip_arc_tags,
)
from newscaster.scrapers.calmatters import calmatters_scraper
from newscaster.scrapers.web import scrape_text


@dataclass
class TopicFinderResult:
    topics: list
    overview: str
    follow_up_prompt_text: str
    challenging_follow_up_prompt_text: str
    arc_context: list = field(default_factory=list)
    ledger: dict = field(default_factory=dict)
    side_story_briefs: list = field(default_factory=list)


def _today_str():
    return datetime.now().strftime("%B %d, %Y")


def headline_extractor(response):
    extraction_prompt = HEADLINE_EXTRACTION_PROMPT + response
    headline = get_llm_response(response, system_prompt=extraction_prompt)
    return headline


def determine_relevance(topic, result):
    prompt = "Given the search engine headline of '{}' and the search result snippet of '{}', do you think that the given website is a news article AND might be relevant to the topic of '{}'? Today is {}. Give a yes or no answer.".format(
        result['headline'], result['snippet'], topic, _today_str())

    response = call_with_default(
        'no', prompt, mode='light',
        _log_label=f"determine-relevance[{result.get('headline', '?')[:60]}]",
    )

    return 'yes' in response.lower()


def summarize_text(text, article):
    text_length = len(text)
    print_and_write(f"Text length: {text_length} characters")
    prompt = (
        'Summarize this news article, but be sure to include where the article is from '
        'and as many important details as you can.\n\n'
        'STRICT SOURCING RULES:\n'
        '1. Only state facts that are literally present in the article text below. Do not '
        'add details from prior knowledge. If the article omits a detail, leave it out — '
        'do not infer, substitute the most likely answer, or normalize an ambiguous name '
        'to a more famous match.\n'
        '2. Preserve the article\'s own grouping of facts. Each claim in your summary '
        'must map to a single contiguous span of the source. If two pieces of information '
        'appear in separate sentences or paragraphs, render them as separate clauses — '
        'do not fuse them with "and," "from," "at," "while," "after," or other connectors '
        'that imply a relationship (causal, locational, temporal, possessive) the source '
        'did not assert. When the source presents facts as independent, your summary must '
        'too.\n\n'
    )

    completion = get_llm_response(prompt + text, mode='standard')
    return completion


def result_piper(summary_prompt, successful_summary_counter, topic, result, i, formatted_date2, articles=None):
    print_and_write('HEADLINE:', result['headline'], '\n')
    relevant = determine_relevance(topic, result)
    print_and_write(relevant)
    if relevant == True:

        print_and_write('ARTICLE SEEMS RELEVANT', '\n')
        url = result['url']
        text = scrape_text(result['url'])
        if text.startswith('Error: '):
            print_and_write('SCRAPE FAILED:', text)
            return summary_prompt, successful_summary_counter
        if not text.strip():
            print_and_write('ARTICLE IS EMPTY')
            return summary_prompt, successful_summary_counter
        try:
            summary = summarize_text(text, 'article')
        except LLMError as e:
            print_and_write(f'summarize_text failed for {url}: {e}; skipping article')
            return summary_prompt, successful_summary_counter
        print_and_write('\nSUMMARY:', summary, '\n')
        relevance_prompt = "Given the topic of '" + topic + "', is there any relevant information in the summary of a news article below? Answer as a yes or no.\n" + summary
        completion = call_with_default(
            'no', relevance_prompt, mode='light',
            _log_label=f'article-relevance[{url}]',
        )
        response = completion.replace('-', '')

        if 'yes' in response.lower():
            print_and_write('ARTICLE IS RELEVANT', '\n')
            news_source_prompt = 'What is the news outlet this url is associated with? Answer after writing "SOURCE:"\n' + url

            news_source_response = call_with_default(
                f'SOURCE: {url}', news_source_prompt, mode='light',
                _log_label=f'news-source[{url}]',
            )

            summary_prompt = summary_prompt + '\n\n---\nArticle ' + str(successful_summary_counter + 1) + '\n'
            summary_prompt = summary_prompt + 'Source: ' + news_source_response + '\n'
            summary_prompt = summary_prompt + summary
            summary_prompt = summary_prompt + '\n---\n'

            outfile = open('segment_summaries/{}_segment{}_article{}_summary.txt'.format(formatted_date2, i, successful_summary_counter), 'w', encoding='utf-8')
            outfile.write(summary)
            outfile.close()

            if articles is not None:
                articles.append({
                    "chunk_id": f"{formatted_date2}_seg{i}_art{successful_summary_counter}",
                    "url": url,
                    "outlet": (news_source_response or "").replace("SOURCE:", "").strip(),
                    "original_headline": result.get("headline"),
                    "published_date": result.get("date"),
                    "retrieved_date": formatted_date2,
                    "surfacing_topic": topic,
                    "summary": summary,
                })

            successful_summary_counter += 1

        else:
            print_and_write('ARTICLE IS NOT RELEVANT', '\n')
    return summary_prompt, successful_summary_counter


def summarize_headline_with_grounding(headline: str) -> str:
    headline_clean = (headline or '').strip()
    if not headline_clean:
        return 'UNVERIFIED: Empty headline received. Please provide a valid headline to summarize.'

    system_prompt = OVERVIEW_SYSTEM_PROMPT_TEMPLATE.format(date=_today_str())
    base_prompt = (
        f"Headline: {headline_clean}\n"
        f"Date: {_today_str()}\n"
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
    overview_headlines = []
    overview_briefs = []
    for i in range(5):
        number = str(i + 1)
        headline_finder_prompt = f'Find story number {number}. Only give the headline of that story.'

        try:
            headline_n = get_llm_response(overview, system_prompt=headline_finder_prompt, mode='light')
        except Exception as e:
            print_and_write(f'Headline extraction failed in overview for story {number}: {e}')
            continue

        story_finder_prompt = "Tell me more about the story behind this headline from today's paper. Include as many details as possible :\n" + headline_n
        try:
            print_and_write(story_finder_prompt)
            story = summarize_headline_with_grounding(headline_n)
            print_and_write(story)
            story_overviews = story_overviews + '\n' + story
            overview_headlines.append(headline_n)
            overview_briefs.append((headline_n, story))
        except Exception as e:
            print_and_write(f'Grounding failed in overview: {e}')

    return story_overviews, overview_headlines, overview_briefs


import re


def _parse_tier1_scores(response):
    """Parse structured SCORE: X | HEADLINE: Y | REASON: Z lines from Tier 1 triage."""
    results = []
    for line in response.strip().split('\n'):
        match = re.match(r'SCORE:\s*(\d+)\s*\|\s*HEADLINE:\s*(.+?)\s*\|\s*REASON:\s*(.+)', line.strip())
        if match:
            score = int(match.group(1))
            headline = match.group(2).strip()
            reason = match.group(3).strip()
            results.append({'score': score, 'headline': headline, 'reason': reason})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def _format_research_briefs(briefs):
    """Assemble individual research memos into a single document."""
    sections = []
    for i, (headline, brief) in enumerate(briefs, 1):
        sections.append(f"--- Brief {i} ---\nHeadline: {headline}\n\n{brief}\n")
    return '\n'.join(sections)


def topic_finder(formatted_date):
    today = date.today()
    formatted_date2 = today.strftime("%Y_%m_%d")

    # Load ledger and fall back to flat descriptions if empty
    ledger = load_ledger()
    ledger = prune_ledger(ledger)
    arc_summaries = format_arcs_for_dedup(ledger)
    use_ledger = bool(arc_summaries)

    recent_story_descriptions, history_found = load_recent_story_descriptions(window_days=7)
    if use_ledger:
        print_and_write("Loaded story ledger for deduping.")
    elif history_found:
        print_and_write("Loaded recent story summaries for deduping (ledger empty).")
    else:
        print_and_write("No recent story summaries found; using full headline set.")

    follow_up_prompt_text = FOLLOW_UP_PROMPT_TEMPLATE.format(date=formatted_date)
    challenging_follow_up_prompt_text = CHALLENGING_FOLLOW_UP_PROMPT_TEMPLATE.format(date=formatted_date)

    base_scraper_prompt = 'What are the latest headlines here released today, {}? If there are none released today, then say that there are none released from the news source today. And mention the news source.\n'.format(formatted_date)

    npr_specific_prompt = "What are the main headlines for NPR's morning news brief here released today, {}? If there are none released today, then say that there are none released from the news source today. And mention the news source. You can be descriptive when talking about the main headlines. \n".format(formatted_date)

    print_and_write('scraping NPR')
    npr_headlines = call_with_default(
        '', npr_specific_prompt, grounding=True, _log_label='scrape-npr',
    ) + '\n'
    print_and_write('scraping AP')
    ap_prompt = (
        base_scraper_prompt
        + "If the page uses relative timestamps like 'Now' or 'minutes ago', assume they refer to today ({}) unless the text explicitly says otherwise. "
        + "Explicit date stamps that fall within the last 24 hours should also be treated as today's headlines. "
        + "Ignore sections that are clearly labeled as historical retrospectives such as 'Today in History'. "
        + "https://apnews.com"
    ).format(formatted_date)
    ap_headlines = call_with_default(
        '', ap_prompt, url_context=True, _log_label='scrape-ap',
    ) + '\n'
    print_and_write('scraping DN')
    dn_headlines = call_with_default(
        '', base_scraper_prompt + 'https://www.democracynow.org', grounding=True, _log_label='scrape-dn',
    ) + '\n'
    print_and_write('scraping PP')
    pp_headlines = call_with_default(
        '', base_scraper_prompt + 'https://www.propublica.org', url_context=True, _log_label='scrape-pp',
    ) + '\n'
    print_and_write('scraping CM')
    calmatters_headlines = calmatters_scraper() + '\n'
    city_of_riverside_headlines = call_with_default(
        '',
        'What are the latest headlines here released in the past two days? Today is {}. If there are none released today, then say that there are none released from the news source today or yesterday. And mention the news source. Do not give anything else. https://www.riversideca.gov/media'.format(formatted_date),
        mode='standard', url_context=True, _log_label='scrape-riverside',
    )

    all_headlines = 'NPR:\n' + npr_headlines + '\n\nThe Associated Press:\n' + ap_headlines + '\n\nDemocracy Now:\n' + dn_headlines + '\n\nProPublica\n' + pp_headlines + '\n\nCalMatters\n' + calmatters_headlines + '\n\nThe City of Riverside\n' + city_of_riverside_headlines

    print_and_write('all headlines')
    print_and_write(all_headlines)

    if use_ledger:
        repetition_remover_system_prompt = LEDGER_REPETITION_REMOVER_TEMPLATE.format(arc_summaries=arc_summaries)
        all_headlines = call_with_default(
            all_headlines, all_headlines, system_prompt=repetition_remover_system_prompt, mode='standard',
            _log_label='dedup-headlines-ledger',
        )
    elif history_found:
        repetition_remover_system_prompt = REPETITION_REMOVER_TEMPLATE.format(recent_stories=recent_story_descriptions)
        all_headlines = call_with_default(
            all_headlines, all_headlines, system_prompt=repetition_remover_system_prompt, mode='standard',
            _log_label='dedup-headlines-history',
        )

    # === TIER 1: Triage — score all headlines ===
    print_and_write('TIER 1: Triaging headlines')
    tier1_response = get_llm_response(all_headlines, system_prompt=TIER1_TRIAGE_PROMPT, mode='standard')
    print_and_write('Tier 1 raw response:', tier1_response)

    scored = _parse_tier1_scores(tier1_response)
    print_and_write(f'Tier 1 parsed {len(scored)} headlines')

    if len(scored) < 5:
        print_and_write('Tier 1 parsing returned < 5 results, passing all headlines to Tier 3')
        shortlisted_headlines = [line.strip() for line in all_headlines.split('\n') if line.strip()]
    else:
        shortlisted_headlines = [s['headline'] for s in scored[:10]]

    for i, h in enumerate(shortlisted_headlines, 1):
        print_and_write(f'  Shortlisted {i}: {h}')

    # === TIER 2: Research — grounded briefs per headline ===
    print_and_write('TIER 2: Researching shortlisted headlines')
    briefs = []
    for headline in shortlisted_headlines:
        print_and_write(f'  Researching: {headline}')
        research_prompt = TIER2_RESEARCH_PROMPT.format(date=formatted_date, headline=headline)
        try:
            brief = get_llm_response(research_prompt, grounding=True, mode='standard')
            briefs.append((headline, brief))
            print_and_write(f'  Brief received ({len(brief)} chars)')
        except Exception as e:
            print_and_write(f'  Research failed for "{headline}": {e}')

    research_document = _format_research_briefs(briefs)
    print_and_write(f'TIER 2: Assembled {len(briefs)} research briefs ({len(research_document)} chars)')

    # === TIER 3: Final picks using enriched context ===
    print_and_write('TIER 3: Selecting stories')

    # Important story (heavy = Claude Opus 4.6)
    important_response = get_llm_response(research_document, system_prompt=TIER3_IMPORTANT_STORY_PROMPT, mode='heavy')
    important_response = important_response.replace('*', '')
    print_and_write()
    print_and_write(important_response)

    important_headline = headline_extractor(important_response)
    print_and_write(important_headline)

    # Everyman story (heavy = Claude Opus 4.6)
    everyman_prompt = TIER3_EVERYMAN_STORY_PROMPT.format(excluded_headline=important_headline)
    everyman_topic_response = get_llm_response(research_document, system_prompt=everyman_prompt, mode='heavy')
    print_and_write('\nimportant topic for average person and why:', everyman_topic_response)
    everyman_headline = headline_extractor(everyman_topic_response)

    # Overview picks (standard = Gemini Flash, cheaper)
    overview_prompt = TIER3_OVERVIEW_PICK_PROMPT.format(
        excluded_headlines=f"'{important_headline}' or '{everyman_headline}'"
    )
    print_and_write('overview_string_prompt', overview_prompt)

    overview = get_llm_response(research_document, system_prompt=overview_prompt, mode='standard')
    print_and_write('\noverview1:', overview)

    overview_raw, overview_headlines, overview_briefs = overview_process(overview)
    print_and_write('\noverview2:', overview_raw)

    overview_text = get_llm_response(overview_raw, system_prompt=OVERVIEW_ANCHOR_PROMPT, mode='standard')

    overview_text = overview_text.replace("*", "")
    print_and_write('\noverview3:', overview_text)

    # --- Extract arc tags before stripping ---
    important_raw = important_headline.strip('\"').strip('\'').strip()
    everyman_raw = everyman_headline.strip('\"').strip('\'').strip()

    important_arc_info = find_matching_arc(important_raw)
    everyman_arc_info = find_matching_arc(everyman_raw)

    important_clean = strip_arc_tags(important_raw)
    everyman_clean = strip_arc_tags(everyman_raw)

    topics = [important_clean, everyman_clean]
    print_and_write()
    print_and_write(topics)

    filename = "stories_chosen/{}_stories_chosen.txt".format(formatted_date2)
    outstring = important_clean + ', ' + everyman_clean
    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(outstring)

    # --- Legacy story summaries (secondary record) ---
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

    # --- Ledger updates ---
    arc_context = []  # parallel to topics: [arc_dict_for_slot_0, arc_dict_for_slot_1]
    main_stories_info = [
        (important_clean, important_summary, important_arc_info, 0),
        (everyman_clean, everyman_summary, everyman_arc_info, 1),
    ]
    for headline, summary, arc_info, slot in main_stories_info:
        if arc_info:
            tag_type, slug = arc_info
            if slug in ledger["arcs"]:
                update_arc(ledger, slug, headline, "main", slot, formatted_date2)
                arc_context.append(ledger["arcs"][slug])
            else:
                new_slug = create_arc(ledger, headline, "main", slot, formatted_date2, summary)
                arc_context.append(ledger["arcs"][new_slug])
        else:
            new_slug = create_arc(ledger, headline, "main", slot, formatted_date2, summary)
            arc_context.append(ledger["arcs"][new_slug])

    # Archive side stories in ledger
    for i, (oh_headline, oh_brief) in enumerate(overview_briefs):
        oh_clean = strip_arc_tags(oh_headline).strip()
        oh_arc_info = find_matching_arc(oh_headline)
        if oh_arc_info:
            tag_type, slug = oh_arc_info
            if slug in ledger["arcs"]:
                update_arc(ledger, slug, oh_clean, "side", i, formatted_date2)
            else:
                create_arc(ledger, oh_clean, "side", i, formatted_date2, oh_brief)
        else:
            create_arc(ledger, oh_clean, "side", i, formatted_date2, oh_brief)

    save_ledger(ledger)
    print_and_write(f'Saved ledger with {len(ledger["arcs"])} arcs')

    return TopicFinderResult(
        topics=topics,
        overview=overview_text,
        follow_up_prompt_text=follow_up_prompt_text,
        challenging_follow_up_prompt_text=challenging_follow_up_prompt_text,
        arc_context=arc_context,
        ledger=ledger,
        side_story_briefs=overview_briefs,
    )
