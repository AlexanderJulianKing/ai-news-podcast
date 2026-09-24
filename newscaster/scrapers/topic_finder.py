from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime

import newscaster.config as _config
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
    TIER1_CALIFORNIA_TRIAGE_PROMPT,
    TIER2_RESEARCH_PROMPT,
    TIER3_IMPORTANT_STORY_PROMPT,
    TIER3_EVERYMAN_STORY_PROMPT,
    TIER3_OVERVIEW_PICK_PROMPT,
    RESEARCH_DEGRADED_NOTICE,
    COVERAGE_NOTES_HEADER,
    EVENT_SCRAPER_PROMPT,
    EVENT_SCRAPER_TIMESTAMP_RULES,
    EVENT_SCRAPER_GROUNDED_TAIL,
)
from newscaster.llm import get_llm_response, call_with_default, LLMError
from newscaster.source_hunter import answer_with_escalation
from newscaster.search import openrouter_web_brief
from newscaster.dedup import (
    _content_tokens,
    apply_coverage_depth,
    format_coverage_notes,
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
    build_headline_arc_map,
    recover_arc_for_headline,
    resolve_arc_identity,
)
from newscaster.scrapers.calmatters import calmatters_scraper
from newscaster.scrapers.dropsite import dropsite_scraper, rss_scraper
from newscaster.scrapers.riverside import riverside_scraper
from newscaster.scrapers.watchlist import beat_scraper, watchlist_scraper
from newscaster.scrapers.web import scrape_text
from newscaster.scrapers.browser import RenderError, prune_screenshots, scrape_rendered
from newscaster.tagger import tag_pool


_NATIONAL_SHORTLIST_LIMIT = 10
_CALIFORNIA_SHORTLIST_LIMIT = 5
_MERGED_SHORTLIST_LIMIT = 16   # was 13; with ~20 events per source the cap bound on the top group alone


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

            # Persist the RAW scraped page text (NOT the LLM summary, which already contains any
            # invented claim) so the pre-TTS fact-finder can ground-check the script against what the
            # writer actually read. Capped to bound corpus size.
            try:
                with open('segment_summaries/{}_segment{}_article{}_source.txt'.format(formatted_date2, i, successful_summary_counter), 'w', encoding='utf-8') as src_file:
                    src_file.write(text[:50000])
            except OSError as src_err:
                print_and_write(f'Could not persist raw source text for {url}: {src_err}')

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


def summarize_headline_with_grounding(headline: str, audience_state: str | None = None) -> str:
    """Backward-compatible name; this now uses controlled source-hunter research.

    When ``audience_state`` is supplied, this side story is a continuation of an
    arc the audience has already heard. The grounding prompt then carries what
    listeners already know plus an instruction to report only what is new, so the
    overview stops re-explaining the same background day after day.
    """
    headline_clean = (headline or '').strip()
    if not headline_clean:
        return 'UNVERIFIED: Empty headline received. Please provide a valid headline to summarize.'

    continuation_block = ""
    if audience_state and audience_state.strip():
        continuation_block = (
            "\nCONTINUATION: This show has covered this story before. Listeners already know:\n"
            f"{audience_state.strip()}\n"
            "- Report ONLY what is genuinely new since then (the latest developments, numbers, rulings, or turns).\n"
            "- Do NOT re-explain background the audience already has.\n"
            "- If there is no meaningful new development, say so in one sentence rather than restating old facts.\n"
        )

    base_prompt = (
        f"Headline: {headline_clean}\n"
        f"Date: {_today_str()}\n"
        f"{continuation_block}"
        "Instructions:\n"
        "- Use controlled source-hunter research to pull multiple reputable sources published today or within the past 48 hours.\n"
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
        story = _source_hunter_answer(prompt, topic=headline_clean, formatted_date=_today_str())
        if story is None:
            # Terminal: the source hunter already escalated (standard -> advanced)
            # and found no current evidence. Re-running the whole apparatus on the
            # near-identical retry prompt cannot recover, so stop and mark unverified.
            break
        if not _grounded_response_needs_retry(story):
            return story
        # Got a real answer that still reads as unverified or denies the story;
        # loop once more with retry_prompt, which pushes past false negatives.

    return (
        "UNVERIFIED: Source-hunter research did not return accepted current evidence. "
        f"Editor should manually verify this headline: {headline_clean}"
    )


def _audience_state_for_arc(arc_info, ledger):
    """audience_state of an already-resolved arc, or None."""
    if not arc_info or not ledger:
        return None
    _tag_type, slug = arc_info
    arc = ledger.get("arcs", {}).get(slug)
    if not arc:
        return None
    return arc.get("audience_state") or None


def overview_process(overview, headline_arc_map=None, ledger=None):
    story_overviews = ''
    overview_headlines = []
    overview_briefs = []
    overview_arc_infos = []  # parallel to overview_briefs; reused by the archival step
    for i in range(5):
        number = str(i + 1)
        headline_finder_prompt = f'Find story number {number}. Only give the headline of that story.'

        try:
            headline_n = get_llm_response(overview, system_prompt=headline_finder_prompt, mode='light')
        except Exception as e:
            print_and_write(f'Headline extraction failed in overview for story {number}: {e}')
            continue

        # Resolve the arc ONCE here so the audience_state lookup (for writing) and the
        # later ledger archival agree and don't each pay for a separate LLM match.
        arc_info = resolve_arc_identity(headline_n, headline_arc_map, ledger) if headline_arc_map else None
        prior_state = _audience_state_for_arc(arc_info, ledger)
        if prior_state:
            print_and_write(f'Side story "{strip_arc_tags(headline_n).strip()[:60]}" is a continuation; injecting prior audience_state')

        story_finder_prompt = "Tell me more about the story behind this headline from today's paper. Include as many details as possible :\n" + headline_n
        try:
            print_and_write(story_finder_prompt)
            story = summarize_headline_with_grounding(headline_n, audience_state=prior_state)
            print_and_write(story)
            story_overviews = story_overviews + '\n' + story
            overview_headlines.append(headline_n)
            overview_briefs.append((headline_n, story))
            overview_arc_infos.append(arc_info)
        except Exception as e:
            print_and_write(f'Overview source-hunter research failed: {e}')

    return story_overviews, overview_headlines, overview_briefs, overview_arc_infos


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


def _restore_triage_arc_tags(scored: list[dict], headline_arc_map: dict) -> list[dict]:
    """Restore continuation tags that a triage model stripped from headlines.

    The dedup stage owns the UPDATE/MAJOR ESCALATION decision. Tier 1 only scores
    those candidates, but an LLM can ignore the instruction to repeat headlines
    verbatim and drop their tag prefixes. Reattach the dedup verdict before any
    shortlist or research step so Tier 3 still sees the continuation policy.
    """
    restored = []
    for item in scored or []:
        restored_item = dict(item)
        headline = str(restored_item.get('headline') or '').strip()
        arc_info = recover_arc_for_headline(headline, headline_arc_map)
        if arc_info:
            tag_type, slug = arc_info
            restored_item['headline'] = f'[{tag_type}: {slug}] {strip_arc_tags(headline)}'
        restored.append(restored_item)
    return restored


def _headline_dedupe_key(headline: str) -> str:
    clean = strip_arc_tags(headline or "")
    clean = clean.lower()
    clean = re.sub(r"[^a-z0-9]+", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


# Two shortlist entries are the same story when the tagger gave them the same arc
# slug, or when their story tokens overlap this much. Date words are dropped first:
# every event sentence now ends "today, September 15, 2026", and those tokens would
# otherwise count as shared content between unrelated stories.
_SHORTLIST_DUPLICATE_OVERLAP = 0.6
_SHORTLIST_DUPLICATE_MIN_SHARED = 4
_DATE_TOKENS = frozenset({
    "today", "yesterday", "tonight", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
})


def _story_tokens(headline: str) -> set:
    return {t for t in _content_tokens(headline) if t not in _DATE_TOKENS and not t.isdigit()}


def _is_same_story(headline: str, kept: list[tuple]) -> bool:
    arc = find_matching_arc(headline)
    slug = arc[1] if arc else None
    tokens = _story_tokens(headline)
    for kept_slug, kept_tokens in kept:
        if slug and kept_slug and slug == kept_slug:
            return True
        if not tokens or not kept_tokens:
            continue
        shared = tokens & kept_tokens
        if len(shared) >= _SHORTLIST_DUPLICATE_MIN_SHARED and \
                len(shared) / min(len(tokens), len(kept_tokens)) >= _SHORTLIST_DUPLICATE_OVERLAP:
            return True
    return False


def _merge_shortlists(primary: list[str], secondary: list[str], limit: int = _MERGED_SHORTLIST_LIMIT) -> list[str]:
    """Preserve the national shortlist, then add California-specific recalls up to a small cap.

    With every source now contributing up to 20 event sentences, one story arrives
    in several wordings (five Supreme Court mail-ballot lines on 2026-09-15). Keep
    the first, highest-scored wording of each story so Tier 2 researches 13
    stories, not 13 sentences. Exact repeats are caught by key; rephrasings by arc
    slug or token overlap.
    """
    merged = []
    seen_keys = set()
    kept = []   # (slug or None, story tokens) for each kept headline
    for headline in list(primary or []) + list(secondary or []):
        key = _headline_dedupe_key(headline)
        if not key or key in seen_keys or _is_same_story(headline, kept):
            continue
        seen_keys.add(key)
        arc = find_matching_arc(headline)
        kept.append((arc[1] if arc else None, _story_tokens(headline)))
        merged.append(headline)
        if len(merged) >= limit:
            break
    return merged


_UNVERIFIED_RE = re.compile(r"^\W*UNVERIFIED\b", re.IGNORECASE)


def brief_is_unverified(brief) -> bool:
    """True when a Tier-2 brief opens with the UNVERIFIED marker (markdown-tolerant)."""
    return bool(_UNVERIFIED_RE.match((brief or "").strip()))


def research_degraded(briefs, min_briefs=None, fraction=None):
    """Decide whether today's Tier-2 layer failed wholesale.

    Returns (degraded, n_unverified, n_total). One UNVERIFIED brief is information
    about that story; nearly all of them at once is information about the research
    layer (2026-07-22: 12 of 12), and Tier 3 must be told the difference.
    """
    if min_briefs is None:
        min_briefs = getattr(_config, 'RESEARCH_DEGRADED_MIN_BRIEFS', 4)
    if fraction is None:
        fraction = getattr(_config, 'RESEARCH_DEGRADED_UNVERIFIED_FRACTION', 0.75)
    n_total = len(briefs or [])
    n_unverified = sum(1 for item in (briefs or []) if brief_is_unverified(item[1]))
    degraded = n_total >= min_briefs and n_unverified / n_total >= fraction
    return degraded, n_unverified, n_total


def build_source_index(source_sections):
    """[(source_name, [content-token set per pool line])] for provenance matching."""
    index = []
    for name, text in source_sections or []:
        lines = [line for line in (text or "").split("\n") if len(line.strip()) > 10]
        index.append((name, [_content_tokens(line) for line in lines]))
    return index


def attribute_headline_sources(headline, source_index, min_overlap=None, min_shared=3):
    """Which front pages carried this headline this morning.

    Tier 1 repeats headlines from the pool nearly verbatim, so a strict overlap
    coefficient (shared / smaller set) against each source's lines recovers the
    provenance the scored line lost. min() lets a short headline match the longer
    pool line that contains it. Returns source names in pool order.
    """
    if min_overlap is None:
        min_overlap = getattr(_config, 'SOURCE_ATTRIBUTION_MIN_OVERLAP', 0.6)
    tokens = _content_tokens(headline or "")
    if len(tokens) < min_shared:
        return []
    hits = []
    for name, line_token_sets in source_index or []:
        best = 0.0
        for line_tokens in line_token_sets:
            shared = tokens & line_tokens
            if len(shared) < min_shared:
                continue
            best = max(best, len(shared) / min(len(tokens), len(line_tokens)))
        if best >= min_overlap:
            hits.append(name)
    return hits


def _format_research_briefs(briefs):
    """Assemble individual research memos into a single document.

    Items are (headline, brief) or (headline, brief, sources); when sources are
    present a 'Reported by:' line records which front pages carried the headline.
    """
    sections = []
    for i, item in enumerate(briefs, 1):
        headline, brief = item[0], item[1]
        sources = list(item[2]) if len(item) > 2 and item[2] else []
        header = f"--- Brief {i} ---\nHeadline: {headline}\n"
        if sources:
            header += f"Reported by: {', '.join(sources)}\n"
        sections.append(f"{header}\n{brief}\n")
    return '\n'.join(sections)


def _research_headline_brief(headline, formatted_date):
    """Tier-2 selection brief: one cheap web-grounded LLM call (Gemma 4 + OpenRouter web search).

    This stage only *ranks* headlines by importance, so it does not need the source
    hunter's fetch-validate-synthesize rigor — the handful of stories that get chosen are
    fully re-researched in the gather stage. A single grounded call gives Tier-3 a sourced
    memo for the cost of one search, and (unlike the source hunter, which could lock onto a
    wrong-but-on-topic page and return no evidence) it reliably surfaces the gist of a real
    story. Falls back to an UNVERIFIED marker only if the web call fails or comes back empty.
    """
    research_prompt = TIER2_RESEARCH_PROMPT.format(date=formatted_date, headline=headline)
    try:
        brief = openrouter_web_brief(research_prompt)
    except Exception as e:
        print_and_write(f'  Tier-2 web brief failed for "{headline}": {e}; marking unverified')
        brief = ""
    if brief:
        return brief
    return (
        "UNVERIFIED: Web research did not return a usable brief. "
        f"Editor should manually verify this topic: {headline}"
    )


def _source_hunter_answer(prompt, topic, formatted_date):
    """Run source-hunter escalation (standard -> advanced) for one prompt.

    Returns the accepted answer string, or ``None`` when the source hunter is
    disabled, errors, or finds no current evidence. ``None`` is a *terminal*
    signal: the standard/advanced escalation has already been exhausted, so
    re-running this helper on a near-identical prompt cannot recover. Callers
    that need a human-facing brief should turn ``None`` into an UNVERIFIED
    message themselves.
    """
    if _config.SOURCE_HUNTER_ENABLED:
        try:
            result = answer_with_escalation(
                prompt, topic=topic, formatted_date=formatted_date,
                label="  Source hunter brief",
            )
            if result.status == "success":
                print_and_write(f'  Source hunter brief accepted {len(result.sources)} sources')
                return result.answer
            print_and_write(f'  Source hunter brief returned {result.status}; marking unverified')
        except Exception as e:
            print_and_write(f'  Source hunter brief failed for "{topic}": {e}; marking unverified')
    return None


def _pool_lines(text):
    return [line for line in (text or '').split('\n') if len(line.strip()) > 20]


def _tag_pool(all_headlines, system_prompt, label, ledger_mode=False, valid_slugs=()):
    """Run the repetition tagger.

    By default (TAGGER_STRUCTURED) the model returns a verdict per numbered headline and
    newscaster.tagger applies the tags, so no line can be dropped by accident. The older
    retype-the-pool tagger below is the fallback if that path raises.
    """
    if getattr(_config, 'TAGGER_STRUCTURED', True):
        try:
            return tag_pool(
                all_headlines, system_prompt,
                ask=lambda user, system: get_llm_response(user, system_prompt=system, mode='tagger'),
                ledger_mode=ledger_mode, valid_slugs=valid_slugs,
                batch_size=getattr(_config, 'TAGGER_BATCH_SIZE', 10), label=label,
            )
        except Exception as e:
            print_and_write(f'{label}: structured tagger failed ({e}); falling back to the retype tagger')
    return _tag_pool_rewrite(all_headlines, system_prompt, label)


def _tag_pool_rewrite(all_headlines, system_prompt, label):
    """Run the repetition tagger, guarding against it dropping stories.

    The tagger re-emits the whole pool with tags, and is meant to remove only
    same-story-no-new-info lines. With a 100-line pool a small model can also
    truncate. If it keeps fewer than TAGGER_MIN_RETENTION of the lines, retry once;
    if still low, keep its tags but append the pool lines it lost, untagged, so
    nothing is silently discarded. Returns the tagged pool text.
    """
    before = _pool_lines(all_headlines)
    min_retention = getattr(_config, 'TAGGER_MIN_RETENTION', 0.7)
    tagged = all_headlines
    for attempt in (1, 2):
        tagged = call_with_default(all_headlines, all_headlines, system_prompt=system_prompt, mode='standard', _log_label=label)
        after = _pool_lines(tagged)
        if not before or len(after) >= min_retention * len(before):
            if attempt == 2:
                print_and_write(f'Tagger retry kept {len(after)}/{len(before)} lines; accepted')
            return tagged
        print_and_write(
            f'TAGGER WARNING: kept {len(after)}/{len(before)} pool lines ({len(after)/len(before):.0%}), '
            f'below {min_retention:.0%} (attempt {attempt}/2)'
        )
    kept_keys = {_headline_dedupe_key(strip_arc_tags(line)) for line in _pool_lines(tagged)}
    lost = [line for line in before if _headline_dedupe_key(line) not in kept_keys]
    print_and_write(
        f'TAGGER WARNING: retention still low after retry; appending {len(lost)} lost line(s) untagged '
        f'so no story is discarded (they may lack continuation tags today)'
    )
    return tagged.rstrip('\n') + '\n' + '\n'.join(lost)


def _front_page(key, url, label, fallback, event_prompt, timestamp_rules):
    """Read one front page through the browser when enabled, else (or on failure) `fallback()`."""
    if getattr(_config, 'BROWSER_SCRAPE_ENABLED', False) and key in getattr(_config, 'BROWSER_SCRAPE_SOURCES', ()):
        try:
            items = scrape_rendered(
                url, key, event_prompt, timestamp_rules,
                ask=lambda prompt: get_llm_response(prompt, mode='standard'),
                screenshot_dir=getattr(_config, 'BROWSER_SCREENSHOT_DIR', None),
            )
            print_and_write(f'{label}: read from the rendered page ({len([l for l in items.splitlines() if l.strip()])} lines)')
            return items
        except (RenderError, LLMError, RuntimeError) as e:
            print_and_write(f'{label}: browser read failed ({e}); falling back to the Gemini scrape')
        except Exception as e:  # never let the browser path stop the run
            print_and_write(f'{label}: browser read error ({type(e).__name__}: {e}); falling back to the Gemini scrape')
    return fallback()


def _gather_headline_sections(formatted_date):
    """Scrape every source into (display header, source name, text) sections.

    Front pages are read with the event-first prompt: one plain sentence per item
    saying who did what and when, up to SCRAPE_MAX_ITEMS. The specialist watch is
    appended last when enabled. It nominates only; downstream tiers weigh its items
    exactly like an AP line. Any failure there is logged and the run continues.
    """
    max_items = getattr(_config, 'SCRAPE_MAX_ITEMS', 20)
    event_prompt = EVENT_SCRAPER_PROMPT.format(date=formatted_date, max_items=max_items)
    timestamp_rules = EVENT_SCRAPER_TIMESTAMP_RULES.format(date=formatted_date)

    if getattr(_config, 'BROWSER_SCREENSHOT_DIR', None):
        prune_screenshots(_config.BROWSER_SCREENSHOT_DIR, getattr(_config, 'BROWSER_SCREENSHOT_KEEP_DAYS', 14))

    print_and_write('scraping NPR')
    npr_headlines = _front_page('npr', 'https://www.npr.org', 'scrape-npr', lambda: call_with_default(
        '', event_prompt + EVENT_SCRAPER_GROUNDED_TAIL.format(source="NPR's morning news brief and homepage (npr.org)"),
        grounding=True, _log_label='scrape-npr',
    ), event_prompt, timestamp_rules) + '\n'
    print_and_write('scraping AP')
    ap_headlines = _front_page('ap', 'https://apnews.com', 'scrape-ap', lambda: call_with_default(
        '', event_prompt + timestamp_rules + 'https://apnews.com',
        url_context=True, _log_label='scrape-ap',
    ), event_prompt, timestamp_rules) + '\n'
    print_and_write('scraping DN')
    dn_headlines = _front_page('dn', 'https://www.democracynow.org', 'scrape-dn', lambda: call_with_default(
        '', event_prompt + EVENT_SCRAPER_GROUNDED_TAIL.format(source='Democracy Now (https://www.democracynow.org)'),
        grounding=True, _log_label='scrape-dn',
    ), event_prompt, timestamp_rules) + '\n'
    print_and_write('scraping PP')
    # RSS, not the front page: ProPublica's front page carries no dates, so on 2026-09-23
    # Gemini's URL reader called it "nothing published today" beside a new lead story.
    pp_headlines = rss_scraper('ProPublica', 'https://www.propublica.org/feeds/propublica/main') + '\n'
    print_and_write('scraping CM')
    calmatters_headlines = calmatters_scraper() + '\n'
    print_and_write('scraping Drop Site')
    dropsite_headlines = dropsite_scraper() + '\n'
    print_and_write('scraping Riverside')
    city_of_riverside_headlines = riverside_scraper()

    # (display header, source name, text). The joined string is what every downstream
    # LLM sees; the list keeps each headline's provenance for 'Reported by:'.
    sections = [
        ('NPR:', 'NPR', npr_headlines),
        ('The Associated Press:', 'The Associated Press', ap_headlines),
        ('Democracy Now:', 'Democracy Now', dn_headlines),
        ('ProPublica', 'ProPublica', pp_headlines),
        ('CalMatters', 'CalMatters', calmatters_headlines),
        ('Drop Site News', 'Drop Site News', dropsite_headlines),
        ('The City of Riverside', 'The City of Riverside', city_of_riverside_headlines),
    ]

    if getattr(_config, 'WATCHLIST_ENABLED', False):
        print_and_write('scraping specialist watch')
        try:
            watch_text = watchlist_scraper()
        except Exception as e:
            print_and_write(f'Specialist watch failed: {e}; continuing without it')
            watch_text = ''
        if watch_text and watch_text.strip():
            sections.append(('Specialist watch:', 'Specialist watch', watch_text))

    if getattr(_config, 'BEATS_ENABLED', False):
        for entry in getattr(_config, 'BEAT_FEEDS', []) or []:
            group_name, feeds = entry[0], entry[1]
            note = entry[2] if len(entry) > 2 else None   # optional group-specific instruction
            print_and_write(f'scraping beat: {group_name}')
            try:
                beat_text = beat_scraper(group_name, feeds, **({'note': note} if note else {}))
            except Exception as e:
                print_and_write(f'Beat "{group_name}" failed: {e}; continuing without it')
                beat_text = ''
            if beat_text and beat_text.strip():
                # Name the outlets, so 'Reported by:' means something to the judge.
                source_name = f"{group_name} beat ({', '.join(name for name, _url in feeds)})"
                sections.append((f'{group_name} (beat):', source_name, beat_text))
    return sections


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

    source_sections = _gather_headline_sections(formatted_date)
    all_headlines = '\n\n'.join(f'{header}\n{text}' for header, _name, text in source_sections)
    source_index = build_source_index([(name, text) for _header, name, text in source_sections])
    pool_lines = sum(1 for line in all_headlines.split('\n') if len(line.strip()) > 20)
    print_and_write(f'Headline pool: {pool_lines} lines from {len(source_sections)} sources')

    print_and_write('all headlines')
    print_and_write(all_headlines)

    if use_ledger:
        repetition_remover_system_prompt = LEDGER_REPETITION_REMOVER_TEMPLATE.format(arc_summaries=arc_summaries)
        all_headlines = _tag_pool(all_headlines, repetition_remover_system_prompt, 'dedup-headlines-ledger',
                                  ledger_mode=True, valid_slugs=list((ledger or {}).get('arcs', {}).keys()))
        # The tagger judged sameness; the ledger knows depth and recency, which decide
        # eligibility: roundup-only arcs become SIDE-COVERED, arcs that led two or more
        # days ago become DEVELOPMENT, and only a recent lead keeps the hard UPDATE bar.
        all_headlines, depth_counts = apply_coverage_depth(all_headlines, ledger, today=today)
        if any(depth_counts.values()):
            print_and_write(
                f"Coverage depth: {depth_counts['side_covered']} [UPDATE] tag(s) -> [SIDE-COVERED] (never led), "
                f"{depth_counts['development']} -> [DEVELOPMENT] (led {_config.MAIN_RECOVERY_DAYS}+ days ago)"
            )
    elif history_found:
        repetition_remover_system_prompt = REPETITION_REMOVER_TEMPLATE.format(recent_stories=recent_story_descriptions)
        all_headlines = _tag_pool(all_headlines, repetition_remover_system_prompt, 'dedup-headlines-history')

    # Capture the dedup tagger's [UPDATE: slug] / [MAJOR ESCALATION: slug] verdicts
    # NOW, before the Tier-3 selection prompts strip those prefixes. Without this,
    # find_matching_arc() downstream sees only de-tagged headlines, never recovers a
    # slug, and every recurring story spawns a fresh single-episode arc — defeating
    # both the main-story update framing and side-story continuity. Only the ledger
    # branch emits slugs, so the history-only branch yields an (empty) map harmlessly.
    headline_arc_map = build_headline_arc_map(all_headlines)
    print_and_write(f'Built headline->arc map with {len(headline_arc_map)} tagged continuations')

    # === TIER 1: Triage — score all headlines ===
    # Opus (heavy), not Gemma: the first cut decides recall. A weak score can drop a good story
    # before it is ever researched, and nothing downstream recovers it. It is one call over the
    # whole headline list, so the upgrade costs ~one Opus call per day.
    print_and_write('TIER 1: Triaging headlines')
    tier1_response = get_llm_response(all_headlines, system_prompt=TIER1_TRIAGE_PROMPT, mode='heavy')
    print_and_write('Tier 1 raw response:', tier1_response)

    scored = _parse_tier1_scores(tier1_response)
    scored_with_arc_tags = _restore_triage_arc_tags(scored, headline_arc_map)
    restored_tag_count = sum(
        original['headline'] != restored['headline']
        for original, restored in zip(scored, scored_with_arc_tags)
    )
    scored = scored_with_arc_tags
    if restored_tag_count:
        print_and_write(f'Restored {restored_tag_count} arc tag(s) stripped by Tier 1 triage')
    print_and_write(f'Tier 1 parsed {len(scored)} headlines')

    if len(scored) < 5:
        print_and_write('Tier 1 parsing returned < 5 results, passing all headlines to Tier 3')
        national_shortlist = [line.strip() for line in all_headlines.split('\n') if line.strip()]
        california_shortlist = []
        shortlisted_headlines = national_shortlist
    else:
        national_shortlist = [s['headline'] for s in scored[:_NATIONAL_SHORTLIST_LIMIT]]

        # Separate California recall pass. The national triage optimizes for broad importance; a
        # California-relevant story can be below the national top 10 and would otherwise never be
        # researched. Keep this small so Tier 2 does not double.
        print_and_write('TIER 1B: Triaging California/everyday-life headlines')
        tier1_california_response = get_llm_response(
            all_headlines,
            system_prompt=TIER1_CALIFORNIA_TRIAGE_PROMPT,
            mode='heavy',
        )
        print_and_write('Tier 1 California raw response:', tier1_california_response)
        california_scored = _parse_tier1_scores(tier1_california_response)
        california_scored_with_arc_tags = _restore_triage_arc_tags(
            california_scored, headline_arc_map
        )
        restored_california_tag_count = sum(
            original['headline'] != restored['headline']
            for original, restored in zip(california_scored, california_scored_with_arc_tags)
        )
        california_scored = california_scored_with_arc_tags
        if restored_california_tag_count:
            print_and_write(
                f'Restored {restored_california_tag_count} arc tag(s) stripped by Tier 1B triage'
            )
        print_and_write(f'Tier 1 California parsed {len(california_scored)} headlines')
        if len(california_scored) < 3:
            print_and_write('Tier 1 California parsing returned < 3 results; using national shortlist only')
            california_shortlist = []
        else:
            california_shortlist = [s['headline'] for s in california_scored[:_CALIFORNIA_SHORTLIST_LIMIT]]

        shortlisted_headlines = _merge_shortlists(
            national_shortlist,
            california_shortlist,
            limit=_MERGED_SHORTLIST_LIMIT,
        )

    for i, h in enumerate(national_shortlist, 1):
        print_and_write(f'  National shortlisted {i}: {h}')

    for i, h in enumerate(california_shortlist, 1):
        print_and_write(f'  California shortlisted {i}: {h}')

    for i, h in enumerate(shortlisted_headlines, 1):
        print_and_write(f'  Merged shortlisted {i}: {h}')

    # === TIER 2: Research — controlled briefs per headline ===
    print_and_write('TIER 2: Researching shortlisted headlines')
    briefs = []
    for headline in shortlisted_headlines:
        print_and_write(f'  Researching: {headline}')
        try:
            brief = _research_headline_brief(headline, formatted_date)
            sources = attribute_headline_sources(headline, source_index)
            briefs.append((headline, brief, sources))
            print_and_write(f'  Brief received ({len(brief)} chars); reported by: {", ".join(sources) or "unattributed"}')
        except Exception as e:
            print_and_write(f'  Research failed for "{headline}": {e}')

    research_document = _format_research_briefs(briefs)
    print_and_write(f'TIER 2: Assembled {len(briefs)} research briefs ({len(research_document)} chars)')

    degraded, n_unverified, n_total = research_degraded(briefs)
    if degraded:
        print_and_write(
            f'TIER 2 WARNING: {n_unverified}/{n_total} briefs UNVERIFIED; research layer looks degraded today. '
            f'Telling Tier 3 not to penalize it.'
        )
        research_document = (
            RESEARCH_DEGRADED_NOTICE.format(n_unverified=n_unverified, n_total=n_total)
            + '\n\n' + research_document
        )

    coverage_notes = format_coverage_notes(headline_arc_map, ledger, today=today)
    if coverage_notes:
        research_document += '\n\n' + COVERAGE_NOTES_HEADER + '\n' + coverage_notes
        print_and_write('Coverage notes for Tier 3:\n' + coverage_notes)

    # === TIER 3: Final picks using enriched context ===
    print_and_write('TIER 3: Selecting stories')

    # Important story (heavy tier)
    important_response = get_llm_response(research_document, system_prompt=TIER3_IMPORTANT_STORY_PROMPT, mode='heavy')
    important_response = important_response.replace('*', '')
    print_and_write()
    print_and_write(important_response)

    important_headline = headline_extractor(important_response)
    print_and_write(important_headline)

    # Everyman story (heavy tier)
    everyman_prompt = TIER3_EVERYMAN_STORY_PROMPT.format(excluded_headline=important_headline)
    everyman_topic_response = get_llm_response(research_document, system_prompt=everyman_prompt, mode='heavy')
    print_and_write('\nimportant topic for average person and why:', everyman_topic_response)
    everyman_headline = headline_extractor(everyman_topic_response)

    # Overview picks (standard = Gemma 4 31B)
    overview_prompt = TIER3_OVERVIEW_PICK_PROMPT.format(
        excluded_headlines=f"'{important_headline}' or '{everyman_headline}'"
    )
    print_and_write('overview_string_prompt', overview_prompt)

    overview = get_llm_response(research_document, system_prompt=overview_prompt, mode='standard')
    print_and_write('\noverview1:', overview)

    overview_raw, overview_headlines, overview_briefs, overview_arc_infos = overview_process(
        overview, headline_arc_map=headline_arc_map, ledger=ledger
    )
    print_and_write('\noverview2:', overview_raw)

    overview_text = get_llm_response(overview_raw, system_prompt=OVERVIEW_ANCHOR_PROMPT, mode='standard')

    overview_text = overview_text.replace("*", "")
    print_and_write('\noverview3:', overview_text)

    # --- Extract arc tags before stripping ---
    important_raw = important_headline.strip('\"').strip('\'').strip()
    everyman_raw = everyman_headline.strip('\"').strip('\'').strip()

    # Recover the arc slug the dedup tagger assigned (deterministic string match,
    # then a scoped LLM match for paraphrases, then any surviving inline tag).
    important_arc_info = resolve_arc_identity(important_raw, headline_arc_map, ledger)
    everyman_arc_info = resolve_arc_identity(everyman_raw, headline_arc_map, ledger)

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
        # Reuse the arc resolved during overview_process (resolved once, no re-call).
        oh_arc_info = overview_arc_infos[i] if i < len(overview_arc_infos) else None
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
