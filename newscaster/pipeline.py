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
    RAG_REFINE_PROMPT,
)
from newscaster.llm import get_llm_response, call_with_default, LLMError
from newscaster.scrapers.topic_finder import topic_finder, TopicFinderResult
from newscaster.search import search_web
from newscaster.scrapers.topic_finder import result_piper
from newscaster.dedup import update_audience_learned, save_ledger
from newscaster.weather import get_daily_temp
from newscaster.script.headlines import story_gatherer, headline_maker
from newscaster.script.intro import intro_writer
from newscaster.script.segments import segments_writer
from newscaster.rag.indexer import index_day, build_research_record
from newscaster.rag.retrieve import retrieve_prior_research
from newscaster.research_agent import run_adaptive_research
from newscaster.source_hunter import answer_with_escalation
# Audio modules are lazy-imported in generate_audio() to avoid pulling
# google.cloud.texttospeech / pydub at module load time (those imports break
# in some test environments and aren't needed for the gather/script stages).


def _run_follow_up_rounds(summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text,
                          followups=None, topic=None, formatted_date=None):
    """Run multiple follow-up question rounds to enrich a story summary.

    Each round: generate a follow-up question, answer it through the controlled
    source-hunter apparatus, and append both to the summary. Runs through light,
    standard, plus, and heavy question-generation modes.
    """
    rounds = [
        (follow_up_prompt_text, 'light', 'Gemini Flash Lite'),
        (challenging_follow_up_prompt_text, 'light', 'Gemini Flash Lite'),
        (follow_up_prompt_text, 'standard', 'Gemma 4 31B'),
        (challenging_follow_up_prompt_text, 'standard', 'Gemma 4 31B'),
        (follow_up_prompt_text, 'plus', 'Gemini Pro 3.1'),
        (challenging_follow_up_prompt_text, 'plus', 'Gemini Pro 3.1'),
        (follow_up_prompt_text, 'heavy', 'Claude Opus 4.8'),
        (challenging_follow_up_prompt_text, 'heavy', 'Claude Opus 4.8'),
    ]

    for prompt_template, mode, asker_name in rounds:
        try:
            follow_up_creator_response = get_llm_response(summary_prompt, system_prompt=prompt_template, mode=mode)
        except LLMError as e:
            print_and_write(f'follow-up question creation failed ({asker_name}): {e}; skipping this round')
            continue

        if mode == 'heavy':
            print_and_write('heavy follow_up_creator_response', follow_up_creator_response)

        if "\"" in follow_up_creator_response or "\'" in follow_up_creator_response:
            follow_up_question = find_quoted_strings(follow_up_creator_response)[0]
        else:
            follow_up_question = follow_up_creator_response

        is_challenging = (prompt_template == challenging_follow_up_prompt_text)
        label = 'challenging follow_up_question' if is_challenging else 'follow_up_question'
        print_and_write(label, follow_up_question)
        try:
            response = _answer_research_question(follow_up_question, topic, formatted_date)
        except LLMError as e:
            print_and_write(f'follow-up source-hunter answer failed ({asker_name}): {e}; skipping this round')
            continue
        print_and_write(response)

        summary_prompt = summary_prompt + '\n' + f'{asker_name} asked:' + follow_up_question + '\nControlled source hunter reported:\n' + response
        if followups is not None:
            followups.append({
                "asker": asker_name,
                "question": follow_up_question,
                "answer": response,
                "challenging": is_challenging,
            })

    return summary_prompt


def _answer_research_question(question, topic=None, formatted_date=None):
    """Answer a research question with source-hunter standard, then advanced."""
    import newscaster.config as _config
    if not _config.SOURCE_HUNTER_ENABLED:
        raise LLMError("source hunter is disabled for research question answering")

    result = answer_with_escalation(question, topic=topic, formatted_date=formatted_date)
    if result.status == "success":
        return result.answer
    raise LLMError(
        f"source hunter could not answer research question (status={result.status})"
    )


def _manifest_path(formatted_date2):
    return "segment_summaries/{}_GATHER_MANIFEST.json".format(formatted_date2)


class _ManifestCorruptError(Exception):
    """Raised when a manifest is unreadable but slot summaries already exist on disk —
    in that case silently treating the manifest as absent would re-run topic_finder
    and pair newly-picked topics with stale slot summaries. Better to fail loudly."""


def _atomic_write_text(path, content):
    """Write text to `path` atomically: write to a sibling tmp file, fsync, then
    os.replace. A crash mid-write leaves either the old file or no file —
    never a half-written truncated artifact."""
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _save_manifest(tf_result, formatted_date2):
    """Persist tf_result to disk atomically so partial reruns can reuse the
    original topic selection without risking a half-written file."""
    if not isinstance(tf_result, TopicFinderResult):
        return
    from dataclasses import asdict
    payload = json.dumps(asdict(tf_result), ensure_ascii=False, indent=2)
    _atomic_write_text(_manifest_path(formatted_date2), payload)


def _existing_slot_summary_paths(formatted_date2):
    import glob as _glob
    return _glob.glob("segment_summaries/{}_segment*_summary.txt".format(formatted_date2))


def _load_manifest(formatted_date2):
    """Load and reconstitute a TopicFinderResult from disk.

    Returns None if the manifest is genuinely absent AND no slot summaries exist
    (a truly fresh state). Raises _ManifestCorruptError if any of these hold while
    slot summaries DO exist on disk — silently treating those cases as absent
    would re-run topic_finder and orphan stale summaries against new picks:
      - manifest file missing
      - manifest unreadable (JSON decode error, IO error)
      - manifest schema doesn't match TopicFinderResult fields
      - manifest top-level isn't a dict (e.g. naked list)
    """
    path = _manifest_path(formatted_date2)
    has_slot_summaries = bool(_existing_slot_summary_paths(formatted_date2))

    if not os.path.exists(path):
        if has_slot_summaries:
            raise _ManifestCorruptError(
                f"Manifest at {path} is missing but slot summaries already exist — "
                f"refusing to silently re-pick topics. Delete the slot summaries (and any matching _research.json sidecars) to force a clean rerun."
            )
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        if has_slot_summaries:
            raise _ManifestCorruptError(
                f"Manifest at {path} is corrupt ({e}) but slot summaries already exist — "
                f"refusing to silently re-pick topics. Delete the slot summaries (and any matching _research.json sidecars) to force a clean rerun, "
                f"or repair the manifest."
            ) from e
        print_and_write(f"Failed to load gather manifest at {path}: {e}; treating as absent (no slot summaries to orphan)")
        return None

    if not isinstance(payload, dict):
        msg = f"Manifest at {path} has wrong top-level type ({type(payload).__name__}); expected dict"
        if has_slot_summaries:
            raise _ManifestCorruptError(msg + " — refusing to silently re-pick topics. Delete slot summaries (and any matching _research.json sidecars) to force a clean rerun.")
        print_and_write(msg + "; treating as absent (no slot summaries to orphan)")
        return None

    # Side story briefs were tuples; JSON round-trips them as lists.
    briefs = payload.get('side_story_briefs') or []
    payload['side_story_briefs'] = [tuple(b) if isinstance(b, list) and len(b) == 2 else b for b in briefs]
    try:
        return TopicFinderResult(**payload)
    except TypeError as e:
        if has_slot_summaries:
            raise _ManifestCorruptError(
                f"Manifest schema mismatch at {path} ({e}) but slot summaries already exist — "
                f"refusing to silently re-pick topics. Delete the slot summaries (and any matching _research.json sidecars) to force a clean rerun."
            ) from e
        print_and_write(f"Manifest schema mismatch at {path}: {e}; treating as absent (no slot summaries to orphan)")
        return None


def gather_news(formatted_date, formatted_date2):
    """Stage 1: Scrape news, search for details, and write segment summaries.

    Idempotent across reruns:
    - GATHER_COMPLETE marker present → skip; load and return the manifest so
      downstream stages still see arc_context / ledger / side_story_briefs.
    - Marker absent but manifest present → reuse the original topic selection
      (don't re-run topic_finder, which is expensive AND non-deterministic, so
      a re-pick could orphan existing slot summaries).
    - Marker absent and manifest absent → fresh run.
    Within the topic loop, slots whose summary file already exists are reused
    rather than re-gathered.
    """
    marker_path = "segment_summaries/{}_GATHER_COMPLETE.flag".format(formatted_date2)

    if os.path.exists(marker_path):
        print_and_write(f"Gather already complete for {formatted_date2} (marker exists); loading manifest")
        manifest = _load_manifest(formatted_date2)
        if manifest is None:
            print_and_write(f"WARNING: marker present but manifest missing/corrupt; downstream arc_context unavailable")
        # Self-heal: a prior run may have written research sidecars but failed to index
        # them (e.g. a transient embedding-API error) before the marker was written.
        # index_day is idempotent and skips already-indexed chunks, so retrying here is
        # cheap and recovers the corpus on the next (same-day) run.
        try:
            index_day(formatted_date2)
        except Exception as e:
            print_and_write(f"RAG index_day (marker-present retry) failed for {formatted_date2}: {e}; continuing")
        return manifest

    # Try to reuse a previous run's topic selection before re-paying for topic_finder.
    tf_result = _load_manifest(formatted_date2)
    if tf_result is not None:
        print_and_write(f"Reusing topic selection from manifest at {_manifest_path(formatted_date2)}")
    else:
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

    # Persist the manifest as soon as topics are pinned, so a crash mid-loop still
    # leaves a recoverable state for the next run.
    _save_manifest(tf_result, formatted_date2)

    stories: dict[int, str] = {}
    slot_records: dict[int, tuple] = {}  # slot -> (articles, followups)

    arc_context = getattr(tf_result, "arc_context", None) or []

    for topic_index, topic in enumerate(topics):
        topic_OG = topic
        existing_summary = "segment_summaries/{}_segment{}_summary.txt".format(formatted_date2, topic_index)
        if os.path.exists(existing_summary):
            with open(existing_summary, 'r', encoding='utf-8') as f:
                existing_text = f.read()
            if existing_text.strip():
                stories[topic_index] = existing_text
                print_and_write(f"Slot {topic_index} already has a summary on disk; reusing (skipping gather)")
                continue
            # Empty/whitespace summary file — treat as absent and re-gather.
            print_and_write(f"Slot {topic_index} summary on disk is empty/whitespace; will re-gather")
        articles, followups = [], []
        try:
            stories[topic_index] = _gather_one_topic(
                topic, topic_index, formatted_date, formatted_date2,
                follow_up_prompt_text, challenging_follow_up_prompt_text,
                articles=articles, followups=followups,
            )
        except LLMError as e:
            print_and_write(
                f'GATHER FAILURE: topic "{topic_OG}" (slot {topic_index}) failed: {e}; '
                f'slot will be empty and skipped downstream'
            )
            continue
        slot_records[topic_index] = (articles, followups)

    for i, summary_text in stories.items():
        _atomic_write_text(
            "segment_summaries/{}_segment{}_summary.txt".format(formatted_date2, i),
            summary_text,
        )

    # Write per-slot research sidecars (provenance + Q&A) for freshly-gathered slots.
    for slot, (articles, followups) in slot_records.items():
        arc = arc_context[slot] if slot < len(arc_context) else None
        arc_slug = arc.get("slug") if isinstance(arc, dict) else None
        topic_str = topics[slot] if slot < len(topics) else ""
        record = build_research_record(
            formatted_date2, slot, topic_str, arc_slug, articles, followups
        )
        _atomic_write_text(
            "segment_summaries/{}_segment{}_research.json".format(formatted_date2, slot),
            json.dumps(record, ensure_ascii=False, indent=2),
        )

    # Index the day's research (non-critical: never let it break gather).
    try:
        indexed = index_day(formatted_date2)
        print_and_write(f"Indexed {indexed} research chunks for {formatted_date2}")
    except Exception as e:
        print_and_write(f"RAG index_day failed for {formatted_date2}: {e}; continuing")

    _atomic_write_text(
        "output_scripts/{}_overview.txt".format(formatted_date2),
        overview,
    )

    # Only mark gather complete if at least one slot was actually gathered.
    # Otherwise an all-failed transient becomes sticky: the next run sees the
    # marker, returns the manifest, and write_scripts then loops forever
    # against zero summaries (it can't write SCRIPTS_COMPLETE either).
    if stories:
        _atomic_write_text(marker_path, formatted_date2)
    else:
        print_and_write(
            f"GATHER WARNING: zero topics produced summaries for {formatted_date2}; "
            f"NOT writing GATHER_COMPLETE marker so the next run will retry"
        )

    return tf_result


def _augment_with_prior_research(super_summary, formatted_date2):
    """Fold dated prior coverage into the draft summary. Gated + fail-safe:
    returns the un-augmented draft when disabled, on no hits, or on any error."""
    import newscaster.config as _config
    if not _config.RAG_AUGMENT_ENABLED:
        return super_summary
    try:
        hits = retrieve_prior_research(super_summary, exclude_date=formatted_date2)
        if not hits:
            return super_summary
        context = "\n\n".join(
            f"[Prior coverage — {h.outlet or 'unknown'}, {h.date}]\n{h.text}" for h in hits
        )
        user_prompt = (
            f"TODAY'S SYNTHESIS:\n{super_summary}\n\n"
            f"BACKGROUND FROM PRIOR COVERAGE:\n{context}"
        )
        enriched = get_llm_response(user_prompt, system_prompt=RAG_REFINE_PROMPT, mode="standard")
        print_and_write(f"RAG: augmented summary with {len(hits)} prior-coverage chunks")
        return enriched
    except Exception as e:
        print_and_write(f"RAG augment failed: {e}; using un-augmented summary")
        return super_summary


def _gather_one_topic(topic, topic_index, formatted_date, formatted_date2,
                     follow_up_prompt_text, challenging_follow_up_prompt_text,
                     articles=None, followups=None):
    """Gather a single topic into a super_summary. Raises LLMError on terminal failure
    so the caller can isolate the slot. Returns the synthesized summary text."""
    topic_OG = topic
    successful_summary_counter = 0
    waiting_time = [0, 5, 60, 60, 300, 600]
    search_results = []
    for waiting_duration in waiting_time:
        time.sleep(waiting_duration)
        try:
            search_results = search_web(topic, 9)
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
        summary_prompt, successful_summary_counter = result_piper(summary_prompt, successful_summary_counter, topic, result, topic_index, formatted_date2, articles=articles)

        if successful_summary_counter == 3:
            break

    if successful_summary_counter < 1:
        search_results = []
        for waiting_duration in waiting_time:
            time.sleep(waiting_duration)
            try:
                search_results = search_web(topic, 9, days_prior=3)
                break
            except Exception as e:
                print_and_write(f'Google search failed: {e}')
        for result in search_results:
            print_and_write(result)
            try:
                summary_prompt, successful_summary_counter = result_piper(summary_prompt, successful_summary_counter, topic, result, topic_index, formatted_date2, articles=articles)
            except Exception as e:
                print_and_write(f'result_piper failed for topic {topic}: {e}')
                pass
            if successful_summary_counter == 3:
                break

    if successful_summary_counter < 1:
        rephrased = call_with_default(
            topic_OG,
            topic,
            system_prompt='Please change the following query into a rephrased google search query that can get more relevant results. Please only give the new query and do not give quotation marks.',
            _log_label=f'rephrase-query[slot={topic_index}]',
        )
        topic = rephrased.strip("\"").strip('\'').strip('\n')
        print_and_write('new topic:', topic)
        search_results = []
        for waiting_duration in waiting_time:
            time.sleep(waiting_duration)
            try:
                search_results = search_web(topic, 9)
                break
            except Exception as e:
                print_and_write(f'Google search failed: {e}')
        for result in search_results:
            print_and_write(result)
            try:
                summary_prompt, successful_summary_counter = result_piper(summary_prompt, successful_summary_counter, topic, result, topic_index, formatted_date2, articles=articles)
            except Exception as e:
                print_and_write(f'result_piper failed for topic {topic}: {e}')
                pass
            if successful_summary_counter == 3:
                break

    if successful_summary_counter < 1:
        from newscaster.llm.errors import LLMRetriesExhaustedError
        raise LLMRetriesExhaustedError(
            f'topic "{topic_OG}" produced 0 articles after all search retries',
            provider='gather', model=f'slot={topic_index}',
        )

    topic = topic_OG
    perplexity_prompt = 'Tell me about this story with as much detail as possible:\n' + topic
    import newscaster.config as _config
    if _config.SOURCE_HUNTER_ENABLED:
        try:
            seed_result = answer_with_escalation(
                perplexity_prompt, topic=topic, formatted_date=formatted_date,
                label="Source hunter seed",
            )
            if seed_result.status == "success":
                summary_prompt = (
                    summary_prompt
                    + "\n\nControlled source-hunter seed context:\n"
                    + seed_result.answer
                )
            else:
                print_and_write(
                    f"Source hunter seed returned {seed_result.status}; "
                    "continuing without seed context"
                )
        except Exception as e:
            print_and_write(f"Source hunter seed failed: {e}; continuing without seed context")
    else:
        print_and_write("Source hunter disabled; continuing without grounded seed context")

    # The research agent's only research tool is the controlled source hunter, so it
    # requires both flags. With the hunter off the agent would otherwise collapse to
    # early termination, so fall back to the fixed follow-up rounds instead.
    if _config.AGENTIC_RESEARCH_ENABLED and _config.SOURCE_HUNTER_ENABLED:
        try:
            adaptive_result = run_adaptive_research(
                topic,
                topic_index,
                formatted_date,
                formatted_date2,
                summary_prompt,
                successful_summary_counter,
                articles=articles,
                followups=followups,
            )
            summary_prompt = adaptive_result.summary_prompt
            successful_summary_counter = adaptive_result.successful_summary_counter
            print_and_write(
                f"Research agent completed slot {topic_index}: {len(adaptive_result.followups)} follow-ups, "
                f"{len(adaptive_result.articles)} articles, reason: {adaptive_result.done_reason or 'complete'}"
            )
        except Exception as e:
            print_and_write(
                f"Research agent failed for slot {topic_index}: {e}; "
                f"falling back to fixed follow-up rounds"
            )
            summary_prompt = _run_follow_up_rounds(
                summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text,
                followups=followups, topic=topic, formatted_date=formatted_date,
            )
    else:
        summary_prompt = _run_follow_up_rounds(
            summary_prompt, follow_up_prompt_text, challenging_follow_up_prompt_text,
            followups=followups, topic=topic, formatted_date=formatted_date,
        )

    print_and_write('SUPER SUMMARY LENGTH:', len(summary_prompt))

    super_summary = get_llm_response(summary_prompt, system_prompt=summary_prompt_alpha, mode='standard')
    print_and_write('SUPER SUMMARY ITERATION 1:', super_summary, '\n')

    irrelevance_full_prompt = irrelevance_prompt + super_summary
    irrelevance_answer = call_with_default(
        'no',
        irrelevance_full_prompt,
        _log_label=f'irrelevance-check[slot={topic_index}]',
    )
    print_and_write('IRRELEVANT INFORMATION?:' + irrelevance_answer)
    if 'yes' in irrelevance_answer.lower():
        summary_prompt2_full = summary_prompt2 + super_summary
        refined = call_with_default(
            super_summary,
            summary_prompt2_full,
            mode='standard',
            _log_label=f'super-summary-refinement[slot={topic_index}]',
        )
        super_summary = refined
        print_and_write('SUPER SUMMARY ITERATION 2:', super_summary, '\n')

    super_summary = _augment_with_prior_research(super_summary, formatted_date2)
    return super_summary


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
    """Stage 2: Generate dialogue scripts from segment summaries.

    Idempotent: gated on a completion marker. segments_writer itself will skip
    individual slots whose script files already exist.
    """
    marker_path = "output_scripts/{}_SCRIPTS_COMPLETE.flag".format(formatted_date2)
    if os.path.exists(marker_path):
        print_and_write(f"Scripts already complete for {formatted_date2} (marker exists); skipping")
        return

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

    # Only mark scripts complete if at least one of THIS run's slots produced
    # a segment script. Check explicit slot indices rather than a wildcard glob
    # so we don't count orphan/legacy files from prior runs as success.
    produced_count = sum(
        1 for slot in stories
        if os.path.exists('output_scripts/{}_segment_{}.txt'.format(formatted_date2, slot))
    )
    if produced_count > 0:
        _atomic_write_text(marker_path, formatted_date2)
    else:
        print_and_write(
            f"SCRIPTS WARNING: no segment scripts produced for any slot in {sorted(stories.keys())}; "
            f"NOT writing SCRIPTS_COMPLETE marker so the next run will retry"
        )


def generate_audio(formatted_date2, voices_list):
    """Stage 3: Synthesize speech, add intro music, and assemble final podcast."""
    from newscaster.audio.tts import text2speech
    from newscaster.audio.intro_music import fun_intromaker
    from newscaster.audio.assembly import assemble_podcast

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
