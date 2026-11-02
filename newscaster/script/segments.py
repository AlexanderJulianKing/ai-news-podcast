import os
import re
import time

from newscaster import edition
from newscaster.llm import get_llm_response, LLMError
from newscaster.llm.errors import LLMMalformedResponseError
from newscaster.logging import print_and_write
from newscaster.dates import format_spoken_date
from newscaster.text_utils import extract_producer_notes
from newscaster.prompts import (
    SEGMENT_SCRIPT_PROMPT_TEMPLATE,
    show_time_note,
    SCRIPT_WRITING_RULES,
    SEGMENT_SCRIPT_SIDE_COVERED_CONTEXT,
    SEGMENT_SCRIPT_UPDATE_CONTEXT,
    SEGMENT_SCRIPT_EARLIER_FACTS,
)

MAX_EARLIER_FACTS = 10

_NAME = re.compile(r"\b[A-Z][\w'&.-]*[A-Za-z0-9](?:\s+(?:of\s+|the\s+|de\s+)?[A-Z][\w'&.-]*[A-Za-z0-9])*")
_NOT_NAMES = {"The", "A", "An", "On", "In", "It", "This", "That", "These", "He", "She", "They", "As", "At",
              "By", "For", "According", "After", "While", "However", "But", "And", "Also", "Its", "His", "Her",
              "Their", "AI", "U.S", "US", "January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday",
              "Thursday", "Friday", "Saturday", "Sunday"}


def _names(text):
    out = set()
    for m in _NAME.finditer(text):
        words = m.group(0).strip(".").split()
        while words and words[0] in _NOT_NAMES:
            words = words[1:]
        name = " ".join(words)
        if len(name) >= 3 and name not in _NOT_NAMES:
            out.add(name)
    return out


def _has_word(word, text):
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def earlier_facts(episodes, story_text, formatted_date2, limit=MAX_EARLIER_FACTS):
    """Facts from this thread's earlier episodes that name something in today's brief, as dated lines.

    The thread's audience_state is one paragraph rewritten after every episode, so on a long thread it
    drops specifics: on 2026-11-02 it no longer mentioned Hugging Face, and the writer retold that breach
    as news. A name counts when it is rare in the thread (a name in most facts, like "OpenAI" on the AI
    thread, says nothing about which ones matter) and is a real name, not a word that is capitalized only
    because it starts a sentence ("Security researchers ...").
    """
    facts = [(ep["date"], fact) for ep in episodes or []
             if ep.get("date") != formatted_date2 for fact in ep.get("audience_learned") or []]
    if not facts:
        return []
    lowered = " ".join(f for _, f in facts) + " " + story_text
    names = [{n for n in _names(f) if " " in n or not _has_word(n.lower(), lowered)} for _, f in facts]
    counts = {}
    for found in names:
        for n in found:
            counts[n] = counts.get(n, 0) + 1
    rare = max(3, 0.05 * len(facts))
    scored = []
    for (date, fact), found in zip(facts, names):
        shared = [n for n in found if counts[n] <= rare and _has_word(n, story_text)]
        if shared:
            scored.append((sum(1.0 / counts[n] for n in shared), date, fact))
    best = sorted(scored, key=lambda s: -s[0])[:limit]
    lines = []
    for _, date, fact in sorted(best, key=lambda s: s[1]):
        d = edition.key_date(date)
        when = f"{d:%B} {d.day}" if d else date
        if edition.key_edition(date) == edition.EVENING:
            when += ", evening edition"
        lines.append(f"- {when}: {fact}")
    return lines


def continuation_context_template(episodes, formatted_date2):
    """Pick the continuation framing for a main story with prior coverage.

    Returns (template, kind). A prior full segment gets the "as we reported..."
    update framing. An arc the audience only met in the side-story roundup gets
    the first-full-telling framing, because a sentence in the roundup is not
    background a listener remembers. Today's own episode is ignored: it is the
    main slot being written now.
    """
    prior_main = any(
        (ep or {}).get("coverage") == "main" and (ep or {}).get("date") != formatted_date2
        for ep in episodes or []
    )
    if prior_main:
        return SEGMENT_SCRIPT_UPDATE_CONTEXT, "update"
    return SEGMENT_SCRIPT_SIDE_COVERED_CONTEXT, "side-covered"


def segment_prompt(reporter_name, formatted_date, story_num, total_stories, which=None):
    """The segment-writing prompt for this edition. Without the edition, a morning script opened
    "Our top story tonight" (2026-10-10)."""
    return SEGMENT_SCRIPT_PROMPT_TEMPLATE.format(
        anchor="evening" if edition.is_evening(which) else "morning",
        reporter_name=reporter_name,
        date=formatted_date,
        story_num=story_num,
        total_stories=total_stories,
    ) + show_time_note(which) + SCRIPT_WRITING_RULES


def segments_writer(stories, formatted_date2, voices_list, formatted_date, arc_context=None):
    """stories: dict[int, str] keyed by slot. Failed slots are absent and skipped."""
    MAX_VARIANTS = 3
    MAX_CONTENT_RETRIES = 5
    MAX_STRAY_RETRIES = 2     # then the stray lines are dropped (and logged) instead
    FIRST_PASS_THRESHOLD = 0.55
    attribution_phrases = (
        "according to",
        "reporting from",
        "as reported by",
        "has uncovered",
        "investigation by"
    )

    def _stray_lines(text, reporter_name):
        """Lines no one in the segment speaks: a writer's note, a title, a stage direction."""
        speakers = ('Grace', reporter_name)
        return [line.strip() for line in text.splitlines()
                if line.strip() and line.split(':', 1)[0].strip(' *_') not in speakers]

    def _fetch_dialogue(story_text, prompt, reporter_name, label):
        """Content-shape retry: re-prompt if the LLM returns a script missing
        the expected Grace/reporter lines, ends mid-sentence, or has lines no one speaks.
        Network/transport failures are already retried by the router — those raise
        LLMError here, which we propagate so the caller can decide what to do with this segment."""
        attempt = 0
        while attempt < MAX_CONTENT_RETRIES:
            attempt += 1
            response = extract_producer_notes(get_llm_response(story_text, system_prompt=prompt, mode='heavy'),
                                              where=label)
            stray = _stray_lines(response, reporter_name)
            if stray and attempt <= MAX_STRAY_RETRIES:
                print_and_write(f'{label}: {len(stray)} line(s) no one speaks, e.g. {stray[0][:120]!r} '
                                f'(attempt {attempt}); retrying')
                time.sleep(2)
                continue

            if 'Grace:' not in response:
                print_and_write(f'Missing Grace line for reporter {reporter_name} (attempt {attempt}); retrying')
                time.sleep(2)
                continue
            if f'{reporter_name}:' not in response:
                print_and_write(f'Missing reporter line ({reporter_name}) in response (attempt {attempt}); retrying')
                time.sleep(2)
                continue
            if not _dialogue_looks_finished(response, reporter_name):
                print_and_write(f'Unfinished dialogue for reporter {reporter_name} (attempt {attempt}); retrying')
                time.sleep(2)
                continue
            return response

        raise LLMMalformedResponseError(
            f'Unable to obtain content-valid script for reporter {reporter_name} after {MAX_CONTENT_RETRIES} attempts',
            provider='script-shape', model=reporter_name,
        )

    def _normalize_dialogue(text, reporter_name=None):
        """'Name: text' lines only, and only for this segment's speakers when reporter_name is given."""
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ':' not in line:
                continue
            speaker, dialogue = line.split(':', 1)
            speaker = speaker.strip(' *_')
            if reporter_name and speaker not in ('Grace', reporter_name):
                print_and_write(f'Dropping a line no one in the segment speaks: {line[:160]!r}')
                continue
            dialogue = dialogue.strip().replace("\u2019", "'")
            lines.append(f'{speaker}: {dialogue}')
        return '\n'.join(lines)

    def _dialogue_looks_finished(text, reporter_name):
        """Check the final spoken line, ignoring non-dialogue source notes."""
        dialogue = _normalize_dialogue(text)
        prefixes = ('Grace:', f'{reporter_name}:')
        spoken_lines = [line for line in dialogue.splitlines() if line.startswith(prefixes)]
        if not spoken_lines:
            return False
        last_text = spoken_lines[-1].split(':', 1)[1].strip()
        return bool(re.search(r'[.!?]["\u201d\'\u2019)\]\*]*$', last_text))

    def _score_dialogue(text, reporter_name):
        if not text:
            return 0.0
        word_count = len(text.split())
        length_score = max(0.0, 1 - abs(word_count - 1500) / 1500)
        lines = [line for line in text.splitlines() if ':' in line]
        grace_lines = sum(1 for line in lines if line.startswith('Grace:'))
        reporter_lines = sum(1 for line in lines if line.startswith(f'{reporter_name}:'))
        dialogue_balance = min(grace_lines, reporter_lines) / max(max(grace_lines, reporter_lines), 1)
        lower_text = text.lower()
        attribution_score = 1.0 if any(phrase in lower_text for phrase in attribution_phrases) else 0.0
        total_score = (length_score * 0.5) + (dialogue_balance * 0.2) + (attribution_score * 0.3)
        return round(total_score, 3)

    os.makedirs('output_scripts', exist_ok=True)

    sorted_slots = sorted(stories.keys())
    total_stories = len(sorted_slots)

    for narrative_idx, slot in enumerate(sorted_slots):
        outfile_name = 'output_scripts/{}_segment_{}.txt'.format(formatted_date2, slot)
        if os.path.exists(outfile_name):
            print_and_write(f'Slot {slot} already has a script on disk; reusing (skipping generation)')
            continue

        story_text = stories[slot]
        reporter_name = voices_list[slot]

        segment_script_prompt = segment_prompt(reporter_name, formatted_date, narrative_idx + 1, total_stories)

        # Append update context if this story is a continuation
        if arc_context and slot < len(arc_context) and arc_context[slot]:
            arc = arc_context[slot]
            episodes = arc.get("episodes", [])
            if len(episodes) > 1:
                audience_state = arc.get("audience_state", "")
                if audience_state:
                    # Find the previous episode's date for spoken format
                    prev_date_str = episodes[-2]["date"] if len(episodes) >= 2 else episodes[0]["date"]
                    prev_date = edition.key_date(prev_date_str)
                    if prev_date is None:
                        last_covered_spoken = prev_date_str
                    elif prev_date == edition.key_date(formatted_date2):
                        last_covered_spoken = "this morning, in today's morning edition"
                    else:
                        last_covered_spoken = format_spoken_date(prev_date) + (
                            ", in the evening edition" if edition.key_edition(prev_date_str) == edition.EVENING else "")
                    context_template, kind = continuation_context_template(episodes, formatted_date2)
                    segment_script_prompt += context_template.format(
                        audience_state=audience_state,
                        last_covered_spoken=last_covered_spoken,
                        reporter_name=reporter_name,
                    )
                    print_and_write(f'Injected {kind} context for slot {slot} (arc: {arc.get("slug", "?")})')
                    aired = earlier_facts(episodes, story_text, formatted_date2) if kind == "update" else []
                    if aired:
                        segment_script_prompt += SEGMENT_SCRIPT_EARLIER_FACTS.format(facts="\n".join(aired))
                        print_and_write(f'Listed {len(aired)} earlier fact(s) the audience heard for slot {slot}')

        quality_records = []
        selected_script = None

        try:
            for variant in range(MAX_VARIANTS):
                raw_script = _fetch_dialogue(story_text, segment_script_prompt, reporter_name,
                                             label=f'{formatted_date2} segment {slot}')
                dialogue = _normalize_dialogue(raw_script, reporter_name)
                score = _score_dialogue(dialogue, reporter_name)
                quality_records.append((score, dialogue))
                print_and_write(f'Segment slot {slot}, attempt {variant + 1}, score {score}')

                if variant == 0 and score >= FIRST_PASS_THRESHOLD:
                    selected_script = dialogue
                    print_and_write(f'First attempt cleared threshold ({score}); skipping extra drafts for slot {slot}')
                    break
        except LLMError as e:
            print_and_write(
                f'SCRIPT FAILURE: slot {slot} dialogue generation failed: {e}; '
                f'slot will have no script and will be skipped during audio assembly'
            )
            continue

        if not selected_script:
            if not quality_records:
                print_and_write(f'SCRIPT FAILURE: slot {slot} produced no candidate scripts; skipping')
                continue
            best_score, selected_script = max(quality_records, key=lambda item: (item[0], len(item[1])))
            print_and_write(f'Selected best scoring draft for slot {slot} with score {best_score}')

        with open(outfile_name, 'w', encoding='utf-8') as outfile:
            outfile.write(selected_script)
