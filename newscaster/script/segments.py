import os
import time
from datetime import datetime

from newscaster.llm import get_llm_response, LLMError
from newscaster.llm.errors import LLMMalformedResponseError
from newscaster.logging import print_and_write
from newscaster.dates import format_spoken_date
from newscaster.prompts import SEGMENT_SCRIPT_PROMPT_TEMPLATE, SEGMENT_SCRIPT_UPDATE_CONTEXT


def segments_writer(stories, formatted_date2, voices_list, formatted_date, arc_context=None):
    """stories: dict[int, str] keyed by slot. Failed slots are absent and skipped."""
    MAX_VARIANTS = 3
    MAX_CONTENT_RETRIES = 5
    FIRST_PASS_THRESHOLD = 0.55
    attribution_phrases = (
        "according to",
        "reporting from",
        "as reported by",
        "has uncovered",
        "investigation by"
    )

    def _fetch_dialogue(story_text, prompt, reporter_name):
        """Content-shape retry: re-prompt if the LLM returns a script missing
        the expected Grace/reporter lines. Network/transport failures are
        already retried by the router — those raise LLMError here, which we
        propagate so the caller can decide what to do with this segment."""
        attempt = 0
        while attempt < MAX_CONTENT_RETRIES:
            attempt += 1
            response = get_llm_response(story_text, system_prompt=prompt, mode='heavy')

            if 'Grace:' not in response:
                print_and_write(f'Missing Grace line for reporter {reporter_name} (attempt {attempt}); retrying')
                time.sleep(2)
                continue
            if f'{reporter_name}:' not in response:
                print_and_write(f'Missing reporter line ({reporter_name}) in response (attempt {attempt}); retrying')
                time.sleep(2)
                continue
            return response

        raise LLMMalformedResponseError(
            f'Unable to obtain content-valid script for reporter {reporter_name} after {MAX_CONTENT_RETRIES} attempts',
            provider='script-shape', model=reporter_name,
        )

    def _normalize_dialogue(text):
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ':' not in line:
                continue
            speaker, dialogue = line.split(':', 1)
            speaker = speaker.strip()
            dialogue = dialogue.strip().replace("\u2019", "'")
            lines.append(f'{speaker}: {dialogue}')
        return '\n'.join(lines)

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

        segment_script_prompt = SEGMENT_SCRIPT_PROMPT_TEMPLATE.format(
            reporter_name=reporter_name,
            date=formatted_date,
            story_num=narrative_idx + 1,
            total_stories=total_stories
        )

        # Append update context if this story is a continuation
        if arc_context and slot < len(arc_context) and arc_context[slot]:
            arc = arc_context[slot]
            episodes = arc.get("episodes", [])
            if len(episodes) > 1:
                audience_state = arc.get("audience_state", "")
                if audience_state:
                    # Find the previous episode's date for spoken format
                    prev_date_str = episodes[-2]["date"] if len(episodes) >= 2 else episodes[0]["date"]
                    try:
                        prev_date = datetime.strptime(prev_date_str, "%Y_%m_%d").date()
                        last_covered_spoken = format_spoken_date(prev_date)
                    except ValueError:
                        last_covered_spoken = prev_date_str
                    segment_script_prompt += SEGMENT_SCRIPT_UPDATE_CONTEXT.format(
                        audience_state=audience_state,
                        last_covered_spoken=last_covered_spoken,
                        reporter_name=reporter_name,
                    )
                    print_and_write(f'Injected update context for slot {slot} (arc: {arc.get("slug", "?")})')

        quality_records = []
        selected_script = None

        try:
            for variant in range(MAX_VARIANTS):
                raw_script = _fetch_dialogue(story_text, segment_script_prompt, reporter_name)
                dialogue = _normalize_dialogue(raw_script)
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
