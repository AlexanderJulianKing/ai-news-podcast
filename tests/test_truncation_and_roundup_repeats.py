"""Offline regressions for truncated scripts and repeated roundup stories."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest

import newscaster.llm.claude  # noqa: F401
from newscaster.llm.errors import LLMMalformedResponseError
from newscaster.script import segments
from newscaster.scrapers import topic_finder as tf


claude_mod = sys.modules['newscaster.llm.claude']


class _FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def _streaming(fn):
    """A fake messages.stream(**kwargs) that returns fn(**kwargs) as the final message."""
    return lambda **kwargs: _FakeStream(fn(**kwargs))


def _anthropic_reply(reason, text='Complete reply.', output_tokens=12):
    return SimpleNamespace(
        stop_reason=reason,
        content=[SimpleNamespace(type='text', text=text)],
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


def test_claude_rejects_output_cap_and_reports_usage():
    client = SimpleNamespace(messages=SimpleNamespace(stream=_streaming(lambda **kwargs: _anthropic_reply('max_tokens', 'Cut off', 32000))))
    with patch.object(claude_mod.anthropic, 'Anthropic', return_value=client):
        with pytest.raises(LLMMalformedResponseError, match=r'output cap.*32000 output tokens'):
            claude_mod.claude('prompt')


def test_claude_end_turn_and_larger_limit_and_timeout():
    captured = {}

    def make_client(**kwargs):
        captured['timeout'] = kwargs['timeout']
        return SimpleNamespace(messages=SimpleNamespace(stream=_streaming(create)))

    def create(**kwargs):
        captured['create'] = kwargs
        return _anthropic_reply('end_turn')

    with patch.object(claude_mod.anthropic, 'Anthropic', side_effect=make_client):
        assert claude_mod.claude('prompt') == 'Complete reply.'
    assert 'client' not in captured['create']
    assert captured['create']['max_tokens'] == 32000
    assert captured['create']['thinking'] == {'type': 'adaptive'}
    assert captured['create']['extra_body'] == {'output_config': {'effort': 'high'}}
    assert captured['timeout'].read == 600.0
    assert captured['timeout'].connect == 5.0


def test_claude_missing_stop_reason_still_returns_text():
    reply = SimpleNamespace(content=[SimpleNamespace(type='text', text='Okay.')])
    client = SimpleNamespace(messages=SimpleNamespace(stream=_streaming(lambda **kwargs: reply)))
    with patch.object(claude_mod.anthropic, 'Anthropic', return_value=client):
        assert claude_mod.claude('prompt') == 'Okay.'


def test_script_retries_mid_word_then_accepts_reporter_thanks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    truncated = "Grace: What happened?\nEthan: Congress's constitutional power over sp"
    complete = 'Grace: Thank you.\nEthan: Thanks, Grace.'
    with patch.object(segments, 'get_llm_response', side_effect=[truncated] + [complete] * 3) as llm, \
         patch.object(segments.time, 'sleep') as sleep:
        segments.segments_writer({0: 'Story'}, '2026_09_26', ['Ethan'], 'September 26, 2026')
    # The short accepted script is still compared with two additional quality drafts.
    assert llm.call_count == 4
    sleep.assert_called_once_with(2)
    assert (tmp_path / 'output_scripts/2026_09_26_segment_0.txt').read_text() == complete


# The one script that aired cut off (2026-09-26, "...power over sp"). Checked across all 424 aired scripts on the
# Pi that day: it was the only one without terminal punctuation.
KNOWN_TRUNCATED = {'2026_09_26_segment_1.txt'}


def test_all_existing_complete_segment_scripts_pass_terminal_check(tmp_path, monkeypatch):
    """Every aired script on this machine passes the guard, except the known cut-off one, which it must catch."""
    files = sorted((Path(__file__).resolve().parents[1] / 'output_scripts').glob('*_segment_*.txt'))
    if not files:
        pytest.skip('No existing segment scripts in this checkout')
    monkeypatch.chdir(tmp_path)
    for index, path in enumerate(files):
        script = path.read_text(encoding='utf-8')
        reporters = [name for name in ('Ethan', 'Chloe') if f'{name}:' in script]
        assert reporters, path.name
        with patch.object(segments, 'get_llm_response', return_value=script), \
             patch.object(segments.time, 'sleep') as sleep:
            segments.segments_writer({0: 'Story'}, f'test_{index}', [reporters[0]], 'Test day')
        written = (tmp_path / f'output_scripts/test_{index}_segment_0.txt').exists()
        if path.name in KNOWN_TRUNCATED:
            assert sleep.called and not written, path.name
        else:
            sleep.assert_not_called()
            assert written, path.name


MAIN = [
    'The Supreme Court allowed the Trump administration to use its revamped federal voter eligibility database for now.',
    'Trump administration claws back nearly $1 billion in congressionally approved spending',
]
CANDIDATES = [
    'OpenAI AI Agent Breached Australian Government Medicare Statistics Portal, Albanese Says',
    'U.S. 10-Year Treasury Yield Hits Its Highest Level Since 2007',
    'Newsom Declares Statewide Emergency to Prepare for El Niño',
    'Supreme Court Revives Controversial Data System for Citizenship Checks',
    'White House Moves to Cancel Nearly $1 Billion in Congressionally Approved Spending',
]
PICKS = '\n'.join(f'- {headline}' for headline in CANDIDATES)


def test_overview_parser_handles_bullets_numbers_bold_and_tags():
    picks = 'Intro line\n- **[UPDATE] First headline**\n* Second headline\n• Third headline\n1. Fourth headline\n2) Fifth headline\n'
    assert tf._parse_overview_candidates(picks) == [
        'First headline', 'Second headline', 'Third headline', 'Fourth headline', 'Fifth headline'
    ]


def test_overview_drops_repeats_from_batched_llm_and_keeps_order():
    with patch.object(tf, 'get_llm_response', return_value='4, 5') as llm:
        kept = tf._filter_overview_candidates(PICKS, MAIN)
    assert kept == CANDIDATES[:3]
    llm.assert_called_once()
    assert llm.call_args.kwargs['mode'] == 'light'
    assert all(headline in llm.call_args.args[0] for headline in MAIN + CANDIDATES)


def test_overview_backfills_to_five_from_seven_picks():
    extra = ['Regional rail workers agree to contract', 'New satellite maps Antarctic ice loss']
    picks = '\n'.join(f'{i}. {headline}' for i, headline in enumerate(CANDIDATES + extra, 1))
    with patch.object(tf, 'get_llm_response', return_value='4 and 5'):
        assert tf._filter_overview_candidates(picks, MAIN) == CANDIDATES[:3] + extra


def test_overview_lexical_fallback_flags_both_rewordings_only():
    with patch.object(tf, 'get_llm_response', side_effect=RuntimeError('offline')):
        assert tf._filter_overview_candidates(PICKS, MAIN) == CANDIDATES[:3]
    for candidate in CANDIDATES[:3]:
        assert not any(tf._overview_lexical_repeat(candidate, main) for main in MAIN)
    for candidate, main in zip(CANDIDATES[3:], MAIN):
        assert tf._overview_lexical_repeat(candidate, main)


def test_overview_unparseable_same_event_answer_uses_lexical_fallback():
    with patch.object(tf, 'get_llm_response', return_value='unclear'):
        assert tf._filter_overview_candidates(PICKS, MAIN) == CANDIDATES[:3]


def test_overview_lexical_fallback_avoids_previous_day_unrelated_items():
    yesterday_main = [
        'What could Iran’s offer mean for the Strait of Hormuz and nuclear talks',
        'Why California’s Supreme Court ordered a sheriff to return seized ballots',
    ]
    yesterday_side = [
        'Australian cybersecurity agency investigates breach of health systems',
        'Danish intelligence warns of possible limited Russian attack on NATO',
        'California Proposition Forty proposes wealth tax on billionaires',
        'Mortgage rates cross seven percent as Treasury yields rise',
        'Newsom signs Sacramento homelessness coordination bill',
    ]
    assert not any(tf._overview_lexical_repeat(side, main)
                   for side in yesterday_side for main in yesterday_main)


def test_overview_arc_identity_drops_same_slug_even_if_llm_says_none():
    arc_map = {
        'trump administration claws back nearly 1 billion in congressionally approved spending': ('UPDATE', 'trump_grant_cuts'),
        'white house moves to cancel nearly 1 billion in congressionally approved spending': ('UPDATE', 'trump_grant_cuts'),
    }
    with patch.object(tf, 'get_llm_response', return_value='none'):
        assert tf._filter_overview_candidates(PICKS, MAIN, arc_map, {}) == CANDIDATES[:4]


def test_overview_process_with_four_headlines_skips_fifth_extraction():
    headlines = CANDIDATES[:4]
    with patch.object(tf, 'get_llm_response') as llm, \
         patch.object(tf, 'summarize_headline_with_grounding', return_value='Verified brief'):
        text, output_headlines, briefs, arcs = tf.overview_process(
            '\n'.join(f'- {h}' for h in headlines), headlines=headlines
        )
    llm.assert_not_called()
    assert output_headlines == headlines
    assert len(briefs) == len(arcs) == 4
    assert text.count('STORY:') == 4


def test_overview_process_legacy_text_with_four_stories_never_requests_fifth():
    prompts = []

    def extract(_overview, system_prompt, mode):
        prompts.append(system_prompt)
        number = system_prompt.split('story number ')[1].split('.')[0]
        return f'Headline {number}'

    with patch.object(tf, 'get_llm_response', side_effect=extract), \
         patch.object(tf, 'summarize_headline_with_grounding', return_value='Verified brief'):
        _, headlines, _, _ = tf.overview_process('1. A\n2. B\n3. C\n4. D')
    assert len(prompts) == 4
    assert headlines == [f'Headline {i}' for i in range(1, 5)]


# ---- review additions (2026-09-26) ----

def test_overview_candidates_fall_back_to_plain_lines():
    from newscaster.scrapers.topic_finder import _parse_overview_candidates
    reply = ("Here are today's picks:\n"
             "OpenAI AI Agent Breached Australian Government Medicare Statistics Portal\n"
             "U.S. 10-Year Treasury Yield Hits Its Highest Level Since 2007\n"
             "Newsom Declares Statewide Emergency to Prepare for El Niño\n")
    assert _parse_overview_candidates(reply) == [
        "OpenAI AI Agent Breached Australian Government Medicare Statistics Portal",
        "U.S. 10-Year Treasury Yield Hits Its Highest Level Since 2007",
        "Newsom Declares Statewide Emergency to Prepare for El Niño",
    ]


def test_same_event_reply_must_be_numbers_only():
    import pytest
    from newscaster.scrapers.topic_finder import _parse_overview_repeat_numbers
    assert _parse_overview_repeat_numbers("4, 5", 7) == {4, 5}
    assert _parse_overview_repeat_numbers("4 and 5.", 7) == {4, 5}
    assert _parse_overview_repeat_numbers("NONE", 7) == set()
    # a stray number inside an explanation must not drop candidate 1
    with pytest.raises(ValueError):
        _parse_overview_repeat_numbers("4, 5 (5 is the $1 billion cuts)", 7)
