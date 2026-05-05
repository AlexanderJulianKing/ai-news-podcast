"""Regression tests for issues caught in the post-refactor review:
empty-response handling, openrouter classification stickiness, and content-shape exhaustion."""
from unittest.mock import patch, MagicMock

import pytest
import requests

import sys

import newscaster.config as _config
# __init__.py re-exports the `gemini` function name, shadowing the submodule
# reference, so we go through sys.modules to get the actual modules.
import newscaster.llm.openrouter  # noqa: F401 — registers in sys.modules
import newscaster.llm.gemini  # noqa: F401
import newscaster.llm.claude  # noqa: F401
openrouter_mod = sys.modules['newscaster.llm.openrouter']
gemini_mod = sys.modules['newscaster.llm.gemini']
claude_mod = sys.modules['newscaster.llm.claude']

from newscaster.llm.errors import (
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMTransportError,
)


# Stub config keys so module-level reads don't blow up under test.
_config.OPENROUTER_API_KEY = "fake"
_config.GOOGLE_GENAI_API_KEY = "fake"
_config.ANTHROPIC_API_KEY = "fake"


def _http_response(status_code=200, json_data=None, text="", headers=None):
    r = MagicMock()
    r.status_code = status_code
    r.ok = 200 <= status_code < 300
    r.headers = headers or {}
    r.content = (text or "").encode() if text else b""
    r.text = text or ""
    if json_data is not None:
        r.json.return_value = json_data
    else:
        r.json.side_effect = ValueError("no json")
    return r


def _patch_session(post_returns=None, post_side_effect=None):
    """Yield a MagicMock that stands in for requests.Session() across all calls."""
    fake_session = MagicMock()
    fake_session.__enter__.return_value = fake_session
    fake_session.__exit__.return_value = False
    fake_session.headers = {}
    if post_side_effect is not None:
        fake_session.post.side_effect = post_side_effect
    elif post_returns is not None:
        fake_session.post.return_value = post_returns
    return fake_session


# ---- openrouter classification fixes ----

def test_openrouter_400_then_malformed_classifies_as_malformed():
    """Earlier 400 must NOT cause a later malformed-body failure to be classified
    as LLMBadRequestError (which would make it non-retryable)."""
    bad_request = _http_response(
        status_code=400, json_data={"error": {"message": "bad"}},
    )
    empty_content = _http_response(
        status_code=200,
        json_data={"choices": [{"message": {"content": ""}}]},
        text='{"choices":[{"message":{"content":""}}]}',
    )
    # First post → 400; all later posts → 200 with empty content.
    sess = _patch_session(post_side_effect=[bad_request] + [empty_content] * 30)

    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMMalformedResponseError):
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)


def test_openrouter_5xx_classifies_as_server_error():
    server_err = _http_response(
        status_code=503, json_data={"error": {"message": "down"}},
    )
    sess = _patch_session(post_returns=server_err)
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMServerError):
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)


def test_openrouter_transport_only_classifies_as_transport_error():
    """All variants raise RequestException → final classification is LLMTransportError, not malformed."""
    sess = _patch_session(post_side_effect=requests.RequestException("network down"))
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMTransportError):
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)


def test_openrouter_408_eventually_raises_timeout():
    """HTTP 408 should classify as LLMTimeoutError after exhausting variant cycling.
    (Earlier versions raised eagerly on first 408; current behavior records-and-continues
    so other transports can try.)"""
    timeout_resp = _http_response(
        status_code=408, json_data={"error": {"message": "request timeout"}},
    )
    sess = _patch_session(post_returns=timeout_resp)
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMTimeoutError):
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)


def test_openrouter_429_does_not_short_circuit_variant_cycling():
    """A 429 on one provider/transport must not abort the variant loop —
    other providers might not be rate-limited."""
    rate_limited = _http_response(
        status_code=429, json_data={"error": {"message": "rate limited"}},
        headers={'Retry-After': '30'},
    )
    success_resp = _http_response(
        status_code=200,
        json_data={"choices": [{"message": {"content": "ok"}}]},
        text='{"choices":[{"message":{"content":"ok"}}]}',
    )
    sess = _patch_session(post_side_effect=[rate_limited, success_resp])
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        result = openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)
    assert result == "ok"


def test_openrouter_429_all_variants_classifies_as_rate_limit_with_retry_after():
    """If every variant hits 429, classify as LLMRateLimitError and preserve the
    Retry-After value."""
    rate_limited = _http_response(
        status_code=429, json_data={"error": {"message": "rate limited"}},
        headers={'Retry-After': '42'},
    )
    sess = _patch_session(post_returns=rate_limited)
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMRateLimitError) as exc_info:
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)
    assert exc_info.value.retry_after == 42.0


def test_openrouter_429_uses_max_retry_after_across_variants():
    """When variants give different Retry-After hints, the final error preserves the
    LARGEST (most conservative) value — under-waiting risks immediate re-rate-limit."""
    short_wait = _http_response(
        status_code=429, json_data={"error": {"message": "first"}},
        headers={'Retry-After': '5'},
    )
    long_wait = _http_response(
        status_code=429, json_data={"error": {"message": "second"}},
        headers={'Retry-After': '60'},
    )
    medium_wait = _http_response(
        status_code=429, json_data={"error": {"message": "third"}},
        headers={'Retry-After': '20'},
    )
    sess = _patch_session(post_side_effect=[short_wait, long_wait, medium_wait] * 5)
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMRateLimitError) as exc_info:
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)
    assert exc_info.value.retry_after == 60.0


def test_openrouter_408_does_not_short_circuit_variant_cycling():
    """A 408 on the first variant must not abort the variant loop — let connection_close /
    identity / other provider modes try first."""
    timeout_resp = _http_response(
        status_code=408, json_data={"error": {"message": "request timeout"}},
    )
    success_resp = _http_response(
        status_code=200,
        json_data={"choices": [{"message": {"content": "ok"}}]},
        text='{"choices":[{"message":{"content":"ok"}}]}',
    )
    # First call → 408, second call → success (different transport variant).
    sess = _patch_session(post_side_effect=[timeout_resp, success_resp])
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        result = openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)
    assert result == "ok", "408 on one variant should let the next variant succeed"


def test_openrouter_whitespace_only_content_raises_malformed():
    whitespace = _http_response(
        status_code=200,
        json_data={"choices": [{"message": {"content": "   \n\t  "}}]},
        text='{"choices":[{"message":{"content":"   "}}]}',
    )
    sess = _patch_session(post_returns=whitespace)
    with patch.object(openrouter_mod.requests, 'Session', return_value=sess):
        with pytest.raises(LLMMalformedResponseError):
            openrouter_mod.get_openrouter_response("p", "openai/gpt-5.5", "GPT-5.5 (low)", False)


# ---- gemini & claude empty-string responses ----

def test_gemini_empty_text_raises_malformed():
    fake_response = MagicMock()
    fake_response.text = "   \n\t   "
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    with patch.object(gemini_mod.genai, 'Client', return_value=fake_client):
        with pytest.raises(LLMMalformedResponseError):
            gemini_mod.gemini("p", model="gemini-x")


def test_claude_empty_text_block_raises_malformed():
    fake_block = MagicMock()
    fake_block.type = 'text'
    fake_block.text = '   '
    fake_message = MagicMock()
    fake_message.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_message

    with patch.object(claude_mod.anthropic, 'Anthropic', return_value=fake_client):
        with pytest.raises(LLMMalformedResponseError):
            claude_mod.claude("p", model_to_use="claude-x")


# ---- segments.py content-shape exhaustion ----

def test_segments_content_shape_exhaustion_skips_slot_instead_of_aborting(tmp_path, monkeypatch):
    """5 consecutive shape-invalid scripts should result in an LLMMalformedResponseError
    that the slot's `except LLMError` catches — instead of an uncaught RuntimeError
    aborting the whole script-writing stage."""
    import os
    from newscaster.script.segments import segments_writer

    monkeypatch.chdir(tmp_path)
    os.makedirs("output_scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    stories = {0: "story content"}
    voices_list = ['Ethan', 'Chloe']
    bad_script = "Random text without the expected speakers"

    with patch('newscaster.script.segments.get_llm_response', return_value=bad_script):
        # No exception should escape — the LLMMalformedResponseError from
        # _fetch_dialogue is caught by the slot's except LLMError.
        segments_writer(stories, "2026_11_05", voices_list, "November 5, 2026", arc_context=[None])

    assert not os.path.exists("output_scripts/2026_11_05_segment_0.txt")
