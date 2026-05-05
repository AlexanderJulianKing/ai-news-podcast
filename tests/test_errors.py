"""Tests for the LLMError classifier."""
import pytest

from newscaster.llm.errors import (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMServerError,
    LLMTransportError,
    LLMAuthError,
    LLMBadRequestError,
    LLMRetryableError,
    LLMNonRetryableError,
    classify,
)


def _named_exception(name):
    cls = type(name, (Exception,), {})
    return cls()


@pytest.mark.parametrize("status,expected", [
    (401, LLMAuthError),
    (403, LLMAuthError),
    (400, LLMBadRequestError),
    (404, LLMBadRequestError),
    (422, LLMBadRequestError),
    (429, LLMRateLimitError),
    (500, LLMServerError),
    (502, LLMServerError),
    (503, LLMServerError),
])
def test_classify_by_status(status, expected):
    assert classify(RuntimeError("boom"), status_code=status) is expected


@pytest.mark.parametrize("name,expected", [
    ('APITimeoutError', LLMTimeoutError),
    ('DeadlineExceeded', LLMTimeoutError),
    ('TimeoutError', LLMTimeoutError),
    ('ReadTimeout', LLMTimeoutError),
    ('RateLimitError', LLMRateLimitError),
    ('ResourceExhausted', LLMRateLimitError),
    ('InternalServerError', LLMServerError),
    ('ServiceUnavailable', LLMServerError),
    ('AuthenticationError', LLMAuthError),
    ('PermissionDeniedError', LLMAuthError),
    ('Unauthenticated', LLMAuthError),
    ('BadRequestError', LLMBadRequestError),
    ('InvalidArgument', LLMBadRequestError),
    ('NotFoundError', LLMBadRequestError),
    ('UnprocessableEntityError', LLMBadRequestError),
    ('APIConnectionError', LLMTransportError),
    ('ConnectionError', LLMTransportError),
    ('RequestException', LLMTransportError),
])
def test_classify_by_exception_name(name, expected):
    assert classify(_named_exception(name)) is expected


def test_classify_unknown_falls_back_to_transport():
    """Unknown exceptions default to transport (retryable) rather than failing closed."""
    assert classify(RuntimeError("???")) is LLMTransportError


def test_classify_status_overrides_name():
    """A 401 trumps a generic exception name like RuntimeError."""
    assert classify(_named_exception('NotFoundError'), status_code=401) is LLMAuthError


def test_retryable_subclass_relationship():
    assert issubclass(LLMTimeoutError, LLMRetryableError)
    assert issubclass(LLMRateLimitError, LLMRetryableError)
    assert issubclass(LLMServerError, LLMRetryableError)
    assert issubclass(LLMAuthError, LLMNonRetryableError)
    assert issubclass(LLMBadRequestError, LLMNonRetryableError)


def test_error_str_includes_context():
    e = LLMServerError("boom", provider="google", model="gemini-x", status_code=503, attempts=3)
    s = str(e)
    assert "provider=google" in s
    assert "model=gemini-x" in s
    assert "status=503" in s
    assert "attempts=3" in s
