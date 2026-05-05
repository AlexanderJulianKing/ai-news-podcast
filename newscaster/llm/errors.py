"""Typed exceptions for LLM provider failures.

All provider adapters (gemini.py, claude.py, openrouter.py) translate their
SDK-specific exceptions into the hierarchy below. The router's retry policy
distinguishes Retryable from NonRetryable to decide whether to retry, fall
back, or re-raise immediately. After retries are exhausted the router raises
LLMRetriesExhaustedError wrapping the last cause.
"""


class LLMError(Exception):
    """Base exception for all LLM provider failures."""

    def __init__(self, message='', *, provider=None, model=None, attempts=None, status_code=None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.attempts = attempts
        self.status_code = status_code

    def __str__(self):
        base = super().__str__()
        parts = []
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.attempts is not None:
            parts.append(f"attempts={self.attempts}")
        suffix = f" [{', '.join(parts)}]" if parts else ''
        return f"{type(self).__name__}: {base}{suffix}"


class LLMRetryableError(LLMError):
    """Failure that may succeed on retry."""


class LLMTimeoutError(LLMRetryableError):
    pass


class LLMRateLimitError(LLMRetryableError):
    """HTTP 429. May carry a retry_after hint (seconds) from the response."""

    def __init__(self, message='', *, retry_after=None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMServerError(LLMRetryableError):
    """HTTP 5xx."""


class LLMTransportError(LLMRetryableError):
    """Network/TLS/socket-level failure."""


class LLMMalformedResponseError(LLMRetryableError):
    """Provider returned a syntactically valid response with no usable content
    (e.g. gemini response.text is None, openrouter empty body, missing text block).
    Treated as retryable because the cause is often transient (model variance,
    backend hiccup)."""


class LLMNonRetryableError(LLMError):
    """Failure that won't succeed on retry; do not waste attempts."""


class LLMAuthError(LLMNonRetryableError):
    """HTTP 401, 403."""


class LLMBadRequestError(LLMNonRetryableError):
    """HTTP 400, 422 — prompt invalid, model name wrong, payload malformed."""


class LLMContentPolicyError(LLMNonRetryableError):
    """Safety / recitation refusal — same prompt will always be refused."""


class LLMRetriesExhaustedError(LLMError):
    """Raised by the router after the central retry policy gives up.
    The original last-encountered exception is available via __cause__."""


def classify(exc, status_code=None):
    """Map an arbitrary provider/SDK exception to the right LLMError subclass.

    Returns the *class*, not an instance — caller raises with provider/model
    context attached. Falls through to LLMTransportError as the safest default
    (retryable) for unknown failures.
    """
    if isinstance(exc, LLMError):
        return type(exc)

    if status_code is not None:
        if status_code in (401, 403):
            return LLMAuthError
        if status_code in (400, 404, 422):
            return LLMBadRequestError
        if status_code == 429:
            return LLMRateLimitError
        if 500 <= status_code < 600:
            return LLMServerError

    name = type(exc).__name__

    timeout_names = {'APITimeoutError', 'DeadlineExceeded', 'Timeout', 'ReadTimeout', 'ConnectTimeout', 'TimeoutError'}
    if name in timeout_names:
        return LLMTimeoutError

    rate_limit_names = {'RateLimitError', 'ResourceExhausted'}
    if name in rate_limit_names:
        return LLMRateLimitError

    server_error_names = {'InternalServerError', 'ServiceUnavailable'}
    if name in server_error_names:
        return LLMServerError

    auth_names = {'AuthenticationError', 'Unauthenticated', 'PermissionDenied', 'PermissionDeniedError'}
    if name in auth_names:
        return LLMAuthError

    bad_request_names = {'BadRequestError', 'InvalidArgument', 'NotFoundError', 'UnprocessableEntityError'}
    if name in bad_request_names:
        return LLMBadRequestError

    transport_names = {'APIConnectionError', 'ConnectionError', 'RequestException', 'HTTPError'}
    if name in transport_names:
        return LLMTransportError

    return LLMTransportError
