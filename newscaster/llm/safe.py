"""Helper for degradable LLM call sites.

A "degradable" call is one whose failure is non-critical: we'd rather have a
sensible default and keep going than surface the error to the user. Examples:
deduping headlines (fall back to keeping all), generating an arc slug (fall
back to a deterministic slug from the headline).

Critical call sites should NOT use this — they should call get_llm_response
directly and let LLMError propagate.
"""

from newscaster.logging import print_and_write
from newscaster.llm.router import get_llm_response
from newscaster.llm.errors import LLMError


def call_with_default(default, *args, _log_label=None, **kwargs):
    """Call get_llm_response, returning `default` if any LLMError is raised.

    Logs LLM-DEGRADED with the label and the error so failures stay audible.
    """
    try:
        return get_llm_response(*args, **kwargs)
    except LLMError as e:
        label = _log_label or 'unlabeled'
        print_and_write(f"LLM-DEGRADED [{label}]: {e}; using default")
        return default
