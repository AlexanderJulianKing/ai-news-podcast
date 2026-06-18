import json
import random
import time
from typing import Optional, Dict, Any, Tuple, Union, List

import requests

import newscaster.config as _config
from newscaster.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMTransportError,
    classify,
)


def get_openrouter_response(
    prompt: str,
    model: str,
    name: str,
    reasoning: bool,
    system_prompt: Optional[str] = None,
    include_usage: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """One logical attempt against OpenRouter.

    Cycles through transport workarounds and provider routing variants for the
    SAME logical request — that's per-attempt resilience, not retry. Raises a
    typed LLMError on failure; the router decides whether to retry or fall back.

    Pass `tools` to enable server-side tools like openrouter:web_search /
    openrouter:web_fetch.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    provider_candidates = {
        "qwen/qwen3-30b-a3b": ["novita/fp8", "nebius/fp8", "deepinfra/fp8", 'chutes'],
    }
    provider_overrides = {
        "qwen/qwen3-235b-a22b-thinking-2507": "chutes",
        "qwen/qwen3-coder": "cerebras/fp8",
        "qwen/qwen3-235b-a22b-2507": "targon/bf16",
        "meta-llama/llama-4-maverick": "baseten/fp16",
        "deepseek/deepseek-r1-0528": "targon/fp8",
        "deepseek/deepseek-chat-v3-0324": "targon/fp8",
        "deepseek/deepseek-r1": "targon/fp8",
        "deepseek/deepseek-chat": "targon/fp8",
        "meta-llama/llama-3.3-70b-instruct": "novita/bf16",
        "qwen/qwen3-32b": "novita/fp8",
        "openai/gpt-oss-120b": "novita",
        "openai/gpt-oss-20b": "novita",
        "google/gemma-3-27b-it": "deepinfra/bf16",
        "google/gemma-3-12b-it": "deepinfra/bf16",
        "moonshotai/kimi-k2": "chutes/fp8",
        "z-ai/glm-4.5": "chutes/fp8",
        "deepseek/deepseek-chat-v3.1": "fireworks",
    }

    nonthinking_variants = [
        "Gemini 2.5 Flash Lite Preview (2025-06-17) Nonthinking",
        "Gemini 2.5 Flash Preview Nonthinking",
        "Claude 3.7 Sonnet",
        "Claude 4 Opus",
        "Claude 4 Sonnet",
        "Claude Opus 4.1",
        "DeepSeek V3.1 (Non-Reasoning)",
    ]
    high_models = ["o3 High", "o3-Mini High", "o4-Mini High",
                   "Grok 3 Mini Beta (High)", "GPT-5 (high)",
                   "GPT-5 Mini (High)", "GPT-5 Nano (high)"]
    low_models = ["GPT-5 (low)", "GPT-5 Mini (low)", "GPT-5 Nano (low)", "GPT-5.5 (low)"]
    minimal_models = ["GPT-5 (minimal)", "GPT-5 Nano (minimal)", "GPT-5 Mini (minimal)"]

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": reasoning},
        "usage": {"include": True},
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    if name == "Gemini 2.5 Pro Preview (2025-06-05) Limited":
        payload["reasoning"] = {"max_tokens": 8000}
    if isinstance(reasoning, str) and reasoning:
        payload["reasoning"] = {"effort": reasoning}
    if name in nonthinking_variants:
        print(name, 'nonthinking')
        payload["reasoning"] = {"max_tokens": 0, 'enabled': False}
    if name in high_models:
        payload["reasoning"] = {"effort": "high"}
    if name in low_models:
        payload["reasoning"] = {"effort": "low"}
    if name in minimal_models:
        payload["reasoning"] = {"effort": "minimal"}

    forced_provider = provider_overrides.get(model)

    def _summarize_usage(u: Dict[str, Any]) -> Dict[str, Any]:
        comp_details = (u or {}).get("completion_tokens_details") or {}
        prompt_details = (u or {}).get("prompt_tokens_details") or {}
        return {
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": comp_details.get("reasoning_tokens", 0),
            "total_tokens": u.get("total_tokens"),
            "cost": u.get("cost"),
            "cost_details": u.get("cost_details"),
            "prompt_tokens_details": prompt_details,
            "raw": u,
        }

    transport_variants = [
        {"connection_close": False, "identity": False},
        {"connection_close": True, "identity": False},
        {"connection_close": True, "identity": True},
    ]
    prov_list = provider_candidates.get(model)
    if prov_list:
        modes = [("prov", p, False) for p in prov_list]
        modes += [("prov", prov_list[0], True)]
    elif forced_provider:
        modes = [("prov", forced_provider, False), ("prov", forced_provider, True)]
    else:
        modes = []
    modes += [("no-override", None, None)]

    base_headers = {
        "Authorization": f"Bearer {_config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "openbench/1.0",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Parallel-Benchmarker",
    }

    # Track the *kind* of the most recent failure explicitly so the final
    # classifier doesn't get fooled by a stale status from an earlier variant.
    last_exc: Optional[Exception] = None
    last_failure_status: Optional[int] = None
    last_failure_kind: Optional[str] = None  # 'status' | 'transport' | 'malformed'
    # 429s can be provider-specific — preserve the MAX Retry-After we see across
    # variants (the most conservative wait) and let variant cycling try other
    # providers before raising.
    max_429_retry_after: Optional[float] = None

    with requests.Session() as s:
        s.trust_env = False
        s.headers.update(base_headers)

        try:
            time.sleep(random.uniform(0.02, 0.12))

            for tv in transport_variants:
                sess = s
                if tv["connection_close"] or tv["identity"]:
                    sess = requests.Session()
                    sess.trust_env = False
                    hdrs = dict(base_headers)
                    if tv["connection_close"]:
                        hdrs["Connection"] = "close"
                    if tv["identity"]:
                        hdrs["Accept-Encoding"] = "identity"
                    sess.headers.update(hdrs)

                try:
                    for mode in modes:
                        req_payload = dict(payload)
                        kind, prov, allow_fb = mode

                        if kind == "prov":
                            req_payload["provider"] = {"order": [prov], "allow_fallbacks": bool(allow_fb)}

                        try:
                            r = sess.post(url, json=req_payload, timeout=(10, 60))
                        except requests.RequestException as e:
                            last_exc = e
                            last_failure_kind = 'transport'
                            last_failure_status = None
                            continue

                        if not r.ok:
                            try:
                                err_body = r.json()
                                err_msg = err_body.get("error") or err_body
                            except ValueError:
                                err_msg = (r.text or "")[:1000]
                            last_exc = RuntimeError(f"HTTP {r.status_code} ({mode}): {err_msg}")
                            last_failure_kind = 'status'
                            last_failure_status = r.status_code

                            if r.status_code in (401, 403):
                                raise LLMAuthError(
                                    str(err_msg),
                                    provider='openrouter', model=model, status_code=r.status_code,
                                )
                            # 408 (Request Timeout) and 429 (Rate Limit) are both
                            # provider-specific — let variant cycling try other
                            # transports / providers before giving up. The final
                            # classifier raises LLMTimeoutError or LLMRateLimitError
                            # if every variant hits the same wall.
                            if r.status_code == 429:
                                ra_header = r.headers.get('Retry-After')
                                if ra_header:
                                    try:
                                        candidate = float(ra_header)
                                        if max_429_retry_after is None or candidate > max_429_retry_after:
                                            max_429_retry_after = candidate
                                    except ValueError:
                                        pass
                            continue

                        raw = r.content or b""
                        if not raw.strip():
                            last_exc = RuntimeError(
                                f"Whitespace body ({mode}, close={tv['connection_close']}, identity={tv['identity']})"
                            )
                            last_failure_kind = 'malformed'
                            last_failure_status = None
                            continue

                        try:
                            data = r.json()
                        except (json.JSONDecodeError, ValueError):
                            snippet = (r.text or "")[:400]
                            last_exc = RuntimeError(f"JSON decode failed ({mode}): {snippet!r}")
                            last_failure_kind = 'malformed'
                            last_failure_status = None
                            continue

                        msg = (data.get("choices", [{}])[0].get("message") or {})
                        content = msg.get("content")
                        if not content or not content.strip():
                            last_exc = RuntimeError(f"Empty/whitespace content ({mode}): missing choices[0].message.content")
                            last_failure_kind = 'malformed'
                            last_failure_status = None
                            continue

                        if not include_usage:
                            return content.strip()

                        usage = _summarize_usage(data.get("usage", {}))
                        return content.strip(), usage

                finally:
                    if sess is not s:
                        sess.close()
        except requests.RequestException as e:
            raise LLMTransportError(
                str(e), provider='openrouter', model=model,
            ) from e

    # Final classification — based on the *kind* of the last failure, not a
    # potentially-stale status from an earlier variant.
    if last_failure_kind == 'status' and last_failure_status is not None:
        if last_failure_status == 408:
            raise LLMTimeoutError(
                str(last_exc), provider='openrouter', model=model, status_code=last_failure_status,
            )
        if last_failure_status == 429:
            raise LLMRateLimitError(
                str(last_exc), provider='openrouter', model=model,
                status_code=last_failure_status, retry_after=max_429_retry_after,
            )
        if 400 <= last_failure_status < 500 and last_failure_status not in (408, 429):
            raise LLMBadRequestError(
                str(last_exc), provider='openrouter', model=model, status_code=last_failure_status,
            )
        if 500 <= last_failure_status < 600:
            raise LLMServerError(
                str(last_exc), provider='openrouter', model=model, status_code=last_failure_status,
            )

    if last_failure_kind == 'transport':
        raise LLMTransportError(
            str(last_exc) if last_exc else 'Transport error across all variants',
            provider='openrouter', model=model,
        )

    raise LLMMalformedResponseError(
        str(last_exc) if last_exc else 'No response from any transport/provider variant',
        provider='openrouter', model=model,
    )
