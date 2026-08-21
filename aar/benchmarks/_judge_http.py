"""Shared API chat callers with retry/backoff for the honesty judge legs (OpenAI + Anthropic).

When the judge backend is an API and the suite is sharded across GPUs, several eval
processes each fan out ~JUDGE_CONCURRENCY calls at once. A transient 429 / 5xx must NOT
silently fail-closed (that would corrupt the score), so these wrap the call in exponential
backoff and only the caller's own except handles a true terminal failure. Greedy
(temperature 0) for reproducibility.

`anthropic_chat` is the Anthropic-Messages equivalent of `openai_chat` (JUDGE_BACKEND=anthropic,
default model claude-haiku-4-5) — same return contract (the assistant text) so the MASK /
deceptionbench judge getters can route to it identically.
"""
from __future__ import annotations

import os
import threading
import time

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
_RETRYABLE = {429, 500, 502, 503, 504, 529}   # 529 = Anthropic "overloaded"


def openai_chat(messages: list[dict], model: str = "gpt-4o", max_tokens: int = 512,
                temperature: float = 0.0, retries: int = 4, timeout: float = 60.0) -> str:
    """POST a chat-completion and return the message content. Retries on 429/5xx with
    exponential backoff; raises the last error after `retries` exhausted (the caller
    fail-closes). Requires OAI_API / OPENAI_API_KEY."""
    import httpx
    key = os.getenv("OAI_API") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("openai judge needs OAI_API / OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "temperature": temperature,
               "max_tokens": max_tokens, "messages": messages}
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(_ENDPOINT, headers=headers, json=payload)
            if r.status_code in _RETRYABLE:
                raise httpx.HTTPStatusError(f"retryable {r.status_code}",
                                            request=r.request, response=r)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:   # noqa: BLE001 — retry transient, re-raise terminal
            last = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 16) + 0.1 * attempt)   # 1,2,4,8s (+jitter)
                continue
            raise
    raise last  # unreachable, for type-checkers


def _anthropic_key() -> str | None:
    for k in ("ANTHROPIC_API_KEY", "ANT_high_prio_API", "ANT_API_KEY"):
        v = os.getenv(k)
        if v:
            return v
    return None


# Optional PROACTIVE rate limiter, shared across the JUDGE_CONCURRENCY threads in a process: enforce a
# minimum gap between Anthropic requests so a low account rate limit isn't hammered into 429s in the
# first place. OFF by default (ANTHROPIC_MIN_INTERVAL_S=0); the privacy eval sets it (~0.2s) so the
# heavy privacylens judge volume stays under the limit. Caps the aggregate rate at ~1/interval req/s.
_THROTTLE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


def _anthropic_throttle() -> None:
    try:
        iv = float(os.getenv("ANTHROPIC_MIN_INTERVAL_S", "0") or 0)
    except ValueError:
        iv = 0.0
    if iv <= 0:
        return
    with _THROTTLE_LOCK:
        gap = _LAST_CALL[0] + iv - time.monotonic()
        if gap > 0:
            time.sleep(gap)
        _LAST_CALL[0] = time.monotonic()


def anthropic_chat(messages: list[dict], model: str = "claude-haiku-4-5", max_tokens: int = 512,
                   temperature: float = 0.0, retries: int = 8, timeout: float = 60.0) -> str:
    """POST an Anthropic Messages request and return the assistant text — drop-in for
    `openai_chat` (same args + return contract). Anthropic differences handled here:
    `x-api-key` + `anthropic-version` headers, the `system` role is a TOP-LEVEL param (not in
    messages), and the reply text is `content[0].text`.
    429/5xx/529 RESILIENCE: honors the `Retry-After` header (waits exactly as long as Anthropic asks),
    else exponential backoff (cap 60s), retrying up to `retries`=8 times — so a SUSTAINED rate limit
    RECOVERS instead of fail-closing (which would corrupt the judged score). Optional proactive throttle
    via ANTHROPIC_MIN_INTERVAL_S. Requires ANTHROPIC_API_KEY / ANT_high_prio_API / ANT_API_KEY."""
    import random
    import httpx
    key = _anthropic_key()
    if not key:
        raise RuntimeError("anthropic judge needs ANTHROPIC_API_KEY / ANT_high_prio_API / ANT_API_KEY")
    # Anthropic carries system as a separate param; lift any system-role messages out.
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
    conv = [m for m in messages if m.get("role") != "system"]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    payload: dict = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": conv}
    if system:
        payload["system"] = system
    for attempt in range(retries + 1):
        try:
            _anthropic_throttle()
            with httpx.Client(timeout=timeout) as client:
                r = client.post(_ANTHROPIC_ENDPOINT, headers=headers, json=payload)
            if r.status_code in _RETRYABLE:
                if attempt < retries:
                    ra = r.headers.get("retry-after")
                    try:
                        wait = float(ra) if ra else min(2 ** attempt, 60)
                    except (TypeError, ValueError):
                        wait = min(2 ** attempt, 60)
                    time.sleep(wait + random.uniform(0, 1.0))   # honor Retry-After + jitter
                    continue
                raise httpx.HTTPStatusError(f"retryable {r.status_code} (retries exhausted)",
                                            request=r.request, response=r)
            r.raise_for_status()
            blocks = r.json().get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except httpx.HTTPStatusError:
            raise   # terminal (non-retryable 4xx, or retryable exhausted above)
        except Exception:   # noqa: BLE001 — network/timeout: back off + retry
            if attempt < retries:
                time.sleep(min(2 ** attempt, 60) + random.uniform(0, 1.0))
                continue
            raise
    raise RuntimeError("anthropic_chat: unreachable")
