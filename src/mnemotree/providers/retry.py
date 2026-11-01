"""Retry policy for the OpenAI-compatible embedder. Needs tenacity (embeddings extra)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from tenacity import RetryCallState
from tenacity.wait import wait_base

# Longest wait a provider may request through Retry-After or a reset header.
_MAX_SERVER_WAIT = 60.0


def retryable(exc: BaseException) -> bool:
    """Rate limits, provider outages and dropped connections; other 4xx fail at once."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


class RetryAfterOrBackoff(wait_base):
    """Wait as long as a 429 asks (capped at 60 s), else back off.

    Honours ``Retry-After`` as delta-seconds or an HTTP-date (RFC 7231), then
    ``x-ratelimit-reset-requests``/``x-ratelimit-reset-tokens`` only when they are
    plain seconds. Negative, non-finite or unparseable values use the fallback.
    """

    def __init__(
        self,
        fallback: wait_base,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.fallback = fallback
        self.now = now

    def __call__(self, retry_state: RetryCallState) -> float:
        import httpx

        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            seconds = _server_wait(exc.response.headers, self.now)
            if seconds is not None:
                return seconds
        return self.fallback(retry_state)


def _server_wait(headers: Mapping[str, str], now: Callable[[], datetime]) -> float | None:
    value = headers.get("retry-after", "").strip()
    seconds = _retry_after(value, now) if value else None
    if seconds is None:
        resets = [
            _seconds(headers.get(name, "").strip())
            for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
        ]
        valid = [reset for reset in resets if reset is not None]
        seconds = max(valid) if valid else None
    return None if seconds is None else min(seconds, _MAX_SERVER_WAIT)


def _retry_after(value: str, now: Callable[[], datetime]) -> float | None:
    try:
        return _valid(float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return _valid((when - now()).total_seconds())


def _seconds(value: str) -> float | None:
    """Plain seconds only; durations like ``6m0s`` or epoch formats are not guessed."""
    try:
        return _valid(float(value))
    except ValueError:
        return None


def _valid(seconds: float) -> float | None:
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
