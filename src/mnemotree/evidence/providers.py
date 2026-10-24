"""Optional OpenAI-compatible embedding transport. No LangChain dependency."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_none,
)
from tenacity.wait import wait_base

from .models import EmbeddingSpec

logger = logging.getLogger(__name__)

# Longest wait a provider may request through Retry-After or a reset header.
_MAX_SERVER_WAIT = 60.0


class OpenAIEmbedder:
    """Embeds through an OpenAI-compatible ``/embeddings`` endpoint.

    Rate limiting is per instance (the service holds one embedder per process):
    ``max_concurrency`` bounds the embedding POSTs in flight across *all*
    concurrent ``embed_documents``/``embed_query`` calls on this instance, and a
    429 or 5xx on any batch puts the whole instance into a shared cool-down, so
    no new POST is sent until it ends. Each batch also retries on its own
    (``max_attempts``, Retry-After or exponential backoff); once its next retry
    would bring its waits, including time spent in the shared cool-down, past
    ``max_total_wait`` it stops and raises the last error.
    """

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        revision: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout: float = 60.0,
        send_dimensions: bool = True,
        max_concurrency: int = 8,
        max_attempts: int = 6,
        backoff_seconds: float = 0.5,
        max_total_wait: float = 90.0,
    ) -> None:
        if not api_key or batch_size < 1 or timeout <= 0 or max_concurrency < 1:
            raise ValueError(
                "api_key and positive batch_size, timeout, and max_concurrency are required"
            )
        if max_attempts < 1 or backoff_seconds < 0 or max_total_wait < 0:
            raise ValueError(
                "max_attempts must be positive and backoff_seconds, max_total_wait non-negative"
            )
        self.base_url = base_url.rstrip("/")
        self.spec = EmbeddingSpec(
            provider=self.base_url, model=model, dimensions=dimensions, revision=revision
        )
        self.api_key, self.batch_size, self.timeout = api_key, batch_size, timeout
        self.send_dimensions = send_dimensions
        # Batches run concurrently (sequential batches made ingestion of long
        # histories latency-bound), but under one limit shared by every call on
        # this instance: per-call limits let 16 concurrent Adds open 128 POSTs.
        self.max_concurrency = max_concurrency
        # A rate limit, a provider outage or a dropped connection on one batch would
        # otherwise fail the whole Add or Search; the platform retries only some of those.
        self.max_attempts, self.backoff_seconds = max_attempts, backoff_seconds
        # The service sits behind Cloudflare, which fails a request with 524 when the
        # origin has not answered within 100 s. Retrying past that only burns quota for
        # a response nobody receives, so a batch's waits are bounded below it, leaving
        # room for the last attempt itself. Adds may legitimately run longer than that
        # in total; this bounds waiting, not the work.
        self.max_total_wait = max_total_wait
        # Injectable time source and sleep so tests never wait real seconds.
        self._clock: Callable[[], float] = time.monotonic
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._limit: asyncio.Semaphore | None = None
        self._limit_loop: asyncio.AbstractEventLoop | None = None
        self._cooldown_until = 0.0
        self._cooldown_started = 0.0
        self._cooldown_announced = 0.0

    def _limiter(self) -> asyncio.Semaphore:
        # A semaphore binds to the event loop it first blocks on; recreate it if the
        # instance is reused under another loop (e.g. successive asyncio.run calls).
        loop = asyncio.get_running_loop()
        if self._limit is None or self._limit_loop is not loop:
            self._limit, self._limit_loop = asyncio.Semaphore(self.max_concurrency), loop
        return self._limit

    def _enter_cooldown(self, seconds: float) -> None:
        now = self._clock()
        if self._cooldown_until <= now:
            self._cooldown_started = now
        self._cooldown_until = max(self._cooldown_until, now + seconds)

    async def _await_cooldown(self) -> float:
        """Hold a POST until the shared cool-down ends; return the seconds waited."""
        started = self._clock()
        while (remaining := self._cooldown_until - self._clock()) > 0:
            await self._sleep(remaining)
        now = self._clock()
        if self._cooldown_announced < self._cooldown_until <= now:
            self._cooldown_announced = self._cooldown_until
            logger.info(
                "embedding cool-down ended cooldown_s=%.2f",
                self._cooldown_until - self._cooldown_started,
            )
        return now - started

    def _retrying(self, batch_size: int, cooldown_waited: list[float]) -> AsyncRetrying:
        import httpx

        def before_sleep(state: RetryCallState) -> None:
            exc = state.outcome.exception() if state.outcome else None
            if exc is None:  # pragma: no cover - only failed attempts are retried
                return
            wait = state.next_action.sleep if state.next_action else 0.0
            status: object = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if wait > 0:
                    self._enter_cooldown(wait)
            # idle_for already includes the upcoming wait.
            total = state.idle_for + cooldown_waited[0]
            if total > self.max_total_wait:
                logger.warning(
                    "embedding giving up attempt=%d status=%s total_wait_s=%.2f batch_size=%d",
                    state.attempt_number,
                    status,
                    total,
                    batch_size,
                )
                raise exc
            logger.warning(
                "embedding retry attempt=%d status=%s wait_s=%.2f batch_size=%d",
                state.attempt_number,
                status,
                wait,
                batch_size,
            )

        async def sleep(seconds: float) -> None:
            await self._sleep(seconds)

        return AsyncRetrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(self.max_attempts),
            wait=_RetryAfterOrBackoff(
                wait_exponential_jitter(initial=self.backoff_seconds, max=16)
                if self.backoff_seconds > 0
                else wait_none()
            ),
            before_sleep=before_sleep,
            sleep=sleep,
            reraise=True,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        limit = self._limiter()

        async def post(
            client: httpx.AsyncClient, body: dict[str, object], batch_size: int
        ) -> httpx.Response:
            # One retry state per batch: a shared AsyncRetrying mixes up the
            # attempt counts of concurrent batches.
            cooldown_waited = [0.0]
            async for attempt in self._retrying(batch_size, cooldown_waited):
                with attempt:
                    async with limit:
                        cooldown_waited[0] += await self._await_cooldown()
                        response = await client.post(
                            self.base_url + "/embeddings",
                            json=body,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                        )
                    response.raise_for_status()
                    return response
            raise AssertionError("unreachable")

        async def embed(client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
            body: dict[str, object] = {"model": self.spec.model, "input": batch}
            if self.send_dimensions:
                body["dimensions"] = self.spec.dimensions
            response = await post(client, body, len(batch))
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(batch))):
                raise ValueError("embedding response has missing or repeated indices")
            return [row["embedding"] for row in rows]

        batches = [
            list(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            results = await asyncio.gather(*(embed(client, batch) for batch in batches))
        return [vector for result in results for vector in result]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def _retryable(exc: BaseException) -> bool:
    """Rate limits, provider outages and dropped connections; other 4xx fail at once."""
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


class _RetryAfterOrBackoff(wait_base):
    """Wait as long as a 429 asks (capped at 60 s), else back off.

    Honours ``Retry-After`` as delta-seconds or an HTTP-date (RFC 7231), then
    ``x-ratelimit-reset-requests``/``x-ratelimit-reset-tokens`` only when they are
    plain seconds. Negative, non-finite or unparseable values use the fallback.
    """

    def __init__(
        self,
        fallback: wait_base,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
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
        when = when.replace(tzinfo=timezone.utc)
    return _valid((when - now()).total_seconds())


def _seconds(value: str) -> float | None:
    """Plain seconds only; durations like ``6m0s`` or epoch formats are not guessed."""
    try:
        return _valid(float(value))
    except ValueError:
        return None


def _valid(seconds: float) -> float | None:
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class SentenceTransformerEmbedder:
    """Optional local semantic baseline with explicit query/document prompts."""

    def __init__(
        self, model: str, *, revision: str, query_prompt: str = "", document_prompt: str = ""
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, revision=revision)
        self.query_prompt, self.document_prompt = query_prompt, document_prompt
        self.spec = EmbeddingSpec(
            provider="sentence-transformers",
            model=model,
            revision=f"{revision}|query={query_prompt}|document={document_prompt}",
            dimensions=self.model.get_sentence_embedding_dimension(),
        )

    async def _encode(self, texts: Sequence[str], prompt: str) -> list[list[float]]:
        result = await asyncio.to_thread(
            self.model.encode, list(texts), prompt=prompt, normalize_embeddings=True
        )
        return result.tolist()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._encode(texts, self.document_prompt)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._encode([text], self.query_prompt))[0]
