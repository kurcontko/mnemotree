"""Optional OpenAI-compatible embedding transport. No LangChain dependency."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from .models import EmbeddingSpec


class OpenAIEmbedder:
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
    ) -> None:
        if not api_key or batch_size < 1 or timeout <= 0 or max_concurrency < 1:
            raise ValueError(
                "api_key and positive batch_size, timeout, and max_concurrency are required"
            )
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("max_attempts must be positive and backoff_seconds non-negative")
        self.base_url = base_url.rstrip("/")
        self.spec = EmbeddingSpec(
            provider=self.base_url, model=model, dimensions=dimensions, revision=revision
        )
        self.api_key, self.batch_size, self.timeout = api_key, batch_size, timeout
        self.send_dimensions = send_dimensions
        # Batches of one call run concurrently; sequential batches made
        # ingestion of long histories latency-bound.
        self.max_concurrency = max_concurrency
        # A rate limit, a provider outage or a dropped connection on one batch would
        # otherwise fail the whole Add or Search; the platform retries only some of those.
        self.max_attempts, self.backoff_seconds = max_attempts, backoff_seconds

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        limit = asyncio.Semaphore(self.max_concurrency)

        retrying = AsyncRetrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(self.max_attempts),
            wait=_RetryAfterOrBackoff(
                wait_exponential_jitter(initial=self.backoff_seconds, max=16)
            ),
            reraise=True,
        )

        async def post(client: httpx.AsyncClient, body: dict[str, object]) -> httpx.Response:
            async for attempt in retrying:
                with attempt:
                    async with limit:
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
            response = await post(client, body)
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
    """Wait as long as a 429's Retry-After header asks (capped at 60 s), else back off."""

    def __init__(self, fallback: wait_base) -> None:
        self.fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        import httpx

        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            header = exc.response.headers.get("retry-after", "")
            try:
                return min(max(float(header), 0.0), 60.0)
            except ValueError:
                pass
        return self.fallback(retry_state)


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
