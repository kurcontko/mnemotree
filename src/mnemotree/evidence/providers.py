"""Optional embedding providers: an OpenAI-compatible transport and local models."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import EmbeddingSpec

if TYPE_CHECKING:
    from tenacity import AsyncRetrying

logger = logging.getLogger(__name__)


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
        # tenacity comes with the embeddings extra; importing it here keeps the module,
        # and so SentenceTransformerEmbedder, importable on a lite install.
        import httpx
        from tenacity import (
            AsyncRetrying,
            RetryCallState,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential_jitter,
            wait_none,
        )

        from .retry import RetryAfterOrBackoff, retryable

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
            retry=retry_if_exception(retryable),
            stop=stop_after_attempt(self.max_attempts),
            wait=RetryAfterOrBackoff(
                wait_exponential_jitter(multiplier=self.backoff_seconds, max=16)
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
            # A batch that fails after its retries cancels its siblings, so none keeps
            # a request in flight on the client being closed. Callers still see the
            # batch's own exception, as with gather.
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(embed(client, batch)) for batch in batches]
            except BaseExceptionGroup as failed:
                raise failed.exceptions[0] from None
        return [vector for task in tasks for vector in task.result()]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class SentenceTransformerEmbedder:
    """Optional local semantic baseline with explicit query/document prompts."""

    def __init__(
        self, model: str, *, revision: str, query_prompt: str = "", document_prompt: str = ""
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, revision=revision)
        self.query_prompt, self.document_prompt = query_prompt, document_prompt
        # sentence-transformers 5.6 renamed the method and warns on the old name;
        # earlier releases only have the old one.
        dimension = getattr(self.model, "get_embedding_dimension", None)
        if dimension is None:
            dimension = self.model.get_sentence_embedding_dimension
        self.spec = EmbeddingSpec(
            provider="sentence-transformers",
            model=model,
            revision=f"{revision}|query={query_prompt}|document={document_prompt}",
            dimensions=dimension(),
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


# The zero-configuration MCP server's model: English, 384 dimensions and an 8192-token
# context, so a whole 1200-token chunk is embedded rather than truncated. The revision
# pins the ONNX export; a new upstream commit would change the vectors under the
# same model name.
DEFAULT_LOCAL_EMBEDDING: dict[str, Any] = {
    "kind": "fastembed",
    "model": "ibm-granite/granite-embedding-small-english-r2",
    "source": "onnx-community/granite-embedding-small-english-r2-ONNX",
    "revision": "1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8",
    "dimensions": 384,
    "cache_dir": "~/.mnemotree/models",
}

# What fastembed's ONNX text models read besides the model files themselves.
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class EmbedderUnavailable(RuntimeError):
    """The embedding model could not be downloaded or loaded; a later call retries."""


class FastEmbedEmbedder:
    """A local ONNX model through fastembed, fetched at a pinned revision on first use.

    fastembed would download the latest revision into the system temp directory, which
    macOS purges, so the files are fetched here with huggingface_hub into ``cache_dir``.
    Nothing is imported or downloaded until the first embedding or ``warm()``, so
    ``spec`` is available at once and a server can answer before the model is ready.
    """

    def __init__(
        self,
        model: str,
        *,
        source: str,
        revision: str,
        dimensions: int,
        cache_dir: str | Path = "~/.mnemotree/models",
        query_prompt: str = "",
        document_prompt: str = "",
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Checked without importing, so a missing package fails at startup, not on
        # the first embedding.
        if importlib.util.find_spec("fastembed") is None:
            raise ImportError(
                "the fastembed embedding kind needs the fastembed package "
                "(install mnemotree-mcp, or pip install fastembed)"
            )
        self.source, self.revision = source, revision
        self.cache_dir = Path(cache_dir).expanduser()
        self.query_prompt, self.document_prompt = query_prompt, document_prompt
        # Attention memory grows with batch size times the square of the longest text,
        # and chunks run to 1200 tokens, so batches stay small.
        self.batch_size = batch_size
        # The source is part of the revision: another ONNX export of the same model
        # gives different vectors.
        self.spec = EmbeddingSpec(
            provider="fastembed",
            model=model,
            revision=f"{source}@{revision}|query={query_prompt}|document={document_prompt}",
            dimensions=dimensions,
        )
        self._lock = threading.Lock()
        self._model: Any = None

    def files(self) -> list[str]:
        """The files to fetch, per fastembed's registry, after checking its dimensions."""
        from fastembed import TextEmbedding

        model = self.spec.model
        entry = next(
            (row for row in TextEmbedding.list_supported_models() if row["model"] == model),
            None,
        )
        if entry is None:
            raise ValueError(f"fastembed does not support the embedding model {model!r}")
        if entry["dim"] != self.spec.dimensions:
            raise ValueError(
                f"{model} has {entry['dim']} dimensions, not the configured {self.spec.dimensions}"
            )
        return [*TOKENIZER_FILES, entry["model_file"], *entry.get("additional_files", [])]

    def warm(self) -> None:
        """Download and load the model now rather than on the first embedding."""
        self._load()

    def _load(self) -> Any:
        # One lock, so concurrent first calls load once. A failure is not cached: the
        # next call retries, so a dropped connection does not disable the process.
        with self._lock:
            if self._model is None:
                self._model = self._open()
            return self._model

    def _open(self) -> Any:
        files = self.files()
        started = time.monotonic()
        try:
            path = self._fetch(files)
            from fastembed import TextEmbedding

            # cache_dir only keeps fastembed from creating its own in the temp directory.
            model = TextEmbedding(
                self.spec.model, cache_dir=str(self.cache_dir), specific_model_path=path
            )
        except Exception as exc:
            raise EmbedderUnavailable(
                f"embedding model {self.spec.model} could not be loaded: {exc}"
            ) from exc
        if model.embedding_size != self.spec.dimensions:
            raise ValueError(
                f"{self.spec.model} loaded with {model.embedding_size} dimensions, not the "
                f"configured {self.spec.dimensions}"
            )
        logger.info(
            "embedding model %s ready in %.1fs", self.spec.model, time.monotonic() - started
        )
        return model

    def _fetch(self, files: list[str]) -> str:
        from huggingface_hub import snapshot_download

        options: dict[str, Any] = {
            "repo_id": self.source,
            "revision": self.revision,
            "cache_dir": str(self.cache_dir),
            "allow_patterns": files,
        }
        # Cached files first, so a restart neither needs the network nor waits on it.
        # An interrupted download leaves the snapshot incomplete, so check every file.
        try:
            path = snapshot_download(**options, local_files_only=True)
            if all((Path(path) / name).is_file() for name in files):
                return path
        except OSError:
            pass
        logger.info(
            "downloading embedding model %s@%s to %s", self.source, self.revision, self.cache_dir
        )
        return snapshot_download(**options)

    def _encode(self, texts: Sequence[str], prompt: str) -> list[list[float]]:
        model = self._load()
        vectors = []
        for vector in model.embed([prompt + text for text in texts], batch_size=self.batch_size):
            values = [float(value) for value in vector]
            if len(values) != self.spec.dimensions:
                raise ValueError(
                    f"{self.spec.model} returned {len(values)} dimensions, not the configured "
                    f"{self.spec.dimensions}"
                )
            # fastembed leaves this model's vectors unnormalized.
            norm = math.sqrt(sum(value * value for value in values))
            vectors.append([value / norm for value in values] if norm else values)
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts, self.document_prompt)

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [text], self.query_prompt))[0]
