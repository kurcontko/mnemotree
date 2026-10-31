"""Textual AML Add/Search contract; the caller owns the Memory lifecycle."""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import logging
import secrets
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi import Request as HTTPRequest
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import ConfigurationError, IngestConflict
from .engine import Memory
from .models import Evidence, Message, SearchOptions, scope

# The platform never retries a 4xx, so an additive contract field must not fail a
# whole run: request models ignore unknown keys (internal Records stay strict), and
# top_k above the contract maximum is clamped rather than rejected.
MAX_TOP_K = 100

logger = logging.getLogger(__name__)
# Per-process identity: a new value in the logs marks a restart, and seq orders the
# requests of one process (the sampled payloads carry the same pair to join on).
BOOT = secrets.token_hex(6)
_SEQ = itertools.count(1)
MAX_DETAIL_ERRORS = 10
MAX_LOGGED_PATH = 64


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one line; never truncates, so a restart keeps the earlier records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def emit(record: dict[str, Any], path: Path | None = None) -> None:
    """One JSON line at INFO, and appended to ``path`` when given.

    Logging must never fail a request, so a file error is reported, not raised.
    """
    line = json.dumps(record, ensure_ascii=False)
    logger.info(line)
    if path is not None:
        try:
            append_jsonl(path, record)
        except OSError as exc:
            logger.warning("request log write failed: %s", type(exc).__name__)


def auth_scheme(authorization: str | None, x_api_key: str | None) -> str:
    """The scheme the caller used, mirroring ``authorize``; never the credential."""
    scheme = (authorization or "").partition(" ")[0].lower()
    if scheme in ("bearer", "token"):
        return scheme
    if x_api_key is not None:
        return "x-api-key"
    # An unknown scheme word could itself be a bare credential, so it is not echoed.
    return "other" if scheme else "none"


def failure(exc: BaseException) -> dict[str, Any]:
    """Class name and provider HTTP status of a failure; never bodies or headers."""
    info: dict[str, Any] = {"error": type(exc).__name__}
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(getattr(current, "response", None), "status_code", None)
        if type(current).__name__ == "HTTPStatusError" and isinstance(status, int):
            info["provider_status"] = status
            break
        current = current.__cause__ or current.__context__
    return info


def returned_role_chars(evidence: Sequence[Evidence]) -> dict[str, int]:
    """Unique source characters by actual role, including merged spans and neighbors.

    Count source offsets, excluding rendered role/date headers and derived date notes.
    Overlaps between returned blocks count once as well as overlaps within a block.
    """
    spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for item in evidence:
        for member in item.members():
            for span in member.spans():
                spans.setdefault((span.source_id, span.role), []).append((span.start, span.end))
    counts: Counter[str] = Counter()
    for (_, role), ranges in spans.items():
        end = 0
        for start, stop in sorted(ranges):
            counts[role] += max(0, stop - max(start, end))
            end = max(end, stop)
    return dict(counts)


class Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class AddMessage(Request):
    role: Literal["user", "assistant"]
    # Public benchmark histories contain empty turns. Rejecting them with 422
    # would stop the evaluation stage (never retried), so they are skipped.
    content: str
    timestamp: StrictInt | None = None

    def source_message(self) -> Message:
        return Message(
            content=self.content,
            role=self.role,
            timestamp=(
                datetime.fromtimestamp(self.timestamp / 1000, tz=UTC)
                if self.timestamp is not None
                else None
            ),
        )


class AddRequest(Request):
    request_id: str
    user_id: str
    session_id: str
    messages: list[AddMessage] = Field(min_length=1)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


class SearchRequest(Request):
    user_id: str
    query: str = Field(min_length=1)
    options: list[str] | None = None
    top_k: int = Field(ge=1, strict=True)

    @field_validator("user_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


# Leave room for the question and answer-model instructions. This is a configurable
# evidence allowance, not a claim about retrieval quality or a provider's context limit.
AML_TOKEN_BUDGET = 8_000


def aml_options(memory: Memory) -> SearchOptions:
    """Service defaults: adjacent turns merged into passages, within 8K tokens."""
    return SearchOptions(
        mode="hybrid" if memory.embedder else "lexical",
        limit=100,
        token_budget=AML_TOKEN_BUDGET,
        neighbor_messages=1,
        merge_blocks=True,
        # Select diverse source anchors before completing long messages. Prefer user
        # turns only in assistant-heavy histories; deepen the pool for long messages.
        chunks_per_source=1,
        role_first="user",
        # Retain the measured hard partition; bounded priors require explicit opt-in.
        role_boost=None,
        role_first_min_assistant_share=0.65,
        model_query_policy="combined",
        candidate_limit=100,
        deep_candidate_limit=300,
        deep_history_density=2.0,
    )


def create_app(
    memory: Memory,
    *,
    api_key: str,
    options: SearchOptions | None = None,
    sample_path: Path | None = None,
    sample_limit: int | None = 50,
    request_log: Path | None = None,
) -> FastAPI:
    """``sample_path`` appends the first ``sample_limit`` (``None``: every) Add and
    Search payloads as JSONL, to inspect what a Smoke run actually sends (for
    example, whether speaker names are inside content). It holds evaluation data:
    AML requires deleting that within 30 days, and it should be off for a Full run.

    Every request is logged as one metadata-only JSON line (no message or query
    text) at INFO on this module's logger, and appended to ``request_log`` if set.
    """
    if not api_key.strip():
        raise ValueError("a nonblank service api_key is required")
    if sample_limit is not None and sample_limit < 0:
        raise ValueError("sample_limit must be nonnegative or None")
    options = options or aml_options(memory)
    sampled = {"add": 0, "search": 0}
    for path in (sample_path, request_log):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def sample(kind: str, payload: Request, http: HTTPRequest) -> None:
        if sample_path is None or (sample_limit is not None and sampled[kind] >= sample_limit):
            return
        sampled[kind] += 1
        state = http.state
        record = {
            "kind": kind,
            "ts": getattr(state, "ts", None),
            "boot": BOOT,
            "seq": getattr(state, "seq", None),
            **payload.model_dump(),
        }
        try:
            append_jsonl(sample_path, record)
        except OSError as exc:
            logger.warning("payload sample write failed: %s", type(exc).__name__)

    def note(http: HTTPRequest, **fields: Any) -> None:
        """Stash request-log fields for the middleware; survives a later failure."""
        state = http.state
        if not hasattr(state, "log"):
            state.log = {}
        state.log.update(fields)

    if options.mode != "lexical" and memory.embedder is None:
        raise ValueError("dense/hybrid service requires an embedder")
    if memory.store.embedding_spec and memory.embedder is None:
        # Every Add would fail, and the client would be told to retry forever.
        raise ValueError("a dense database requires its configured embedder")
    if options.representation != "raw" and memory.store.extractor_config is None:
        raise ConfigurationError("atom retrieval service requires an extracted corpus")
    app = FastAPI(
        title="Mnemotree evidence memory", docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.middleware("http")
    async def log_requests(
        http: HTTPRequest, call_next: Callable[[HTTPRequest], Awaitable[Response]]
    ) -> Response:
        started = perf_counter()
        http.state.ts = utc_now()
        http.state.seq = next(_SEQ)
        record: dict[str, Any] = {
            "ts": http.state.ts,
            "boot": BOOT,
            "seq": http.state.seq,
            # Unmatched paths are caller-chosen; keep only a bounded prefix of them.
            "route": http.url.path[:MAX_LOGGED_PATH],
            "method": http.method,
            "auth_scheme": auth_scheme(
                http.headers.get("authorization"), http.headers.get("x-api-key")
            ),
        }
        status = 500
        try:
            response = await call_next(http)
            status = response.status_code
            return response
        except Exception as exc:
            record.update(failure(exc))
            raise
        finally:
            if (route := http.scope.get("route")) is not None:
                record["route"] = route.path
            record["status"] = status
            record["ms"] = round((perf_counter() - started) * 1000, 1)
            record.update(getattr(http.state, "log", {}))
            emit(record, request_log)

    @app.exception_handler(RequestValidationError)
    async def validation_failed(http: HTTPRequest, exc: RequestValidationError) -> Response:
        # Locations and messages only: an error's input may be message content.
        errors = exc.errors()
        detail = [
            f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}"
            for error in errors[:MAX_DETAIL_ERRORS]
        ]
        note(http, detail=detail, **({"errors": len(errors)} if len(errors) > len(detail) else {}))
        return await request_validation_exception_handler(http, exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_failed(http: HTTPRequest, exc: StarletteHTTPException) -> Response:
        note(http, detail=exc.detail)
        return await http_exception_handler(http, exc)

    expected = api_key.encode()

    def authorize(
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        # AML sends the registered scheme: Bearer, Token or an X-Api-Key header.
        scheme, _, token = (authorization or "").partition(" ")
        presented = token if scheme.lower() in ("bearer", "token") else x_api_key
        if not hmac.compare_digest((presented or "").encode(), expected):
            raise HTTPException(status_code=401, detail="invalid credentials")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # The platform probes both routes with HEAD (and the key) before a run; a 405
    # there is noise in its logs and ours, so the probe answers 200 with no body.
    @app.api_route("/v1/memories/add", methods=["HEAD"], dependencies=[Depends(authorize)])
    @app.api_route("/v1/memories/search", methods=["HEAD"], dependencies=[Depends(authorize)])
    def probe() -> Response:
        return Response(status_code=200)

    @app.post("/v1/memories/add", dependencies=[Depends(authorize)])
    async def add(request: AddRequest, http: HTTPRequest) -> dict[str, Any]:
        sample("add", request, http)
        received = request.messages
        note(
            http,
            user_id=request.user_id,
            request_id=request.request_id,
            session_id=request.session_id,
            messages=len(received),
            blank_skipped=sum(1 for message in received if not message.content.strip()),
            words=sum(len(message.content.split()) for message in received),
            chars=sum(len(message.content) for message in received),
            roles=dict(Counter(message.role for message in received)),
            with_timestamp=sum(1 for message in received if message.timestamp is not None),
        )
        try:
            messages = [
                message.source_message() for message in request.messages if message.content.strip()
            ]
        except (ValueError, OverflowError, OSError) as exc:
            raise HTTPException(status_code=422, detail="invalid message or timestamp") from exc
        add_started = perf_counter()
        try:
            if messages:
                receipt = await memory.add(
                    request.user_id,
                    messages,
                    request_id=request.request_id,
                    session_id=request.session_id,
                )
                note(http, replay=receipt.replayed, chunks=receipt.chunk_count)
        except IngestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            # Provider responses may contain credentials; never echo them.
            note(http, **failure(exc))
            raise HTTPException(
                status_code=503, detail="ingestion failed; retry this request"
            ) from exc
        finally:
            if messages:
                note(http, add_ms=round((perf_counter() - add_started) * 1000, 1))
        return {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

    @app.post("/v1/memories/search", dependencies=[Depends(authorize)])
    async def search(request: SearchRequest, http: HTTPRequest) -> dict[str, Any]:
        sample("search", request, http)
        limit = min(request.top_k, MAX_TOP_K)
        note(
            http,
            user_id=request.user_id,
            query_chars=len(request.query),
            options=len(request.options) if request.options is not None else None,
            top_k=request.top_k,
            limit=limit,
        )
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must not be blank")
        # Keep explicit option boundaries for the bounded lexical term allocation.
        # Memory still combines the stem and options in order for model inputs.
        # Returning fewer than top_k is valid; returning more is a contract error.
        search_options = options.model_copy(
            update={"limit": limit, "candidate_limit": max(options.candidate_limit, limit)}
        )
        try:
            result = await memory.search(
                request.user_id,
                request.query,
                options=search_options,
                query_options=request.options,
            )
        except Exception as exc:
            note(http, **failure(exc))
            raise HTTPException(
                status_code=503, detail="retrieval failed; retry this request"
            ) from exc
        trace = result.trace
        ids = [item.id for item in result.evidence]
        note(
            http,
            items=len(ids),
            sources=len({source for item in result.evidence for source in item.source_ids()}),
            packed_tokens=trace.packed_tokens,
            evidence_tokens=trace.evidence_tokens,
            candidate_limit=trace.candidate_limit,
            chunk_density=(
                round(trace.chunk_density, 3) if trace.chunk_density is not None else None
            ),
            original_model_query_tokens=trace.original_model_query_tokens,
            model_query_tokens=trace.model_query_tokens,
            model_query_truncated=trace.model_query_truncated,
            tokenizer=trace.tokenizer,
            assistant_share=trace.assistant_share,
            role_first_applied=trace.role_first_applied,
            role_boost_applied=trace.role_boost_applied,
            model_query_policy=options.model_query_policy,
            channel_counts=trace.channel_counts,
            timings_ms={key: round(value, 1) for key, value in trace.timings_ms.items()},
            returned_roles=dict(Counter(item.role for item in result.evidence)),
            returned_role_chars=returned_role_chars(result.evidence),
            ids_digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
        )
        return {"data": [response_item(item) for item in result.evidence]}

    return app


def response_item(item: Evidence) -> dict[str, Any]:
    value: dict[str, object] = {
        "id": item.id,
        "content": item.render(),
        "score": item.rerank_score if item.rerank_score is not None else item.score,
    }
    # Some AML answer prompts format memories as "- [created_at] text". A merged
    # block starts at its earliest dated message, whichever anchor ranked first.
    timestamps = [
        span.timestamp
        for member in item.members()
        for span in member.spans()
        if span.timestamp is not None
    ]
    if timestamps:
        value["created_at"] = min(timestamps).isoformat()
    return value
