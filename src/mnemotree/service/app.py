"""Textual AML Add/Search contract; the caller owns the engine's lifecycle."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi import Request as HTTPRequest
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.models import Evidence, SearchOptions
from ..engine import EvidenceEngine
from ..errors import ConfigurationError, IngestConflict
from .presets import aml_options
from .request_log import (
    BOOT,
    MAX_LOGGED_PATH,
    auth_scheme,
    emit,
    failure,
    next_seq,
    returned_role_chars,
    utc_now,
)
from .schemas import MAX_TOP_K, AddRequest, SearchRequest

MAX_DETAIL_ERRORS = 10
# Naming 422 replaces FastAPI's generated entry for body validation, so restate it.
VALIDATION_ERROR: dict[str, Any] = {
    "description": "Validation Error",
    "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
    },
}


def create_app(
    memory: EvidenceEngine,
    *,
    api_key: str,
    options: SearchOptions | None = None,
    request_log: Path | None = None,
) -> FastAPI:
    """Every request is logged as one metadata-only JSON line (no message or query
    text) at INFO on this module's logger, and appended to ``request_log`` if set.
    """
    if not api_key.strip():
        raise ValueError("a nonblank service api_key is required")
    options = options or aml_options(memory)
    if request_log is not None:
        request_log.parent.mkdir(parents=True, exist_ok=True)
    check_service(memory, options)
    app = FastAPI(
        title="Mnemotree evidence memory", docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.middleware("http")
    async def log_requests(
        http: HTTPRequest, call_next: Callable[[HTTPRequest], Awaitable[Response]]
    ) -> Response:
        return await log_request(http, call_next, request_log)

    app.exception_handler(RequestValidationError)(validation_failed)
    app.exception_handler(StarletteHTTPException)(http_failed)

    expected = api_key.encode()

    def authorize(
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        require_key(authorization, x_api_key, expected)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # The platform probes both routes with HEAD (and the key) before a run; a 405
    # there is noise in its logs and ours, so the probe answers 200 with no body.
    @app.api_route("/v1/memories/add", methods=["HEAD"], dependencies=[Depends(authorize)])
    @app.api_route("/v1/memories/search", methods=["HEAD"], dependencies=[Depends(authorize)])
    def probe() -> Response:
        return Response(status_code=200)

    @app.post(
        "/v1/memories/add",
        dependencies=[Depends(authorize)],
        responses={
            422: VALIDATION_ERROR,
            409: {"description": "A request or source ID was reused for a different payload"},
            503: {"description": "Ingestion failed; retry the same request"},
        },
    )
    async def add(request: AddRequest, http: HTTPRequest) -> dict[str, Any]:
        return await add_messages(memory, request, http)

    @app.post(
        "/v1/memories/search",
        dependencies=[Depends(authorize)],
        responses={
            422: VALIDATION_ERROR,
            503: {"description": "Retrieval failed; retry the same request"},
        },
    )
    async def search(request: SearchRequest, http: HTTPRequest) -> dict[str, Any]:
        return await search_memory(memory, options, request, http)

    return app


def check_service(memory: EvidenceEngine, options: SearchOptions) -> None:
    if options.mode != "lexical" and memory.embedder is None:
        raise ValueError("dense/hybrid service requires an embedder")
    if memory.store.embedding_spec and memory.embedder is None:
        # Every Add would fail, and the client would be told to retry forever.
        raise ValueError("a dense database requires its configured embedder")
    if options.representation != "raw" and memory.store.extractor_config is None:
        raise ConfigurationError("atom retrieval service requires an extracted corpus")


def note(http: HTTPRequest, **fields: Any) -> None:
    """Stash request-log fields for the middleware; survives a later failure."""
    state = http.state
    if not hasattr(state, "log"):
        state.log = {}
    state.log.update(fields)


async def log_request(
    http: HTTPRequest,
    call_next: Callable[[HTTPRequest], Awaitable[Response]],
    request_log: Path | None,
) -> Response:
    started = perf_counter()
    http.state.ts = utc_now()
    http.state.seq = next_seq()
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


async def validation_failed(http: HTTPRequest, exc: RequestValidationError) -> Response:
    # Locations and messages only: an error's input may be message content.
    errors = exc.errors()
    detail = [
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}"
        for error in errors[:MAX_DETAIL_ERRORS]
    ]
    note(http, detail=detail, **({"errors": len(errors)} if len(errors) > len(detail) else {}))
    return await request_validation_exception_handler(http, exc)


async def http_failed(http: HTTPRequest, exc: StarletteHTTPException) -> Response:
    note(http, detail=exc.detail)
    return await http_exception_handler(http, exc)


def require_key(authorization: str | None, x_api_key: str | None, expected: bytes) -> None:
    # AML sends the registered scheme: Bearer, Token or an X-Api-Key header.
    scheme, _, token = (authorization or "").partition(" ")
    presented = token if scheme.lower() in ("bearer", "token") else x_api_key
    if not hmac.compare_digest((presented or "").encode(), expected):
        raise HTTPException(status_code=401, detail="invalid credentials")


async def add_messages(
    memory: EvidenceEngine, request: AddRequest, http: HTTPRequest
) -> dict[str, Any]:
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
        raise HTTPException(status_code=503, detail="ingestion failed; retry this request") from exc
    finally:
        if messages:
            note(http, add_ms=round((perf_counter() - add_started) * 1000, 1))
    return {
        "success": True,
        "request_id": request.request_id,
        "user_id": request.user_id,
        "session_id": request.session_id,
    }


async def search_memory(
    memory: EvidenceEngine, options: SearchOptions, request: SearchRequest, http: HTTPRequest
) -> dict[str, Any]:
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
    # The engine still combines the stem and options in order for model inputs.
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
        raise HTTPException(status_code=503, detail="retrieval failed; retry this request") from exc
    trace = result.trace
    ids = [item.id for item in result.evidence]
    note(
        http,
        items=len(ids),
        sources=len({source for item in result.evidence for source in item.source_ids()}),
        packed_tokens=trace.packed_tokens,
        evidence_tokens=trace.evidence_tokens,
        candidate_limit=trace.candidate_limit,
        chunk_density=(round(trace.chunk_density, 3) if trace.chunk_density is not None else None),
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
