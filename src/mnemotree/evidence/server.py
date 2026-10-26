"""Textual AML Add/Search contract; the caller owns the Memory lifecycle."""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import Field, StrictInt, field_validator

from .engine import Memory
from .models import Message, Record, SearchOptions, scope
from .store import IngestConflict


class AddMessage(Record):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
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


class AddRequest(Record):
    request_id: str
    user_id: str
    session_id: str
    messages: list[AddMessage] = Field(min_length=1)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


class SearchRequest(Record):
    user_id: str
    query: str = Field(min_length=1)
    options: list[str] | None = None
    top_k: int = Field(ge=1, le=100, strict=True)

    @field_validator("user_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


def create_app(memory: Memory, *, api_key: str, options: SearchOptions | None = None) -> FastAPI:
    if not api_key.strip():
        raise ValueError("a nonblank service api_key is required")
    options = options or SearchOptions(mode="hybrid" if memory.embedder else "lexical")
    if options.mode != "lexical" and memory.embedder is None:
        raise ValueError("dense/hybrid service requires an embedder")
    if options.representation != "raw" and memory.store.extractor_config is None:
        raise ValueError("atom retrieval service requires an extracted corpus")
    app = FastAPI(title="Mnemotree evidence memory", docs_url=None, redoc_url=None)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if not hmac.compare_digest((authorization or "").encode(), f"Bearer {api_key}".encode()):
            raise HTTPException(status_code=401, detail="invalid credentials")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/memories/add", dependencies=[Depends(authorize)])
    async def add(request: AddRequest):
        try:
            messages = [message.source_message() for message in request.messages]
        except (ValueError, OverflowError, OSError) as exc:
            raise HTTPException(status_code=422, detail="invalid message or timestamp") from exc
        try:
            await memory.add(
                request.user_id,
                messages,
                request_id=request.request_id,
                session_id=request.session_id,
            )
        except IngestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            # Provider responses may contain credentials; never echo them.
            raise HTTPException(
                status_code=503, detail="ingestion failed; retry this request"
            ) from exc
        return {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

    @app.post("/v1/memories/search", dependencies=[Depends(authorize)])
    async def search(request: SearchRequest):
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must not be blank")
        # Options are supplied by the benchmark, not gold labels. Retain their
        # ordering and include them in the retrieval query for lexical coverage.
        query = request.query
        if request.options:
            query += "\n" + "\n".join(request.options)
        search_options = options.model_copy(
            update={
                "limit": request.top_k,
                "candidate_limit": max(options.candidate_limit, request.top_k),
            }
        )
        try:
            result = await memory.search(request.user_id, query, options=search_options)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="retrieval failed; retry this request"
            ) from exc
        return {"data": [{"id": item.id, "content": item.render()} for item in result.evidence]}

    return app
