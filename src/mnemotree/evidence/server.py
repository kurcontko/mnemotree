"""Textual AML Add/Search contract; the caller owns the Memory lifecycle."""

from __future__ import annotations

import hmac
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import Field, StrictInt, field_validator

from .engine import Memory
from .models import Evidence, Message, Record, SearchOptions, scope
from .store import IngestConflict


class AddMessage(Record):
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


# AML keeps a 117,760-token prefix of the returned evidence, but filling it does
# not pay: with qwen3.6 answering, 16K never beat 8K on LoCoMo, LongMemEval-S,
# BEAM, ScriptMem or PersonaMem, and the extra turns cost abstention on BEAM.
AML_TOKEN_BUDGET = 8_000


def aml_options(memory: Memory) -> SearchOptions:
    """Service defaults: adjacent turns merged into passages, within 8K tokens.

    Merging with one neighbor raised LoCoMo all-evidence coverage at 8K from
    72.7% to 86.7%, and answer accuracy on LoCoMo, BEAM and ScriptMem by 2-8 points.
    """
    return SearchOptions(
        mode="hybrid" if memory.embedder else "lexical",
        limit=100,
        token_budget=AML_TOKEN_BUDGET,
        neighbor_messages=1,
        merge_blocks=True,
        # Role stack (24 Oct, gpt-4o-mini answers on the sweep contexts): one chunk per
        # message and user turns first are neutral-to-positive on every public set; the
        # deeper pool pays only where long assistant turns crowd the candidates (BEAM
        # +2 to +3) and costs 3-6 points on short-session sets, so it is gated on the
        # history's chunks per message (BEAM 2.1-3.7, LME <= 1.7, LoCoMo 1.0).
        chunks_per_source=1,
        role_first="user",
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
    sample_limit: int = 50,
) -> FastAPI:
    """``sample_path`` appends the first ``sample_limit`` Add and Search payloads
    as JSONL, to inspect what a Smoke run actually sends (for example, whether
    speaker names are inside content). It holds evaluation data: AML requires
    deleting that within 30 days, and it should be off for a Full run.
    """
    if not api_key.strip():
        raise ValueError("a nonblank service api_key is required")
    options = options or aml_options(memory)
    sampled = {"add": 0, "search": 0}

    def sample(kind: str, payload: Record) -> None:
        if sample_path is None or sampled[kind] >= sample_limit:
            return
        sampled[kind] += 1
        line = json.dumps({"kind": kind, **payload.model_dump()}, ensure_ascii=False)
        with sample_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

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
        sample("add", request)
        try:
            messages = [
                message.source_message() for message in request.messages if message.content.strip()
            ]
        except (ValueError, OverflowError, OSError) as exc:
            raise HTTPException(status_code=422, detail="invalid message or timestamp") from exc
        try:
            if messages:
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
        sample("search", request)
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
        return {"data": [response_item(item) for item in result.evidence]}

    return app


def response_item(item: Evidence) -> dict:
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
        for span in (member, *member.neighbors)
        if span.timestamp is not None
    ]
    if timestamps:
        value["created_at"] = min(timestamps).isoformat()
    return value
