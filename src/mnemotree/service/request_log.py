"""Metadata-only request logging for the service: one JSON line per request.

Message and query text never reach these records.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import secrets
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.models import Evidence

if TYPE_CHECKING:
    from starlette.requests import Request as HTTPRequest

    from ..core.models import SearchOptions, SearchResult
    from .schemas import AddRequest, SearchRequest

# The 1.0 logger name: logging configuration addresses it, and the CLI
# attaches its handler to "mnemotree.evidence".
logger = logging.getLogger("mnemotree.evidence.server")
# Per-process identity: a new value in the logs marks a restart, and seq orders the
# requests of one process.
BOOT = secrets.token_hex(6)
_SEQ = itertools.count(1)
# Unmatched paths are caller-chosen; a log line keeps only this much of them.
MAX_LOGGED_PATH = 64


def next_seq() -> int:
    return next(_SEQ)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def note(http: HTTPRequest, **fields: Any) -> None:
    """Stash request-log fields for the middleware; survives a later failure."""
    state = http.state
    if not hasattr(state, "log"):
        state.log = {}
    state.log.update(fields)


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


def add_fields(request: AddRequest) -> dict[str, Any]:
    """What an Add request carried: its IDs and the shape of its messages."""
    received = request.messages
    return {
        "user_id": request.user_id,
        "request_id": request.request_id,
        "session_id": request.session_id,
        "messages": len(received),
        "blank_skipped": sum(1 for message in received if not message.content.strip()),
        "words": sum(len(message.content.split()) for message in received),
        "chars": sum(len(message.content) for message in received),
        "roles": dict(Counter(message.role for message in received)),
        "with_timestamp": sum(1 for message in received if message.timestamp is not None),
    }


def query_fields(request: SearchRequest, limit: int) -> dict[str, Any]:
    """What a Search request asked for, and the limit it is served with."""
    return {
        "user_id": request.user_id,
        "query_chars": len(request.query),
        "options": len(request.options) if request.options is not None else None,
        "top_k": request.top_k,
        "limit": limit,
    }


def result_fields(result: SearchResult, options: SearchOptions) -> dict[str, Any]:
    """What a search returned and how; ``options`` are the service's own defaults."""
    trace = result.trace
    ids = [item.id for item in result.evidence]
    return {
        "items": len(ids),
        "sources": len({source for item in result.evidence for source in item.source_ids()}),
        "packed_tokens": trace.packed_tokens,
        "evidence_tokens": trace.evidence_tokens,
        "candidate_limit": trace.candidate_limit,
        "chunk_density": (
            round(trace.chunk_density, 3) if trace.chunk_density is not None else None
        ),
        "original_model_query_tokens": trace.original_model_query_tokens,
        "model_query_tokens": trace.model_query_tokens,
        "model_query_truncated": trace.model_query_truncated,
        "tokenizer": trace.tokenizer,
        "assistant_share": trace.assistant_share,
        "role_first_applied": trace.role_first_applied,
        "role_boost_applied": trace.role_boost_applied,
        "model_query_policy": options.model_query_policy,
        "channel_counts": trace.channel_counts,
        "timings_ms": {key: round(value, 1) for key, value in trace.timings_ms.items()},
        "returned_roles": dict(Counter(item.role for item in result.evidence)),
        "returned_role_chars": returned_role_chars(result.evidence),
        "ids_digest": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
    }
