"""Match submitted prompts to actual user messages in Claude Code's transcript."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.models import Message


@dataclass(frozen=True)
class Submission:
    key: str
    fingerprint: str
    offset: int
    submitted: str


def fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def transcript_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0  # The first submit can precede a readable transcript.


def matching_text(content: Any, expected: str) -> str | None:
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
    else:
        return None
    return next((text for text in texts if fingerprint(text) == expected), None)


def transcript_records(path: Path, session: str, offset: int) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(offset if offset <= size else 0)
        for line in stream:
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if (
                isinstance(record, dict)
                and record.get("sessionId") == session
                and not record.get("isSidechain")
                and not record.get("isMeta")
            ):
                yield record


def accepted_prompts(
    path: Path, session: str, pending: Sequence[Submission]
) -> tuple[str, list[Message]] | None:
    """Confirm the stopped prompt and any interrupted prompt Claude actually answered.

    A matching Stop confirms its prompt ID. Earlier prompts and clients without
    prompt IDs also need an assistant record linked to that user's message UUID.
    """
    submitted = {p.key: p for p in pending}
    current = pending[-1].key
    found: dict[str, tuple[str, Message]] = {}
    answered = set()
    for record in transcript_records(path, session, min(p.offset for p in pending)):
        if record.get("type") == "assistant" and not record.get("isApiErrorMessage"):
            answered.add(record.get("parentUuid"))
            continue
        message = record.get("message")
        if (
            record.get("type") != "user"
            or not isinstance(message, Mapping)
            or message.get("role") != "user"
        ):
            continue
        key, uuid = record.get("promptId") or "", record.get("uuid")
        if not isinstance(key, str) or key not in submitted or not isinstance(uuid, str):
            continue
        prompt = submitted[key]
        text = matching_text(message.get("content"), prompt.fingerprint)
        if text is not None:
            found[key] = (
                uuid,
                Message(content=text, timestamp=datetime.fromisoformat(prompt.submitted)),
            )

    if current not in found or (not current and found[current][0] not in answered):
        return None
    messages = [
        found[p.key][1]
        for p in pending
        if p.key in found and (p.key == current or found[p.key][0] in answered)
    ]
    return found[current][0], messages
