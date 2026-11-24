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


# The transcript path below is the one Claude Code names in its own hook event. The
# hook only reads it, and keeps a line only when it belongs to the session and
# carries text whose SHA-256 the submit hook already recorded.
def transcript_size(path: Path) -> int:
    try:
        return path.stat().st_size  # NOSONAR
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
    with path.open("rb") as stream:  # NOSONAR
        size = stream.seek(0, 2)
        stream.seek(offset if offset <= size else 0)
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:  # Bad JSON or bad UTF-8: UnicodeError is a ValueError.
                continue
            if (
                isinstance(record, dict)
                and record.get("sessionId") == session
                and not record.get("isSidechain")
                and not record.get("isMeta")
            ):
                yield record


def submitted_message(
    record: Mapping[str, Any], submitted: Mapping[str, Submission]
) -> tuple[str, str, str] | None:
    """The prompt ID, message UUID and text of a user record that matches a submission."""
    message = record.get("message")
    if (
        record.get("type") != "user"
        or not isinstance(message, Mapping)
        or message.get("role") != "user"
    ):
        return None
    key, uuid = record.get("promptId") or "", record.get("uuid")
    if not isinstance(key, str) or key not in submitted or not isinstance(uuid, str):
        return None
    text = matching_text(message.get("content"), submitted[key].fingerprint)
    return None if text is None else (key, uuid, text)


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
        match = submitted_message(record, submitted)
        if match is not None:
            key, uuid, text = match
            sent = datetime.fromisoformat(submitted[key].submitted)
            found[key] = (uuid, Message(content=text, timestamp=sent))

    if current not in found or (not current and found[current][0] not in answered):
        return None
    messages = [
        found[p.key][1]
        for p in pending
        if p.key in found and (p.key == current or found[p.key][0] in answered)
    ]
    return found[current][0], messages
