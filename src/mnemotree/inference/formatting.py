"""Shared formatting helpers for router & extractor models.

Used at both training time (scripts/training/) and inference time.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Extractor system prompt (shared between training & inference)
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Given a conversation turn with a "
    "type prefix (<|semantic|>, <|episodic|>, or <|procedural|>), extract "
    "structured information as a JSON object. Output ONLY valid JSON, no explanation."
)

# Label names — order matches the [episodic, semantic, procedural] vector
ROUTER_LABEL_NAMES: tuple[str, ...] = ("episodic", "semantic", "procedural")
EXTRACTOR_TYPE_NAMES: tuple[str, ...] = ("semantic", "episodic", "procedural")


# ---------------------------------------------------------------------------
# Router formatting
# ---------------------------------------------------------------------------


def format_router_input(record: dict) -> str:
    """Format a router record into a single string for encoder input.

    Output: "{speaker}: {text} [SEP] {ctx_speaker}: {ctx_text} [SEP] ..."
    """
    parts = [f'{record["speaker"]}: {record["text"]}']
    for turn in record.get("context", []):
        parts.append(f'{turn["speaker"]}: {turn["text"]}')
    return " [SEP] ".join(parts)


def build_router_record(content: str, context: dict[str, Any] | None = None) -> dict:
    """Adapt ``remember()`` arguments into a router input record.

    The router expects ``{"speaker": ..., "text": ..., "context": [...]}``.
    When *context* contains a ``"turns"`` key (list of speaker/text dicts)
    they are passed through; otherwise the content is treated as a single
    user utterance with no surrounding context.
    """
    ctx = context or {}
    turns: list[dict[str, str]] = ctx.get("turns", [])
    return {
        "speaker": ctx.get("speaker", "user"),
        "text": content,
        "context": turns,
    }


# ---------------------------------------------------------------------------
# Extractor formatting
# ---------------------------------------------------------------------------


def format_extractor_user_message(record: dict) -> str:
    """Format the user portion of an extractor chat message.

    Includes optional context block followed by the type-prefixed input.
    """
    parts = []
    context = record.get("context", [])
    if context:
        parts.append("Context:")
        for turn in context:
            parts.append(f'  {turn["speaker"]}: "{turn["text"]}"')
        parts.append("")
    parts.append(record["input"])
    return "\n".join(parts)


def format_extractor_messages(record: dict) -> list[dict]:
    """Build a full chat message list (system/user/assistant) for an extractor record."""
    return [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": format_extractor_user_message(record)},
        {
            "role": "assistant",
            "content": json.dumps(record["output"], ensure_ascii=False),
        },
    ]


def build_extractor_record(
    content: str,
    memory_type: str,
    context: dict[str, Any] | None = None,
) -> dict:
    """Build an extractor input record from ``remember()`` arguments.

    The extractor expects a ``{"input": "<|type|> text", "context": [...]}``
    record.  This helper constructs that from the raw content + classified type.
    """
    ctx = context or {}
    turns: list[dict[str, str]] = ctx.get("turns", [])
    return {
        "input": f"<|{memory_type}|> {content}",
        "context": turns,
    }
