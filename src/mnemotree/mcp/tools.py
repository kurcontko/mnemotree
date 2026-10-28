"""MCP tools over one namespace-bound Memory, independent of the transport.

Arguments are Pydantic models; their JSON Schemas are what tools/list advertises,
with nested definitions inlined because some clients do not resolve ``$ref``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..evidence.models import Message
from ..memory import Memory, MemoryItem


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


TIMESTAMP = "When it was said: ISO 8601 with a timezone, e.g. 2026-10-28T09:30:00+01:00."


class RememberArguments(Arguments):
    content: str = Field(min_length=1, description="The text to remember, stored exactly as given.")
    role: str = Field(default="user", description="Who said it, e.g. user or assistant.")
    session: str | None = Field(
        default=None,
        description="Conversation it belongs to. Omit for a standalone memory.",
    )
    timestamp: datetime | None = Field(default=None, description=TIMESTAMP)
    id: str | None = Field(default=None, description="Stable ID; generated when omitted.")
    request_id: str | None = Field(
        default=None,
        description="Reuse to retry safely: the same request is stored once.",
    )


class Turn(Arguments):
    content: str = Field(min_length=1)
    role: str = "user"
    id: str | None = None
    timestamp: datetime | None = Field(default=None, description=TIMESTAMP)


class RememberManyArguments(Arguments):
    session: str = Field(min_length=1, description="Conversation the turns belong to.")
    messages: list[Turn] = Field(min_length=1, description="Consecutive turns, in order.")
    request_id: str | None = Field(
        default=None,
        description="Reuse to retry safely: the same request is stored once.",
    )


class RecallArguments(Arguments):
    query: str = Field(min_length=1, description="What to look for.")
    limit: int | None = Field(
        default=None, ge=1, le=100, description="Maximum number of evidence items."
    )


class GetArguments(Arguments):
    ids: list[str] = Field(min_length=1, max_length=100, description="Memory IDs to fetch.")


class ForgetArguments(Arguments):
    id: str = Field(min_length=1, description="ID of the memory to delete.")


def item(memory: MemoryItem) -> dict[str, Any]:
    return memory.model_dump(mode="json")


async def remember(memory: Memory, args: RememberArguments) -> dict[str, Any]:
    return {"memory": item(await memory.remember(**args.model_dump()))}


async def remember_many(memory: Memory, args: RememberManyArguments) -> dict[str, Any]:
    items = await memory.remember_many(
        [Message(**turn.model_dump()) for turn in args.messages],
        session=args.session,
        request_id=args.request_id,
    )
    return {"memories": [item(found) for found in items]}


async def recall(memory: Memory, args: RecallArguments) -> dict[str, Any]:
    result = await memory.recall(args.query, limit=args.limit)
    return {
        "context": result.context,
        "memories": [item(found) for found in result.items],
        "packed_tokens": result.trace.packed_tokens,
    }


async def get(memory: Memory, args: GetArguments) -> dict[str, Any]:
    found = {id: await memory.get(id) for id in dict.fromkeys(args.ids)}
    return {
        "memories": [item(value) for value in found.values() if value is not None],
        "missing": [id for id, value in found.items() if value is None],
    }


async def forget(memory: Memory, args: ForgetArguments) -> dict[str, Any]:
    return {"forgotten": await memory.forget(args.id)}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[Arguments]
    run: Callable[[Memory, Any], Awaitable[dict[str, Any]]]
    annotations: dict[str, bool] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": inline_refs(self.arguments.model_json_schema()),
            "annotations": self.annotations,
        }


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(definitions[node["$ref"].rsplit("/", 1)[-1]])
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(value) for value in node]
        return node

    return resolve(schema)


WRITE = {"readOnlyHint": False, "destructiveHint": False}
READ = {"readOnlyHint": True}

TOOLS = {
    tool.name: tool
    for tool in (
        Tool(
            "remember",
            "Store one message in long-term memory, verbatim.",
            RememberArguments,
            remember,
            WRITE,
        ),
        Tool(
            "remember_many",
            "Store consecutive turns of one conversation, in order, as one write.",
            RememberManyArguments,
            remember_many,
            WRITE,
        ),
        Tool(
            "recall",
            "Search long-term memory. Returns packed context to read and the memories "
            "it came from. Read-only.",
            RecallArguments,
            recall,
            READ,
        ),
        Tool("get", "Fetch memories by ID. Read-only.", GetArguments, get, READ),
        Tool(
            "forget",
            "Delete a memory permanently.",
            ForgetArguments,
            forget,
            {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
        ),
    )
}
