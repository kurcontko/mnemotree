"""The AML Add/Search request contract."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from ..core.models import Message, scope

# The platform never retries a 4xx, so an additive contract field must not fail a
# whole run: request models ignore unknown keys (internal Records stay strict), and
# top_k above the contract maximum is clamped rather than rejected.
MAX_TOP_K = 100


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
