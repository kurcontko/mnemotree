"""The AML Add/Search request contract, plus the memory management routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from ..core.models import Message, scope

# The platform never retries a 4xx, so an additive contract field must not fail a
# whole run: request models ignore unknown keys (internal Records stay strict), and
# top_k above the contract maximum is clamped rather than rejected.
MAX_TOP_K = 100
MAX_GET_IDS = 100
MAX_LIST = 100
MAX_ROWID = 2**63 - 1


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


# A memory ID is a stored message's source ID, as Add and Search return it.
MemoryId = Annotated[str, Field(min_length=1, max_length=256)]


class GetRequest(Request):
    user_id: str
    ids: list[MemoryId] = Field(min_length=1, max_length=MAX_GET_IDS)

    @field_validator("user_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


class DeleteRequest(Request):
    user_id: str
    id: MemoryId

    @field_validator("user_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


class UserRequest(Request):
    """A request about one user's memories as a whole."""

    user_id: str

    @field_validator("user_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return scope(value)


class ListRequest(UserRequest):
    session_id: str | None = None
    # Clamped to MAX_LIST like top_k: fewer than asked is a valid page.
    limit: int = Field(default=MAX_LIST, ge=1, strict=True)
    # The next_cursor of the previous page; opaque to the caller.
    cursor: str | None = Field(default=None, pattern=r"^[0-9]{1,19}$")

    @field_validator("session_id")
    @classmethod
    def valid_session(cls, value: str | None) -> str | None:
        return scope(value) if value is not None else None

    @field_validator("cursor")
    @classmethod
    def valid_cursor(cls, value: str | None) -> str | None:
        # A rowid is a SQLite integer; a larger cursor could only fail in the store,
        # and the client must hear 422, not a 503 it would retry forever.
        if value is not None and int(value) > MAX_ROWID:
            raise ValueError("cursor is not one this service returned")
        return value
