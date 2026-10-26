"""Persistent source-first memory: SQLite, FTS5, sqlite-vec, and explicit retrieval."""

from .engine import Memory
from .models import EmbeddingSpec, Evidence, IngestReceipt, Message, SearchOptions, SearchResult
from .store import IngestConflict

__all__ = [
    "Memory",
    "Message",
    "SearchOptions",
    "SearchResult",
    "EmbeddingSpec",
    "Evidence",
    "IngestReceipt",
    "IngestConflict",
]
