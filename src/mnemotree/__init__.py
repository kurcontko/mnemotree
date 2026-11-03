"""Mnemotree: persistent, source-first memory for LLM agents.

`Memory` is the namespace-bound API. The engine underneath, with per-call
namespaces and the full retrieval options, is `mnemotree.evidence.Memory`.
"""

from .errors import ConfigurationError, InvalidInput, MnemotreeError
from .evidence import (
    EmbeddingSpec,
    Evidence,
    IngestConflict,
    IngestReceipt,
    Message,
    SearchOptions,
    SearchResult,
)
from .memory import Memory, MemoryItem, Recall

__all__ = [
    "ConfigurationError",
    "EmbeddingSpec",
    "Evidence",
    "IngestConflict",
    "IngestReceipt",
    "InvalidInput",
    "Memory",
    "MemoryItem",
    "Message",
    "MnemotreeError",
    "Recall",
    "SearchOptions",
    "SearchResult",
]
