"""Mnemotree: persistent, source-first memory for LLM agents.

`Memory` is the namespace-bound API. The engine underneath, with per-call
namespaces and the full retrieval options, is `EvidenceEngine`.
"""

from .core.models import (
    EmbeddingSpec,
    Evidence,
    IngestReceipt,
    Message,
    SearchOptions,
    SearchResult,
)
from .engine import EvidenceEngine
from .errors import ConfigurationError, IngestConflict, InvalidInput, MnemotreeError
from .memory import Memory, MemoryItem, Recall

__all__ = [
    "ConfigurationError",
    "EmbeddingSpec",
    "Evidence",
    "EvidenceEngine",
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
