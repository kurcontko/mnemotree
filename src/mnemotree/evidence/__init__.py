"""The 1.0 import paths, kept importable until 2.0.

The engine's modules moved to ``mnemotree.engine``, ``mnemotree.core``, ``.ingest``,
``.retrieval``, ``.storage``, ``.providers``, ``.service``, ``.cli.evidence`` and
``.evaluation``. Each ``mnemotree.evidence.<module>`` re-exports the names its 1.0
module offered and is imported only when used. Nothing outside this package imports it.
"""

from ..core.models import (
    EmbeddingSpec,
    Evidence,
    IngestReceipt,
    Message,
    SearchOptions,
    SearchResult,
)
from ..engine import EvidenceEngine
from ..errors import ConfigurationError, IngestConflict, InvalidInput, MnemotreeError

# The engine's 1.0 name. ``mnemotree.Memory`` is the namespace-bound API.
Memory = EvidenceEngine

__all__ = [
    "ConfigurationError",
    "EmbeddingSpec",
    "Evidence",
    "EvidenceEngine",
    "IngestConflict",
    "IngestReceipt",
    "InvalidInput",
    "Memory",
    "Message",
    "MnemotreeError",
    "SearchOptions",
    "SearchResult",
]
