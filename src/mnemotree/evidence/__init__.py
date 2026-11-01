"""The 1.0 import paths of the engine, kept importable.

The modules moved to ``mnemotree.engine``, ``mnemotree.core``, ``.ingest``,
``.retrieval``, ``.storage``, ``.providers``, ``.service``, ``.cli.evidence`` and
``.evaluation``. Each ``mnemotree.evidence.<module>`` name is a small alias that
resolves to the moved module itself, imported only when it is used.
"""

from ..core.models import (
    EmbeddingSpec,
    Evidence,
    IngestReceipt,
    Message,
    SearchOptions,
    SearchResult,
)
from ..engine import EvidenceEngine, Memory
from ..errors import ConfigurationError, IngestConflict, InvalidInput, MnemotreeError

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
