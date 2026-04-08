"""Optional OpenTelemetry observability for Mnemotree.

When ``opentelemetry-api`` is installed, spans and metrics are emitted for key
operations (vector search, NER, BM25, reranking, RRF, access updates).  When
it is *not* installed every helper degrades to a lightweight no-op so there is
zero overhead in production if you do not want tracing.

Quick start::

    pip install opentelemetry-api opentelemetry-sdk
    from mnemotree.observability import tracer, record_latency

PII redaction is on by default — memory content is never attached to span
attributes unless you explicitly opt-in via ``set_pii_logging(True)``.
"""

from .tracing import (
    meter,
    record_latency,
    set_pii_logging,
    span,
    tracer,
)

__all__ = [
    "meter",
    "record_latency",
    "set_pii_logging",
    "span",
    "tracer",
]
