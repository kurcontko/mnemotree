"""Tracing and metrics helpers with graceful no-op fallback.

If ``opentelemetry-api`` is not installed every exported symbol still
works — ``tracer`` and ``meter`` become thin wrappers whose ``start_span``
/ ``create_histogram`` calls return context-manager stubs that do nothing.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_PII_LOGGING = False


def set_pii_logging(enabled: bool) -> None:
    """Toggle whether memory content is attached to span attributes."""
    global _PII_LOGGING  # noqa: PLW0603
    _PII_LOGGING = enabled


def pii_logging_enabled() -> bool:
    return _PII_LOGGING


# ---------------------------------------------------------------------------
# Try to import OpenTelemetry; fall back to no-ops.
# ---------------------------------------------------------------------------

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace

    tracer = otel_trace.get_tracer("mnemotree")
    meter = otel_metrics.get_meter("mnemotree")

    _HAS_OTEL = True

except ImportError:  # pragma: no cover — tested via mock
    _HAS_OTEL = False

    class _NoOpSpan:
        """Minimal span-like object for the no-op path."""

        def set_attribute(self, key: str, value: Any) -> None:
            pass

        def set_status(self, status: Any, description: str | None = None) -> None:
            pass

        def record_exception(self, exception: BaseException) -> None:
            pass

        def end(self) -> None:
            pass

        def __enter__(self) -> _NoOpSpan:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    class _NoOpTracer:
        """Drop-in replacement for ``opentelemetry.trace.Tracer``."""

        def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
            return _NoOpSpan()

        def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
            return contextlib.nullcontext(_NoOpSpan())

    class _NoOpHistogram:
        def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
            pass

    class _NoOpMeter:
        def create_histogram(self, name: str, **kwargs: Any) -> _NoOpHistogram:
            return _NoOpHistogram()

        def create_counter(self, name: str, **kwargs: Any) -> _NoOpHistogram:
            return _NoOpHistogram()

    tracer = _NoOpTracer()
    meter = _NoOpMeter()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

# Pre-built histograms for common operations.
_latency_histograms: dict[str, Any] = {}


def _get_histogram(name: str) -> Any:
    if name not in _latency_histograms:
        _latency_histograms[name] = meter.create_histogram(
            name=f"mnemotree.{name}.duration_ms",
            description=f"Latency of {name} in milliseconds",
        )
    return _latency_histograms[name]


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context-manager that opens a tracing span *and* records latency.

    Usage::

        with span("vector_search", {"top_k": 10}) as s:
            results = await store.get_similar_memories(...)
            s.set_attribute("result_count", len(results))
    """
    histogram = _get_histogram(name)
    started = time.perf_counter()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, v)
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            histogram.record(elapsed_ms)
            s.set_attribute("duration_ms", elapsed_ms)


def record_latency(name: str, elapsed_ms: float, attributes: dict[str, Any] | None = None) -> None:
    """Record a latency measurement without opening a span."""
    histogram = _get_histogram(name)
    histogram.record(elapsed_ms, attributes=attributes)
