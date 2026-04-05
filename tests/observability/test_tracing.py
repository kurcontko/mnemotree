"""Tests for observability module (no-op fallback path)."""

import pytest

from mnemotree.observability import (
    meter,
    record_latency,
    set_pii_logging,
    span,
    tracer,
)
from mnemotree.observability.tracing import pii_logging_enabled


class TestNoOpFallback:
    def test_tracer_exists(self) -> None:
        assert tracer is not None

    def test_meter_exists(self) -> None:
        assert meter is not None

    def test_span_context_manager(self) -> None:
        with span("test_op", {"key": "value"}) as s:
            s.set_attribute("result_count", 42)

    def test_span_records_exception(self) -> None:
        with pytest.raises(ValueError, match="test error"), span("failing_op"):
            raise ValueError("test error")

    def test_record_latency(self) -> None:
        record_latency("vector_search", 12.5, {"top_k": 10})

    def test_histogram_caching(self) -> None:
        from mnemotree.observability.tracing import _get_histogram

        h1 = _get_histogram("search")
        h2 = _get_histogram("search")
        assert h1 is h2


class TestPIILogging:
    def test_default_off(self) -> None:
        assert pii_logging_enabled() is False

    def test_toggle(self) -> None:
        set_pii_logging(True)
        assert pii_logging_enabled() is True
        set_pii_logging(False)
        assert pii_logging_enabled() is False
