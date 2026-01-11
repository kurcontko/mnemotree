"""Tests for store serialization helpers."""

from datetime import datetime, timezone
from enum import Enum

import pytest

from mnemotree.store.serialization import (
    normalize_entity_text,
    normalize_filter_value,
    safe_load_context,
    serialize_datetime,
    serialize_datetime_list,
)


class TestSerializeDatetime:
    """Tests for serialize_datetime()."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert serialize_datetime(None) is None

    def test_datetime_returns_isoformat(self):
        """datetime object returns ISO format string."""
        dt = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        result = serialize_datetime(dt)
        assert result == "2024-06-15T12:30:45+00:00"

    def test_string_passthrough(self):
        """String input passes through as-is."""
        assert serialize_datetime("2024-06-15") == "2024-06-15"

    def test_non_string_converts_to_str(self):
        """Non-string, non-datetime becomes str()."""
        assert serialize_datetime(12345) == "12345"  # type: ignore[arg-type]


class TestSerializeDatetimeList:
    """Tests for serialize_datetime_list()."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert serialize_datetime_list([]) == []

    def test_mixed_list(self):
        """Mixed datetime and string values."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = serialize_datetime_list([dt, "2024-02-02"])
        assert result == ["2024-01-01T00:00:00+00:00", "2024-02-02"]


class TestSafeLoadContext:
    """Tests for safe_load_context()."""

    def test_none_returns_empty_dict(self):
        """None returns empty dict."""
        assert safe_load_context(None) == {}

    def test_empty_string_returns_empty_dict(self):
        """Empty string returns empty dict."""
        assert safe_load_context("") == {}

    def test_valid_json_object(self):
        """Valid JSON object parses correctly."""
        result = safe_load_context('{"key": "value", "nested": {"a": 1}}')
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_invalid_json_returns_empty_dict(self):
        """Invalid JSON returns empty dict."""
        assert safe_load_context("not valid json") == {}

    def test_json_array_returns_empty_dict(self):
        """JSON array (non-dict) returns empty dict."""
        assert safe_load_context("[1, 2, 3]") == {}

    def test_json_primitive_returns_empty_dict(self):
        """JSON primitive returns empty dict."""
        assert safe_load_context('"just a string"') == {}


class TestNormalizeEntityText:
    """Tests for normalize_entity_text()."""

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        assert normalize_entity_text("  Hello  ") == "hello"

    def test_lowercases(self):
        """Text is lowercased."""
        assert normalize_entity_text("UPPERCASE") == "uppercase"

    def test_removes_article_the(self):
        """Removes leading 'the ' article."""
        assert normalize_entity_text("The Quick Brown Fox") == "quick brown fox"

    def test_removes_article_a(self):
        """Removes leading 'a ' article."""
        assert normalize_entity_text("A nice day") == "nice day"

    def test_removes_article_an(self):
        """Removes leading 'an ' article."""
        assert normalize_entity_text("An apple") == "apple"

    def test_no_article(self):
        """Text without article stays unchanged (except lowercase)."""
        assert normalize_entity_text("Python") == "python"


class SampleEnum(Enum):
    """Sample enum for testing."""

    FOO = "foo_value"
    BAR = 42


class TestNormalizeFilterValue:
    """Tests for normalize_filter_value()."""

    def test_datetime_to_isoformat(self):
        """datetime becomes ISO format string."""
        dt = datetime(2024, 3, 15, 10, 20, 30, tzinfo=timezone.utc)
        result = normalize_filter_value(dt)
        assert result == "2024-03-15T10:20:30+00:00"

    def test_enum_to_value(self):
        """Enum becomes its value."""
        assert normalize_filter_value(SampleEnum.FOO) == "foo_value"
        assert normalize_filter_value(SampleEnum.BAR) == 42

    def test_list_recursive(self):
        """List elements are recursively normalized."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = normalize_filter_value([dt, SampleEnum.FOO, "plain"])
        assert result == ["2024-01-01T00:00:00+00:00", "foo_value", "plain"]

    def test_plain_values_passthrough(self):
        """Plain values pass through unchanged."""
        assert normalize_filter_value("hello") == "hello"
        assert normalize_filter_value(123) == 123
        assert normalize_filter_value(True) is True
        assert normalize_filter_value(None) is None

    def test_nested_list(self):
        """Nested lists are recursively normalized."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = normalize_filter_value([[dt], [SampleEnum.FOO]])
        assert result == [["2024-01-01T00:00:00+00:00"], ["foo_value"]]
