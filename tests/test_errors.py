"""Tests for Mnemotree error classes.

This module tests the exception hierarchy and error formatting.
"""

import pytest

from mnemotree.errors import (
    ConfigurationError,
    DependencyError,
    IndexError,
    InvalidQueryError,
    MemoryNotFoundError,
    MnemotreeError,
    SerializationError,
    StoreError,
)


class TestMnemotreeErrorHierarchy:
    """Tests for exception inheritance."""

    def test_all_errors_inherit_from_mnemotree_error(self):
        """All custom exceptions should inherit from MnemotreeError."""
        assert issubclass(StoreError, MnemotreeError)
        assert issubclass(SerializationError, MnemotreeError)
        assert issubclass(InvalidQueryError, MnemotreeError)
        assert issubclass(DependencyError, MnemotreeError)
        assert issubclass(MemoryNotFoundError, MnemotreeError)
        assert issubclass(ConfigurationError, MnemotreeError)
        assert issubclass(IndexError, MnemotreeError)

    def test_mnemotree_error_inherits_from_exception(self):
        """MnemotreeError should inherit from Exception."""
        assert issubclass(MnemotreeError, Exception)

    def test_catch_all_with_mnemotree_error(self):
        """Can catch all Mnemotree errors with MnemotreeError."""
        errors = [
            StoreError("test", store_type="test"),
            SerializationError("test"),
            InvalidQueryError("test"),
            DependencyError("test", dependency="test"),
            MemoryNotFoundError("memory-123"),
            ConfigurationError("test"),
            IndexError("test"),
        ]
        for error in errors:
            with pytest.raises(MnemotreeError):
                raise error


class TestStoreError:
    """Tests for StoreError."""

    def test_basic_instantiation(self):
        """StoreError can be created with required parameters."""
        error = StoreError("Connection failed", store_type="neo4j")
        assert "Connection failed" in str(error)
        assert "neo4j" in str(error)

    def test_with_memory_id(self):
        """StoreError includes memory_id in string representation."""
        error = StoreError(
            "Failed to store",
            store_type="chroma",
            memory_id="mem-abc123",
        )
        assert "chroma" in str(error)
        assert "mem-abc123" in str(error)

    def test_with_original_error(self):
        """StoreError preserves original exception."""
        original = ConnectionError("Network timeout")
        error = StoreError(
            "Store failed",
            store_type="sqlite",
            original_error=original,
        )
        assert error.original_error is original

    def test_attributes(self):
        """StoreError attributes are accessible."""
        error = StoreError(
            "message",
            store_type="milvus",
            memory_id="id-123",
            original_error=ValueError("test"),
        )
        assert error.store_type == "milvus"
        assert error.memory_id == "id-123"
        assert isinstance(error.original_error, ValueError)


class TestSerializationError:
    """Tests for SerializationError."""

    def test_basic_instantiation(self):
        """SerializationError can be created."""
        error = SerializationError("Failed to parse JSON")
        assert "Failed to parse JSON" in str(error)

    def test_can_be_raised_and_caught(self):
        """SerializationError can be raised and caught."""
        with pytest.raises(SerializationError, match="datetime"):
            raise SerializationError("Invalid datetime format")


class TestInvalidQueryError:
    """Tests for InvalidQueryError."""

    def test_basic_instantiation(self):
        """InvalidQueryError can be created."""
        error = InvalidQueryError("Unsupported operator: $regex")
        assert "Unsupported operator" in str(error)


class TestDependencyError:
    """Tests for DependencyError."""

    def test_basic_instantiation(self):
        """DependencyError can be created with dependency name."""
        error = DependencyError("GLiNER not installed", dependency="gliner")
        assert "GLiNER not installed" in str(error)
        assert "gliner" in str(error)

    def test_dependency_attribute(self):
        """DependencyError has dependency attribute."""
        error = DependencyError("Missing", dependency="sentence-transformers")
        assert error.dependency == "sentence-transformers"

    def test_str_format(self):
        """DependencyError __str__ includes dependency name."""
        error = DependencyError("Neo4j driver required", dependency="neo4j")
        result = str(error)
        assert "Neo4j driver required" in result
        assert "dependency: neo4j" in result


class TestMemoryNotFoundError:
    """Tests for MemoryNotFoundError."""

    def test_basic_instantiation(self):
        """MemoryNotFoundError can be created with memory_id."""
        error = MemoryNotFoundError("mem-12345")
        assert "mem-12345" in str(error)
        assert "not found" in str(error).lower()

    def test_memory_id_attribute(self):
        """MemoryNotFoundError has memory_id attribute."""
        error = MemoryNotFoundError("uuid-abc-123")
        assert error.memory_id == "uuid-abc-123"

    def test_auto_formatted_message(self):
        """MemoryNotFoundError auto-formats the error message."""
        error = MemoryNotFoundError("test-id")
        assert str(error) == "Memory not found: test-id"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_instantiation(self):
        """ConfigurationError can be created."""
        error = ConfigurationError("Invalid embedding dimension")
        assert "Invalid embedding dimension" in str(error)


class TestIndexError:
    """Tests for IndexError."""

    def test_basic_instantiation(self):
        """IndexError can be created."""
        error = IndexError("BM25 index corrupt")
        assert "BM25 index corrupt" in str(error)
