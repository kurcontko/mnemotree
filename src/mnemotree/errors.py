"""
Error taxonomy for Mnemotree.

Defines a consistent hierarchy of exceptions used throughout the codebase
to provide clear error semantics and improve debugging.
"""

from __future__ import annotations


class MnemotreeError(Exception):
    """Base exception for all Mnemotree errors.
    
    All custom exceptions in Mnemotree should inherit from this class
    to enable catch-all error handling when needed.
    """

    pass


class StoreError(MnemotreeError):
    """Store operation failed.
    
    Raised when a storage backend operation fails due to connectivity,
    database errors, or other infrastructure issues.
    
    Attributes:
        message: Human-readable error description
        store_type: Type of store that failed (e.g., "chroma", "neo4j")
        memory_id: Optional memory ID associated with the error
        original_error: Optional original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        store_type: str,
        memory_id: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.store_type = store_type
        self.memory_id = memory_id
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [self.args[0]]
        if self.store_type:
            parts.append(f"store={self.store_type}")
        if self.memory_id:
            parts.append(f"memory_id={self.memory_id}")
        return f"{': '.join(parts)}"


class SerializationError(MnemotreeError):
    """Failed to serialize or deserialize data.
    
    Raised when data cannot be converted between different formats
    (e.g., JSON parsing errors, datetime conversion failures).
    
    This typically indicates data corruption or format incompatibility.
    """

    pass


class InvalidQueryError(MnemotreeError):
    """Invalid query parameters or unsupported query operation.
    
    Raised when a query contains invalid parameters, uses unsupported
    operators, or violates query constraints.
    
    This is a user error and should be handled by validating input.
    """

    pass


class DependencyError(MnemotreeError):
    """Required dependency not available or failed to load.
    
    Raised when an optional dependency is required but not installed,
    or when a dependency fails to initialize properly.
    
    Example: NER backend not installed, embedding model not available.
    """

    def __init__(self, message: str, dependency: str):
        super().__init__(message)
        self.dependency = dependency

    def __str__(self) -> str:
        return f"{self.args[0]} (dependency: {self.dependency})"


class MemoryNotFoundError(MnemotreeError):
    """Requested memory does not exist.
    
    Raised when attempting to retrieve, update, or delete a memory
    that doesn't exist in the store.
    
    Note: get_memory() returns None instead of raising this exception
    to distinguish between "not found" and "error during retrieval".
    """

    def __init__(self, memory_id: str):
        super().__init__(f"Memory not found: {memory_id}")
        self.memory_id = memory_id


class ConfigurationError(MnemotreeError):
    """Invalid configuration or settings.
    
    Raised when configuration values are invalid, conflicting,
    or missing required parameters.
    """

    pass


class IndexError(MnemotreeError):
    """Index operation failed.
    
    Raised when building or querying an index (e.g., BM25, graph index)
    fails due to data issues or index corruption.
    """

    pass


# Error handling guidelines:
#
# 1. Store CRUD operations:
#    - get_memory() -> Return None for "not found" (expected case)
#    - get_memory() -> Raise StoreError for infrastructure failures
#    - store_memory() -> Raise StoreError on failure
#    - delete_memory() -> Return False for "not found", True for deleted
#    - delete_memory() -> Raise StoreError for infrastructure failures
#
# 2. Partial failures in batch operations:
#    - Continue processing remaining items
#    - Log warnings with context for each failure
#    - Track failure count and log summary at end
#    - Preserve last error for debugging (exc_info=last_error)
#
# 3. Error context:
#    - Always include store_type in store errors
#    - Include memory_id when available
#    - Use logger.exception() to preserve stack traces
#    - Add structured logging context via extra={}
#
# 4. Re-raising:
#    - Use "raise StoreError(...) from e" to preserve cause chain
#    - Don't catch-and-reraise without adding context
#
# 5. Silent failures are forbidden:
#    - Never use bare "except: pass" or "except Exception: continue"
#    - Always log failures, even if continuing
#    - Make partial results visible through logging
