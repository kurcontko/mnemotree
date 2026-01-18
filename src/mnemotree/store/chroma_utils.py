"""Shared ChromaDB client creation utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)


def create_chroma_client(
    *,
    host: str | None = None,
    port: int | None = None,
    ssl: bool = False,
    persist_directory: str | None = None,
    headers: dict[str, str] | None = None,
    store_type: str = "chroma",
) -> ClientAPI:
    """Create a ChromaDB client based on provided configuration.

    Args:
        host: Optional host address for remote ChromaDB instance
        port: Optional port for remote ChromaDB instance
        ssl: Whether to use SSL for remote connection
        persist_directory: Optional directory for local persistence
        headers: Optional headers for authentication/authorization
        store_type: Store type label for logging

    Returns:
        Configured ChromaDB client
    """
    from .logging import store_log_context

    if host and port:
        # Connect to remote ChromaDB instance
        client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers or {})
        logger.info(
            "Initialized remote ChromaDB client at %s:%s",
            host,
            port,
            extra=store_log_context(store_type),
        )
    elif persist_directory:
        # Use local persistence
        client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        logger.info(
            "Initialized local ChromaDB client at %s",
            persist_directory,
            extra=store_log_context(store_type),
        )
    else:
        # In-memory client for testing
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        logger.info(
            "Initialized in-memory ChromaDB client",
            extra=store_log_context(store_type),
        )

    return client
