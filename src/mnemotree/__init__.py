from __future__ import annotations

from importlib import import_module

from .errors import (
    ConfigurationError,
    DependencyError,
    IndexError,
    InvalidQueryError,
    MemoryNotFoundError,
    MnemotreeError,
    SerializationError,
    StoreError,
)

__all__ = [
    "Memory",
    "Message",
    "SearchOptions",
    "MemoryCore",
    "MemoryCoreBuilder",
    "MemoryMode",
    "RememberOptions",
    "RecallFilters",
    "RecallOptions",
    # Knowledge graph types
    "LinkType",
    "MemoryLink",
    # Error types
    "MnemotreeError",
    "StoreError",
    "SerializationError",
    "InvalidQueryError",
    "DependencyError",
    "MemoryNotFoundError",
    "ConfigurationError",
    "IndexError",
]


_EXPORTS = {
    "Memory": ".evidence",
    "Message": ".evidence",
    "SearchOptions": ".evidence",
    "MemoryCoreBuilder": ".core.builder",
    **dict.fromkeys(
        ("MemoryCore", "MemoryMode", "RecallFilters", "RecallOptions", "RememberOptions"),
        ".core.memory",
    ),
    "LinkType": ".core.models",
    "MemoryLink": ".core.models",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
    except ModuleNotFoundError as exc:
        if name not in {"Memory", "Message", "SearchOptions"}:
            raise ModuleNotFoundError(
                "The legacy MemoryCore requires pip install 'mnemotree[legacy]'."
            ) from exc
        raise
    globals()[name] = value
    return value
