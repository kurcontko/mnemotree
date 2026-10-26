from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from .core.builder import MemoryCoreBuilder
    from .core.memory import (
        MemoryCore,
        MemoryMode,
        RecallFilters,
        RecallOptions,
        RememberOptions,
    )
    from .core.models import LinkType, MemoryLink
    from .evidence import Memory, Message, SearchOptions

_EVIDENCE = ("Memory", "Message", "SearchOptions")
_LEGACY = (
    "MemoryCore",
    "MemoryCoreBuilder",
    "MemoryMode",
    "RememberOptions",
    "RecallFilters",
    "RecallOptions",
    # Knowledge graph types
    "LinkType",
    "MemoryLink",
)
_ERRORS = (
    "MnemotreeError",
    "StoreError",
    "SerializationError",
    "InvalidQueryError",
    "DependencyError",
    "MemoryNotFoundError",
    "ConfigurationError",
    "IndexError",
)

# Star-imports resolve every name in __all__, so the legacy names are listed
# only where the legacy extra is installed. Other extras (anthropic, google,
# aws, huggingface) bring langchain_core alone, which is not enough.
_LEGACY_MODULES = (
    "langchain",
    "langchain_core",
    "langchain_openai",
    "numpy",
    "sklearn",
    "networkx",
    "spacy",
    "openai",
)
__all__ = [
    *_EVIDENCE,
    *(_LEGACY if all(find_spec(module) for module in _LEGACY_MODULES) else ()),
    *_ERRORS,
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


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
    except ModuleNotFoundError as exc:
        if name not in _EVIDENCE:
            raise ModuleNotFoundError(
                "The legacy MemoryCore requires pip install 'mnemotree[legacy]'."
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
