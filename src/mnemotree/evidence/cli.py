"""The 1.0 path of :mod:`mnemotree.cli.evidence`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere, and keeps
the 1.0 behaviour of ``secret`` and ``open_memory``: a missing credential exits.
"""

from pathlib import Path

from mnemotree import config as _config
from mnemotree.cli.evidence import *
from mnemotree.config import Config as Config
from mnemotree.core.models import Record as Record
from mnemotree.core.protocols import Embedder as Embedder
from mnemotree.engine import EvidenceEngine
from mnemotree.errors import ConfigurationError
from mnemotree.retrieval.packing import TiktokenCounter as TiktokenCounter

# The engine's 1.0 name.
Memory = EvidenceEngine


def secret(name: str) -> str:
    try:
        return _config.secret(name)
    except ConfigurationError as exc:
        raise SystemExit(f"mnemotree-evidence: {exc}") from None


def open_memory(path: Path, config: Config) -> EvidenceEngine:
    """``Config.open``, checking the provider credentials first as 1.0 did."""
    for provider in (config.embedding, config.extraction):
        name = getattr(provider, "api_key_env", None)
        if name is not None:
            secret(name)
    return config.open(path)


# `python -m mnemotree.evidence.cli` still runs the CLI.
if __name__ == "__main__":
    from mnemotree.cli import evidence as cli

    cli.main()
