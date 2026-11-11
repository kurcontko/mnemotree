"""The 1.0 path of :mod:`mnemotree.cli.evidence`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere, and keeps
the 1.0 behaviour of ``secret`` and ``open_memory``: a missing credential exits.
"""

from pathlib import Path as _Path

from mnemotree import config as _config
from mnemotree.cli.evidence import code_identity as code_identity
from mnemotree.cli.evidence import configure_logging as configure_logging
from mnemotree.cli.evidence import main as main
from mnemotree.cli.evidence import positive as positive
from mnemotree.cli.evidence import run as run
from mnemotree.config import Config as Config
from mnemotree.core.models import Record as Record
from mnemotree.core.models import SearchOptions as SearchOptions
from mnemotree.core.protocols import Embedder as Embedder
from mnemotree.engine import EvidenceEngine as Memory  # the engine's 1.0 name
from mnemotree.errors import ConfigurationError as _ConfigurationError
from mnemotree.evaluation import CorpusRecord as CorpusRecord
from mnemotree.evaluation import ablate as ablate
from mnemotree.evaluation import ablate_context as ablate_context
from mnemotree.evaluation import ablate_retrieval as ablate_retrieval
from mnemotree.evaluation import evaluate as evaluate
from mnemotree.evaluation import read_jsonl as read_jsonl
from mnemotree.retrieval.packing import TiktokenCounter as TiktokenCounter


def secret(name: str) -> str:
    try:
        return _config.secret(name)
    except _ConfigurationError as exc:
        raise SystemExit(f"mnemotree-evidence: {exc}") from None


def open_memory(path: _Path, config: Config) -> Memory:
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
