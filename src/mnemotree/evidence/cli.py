"""The 1.0 path of :mod:`mnemotree.cli.evidence`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.cli.evidence import *
from mnemotree.config import Config as Config
from mnemotree.config import open_memory as open_memory
from mnemotree.config import secret as secret
from mnemotree.core.models import Record as Record
from mnemotree.core.protocols import Embedder as Embedder
from mnemotree.engine import EvidenceEngine
from mnemotree.retrieval.packing import TiktokenCounter as TiktokenCounter

# The engine's 1.0 name.
Memory = EvidenceEngine

# `python -m mnemotree.evidence.cli` still runs the CLI.
if __name__ == "__main__":
    from mnemotree.cli import evidence as cli

    cli.main()
