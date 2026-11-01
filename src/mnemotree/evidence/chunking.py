"""Moved to :mod:`mnemotree.ingest.chunking`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.ingest import chunking as _module

if TYPE_CHECKING:
    from mnemotree.ingest.chunking import *  # noqa: F403

sys.modules[__name__] = _module
