"""Moved to :mod:`mnemotree.retrieval.ranking`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.retrieval import ranking as _module

if TYPE_CHECKING:
    from mnemotree.retrieval.ranking import *  # noqa: F403

sys.modules[__name__] = _module
