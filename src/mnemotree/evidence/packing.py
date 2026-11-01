"""Moved to :mod:`mnemotree.retrieval.packing`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.retrieval import packing as _module

if TYPE_CHECKING:
    from mnemotree.retrieval.packing import *  # noqa: F403

sys.modules[__name__] = _module
