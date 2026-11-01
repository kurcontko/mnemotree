"""Moved to :mod:`mnemotree.evaluation`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree import evaluation as _module

if TYPE_CHECKING:
    from mnemotree.evaluation import *  # noqa: F403

sys.modules[__name__] = _module
