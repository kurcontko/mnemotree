"""Moved to :mod:`mnemotree.engine`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree import engine as _module

if TYPE_CHECKING:
    from mnemotree.engine import *  # noqa: F403

sys.modules[__name__] = _module
