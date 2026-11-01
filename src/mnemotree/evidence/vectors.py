"""Moved to :mod:`mnemotree.storage.vectors`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.storage import vectors as _module

if TYPE_CHECKING:
    from mnemotree.storage.vectors import *  # noqa: F403

sys.modules[__name__] = _module
