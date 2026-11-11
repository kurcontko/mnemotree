"""Moved to :mod:`mnemotree.storage.schema`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.storage import schema as _module

if TYPE_CHECKING:
    from mnemotree.storage.schema import *  # noqa: F403

sys.modules[__name__] = _module
