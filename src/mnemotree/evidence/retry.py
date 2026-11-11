"""Moved to :mod:`mnemotree.providers.retry`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.providers import retry as _module

if TYPE_CHECKING:
    from mnemotree.providers.retry import *  # noqa: F403

sys.modules[__name__] = _module
