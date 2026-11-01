"""Moved to :mod:`mnemotree.service.app`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.service import app as _module

if TYPE_CHECKING:
    from mnemotree.service.app import *  # noqa: F403

sys.modules[__name__] = _module
