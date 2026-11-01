"""Moved to :mod:`mnemotree.core.models`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.core import models as _module

if TYPE_CHECKING:
    from mnemotree.core.models import *  # noqa: F403

sys.modules[__name__] = _module
