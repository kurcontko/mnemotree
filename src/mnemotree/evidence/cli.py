"""Moved to :mod:`mnemotree.cli.evidence`; this 1.0 path stays importable as the same module."""

import sys
from typing import TYPE_CHECKING

from mnemotree.cli import evidence as _module

if TYPE_CHECKING:
    from mnemotree.cli.evidence import *  # noqa: F403

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
