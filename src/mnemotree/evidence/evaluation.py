"""The 1.0 path of :mod:`mnemotree.evaluation`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.engine import EvidenceEngine
from mnemotree.evaluation import *

# The engine's 1.0 name.
Memory = EvidenceEngine
