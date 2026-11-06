"""The 1.0 path of :mod:`mnemotree.engine`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.core.models import ChannelScore as ChannelScore
from mnemotree.engine import *
from mnemotree.engine import EvidenceEngine
from mnemotree.retrieval.ranking import QUERY_EMBED_CHARS as QUERY_EMBED_CHARS
from mnemotree.retrieval.ranking import QUERY_EMBED_TOKENS as QUERY_EMBED_TOKENS
from mnemotree.retrieval.ranking import cap_per_source as cap_per_source
from mnemotree.retrieval.ranking import chronological as chronological
from mnemotree.retrieval.ranking import fuse as fuse
from mnemotree.retrieval.ranking import model_query as model_query
from mnemotree.retrieval.ranking import role_first as role_first

# The engine's 1.0 name.
Memory = EvidenceEngine
