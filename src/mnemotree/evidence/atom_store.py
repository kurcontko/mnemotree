"""The 1.0 path of :mod:`mnemotree.storage.atom_store`, kept importable until 2.0."""

from mnemotree.core.models import Atom as Atom
from mnemotree.core.models import Chunk as Chunk
from mnemotree.core.models import scope as scope
from mnemotree.storage.atom_store import ATOM_FTS as ATOM_FTS
from mnemotree.storage.atom_store import SCHEMA as SCHEMA
from mnemotree.storage.atom_store import AtomStore as AtomStore
from mnemotree.storage.lexical import create_fts as create_fts
from mnemotree.storage.lexical import fts_table as fts_table
from mnemotree.storage.lexical import index_text as index_text
from mnemotree.storage.lexical import match_expression as match_expression
