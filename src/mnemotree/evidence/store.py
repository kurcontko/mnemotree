"""The 1.0 path of :mod:`mnemotree.storage.store`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.core.models import Atom as Atom
from mnemotree.core.models import Chunk as Chunk
from mnemotree.core.models import ContextSpan as ContextSpan
from mnemotree.core.models import EmbeddingSpec as EmbeddingSpec
from mnemotree.core.models import IngestReceipt as IngestReceipt
from mnemotree.core.models import Source as Source
from mnemotree.core.models import digest as digest
from mnemotree.core.models import scope as scope
from mnemotree.errors import IngestConflict as IngestConflict
from mnemotree.storage.atom_store import AtomStore as AtomStore
from mnemotree.storage.lexical import create_fts as create_fts
from mnemotree.storage.lexical import fts_table as fts_table
from mnemotree.storage.lexical import index_text as index_text
from mnemotree.storage.lexical import match_expression as match_expression
from mnemotree.storage.schema import CHUNK_FTS as CHUNK_FTS
from mnemotree.storage.schema import SCHEMA_VERSION as SCHEMA_VERSION
from mnemotree.storage.store import SQLiteEvidenceStore as SQLiteEvidenceStore
from mnemotree.storage.store import now as now
from mnemotree.storage.vectors import vector_blob as vector_blob
