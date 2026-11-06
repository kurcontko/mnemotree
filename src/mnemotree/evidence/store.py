"""The 1.0 path of :mod:`mnemotree.storage.store`, kept importable until 2.0.

Also re-exports the names the 1.0 module offered that now live elsewhere.
"""

from mnemotree.errors import IngestConflict as IngestConflict
from mnemotree.storage.schema import CHUNK_FTS as CHUNK_FTS
from mnemotree.storage.schema import SCHEMA_VERSION as SCHEMA_VERSION
from mnemotree.storage.store import *
from mnemotree.storage.vectors import vector_blob as vector_blob
