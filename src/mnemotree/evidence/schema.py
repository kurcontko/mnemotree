"""The evidence database: connection settings, tables, version and embedding spec."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from ..errors import ConfigurationError
from .models import EmbeddingSpec

SCHEMA_VERSION = "5"

# One FTS5 table per namespace, created by its first chunk. It holds
# lexical.index_text(chunk text), keyed by chunk rowid, so CJK bigrams are indexed
# without changing the stored source spans.
CHUNK_FTS = "evidence_fts"

TABLES = """
CREATE TABLE IF NOT EXISTS evidence_config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_requests(
    namespace TEXT NOT NULL, request_id TEXT NOT NULL, body_hash TEXT NOT NULL,
    receipt TEXT NOT NULL, PRIMARY KEY(namespace, request_id)
);
CREATE TABLE IF NOT EXISTS evidence_sources(
    rowid INTEGER PRIMARY KEY, namespace TEXT NOT NULL, id TEXT NOT NULL,
    request_id TEXT NOT NULL, session_id TEXT NOT NULL, position INTEGER NOT NULL,
    -- Order within the session across Add requests; rowids interleave sessions.
    session_seq INTEGER NOT NULL,
    content TEXT NOT NULL, role TEXT NOT NULL, timestamp TEXT,
    ingested_at TEXT NOT NULL, UNIQUE(namespace, id),
    UNIQUE(namespace, session_id, session_seq)
);
CREATE TABLE IF NOT EXISTS evidence_chunks(
    rowid INTEGER PRIMARY KEY, namespace TEXT NOT NULL, id TEXT NOT NULL,
    source_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL,
    FOREIGN KEY(namespace, source_id) REFERENCES evidence_sources(namespace, id) ON DELETE CASCADE,
    UNIQUE(namespace, id), UNIQUE(namespace, source_id, chunk_index)
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database in WAL mode with sqlite-vec loaded; autocommit, any thread."""
    db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.enable_load_extension(True)
        try:
            sqlite_vec.load(db)
        finally:
            db.enable_load_extension(False)
    except BaseException:
        db.close()
        raise
    return db


def create_tables(db: sqlite3.Connection) -> None:
    """Create the shared tables, refusing a database written by another schema version.

    The version is checked before creating anything: an older lexical index must
    not be extended in place with a different tokenization. FTS5 and extension
    failures are explicit startup failures, never silent fallbacks.
    """
    if db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_config'"
    ).fetchone():
        version = db.execute(
            "SELECT value FROM evidence_config WHERE key='schema_version'"
        ).fetchone()
        if version and version[0] != SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported evidence schema version {version[0]!r} (this build reads "
                f"{SCHEMA_VERSION!r}); rebuild into a fresh database"
            )
    db.executescript(TABLES)


def stored_embedding_spec(db: sqlite3.Connection) -> EmbeddingSpec | None:
    row = db.execute("SELECT value FROM evidence_config WHERE key='embedding_spec'").fetchone()
    return EmbeddingSpec.model_validate_json(row[0]) if row else None


def configure(db: sqlite3.Connection, requested: EmbeddingSpec | None) -> EmbeddingSpec | None:
    """Record the schema version and embedding spec; return the spec the database uses.

    Runs inside a write transaction. A database keeps the first spec it was given:
    a different one, or a spec for a corpus already indexed without vectors, needs
    a rebuild. Opening a vector database without a spec still reads it.
    """
    db.execute(
        "INSERT OR IGNORE INTO evidence_config VALUES('schema_version', ?)", (SCHEMA_VERSION,)
    )
    spec = stored_embedding_spec(db)
    if requested and spec and requested != spec:
        raise ConfigurationError("embedding signature mismatch; use a separately rebuilt database")
    if requested and not spec:
        if db.execute("SELECT 1 FROM evidence_chunks LIMIT 1").fetchone():
            raise ConfigurationError("existing lexical corpus needs an explicit vector rebuild")
        spec = requested
        db.execute(
            "INSERT INTO evidence_config VALUES('embedding_spec', ?)", (spec.model_dump_json(),)
        )
    if spec:
        # Partitioned by namespace: a KNN query reads only that namespace's vectors,
        # contiguously. Joining chunks to vec0 row by row instead took ~0.4 ms per
        # chunk (1 s for a 500K-token conversation).
        db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS evidence_vec USING vec0("
            f"chunk_rowid INTEGER PRIMARY KEY, namespace TEXT PARTITION KEY, "
            f"embedding float[{spec.dimensions}] distance_metric=cosine)"
        )
        db.execute(
            "CREATE TRIGGER IF NOT EXISTS evidence_vectors_delete "
            "AFTER DELETE ON evidence_chunks BEGIN "
            "DELETE FROM evidence_vec WHERE chunk_rowid=old.rowid; END"
        )
    return spec
