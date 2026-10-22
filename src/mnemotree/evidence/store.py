"""One SQLite transaction owns sources, FTS5, vectors, and ingestion receipts."""

from __future__ import annotations

import math
import re
import sqlite3
import struct
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from .atom_store import AtomStore
from .atoms import Atom
from .models import Chunk, EmbeddingSpec, IngestReceipt, Source, digest, scope

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_requests(
    namespace TEXT NOT NULL, request_id TEXT NOT NULL, body_hash TEXT NOT NULL,
    receipt TEXT NOT NULL, PRIMARY KEY(namespace, request_id)
);
CREATE TABLE IF NOT EXISTS evidence_sources(
    rowid INTEGER PRIMARY KEY, namespace TEXT NOT NULL, id TEXT NOT NULL,
    request_id TEXT NOT NULL, session_id TEXT NOT NULL, position INTEGER NOT NULL,
    content TEXT NOT NULL, role TEXT NOT NULL, timestamp TEXT,
    ingested_at TEXT NOT NULL, UNIQUE(namespace, id)
);
CREATE INDEX IF NOT EXISTS evidence_sources_session
    ON evidence_sources(namespace, session_id, rowid);
CREATE TABLE IF NOT EXISTS evidence_chunks(
    rowid INTEGER PRIMARY KEY, namespace TEXT NOT NULL, id TEXT NOT NULL,
    source_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL,
    FOREIGN KEY(namespace, source_id) REFERENCES evidence_sources(namespace, id) ON DELETE CASCADE,
    UNIQUE(namespace, id), UNIQUE(namespace, source_id, chunk_index)
);
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    text, content='evidence_chunks', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS evidence_chunks_insert AFTER INSERT ON evidence_chunks BEGIN
    INSERT INTO evidence_fts(rowid, text) VALUES(new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS evidence_chunks_delete AFTER DELETE ON evidence_chunks BEGIN
    INSERT INTO evidence_fts(evidence_fts, rowid, text) VALUES('delete', old.rowid, old.text);
END;
"""


class IngestConflict(ValueError):
    """A request or source ID was reused for a different occurrence."""


def vector_blob(values: Sequence[float], dimensions: int) -> bytes:
    if len(values) != dimensions:
        raise ValueError(f"expected {dimensions} embedding dimensions, got {len(values)}")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    try:
        blob = struct.pack(f"<{dimensions}f", *vector)
    except (OverflowError, struct.error) as exc:
        raise ValueError("embedding must fit float32") from exc
    floats = struct.unpack(f"<{dimensions}f", blob)
    if not all(math.isfinite(value) for value in floats) or not any(floats):
        raise ValueError("cosine embeddings must be finite and nonzero in float32")
    # sqlite-vec accumulates cosine products in float32. Unit-normalizing here
    # prevents finite large values from overflowing those products.
    norm = math.hypot(*floats)
    return struct.pack(f"<{dimensions}f", *(value / norm for value in floats))


class SQLiteEvidenceStore:
    """Persistent local store. Exact cosine search filters the corpus before top-k.

    No ANN approximation or query-time reembedding is hidden in the baseline.
    Each instance owns a connection; WAL permits independent readers and writers.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_spec: EmbeddingSpec | None = None,
        extractor_config: dict | None = None,
    ):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA busy_timeout=10000")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.enable_load_extension(True)
            try:
                sqlite_vec.load(self.db)
            finally:
                self.db.enable_load_extension(False)
            # FTS5/extension failures are explicit startup failures, never silent fallbacks.
            self.db.executescript(_SCHEMA)
            with self.transaction():
                version = self.db.execute(
                    "SELECT value FROM evidence_config WHERE key='schema_version'"
                ).fetchone()
                if version and version[0] != "1":
                    raise ValueError("unsupported evidence schema version")
                self.db.execute(
                    "INSERT OR IGNORE INTO evidence_config VALUES('schema_version', '1')"
                )
                row = self.db.execute(
                    "SELECT value FROM evidence_config WHERE key='embedding_spec'"
                ).fetchone()
                self.embedding_spec = EmbeddingSpec.model_validate_json(row[0]) if row else None
                if embedding_spec and self.embedding_spec and embedding_spec != self.embedding_spec:
                    raise ValueError(
                        "embedding signature mismatch; use a separately rebuilt database"
                    )
                if embedding_spec and not self.embedding_spec:
                    if self.db.execute("SELECT 1 FROM evidence_chunks LIMIT 1").fetchone():
                        raise ValueError("existing lexical corpus needs an explicit vector rebuild")
                    self.embedding_spec = embedding_spec
                    self.db.execute(
                        "INSERT INTO evidence_config VALUES('embedding_spec', ?)",
                        (embedding_spec.model_dump_json(),),
                    )
                if self.embedding_spec:
                    dimensions = self.embedding_spec.dimensions
                    self.db.execute(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS evidence_vec USING vec0("
                        f"chunk_rowid INTEGER PRIMARY KEY, embedding float[{dimensions}] "
                        "distance_metric=cosine)"
                    )
                    self.db.execute(
                        "CREATE TRIGGER IF NOT EXISTS evidence_vectors_delete "
                        "AFTER DELETE ON evidence_chunks BEGIN "
                        "DELETE FROM evidence_vec WHERE chunk_rowid=old.rowid; END"
                    )
            self.atoms = AtomStore(
                self.db, self.embedding_spec.dimensions if self.embedding_spec else None
            )
            with self.transaction():
                self.extractor_config = self.atoms.configure(extractor_config)
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self, *, read: bool = False) -> Iterator[None]:
        with self._lock:
            self.db.execute("BEGIN" if read else "BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def receipt(self, namespace: str, request_id: str, body_hash: str) -> IngestReceipt | None:
        scope(namespace)
        with self._lock:
            row = self.db.execute(
                "SELECT body_hash, receipt FROM evidence_requests WHERE namespace=? AND request_id=?",
                (namespace, request_id),
            ).fetchone()
            if row is None:
                return None
            if row["body_hash"] != body_hash:
                raise IngestConflict("request_id already belongs to a different payload")
            receipt = IngestReceipt.model_validate_json(row["receipt"])
            count = self.db.execute(
                "SELECT count(*) FROM evidence_sources WHERE namespace=? AND request_id=?",
                (namespace, request_id),
            ).fetchone()[0]
            if count != len(receipt.source_ids):
                raise IngestConflict(
                    "this request contains forgotten sources; replay cannot restore them"
                )
            return receipt.model_copy(update={"replayed": True})

    def ingest(
        self,
        namespace: str,
        request_id: str,
        session_id: str,
        body_hash: str,
        sources: Sequence[Source],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]] | None,
        atoms: Sequence[Atom] = (),
        atom_vectors: Sequence[Sequence[float]] | None = None,
    ) -> IngestReceipt:
        scope(namespace)
        if not sources or len({source.id for source in sources}) != len(sources):
            raise ValueError("a request needs sources with distinct IDs")
        if any(
            source.namespace != namespace
            or source.request_id != request_id
            or source.session_id != session_id
            for source in sources
        ):
            raise ValueError("source scope does not match ingestion request")
        source_map = {source.id: source for source in sources}
        for chunk in chunks:
            source = source_map.get(chunk.source_id)
            if source is None or not 0 <= chunk.start < chunk.end <= len(source.content):
                raise ValueError("invalid chunk source span")
            if source.content[chunk.start : chunk.end] != chunk.text:
                raise ValueError("chunk is not an exact source span")
        if {chunk.source_id for chunk in chunks} != set(source_map):
            raise ValueError("every source must have searchable chunks")
        blobs: list[bytes] = []
        if self.embedding_spec:
            if vectors is None or len(vectors) != len(chunks):
                raise ValueError("every chunk requires an embedding before acknowledging ingestion")
            blobs = [vector_blob(vector, self.embedding_spec.dimensions) for vector in vectors]
        elif vectors is not None:
            raise ValueError("vectors supplied without an embedding specification")
        atom_blobs: list[bytes] = []
        for atom in atoms:
            if self.extractor_config is None or atom.extractor_signature != digest(
                self.extractor_config
            ):
                raise ValueError("atom extractor signature does not match the database")
            if not atom.spans:
                raise ValueError("atoms require source evidence")
            for span in atom.spans:
                source = source_map.get(span.source_id)
                if source is None or not 0 <= span.start < span.end <= len(source.content):
                    raise ValueError("atom evidence must belong to this ingestion request")
        if self.embedding_spec and atoms:
            if atom_vectors is None or len(atom_vectors) != len(atoms):
                raise ValueError("every atom requires an embedding")
            atom_blobs = [
                vector_blob(vector, self.embedding_spec.dimensions) for vector in atom_vectors
            ]
        elif atom_vectors:
            raise ValueError("unexpected atom vectors")

        with self.transaction():
            current_embedding = self.db.execute(
                "SELECT value FROM evidence_config WHERE key='embedding_spec'"
            ).fetchone()
            if (
                EmbeddingSpec.model_validate_json(current_embedding[0])
                if current_embedding
                else None
            ) != self.embedding_spec:
                raise ValueError("database configuration changed; reopen this Memory instance")
            if self.atoms.configure(self.extractor_config) != self.extractor_config:
                raise ValueError("database configuration changed; reopen this Memory instance")
            replay = self.receipt(namespace, request_id, body_hash)
            if replay:
                return replay
            try:
                for source in sources:
                    self.db.execute(
                        "INSERT INTO evidence_sources(namespace,id,request_id,session_id,position,"
                        "content,role,timestamp,ingested_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            namespace,
                            source.id,
                            request_id,
                            session_id,
                            source.position,
                            source.content,
                            source.role,
                            source.timestamp.isoformat() if source.timestamp else None,
                            source.ingested_at.isoformat(),
                        ),
                    )
                for index, chunk in enumerate(chunks):
                    cursor = self.db.execute(
                        "INSERT INTO evidence_chunks(namespace,id,source_id,chunk_index,start,end,text) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (
                            namespace,
                            chunk.id,
                            chunk.source_id,
                            chunk.index,
                            chunk.start,
                            chunk.end,
                            chunk.text,
                        ),
                    )
                    if blobs:
                        self.db.execute(
                            "INSERT INTO evidence_vec(chunk_rowid,embedding) VALUES(?,?)",
                            (cursor.lastrowid, blobs[index]),
                        )
                self.atoms.insert(namespace, atoms, atom_blobs)
            except sqlite3.IntegrityError as exc:
                raise IngestConflict(
                    "source IDs must identify distinct occurrences within a namespace"
                ) from exc
            receipt = IngestReceipt(
                namespace=namespace,
                request_id=request_id,
                session_id=session_id,
                source_ids=tuple(source.id for source in sources),
                chunk_count=len(chunks),
                atom_count=len(atoms),
            )
            self.db.execute(
                "INSERT INTO evidence_requests VALUES(?,?,?,?)",
                (namespace, request_id, body_hash, receipt.model_dump_json()),
            )
            return receipt

    def get_source(self, namespace: str, source_id: str) -> Source | None:
        scope(namespace)
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM evidence_sources WHERE namespace=? AND id=?", (namespace, source_id)
            ).fetchone()
            if row is None:
                return None
            return Source.model_validate({key: row[key] for key in Source.model_fields})

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source_id=row["source_id"],
            index=row["chunk_index"],
            start=row["start"],
            end=row["end"],
            text=row["text"],
        )

    def lexical(self, namespace: str, query: str, limit: int) -> list[tuple[Chunk, float]]:
        scope(namespace)
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:64]
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self._lock:
            rows = self.db.execute(
                "SELECT c.*, bm25(evidence_fts) AS score FROM evidence_fts "
                "JOIN evidence_chunks c ON c.rowid=evidence_fts.rowid "
                "WHERE evidence_fts MATCH ? AND c.namespace=? ORDER BY score,c.id LIMIT ?",
                (match, namespace, limit),
            ).fetchall()
        return [(self._chunk(row), -float(row["score"])) for row in rows]

    def dense(
        self, namespace: str, vector: Sequence[float], limit: int
    ) -> list[tuple[Chunk, float]]:
        scope(namespace)
        if self.embedding_spec is None:
            raise ValueError("dense retrieval needs an indexed embedding model")
        blob = vector_blob(vector, self.embedding_spec.dimensions)
        with self._lock:
            rows = self.db.execute(
                "SELECT c.*, vec_distance_cosine(v.embedding, ?) AS distance "
                "FROM evidence_chunks c JOIN evidence_vec v ON v.chunk_rowid=c.rowid "
                "WHERE c.namespace=? ORDER BY distance,c.id LIMIT ?",
                (blob, namespace, limit),
            ).fetchall()
        return [(self._chunk(row), 1.0 - float(row["distance"])) for row in rows]

    def window(self, namespace: str, chunk: Chunk, neighbors: int) -> tuple[Source, int, int]:
        scope(namespace)
        with self._lock:
            source = self.get_source(namespace, chunk.source_id)
            if source is None:
                raise ValueError("source is not present in this namespace")
            row = self.db.execute(
                "SELECT min(start),max(end) FROM evidence_chunks "
                "WHERE namespace=? AND source_id=? AND chunk_index BETWEEN ? AND ?",
                (namespace, chunk.source_id, chunk.index - neighbors, chunk.index + neighbors),
            ).fetchone()
            return source, row[0], row[1]

    def forget(self, namespace: str, source_id: str) -> bool:
        scope(namespace)
        with self.transaction():
            cursor = self.db.execute(
                "DELETE FROM evidence_sources WHERE namespace=? AND id=?", (namespace, source_id)
            )
            return cursor.rowcount > 0

    def stats(self, namespace: str) -> dict[str, int]:
        scope(namespace)
        with self.transaction(read=True):
            return {
                table: self.db.execute(
                    f"SELECT count(*) FROM evidence_{table} WHERE namespace=?", (namespace,)
                ).fetchone()[0]
                for table in ("sources", "chunks", "requests")
            }


def now() -> datetime:
    return datetime.now(timezone.utc)
