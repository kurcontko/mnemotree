"""Atomic projections on the same connection/transaction as source ingestion."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import Any

from ..core.models import Chunk, scope
from ..errors import ConfigurationError
from ..ingest.atoms import Atom
from .lexical import create_fts, fts_table, index_text, match_expression

SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_atoms(
 rowid INTEGER PRIMARY KEY, namespace TEXT NOT NULL, id TEXT NOT NULL,
 text TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(namespace,id));
CREATE TABLE IF NOT EXISTS evidence_atom_spans(
 namespace TEXT NOT NULL, atom_id TEXT NOT NULL, source_id TEXT NOT NULL,
 start INTEGER NOT NULL, end INTEGER NOT NULL,
 PRIMARY KEY(namespace,atom_id,source_id,start,end),
 FOREIGN KEY(namespace,atom_id) REFERENCES evidence_atoms(namespace,id) ON DELETE CASCADE,
 FOREIGN KEY(namespace,source_id) REFERENCES evidence_sources(namespace,id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS evidence_atom_source ON evidence_atom_spans(namespace,source_id);
CREATE TRIGGER IF NOT EXISTS evidence_source_atoms_delete BEFORE DELETE ON evidence_sources BEGIN
 DELETE FROM evidence_atoms WHERE namespace=old.namespace AND id IN
 (SELECT atom_id FROM evidence_atom_spans WHERE namespace=old.namespace AND source_id=old.id); END;
"""

# One FTS5 table per namespace, as for chunks; see store.CHUNK_FTS.
ATOM_FTS = "evidence_atom_fts"


class AtomStore:
    """Internal helper; caller holds the parent store lock/transaction."""

    def __init__(self, db: sqlite3.Connection, dimensions: int | None):
        self.db = db
        db.executescript(SCHEMA)
        if dimensions is not None:
            db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS evidence_atom_vec USING vec0("
                f"atom_rowid INTEGER PRIMARY KEY, namespace TEXT PARTITION KEY, "
                f"embedding float[{dimensions}] distance_metric=cosine)"
            )
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS evidence_atom_vectors_delete "
                "AFTER DELETE ON evidence_atoms BEGIN DELETE FROM evidence_atom_vec "
                "WHERE atom_rowid=old.rowid; END"
            )

    def configure(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        row = self.db.execute("SELECT value FROM evidence_config WHERE key='extractor'").fetchone()
        saved = json.loads(row[0]) if row else None
        if config is not None:
            if saved is not None and saved != config:
                raise ConfigurationError(
                    "extractor signature mismatch; rebuild into a fresh database"
                )
            if saved is None:
                if self.db.execute("SELECT 1 FROM evidence_requests LIMIT 1").fetchone():
                    raise ConfigurationError("existing corpus needs an explicit extraction rebuild")
                self.db.execute(
                    "INSERT INTO evidence_config VALUES('extractor',?)", (json.dumps(config),)
                )
                saved = config
        return saved

    def insert(self, namespace: str, atoms: Sequence[Atom], blobs: Sequence[bytes]) -> None:
        table = fts_table(ATOM_FTS, namespace)
        if atoms:
            self.db.execute(create_fts(table))
        for index, atom in enumerate(atoms):
            cursor = self.db.execute(
                "INSERT INTO evidence_atoms(namespace,id,text,payload) VALUES(?,?,?,?)",
                (namespace, atom.id, atom.text, atom.model_dump_json()),
            )
            self.db.execute(
                f"INSERT INTO {table}(rowid,text) VALUES(?,?)",
                (cursor.lastrowid, index_text(atom.text)),
            )
            for span in atom.spans:
                self.db.execute(
                    "INSERT INTO evidence_atom_spans VALUES(?,?,?,?,?)",
                    (namespace, atom.id, span.source_id, span.start, span.end),
                )
            if blobs:
                self.db.execute(
                    "INSERT INTO evidence_atom_vec(atom_rowid,namespace,embedding) VALUES(?,?,?)",
                    (cursor.lastrowid, namespace, blobs[index]),
                )

    def lexical(
        self, namespace: str, query: str, limit: int, *, options: Sequence[str] | None = None
    ) -> list[tuple[Atom, float]]:
        scope(namespace)
        match = match_expression(query, options=options)
        if match is None:
            return []
        if not self.db.execute(
            "SELECT 1 FROM evidence_atoms WHERE namespace=? LIMIT 1", (namespace,)
        ).fetchone():
            return []
        table = fts_table(ATOM_FTS, namespace)
        rows = self.db.execute(
            f"SELECT a.payload, bm25({table}) AS score FROM {table} "
            f"JOIN evidence_atoms a ON a.rowid={table}.rowid "
            f"WHERE {table} MATCH ? AND a.namespace=? ORDER BY score,a.id LIMIT ?",
            (match, namespace, limit),
        ).fetchall()
        return [(Atom.model_validate_json(row[0]), -float(row[1])) for row in rows]

    def source_rows(self, namespace: str, source_id: str) -> list[tuple[int]]:
        """Rowids of the atoms that deleting this source removes."""
        rows = self.db.execute(
            "SELECT DISTINCT a.rowid FROM evidence_atom_spans s JOIN evidence_atoms a "
            "ON a.namespace=s.namespace AND a.id=s.atom_id WHERE s.namespace=? AND s.source_id=?",
            (namespace, source_id),
        ).fetchall()
        return [(row[0],) for row in rows]

    def unindex(self, namespace: str, rows: Sequence[tuple[int]]) -> None:
        if rows:
            self.db.executemany(f"DELETE FROM {fts_table(ATOM_FTS, namespace)} WHERE rowid=?", rows)

    def dense(self, namespace: str, blob: bytes, limit: int) -> list[tuple[Atom, float]]:
        scope(namespace)
        rows = self.db.execute(
            "SELECT a.payload, v.distance FROM (SELECT atom_rowid, distance FROM evidence_atom_vec "
            "WHERE embedding MATCH ? AND namespace=? AND k=?) v "
            "JOIN evidence_atoms a ON a.rowid=v.atom_rowid ORDER BY v.distance,a.id",
            (blob, namespace, limit),
        ).fetchall()
        return [(Atom.model_validate_json(row[0]), 1.0 - float(row[1])) for row in rows]

    def chunks(self, namespace: str, atom: Atom) -> list[Chunk]:
        scope(namespace)
        rows = self.db.execute(
            "SELECT DISTINCT c.* FROM evidence_atom_spans s JOIN evidence_chunks c "
            "ON c.namespace=s.namespace AND c.source_id=s.source_id AND c.start<s.end AND c.end>s.start "
            "WHERE s.namespace=? AND s.atom_id=? ORDER BY c.source_id,c.chunk_index",
            (namespace, atom.id),
        ).fetchall()
        return [
            Chunk(
                id=row["id"],
                source_id=row["source_id"],
                index=row["chunk_index"],
                start=row["start"],
                end=row["end"],
                text=row["text"],
            )
            for row in rows
        ]
