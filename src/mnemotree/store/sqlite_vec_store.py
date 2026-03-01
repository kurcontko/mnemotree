from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from asyncio import Lock
from collections import deque
from pathlib import Path
from typing import Any, Literal

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - optional dependency
    sqlite_vec = None

from ..core.models import LinkType, MemoryItem, MemoryLink
from ..core.query import MemoryQuery
from ..utils.serialization import json_dumps_safe, json_loads_dict
from ._filters import build_sqlite_filter_clauses, normalize_filter_value
from ._queries import build_entity_set
from ._records import sqlite_memory_from_row, sqlite_record_from_memory
from ._schema import (
    create_sqlite_schema,
    ensure_sqlite_vector_table,
    migrate_sqlite_add_stability,
    migrate_sqlite_phase1_fields,
)
from .base import BaseMemoryStore
from .logging import elapsed_ms, store_log_context
from .query_builders import UnsupportedQueryError
from .serialization import serialize_datetime

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$", re.ASCII)


class SQLiteVecMemoryStore(BaseMemoryStore):
    store_type = "sqlite-vec"

    def __init__(
        self,
        db_path: str | Path = ".mnemotree/mnemotree.sqlite",
        *,
        collection_name: str = "memories",
        embedding_dim: int | None = None,
    ) -> None:
        if not _IDENTIFIER_RE.match(collection_name):
            raise ValueError(
                "collection_name must be a valid SQLite identifier (letters, numbers, underscore)."
            )
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self._vector_table = f"{collection_name}_vectors"
        self._meta_table = f"{collection_name}_meta"
        self._entity_table = f"{collection_name}_entity_index"
        self._link_table = f"{collection_name}_links"
        self._entity_index_version = 1
        self._conn: sqlite3.Connection | None = None
        self._embedding_dim = embedding_dim
        self._initialized = False
        self._init_lock = Lock()
        self._lock = Lock()

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            if sqlite_vec is None:
                raise ModuleNotFoundError(
                    "SQLiteVecMemoryStore requires sqlite-vec. "
                    "Install with `mnemotree[sqlite_vec]`."
                )

            start = time.perf_counter()
            if self.db_path != Path(":memory:"):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

            self._create_schema()
            stored_dim = self._get_meta_int("embedding_dim")
            if stored_dim is not None:
                if self._embedding_dim is not None and stored_dim != self._embedding_dim:
                    raise ValueError(
                        f"Stored embedding dimension {stored_dim} does not match "
                        f"requested dimension {self._embedding_dim}."
                    )
                self._embedding_dim = stored_dim
            if self._embedding_dim is not None:
                self._ensure_vector_table(self._embedding_dim)

            self._ensure_entity_index()

            self._initialized = True
            logger.info(
                "Initialized SQLite vec store at %s",
                self.db_path,
                extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start)),
            )

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteVecMemoryStore is not initialized.")
        return self._conn

    def _create_schema(self) -> None:
        conn = self._require_conn()
        create_sqlite_schema(
            conn,
            collection_name=self.collection_name,
            meta_table=self._meta_table,
        )
        migrate_sqlite_add_stability(conn, self.collection_name)
        migrate_sqlite_phase1_fields(conn, self.collection_name)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self._entity_table}" (
                memory_id TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                PRIMARY KEY (memory_id, normalized_name)
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS "{self._entity_table}_name_idx"
            ON "{self._entity_table}" (normalized_name)
            """
        )
        # Knowledge graph link table
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self._link_table}" (
                link_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 1.0,
                context TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                created_by TEXT DEFAULT 'user',
                similarity_score REAL,
                metadata TEXT,
                UNIQUE(source_id, target_id, link_type)
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS "{self._link_table}_source_idx"
            ON "{self._link_table}" (source_id)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS "{self._link_table}_target_idx"
            ON "{self._link_table}" (target_id)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS "{self._link_table}_type_idx"
            ON "{self._link_table}" (link_type)
            """
        )
        # FTS5 virtual table for full-text search on entities/tags (SwiftMem sub-linear)
        self._fts_table = f"{self.collection_name}_fts"
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS "{self._fts_table}"
            USING fts5(memory_id, entities, tags, content, tokenize='porter unicode61')
            """
        )
        conn.commit()

    def _get_meta_int(self, key: str) -> int | None:
        conn = self._require_conn()
        row = conn.execute(
            f'SELECT value FROM "{self._meta_table}" WHERE key = ?', (key,)
        ).fetchone()
        if not row:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def _set_meta(self, key: str, value: str) -> None:
        conn = self._require_conn()
        conn.execute(
            f'INSERT INTO "{self._meta_table}" (key, value) VALUES (?, ?) '
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()

    def _ensure_vector_table(self, dimension: int) -> None:
        conn = self._require_conn()
        ensure_sqlite_vector_table(conn, vector_table=self._vector_table, dimension=dimension)
        self._set_meta("embedding_dim", str(dimension))

    def _ensure_entity_index(self) -> None:
        conn = self._require_conn()
        stored_version = self._get_meta_int("entity_index_version")
        if stored_version == self._entity_index_version:
            return

        conn.execute(f'DELETE FROM "{self._entity_table}"')
        rows = conn.execute(f'SELECT memory_id, entities FROM "{self.collection_name}"').fetchall()
        entries: list[tuple[str, str]] = []
        for row in rows:
            stored_entities = json_loads_dict(row["entities"])
            entity_set = build_entity_set(stored_entities)
            if not entity_set:
                continue
            entries.extend((row["memory_id"], entity) for entity in entity_set)
        if entries:
            conn.executemany(
                f'INSERT OR IGNORE INTO "{self._entity_table}" '
                "(memory_id, normalized_name) VALUES (?, ?)",
                entries,
            )
        self._set_meta("entity_index_version", str(self._entity_index_version))

    def _update_entity_index(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        entities: dict[str, str] | list[str] | None,
    ) -> None:
        conn.execute(
            f'DELETE FROM "{self._entity_table}" WHERE memory_id = ?',
            (memory_id,),
        )
        if not entities:
            return
        entity_set = build_entity_set(entities)
        if not entity_set:
            return
        conn.executemany(
            f'INSERT OR IGNORE INTO "{self._entity_table}" '
            "(memory_id, normalized_name) VALUES (?, ?)",
            [(memory_id, entity) for entity in entity_set],
        )

    def _update_fts_index(
        self,
        conn: sqlite3.Connection,
        memory: MemoryItem,
    ) -> None:
        """Update the FTS5 full-text search index for a memory."""
        fts_table = f"{self.collection_name}_fts"
        conn.execute(
            f'DELETE FROM "{fts_table}" WHERE memory_id = ?',
            (memory.memory_id,),
        )
        entities_text = " ".join(memory.entities.keys()) if memory.entities else ""
        tags_text = " ".join(memory.tags) if memory.tags else ""
        conn.execute(
            f'INSERT INTO "{fts_table}" (memory_id, entities, tags, content) VALUES (?, ?, ?, ?)',
            (memory.memory_id, entities_text, tags_text, memory.content),
        )

    async def search_fts(
        self,
        query: str,
        *,
        limit: int = 50,
        columns: str = "entities tags content",
    ) -> list[str]:
        """Full-text search on entities/tags/content. Returns matching memory_ids.

        Use this as a pre-filter before vector search to reduce scan from N to N/k.
        """
        await self.initialize()
        conn = self._require_conn()
        fts_table = f"{self.collection_name}_fts"
        try:
            # Use column filter syntax: {column}: query
            match_expr = query
            if columns != "entities tags content":
                match_expr = f"{{{columns}}}: {query}"
            rows = conn.execute(
                f'SELECT memory_id FROM "{fts_table}" WHERE "{fts_table}" MATCH ? '
                f"ORDER BY rank LIMIT ?",
                (match_expr, limit),
            ).fetchall()
            return [row["memory_id"] for row in rows]
        except sqlite3.OperationalError:
            logger.debug("FTS search failed for query=%s", query, exc_info=True)
            return []

    async def store_memory(self, memory: MemoryItem) -> None:
        await self.initialize()
        if memory.embedding is None:
            raise ValueError("SQLiteVecMemoryStore requires embeddings to store memories.")

        start = time.perf_counter()
        async with self._lock:
            conn = self._require_conn()
            try:
                embedding_dim = len(memory.embedding)
                if self._embedding_dim is None:
                    self._embedding_dim = embedding_dim
                    self._ensure_vector_table(embedding_dim)
                elif embedding_dim != self._embedding_dim:
                    raise ValueError(
                        f"Embedding dimension {embedding_dim} does not match "
                        f"store dimension {self._embedding_dim}."
                    )

                record = sqlite_record_from_memory(memory)
                columns = list(record.keys())
                values = list(record.values())

                row = conn.execute(
                    f'SELECT id FROM "{self.collection_name}" WHERE memory_id = ?',
                    (memory.memory_id,),
                ).fetchone()
                if row:
                    memory_row_id = row["id"]
                    set_clause = ", ".join(f"{col} = ?" for col in columns)
                    conn.execute(
                        f'UPDATE "{self.collection_name}" SET {set_clause} WHERE id = ?',
                        values + [memory_row_id],
                    )
                else:
                    placeholders = ", ".join("?" for _ in columns)
                    cursor = conn.execute(
                        f'INSERT INTO "{self.collection_name}" ({", ".join(columns)}) '
                        f"VALUES ({placeholders})",
                        values,
                    )
                    last_id = cursor.lastrowid
                    memory_row_id = int(last_id) if last_id is not None else 0

                conn.execute(
                    f'DELETE FROM "{self._vector_table}" WHERE rowid = ?',
                    (memory_row_id,),
                )
                conn.execute(
                    f'INSERT INTO "{self._vector_table}" (rowid, embedding) VALUES (?, ?)',
                    (memory_row_id, sqlite_vec.serialize_float32(memory.embedding)),
                )
                self._update_entity_index(conn, memory.memory_id, memory.entities)
                self._update_fts_index(conn, memory)
                conn.commit()

                logger.info(
                    "Successfully stored memory %s",
                    memory.memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory.memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
            except sqlite3.Error:
                conn.rollback()
                logger.exception(
                    "Failed to store memory %s",
                    memory.memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory.memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def get_memory(self, memory_id: str) -> MemoryItem | None:
        await self.initialize()
        start = time.perf_counter()
        async with self._lock:
            conn = self._require_conn()
            try:
                row = conn.execute(
                    f'SELECT * FROM "{self.collection_name}" WHERE memory_id = ?',
                    (memory_id,),
                ).fetchone()
                if not row:
                    return None
                return sqlite_memory_from_row(row)
            except sqlite3.Error:
                logger.exception(
                    "Failed to retrieve memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def delete_memory(self, memory_id: str, *, cascade: bool = False) -> bool:
        await self.initialize()
        start = time.perf_counter()
        async with self._lock:
            conn = self._require_conn()
            try:
                row = conn.execute(
                    f'SELECT id FROM "{self.collection_name}" WHERE memory_id = ?',
                    (memory_id,),
                ).fetchone()
                if not row:
                    return False
                memory_row_id = row["id"]
                conn.execute(
                    f'DELETE FROM "{self.collection_name}" WHERE id = ?',
                    (memory_row_id,),
                )
                conn.execute(
                    f'DELETE FROM "{self._vector_table}" WHERE rowid = ?',
                    (memory_row_id,),
                )
                self._update_entity_index(conn, memory_id, None)
                conn.commit()
                logger.info(
                    "Deleted memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return True
            except sqlite3.Error:
                conn.rollback()
                logger.exception(
                    "Failed to delete memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def query_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        await self.initialize()
        if query.relationships:
            raise UnsupportedQueryError("Relationship queries are not supported.")

        start = time.perf_counter()
        async with self._lock:
            conn = self._require_conn()
            try:
                where_clauses: list[str] = []
                params: list[Any] = []
                if query.filters:
                    built_clauses, built_params = build_sqlite_filter_clauses(query.filters)
                    where_clauses.extend(built_clauses)
                    params.extend(built_params)
                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                order_sql = ""
                if query.vector is None and query.sort_by:
                    if query.sort_by not in {
                        "timestamp",
                        "importance",
                        "confidence",
                        "memory_type",
                    }:
                        raise UnsupportedQueryError(f"Unsupported sort field: {query.sort_by!r}.")
                    order_sql = f"ORDER BY {query.sort_by} {query.sort_order.value.upper()}"

                limit = query.limit or 10
                offset = query.offset or 0

                if query.vector is not None:
                    if self._embedding_dim is None:
                        return []  # No vector table yet
                    vector_blob = sqlite_vec.serialize_float32(query.vector)
                    # Use KNN MATCH syntax (sqlite-vec requires this)
                    # Fetch extra candidates to allow for post-filter + offset
                    fetch_k = limit + offset + 50
                    sql = (
                        f"SELECT m.*, v.distance "
                        f'FROM "{self._vector_table}" v '
                        f'JOIN "{self.collection_name}" m ON m.id = v.rowid '
                        f"WHERE v.embedding MATCH ? AND k = ? "
                        f"{('AND ' + ' AND '.join(where_clauses)) if where_clauses else ''} "
                        "ORDER BY v.distance "
                        "LIMIT ? OFFSET ?"
                    )
                    rows = conn.execute(sql, [vector_blob, fetch_k, *params, limit, offset]).fetchall()
                else:
                    sql = (
                        f'SELECT * FROM "{self.collection_name}" '
                        f"{where_sql} "
                        f"{order_sql} "
                        "LIMIT ? OFFSET ?"
                    )
                    rows = conn.execute(sql, [*params, limit, offset]).fetchall()

                return [sqlite_memory_from_row(row) for row in rows]
            except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
                logger.exception(
                    "Failed to query memories",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def get_similar_memories(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        await self.initialize()
        if self._embedding_dim is None:
            return []  # No vector table yet — no memories to search
        start = time.perf_counter()
        async with self._lock:
            conn = self._require_conn()
            try:
                where_clauses = []
                params: list[Any] = []
                if filters:
                    for key, value in filters.items():
                        if key not in {"memory_type", "source"}:
                            raise UnsupportedQueryError(f"Unsupported filter field: {key!r}")
                        where_clauses.append(f"{key} = ?")
                        params.append(normalize_filter_value(value))

                vector_blob = sqlite_vec.serialize_float32(query_embedding)
                # Use KNN MATCH syntax (sqlite-vec requires this)
                fetch_k = top_k + 50  # extra candidates for post-filtering
                sql = (
                    f"SELECT m.*, v.distance "
                    f'FROM "{self._vector_table}" v '
                    f'JOIN "{self.collection_name}" m ON m.id = v.rowid '
                    f"WHERE v.embedding MATCH ? AND k = ? "
                    f"{('AND ' + ' AND '.join(where_clauses)) if where_clauses else ''} "
                    "ORDER BY v.distance "
                    "LIMIT ?"
                )
                rows = conn.execute(sql, [vector_blob, fetch_k, *params, top_k]).fetchall()
                return [sqlite_memory_from_row(row) for row in rows]
            except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
                logger.exception(
                    "Failed to get similar memories",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def query_by_entities(
        self,
        entities: dict[str, str] | list[str],
        limit: int = 10,
    ) -> list[MemoryItem]:
        await self.initialize()
        start = time.perf_counter()
        async with self._lock:
            conn = self._require_conn()
            try:
                if not entities:
                    return []

                entity_set = build_entity_set(entities)
                if not entity_set:
                    return []

                exact_entities = list(entity_set)
                substring_entities = [entity for entity in entity_set if len(entity) >= 4]

                where_clauses: list[str] = []
                params: list[Any] = []
                if exact_entities:
                    placeholders = ", ".join("?" for _ in exact_entities)
                    where_clauses.append(f"e.normalized_name IN ({placeholders})")
                    params.extend(exact_entities)
                if substring_entities:
                    like_clauses = []
                    for entity in substring_entities:
                        like_clauses.append("e.normalized_name LIKE ?")
                        params.append(f"%{entity}%")
                    where_clauses.append(
                        f"(length(e.normalized_name) >= 4 AND ({' OR '.join(like_clauses)}))"
                    )

                where_sql = " OR ".join(where_clauses)
                sql = (
                    f'SELECT DISTINCT m.* FROM "{self.collection_name}" m '
                    f'JOIN "{self._entity_table}" e ON e.memory_id = m.memory_id '
                    f"WHERE {where_sql} "
                    "ORDER BY m.id "
                    "LIMIT ?"
                )
                rows = conn.execute(sql, [*params, limit]).fetchall()
                return [sqlite_memory_from_row(row) for row in rows]
            except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
                logger.exception(
                    "Failed to query by entities",
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def update_connections(
        self,
        memory_id: str,
        *,
        related_ids: list[str] | None = None,
        conflict_ids: list[str] | None = None,
        previous_id: str | None = None,
        next_id: str | None = None,
    ) -> None:
        start = time.perf_counter()
        try:
            memory = await self.get_memory(memory_id)
            if not memory:
                logger.warning(
                    "Attempted to update connections for non-existent memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return
            if related_ids is not None:
                memory.associations = related_ids
            if conflict_ids is not None:
                memory.conflicts_with = conflict_ids
            if previous_id is not None:
                memory.previous_event_id = previous_id
            if next_id is not None:
                memory.next_event_id = next_id
            await self.store_memory(memory)
        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
            logger.exception(
                "Failed to update connections for memory %s",
                memory_id,
                extra=store_log_context(
                    self.store_type,
                    memory_id=memory_id,
                    duration_ms=elapsed_ms(start),
                ),
            )
            raise

    # -- Knowledge Graph Operations --

    def _link_from_row(self, row: sqlite3.Row) -> MemoryLink:
        metadata_str = row["metadata"]
        return MemoryLink(
            link_id=row["link_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            link_type=LinkType(row["link_type"]),
            strength=row["strength"],
            context=row["context"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            created_by=row["created_by"],
            similarity_score=row["similarity_score"],
            metadata=json.loads(metadata_str) if metadata_str else {},
        )

    async def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        *,
        strength: float = 1.0,
        context: str | None = None,
        bidirectional: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryLink:
        await self.initialize()
        start = time.perf_counter()

        link = MemoryLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            strength=strength,
            context=context,
            metadata=metadata or {},
        )

        async with self._lock:
            conn = self._require_conn()
            try:
                now = serialize_datetime(link.created_at)
                meta_json = json_dumps_safe(link.metadata)

                conn.execute(
                    f'INSERT OR REPLACE INTO "{self._link_table}" '
                    "(link_id, source_id, target_id, link_type, strength, context, "
                    "created_at, last_accessed, access_count, created_by, "
                    "similarity_score, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        link.link_id,
                        source_id,
                        target_id,
                        link_type.value,
                        strength,
                        context,
                        now,
                        now,
                        0,
                        link.created_by,
                        link.similarity_score,
                        meta_json,
                    ),
                )

                if bidirectional:
                    reverse = MemoryLink(
                        source_id=target_id,
                        target_id=source_id,
                        link_type=link_type,
                        strength=strength,
                        context=context,
                        metadata=metadata or {},
                    )
                    conn.execute(
                        f'INSERT OR REPLACE INTO "{self._link_table}" '
                        "(link_id, source_id, target_id, link_type, strength, context, "
                        "created_at, last_accessed, access_count, created_by, "
                        "similarity_score, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            reverse.link_id,
                            target_id,
                            source_id,
                            link_type.value,
                            strength,
                            context,
                            serialize_datetime(reverse.created_at),
                            serialize_datetime(reverse.last_accessed),
                            0,
                            reverse.created_by,
                            reverse.similarity_score,
                            meta_json,
                        ),
                    )

                conn.commit()
                logger.info(
                    "Created link %s from %s to %s",
                    link.link_id,
                    source_id,
                    target_id,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return link
            except sqlite3.Error:
                conn.rollback()
                logger.exception(
                    "Failed to create link from %s to %s",
                    source_id,
                    target_id,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def get_links(
        self,
        memory_id: str,
        *,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        link_types: list[LinkType] | None = None,
        min_strength: float = 0.0,
    ) -> list[MemoryLink]:
        await self.initialize()
        start = time.perf_counter()

        async with self._lock:
            conn = self._require_conn()
            try:
                where_clauses: list[str] = []
                params: list[Any] = []

                if direction == "outgoing":
                    where_clauses.append("source_id = ?")
                    params.append(memory_id)
                elif direction == "incoming":
                    where_clauses.append("target_id = ?")
                    params.append(memory_id)
                else:
                    where_clauses.append("(source_id = ? OR target_id = ?)")
                    params.extend([memory_id, memory_id])

                where_clauses.append("strength >= ?")
                params.append(min_strength)

                if link_types:
                    placeholders = ", ".join("?" for _ in link_types)
                    where_clauses.append(f"link_type IN ({placeholders})")
                    params.extend(lt.value for lt in link_types)

                where_sql = " AND ".join(where_clauses)
                rows = conn.execute(
                    f'SELECT * FROM "{self._link_table}" WHERE {where_sql}',
                    params,
                ).fetchall()

                links = [self._link_from_row(row) for row in rows]
                logger.info(
                    "Retrieved %d links for memory %s",
                    len(links),
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return links
            except sqlite3.Error:
                logger.exception(
                    "Failed to get links for memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def get_backlinks(
        self,
        memory_id: str,
        *,
        link_types: list[LinkType] | None = None,
    ) -> list[MemoryItem]:
        await self.initialize()
        start = time.perf_counter()

        async with self._lock:
            conn = self._require_conn()
            try:
                where_clauses = ["l.target_id = ?"]
                params: list[Any] = [memory_id]

                if link_types:
                    placeholders = ", ".join("?" for _ in link_types)
                    where_clauses.append(f"l.link_type IN ({placeholders})")
                    params.extend(lt.value for lt in link_types)

                where_sql = " AND ".join(where_clauses)
                sql = (
                    f'SELECT m.* FROM "{self.collection_name}" m '
                    f'JOIN "{self._link_table}" l ON l.source_id = m.memory_id '
                    f"WHERE {where_sql}"
                )
                rows = conn.execute(sql, params).fetchall()
                memories = [sqlite_memory_from_row(row) for row in rows]

                logger.info(
                    "Retrieved %d backlinks for memory %s",
                    len(memories),
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return memories
            except sqlite3.Error:
                logger.exception(
                    "Failed to get backlinks for memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def traverse_graph(
        self,
        start_id: str,
        *,
        max_depth: int = 3,
        link_types: list[LinkType] | None = None,
        strategy: Literal["bfs", "dfs"] = "bfs",
    ) -> list[tuple[MemoryItem, int, list[MemoryLink]]]:
        await self.initialize()
        start_time = time.perf_counter()
        max_depth = min(max_depth, 5)

        type_values = {lt.value for lt in link_types} if link_types else None

        result: list[tuple[MemoryItem, int, list[MemoryLink]]] = []
        visited: set[str] = {start_id}

        # (current_memory_id, current_depth, path_links)
        frontier: deque[tuple[str, int, list[MemoryLink]]] = deque()
        frontier.append((start_id, 0, []))

        async with self._lock:
            conn = self._require_conn()
            try:
                while frontier:
                    if strategy == "bfs":
                        current_id, depth, path_links = frontier.popleft()
                    else:
                        current_id, depth, path_links = frontier.pop()

                    if depth >= max_depth:
                        continue

                    # Get outgoing links from current node
                    rows = conn.execute(
                        f'SELECT * FROM "{self._link_table}" WHERE source_id = ?',
                        (current_id,),
                    ).fetchall()

                    for row in rows:
                        link = self._link_from_row(row)
                        if type_values and link.link_type.value not in type_values:
                            continue
                        target_id = link.target_id
                        if target_id in visited:
                            continue
                        visited.add(target_id)

                        # Fetch the target memory
                        mem_row = conn.execute(
                            f'SELECT * FROM "{self.collection_name}" WHERE memory_id = ?',
                            (target_id,),
                        ).fetchone()
                        if not mem_row:
                            logger.debug("Linked memory %s not found, skipping", target_id)
                            continue
                        memory = sqlite_memory_from_row(mem_row)
                        new_path = path_links + [link]
                        result.append((memory, depth + 1, new_path))
                        frontier.append((target_id, depth + 1, new_path))

                logger.info(
                    "Traversed graph from %s, found %d nodes",
                    start_id,
                    len(result),
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start_time),
                    ),
                )
                return result
            except sqlite3.Error:
                logger.exception(
                    "Failed to traverse graph from %s",
                    start_id,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start_time),
                    ),
                )
                raise

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 5,
    ) -> list[tuple[MemoryItem, MemoryLink]] | None:
        await self.initialize()
        start_time = time.perf_counter()
        max_depth = min(max_depth, 10)

        async with self._lock:
            conn = self._require_conn()
            try:
                # BFS to find shortest path
                visited: set[str] = {source_id}
                # Each entry: (current_id, path_of_(memory_id, link) pairs)
                queue: deque[tuple[str, list[tuple[str, MemoryLink]]]] = deque()
                queue.append((source_id, []))

                while queue:
                    current_id, path = queue.popleft()
                    if len(path) >= max_depth:
                        continue

                    rows = conn.execute(
                        f'SELECT * FROM "{self._link_table}" WHERE source_id = ?',
                        (current_id,),
                    ).fetchall()

                    for row in rows:
                        link = self._link_from_row(row)
                        next_id = link.target_id
                        if next_id in visited:
                            continue
                        visited.add(next_id)

                        new_path = path + [(next_id, link)]

                        if next_id == target_id:
                            # Build result: fetch memories for each step
                            result: list[tuple[MemoryItem, MemoryLink]] = []
                            for mem_id, step_link in new_path:
                                mem_row = conn.execute(
                                    f'SELECT * FROM "{self.collection_name}" WHERE memory_id = ?',
                                    (mem_id,),
                                ).fetchone()
                                if mem_row:
                                    result.append((sqlite_memory_from_row(mem_row), step_link))
                                else:
                                    logger.debug("Path memory %s not found, skipping", mem_id)
                            logger.info(
                                "Found path from %s to %s with %d hops",
                                source_id,
                                target_id,
                                len(result),
                                extra=store_log_context(
                                    self.store_type,
                                    duration_ms=elapsed_ms(start_time),
                                ),
                            )
                            return result

                        queue.append((next_id, new_path))

                return None
            except sqlite3.Error:
                logger.exception(
                    "Failed to find path from %s to %s",
                    source_id,
                    target_id,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start_time),
                    ),
                )
                raise

    async def suggest_links(
        self,
        memory_id: str,
        *,
        limit: int = 10,
        min_similarity: float = 0.7,
        use_llm: bool = False,
    ) -> list[tuple[MemoryItem, LinkType, float, str | None]]:
        await self.initialize()
        start = time.perf_counter()

        async with self._lock:
            conn = self._require_conn()
            try:
                # Get the source memory's embedding
                mem_row = conn.execute(
                    f'SELECT id FROM "{self.collection_name}" WHERE memory_id = ?',
                    (memory_id,),
                ).fetchone()
                if not mem_row or self._embedding_dim is None:
                    return []

                source_rowid = mem_row["id"]

                # Get embedding for source memory
                vec_row = conn.execute(
                    f'SELECT embedding FROM "{self._vector_table}" WHERE rowid = ?',
                    (source_rowid,),
                ).fetchone()
                if not vec_row:
                    return []

                source_embedding = vec_row["embedding"]

                # Get already-linked memory IDs to exclude
                linked_rows = conn.execute(
                    f'SELECT target_id AS linked_id FROM "{self._link_table}" WHERE source_id = ? '
                    f'UNION SELECT source_id AS linked_id FROM "{self._link_table}" WHERE target_id = ?',
                    (memory_id, memory_id),
                ).fetchall()
                excluded_ids = {memory_id} | {row[0] for row in linked_rows}

                # Find similar memories using vector search
                # sqlite-vec distance() returns L2 distance; lower = more similar
                # We fetch extra candidates and filter out linked ones
                fetch_limit = limit + len(excluded_ids) + 5
                vector_blob = source_embedding
                similar_rows = conn.execute(
                    f"SELECT m.*, v.distance AS dist "
                    f'FROM "{self._vector_table}" v '
                    f'JOIN "{self.collection_name}" m ON m.id = v.rowid '
                    "WHERE v.embedding MATCH ? AND k = ? "
                    "ORDER BY v.distance",
                    (vector_blob, fetch_limit),
                ).fetchall()

                suggestions: list[tuple[MemoryItem, LinkType, float, str | None]] = []
                for row in similar_rows:
                    mid = row["memory_id"]
                    if mid in excluded_ids:
                        continue
                    # Convert L2 distance to a 0-1 similarity score
                    # Using 1/(1+dist) as a simple conversion
                    dist = row["dist"]
                    similarity = 1.0 / (1.0 + dist)
                    if similarity < min_similarity:
                        continue
                    memory = sqlite_memory_from_row(row)
                    suggestions.append((memory, LinkType.SIMILAR_TO, similarity, None))
                    if len(suggestions) >= limit:
                        break

                logger.info(
                    "Suggested %d links for memory %s",
                    len(suggestions),
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                return suggestions
            except sqlite3.Error:
                logger.exception(
                    "Failed to suggest links for memory %s",
                    memory_id,
                    extra=store_log_context(
                        self.store_type,
                        memory_id=memory_id,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def update_link_strength(
        self,
        link_id: str,
        strength: float,
    ) -> bool:
        await self.initialize()
        start = time.perf_counter()

        async with self._lock:
            conn = self._require_conn()
            try:
                cursor = conn.execute(
                    f'UPDATE "{self._link_table}" SET strength = ? WHERE link_id = ?',
                    (strength, link_id),
                )
                conn.commit()
                success = cursor.rowcount > 0
                if success:
                    logger.info(
                        "Updated link strength for %s to %f",
                        link_id,
                        strength,
                        extra=store_log_context(
                            self.store_type,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                return success
            except sqlite3.Error:
                conn.rollback()
                logger.exception(
                    "Failed to update link strength for %s",
                    link_id,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def delete_link(
        self,
        link_id: str,
    ) -> bool:
        await self.initialize()
        start = time.perf_counter()

        async with self._lock:
            conn = self._require_conn()
            try:
                cursor = conn.execute(
                    f'DELETE FROM "{self._link_table}" WHERE link_id = ?',
                    (link_id,),
                )
                conn.commit()
                success = cursor.rowcount > 0
                if success:
                    logger.info(
                        "Deleted link %s",
                        link_id,
                        extra=store_log_context(
                            self.store_type,
                            duration_ms=elapsed_ms(start),
                        ),
                    )
                return success
            except sqlite3.Error:
                conn.rollback()
                logger.exception(
                    "Failed to delete link %s",
                    link_id,
                    extra=store_log_context(
                        self.store_type,
                        duration_ms=elapsed_ms(start),
                    ),
                )
                raise

    async def get_neighborhood_links(
        self,
        memory_ids: list[str],
        max_depth: int = 3,
        link_types: list[LinkType] | None = None,
    ) -> list[MemoryLink]:
        """Return all links reachable within max_depth hops from seed ids.

        Uses an iterative BFS in Python over the link table to avoid recursive
        CTE compatibility issues across SQLite versions.
        """
        await self.initialize()
        start = time.perf_counter()
        max_depth = min(max_depth, 5)

        type_values = {lt.value for lt in link_types} if link_types else None
        collected: dict[str, MemoryLink] = {}  # link_id → MemoryLink
        frontier: set[str] = set(memory_ids)
        visited_nodes: set[str] = set(memory_ids)

        async with self._lock:
            conn = self._require_conn()
            try:
                for _depth in range(max_depth):
                    if not frontier:
                        break

                    placeholders = ", ".join("?" for _ in frontier)
                    rows = conn.execute(
                        f'SELECT * FROM "{self._link_table}" WHERE source_id IN ({placeholders})',
                        list(frontier),
                    ).fetchall()

                    next_frontier: set[str] = set()
                    for row in rows:
                        link = self._link_from_row(row)
                        if type_values and link.link_type.value not in type_values:
                            continue
                        if link.link_id not in collected:
                            collected[link.link_id] = link
                        if link.target_id not in visited_nodes:
                            visited_nodes.add(link.target_id)
                            next_frontier.add(link.target_id)

                    frontier = next_frontier

                links = list(collected.values())
                logger.info(
                    "get_neighborhood_links seeds=%d depth=%d links=%d",
                    len(memory_ids),
                    max_depth,
                    len(links),
                    extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start)),
                )
                return links
            except sqlite3.Error:
                logger.exception(
                    "Failed to get neighborhood links",
                    extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start)),
                )
                raise

    async def traverse_typed_path(
        self,
        start_id: str,
        path_pattern: list[LinkType],
        max_results: int = 20,
    ) -> list[tuple[MemoryItem, list[MemoryLink]]]:
        """Traverse a typed-path sequence from start_id.

        Each element of path_pattern constrains the link type for that hop.
        """
        await self.initialize()
        start_time = time.perf_counter()

        if not path_pattern:
            return []

        async with self._lock:
            conn = self._require_conn()
            try:
                # Iterative hop traversal: at each hop filter by the required link type
                # current_frontier: list of (current_node_id, path_links)
                current_frontier: list[tuple[str, list[MemoryLink]]] = [(start_id, [])]

                for _hop_idx, link_type in enumerate(path_pattern):
                    next_frontier: list[tuple[str, list[MemoryLink]]] = []
                    type_val = link_type.value

                    for node_id, path_so_far in current_frontier:
                        rows = conn.execute(
                            f'SELECT * FROM "{self._link_table}" '
                            "WHERE source_id = ? AND link_type = ?",
                            (node_id, type_val),
                        ).fetchall()

                        for row in rows:
                            link = self._link_from_row(row)
                            next_frontier.append((link.target_id, path_so_far + [link]))
                            if len(next_frontier) >= max_results * 10:
                                break

                    current_frontier = next_frontier
                    if not current_frontier:
                        break

                # Fetch end-node memories
                results: list[tuple[MemoryItem, list[MemoryLink]]] = []
                seen_end: set[str] = set()
                for end_id, path_links in current_frontier:
                    if end_id in seen_end:
                        continue
                    seen_end.add(end_id)
                    mem_row = conn.execute(
                        f'SELECT * FROM "{self.collection_name}" WHERE memory_id = ?',
                        (end_id,),
                    ).fetchone()
                    if mem_row:
                        results.append((sqlite_memory_from_row(mem_row), path_links))
                    else:
                        logger.debug("Typed-path end node %s not found, skipping", end_id)
                    if len(results) >= max_results:
                        break

                logger.info(
                    "traverse_typed_path start=%s pattern=%s results=%d",
                    start_id,
                    [lt.value for lt in path_pattern],
                    len(results),
                    extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start_time)),
                )
                return results
            except sqlite3.Error:
                logger.exception(
                    "Failed to traverse typed path from %s",
                    start_id,
                    extra=store_log_context(self.store_type, duration_ms=elapsed_ms(start_time)),
                )
                raise

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._initialized = False
