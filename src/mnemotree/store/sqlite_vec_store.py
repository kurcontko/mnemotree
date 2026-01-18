from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from asyncio import Lock
from pathlib import Path
from typing import Any

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - optional dependency
    sqlite_vec = None

from ..core.models import MemoryItem
from ..core.query import MemoryQuery
from ..utils.serialization import json_loads_dict
from ._filters import build_sqlite_filter_clauses, normalize_filter_value
from ._queries import build_entity_set
from ._records import sqlite_memory_from_row, sqlite_record_from_memory
from ._schema import create_sqlite_schema, ensure_sqlite_vector_table
from .base import BaseMemoryStore
from .logging import elapsed_ms, store_log_context
from .query_builders import UnsupportedQueryError

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
                    vector_blob = sqlite_vec.serialize_float32(query.vector)
                    sql = (
                        f"SELECT m.*, distance(v.embedding, ?) AS distance "
                        f'FROM "{self._vector_table}" v '
                        f'JOIN "{self.collection_name}" m ON m.id = v.rowid '
                        f"{where_sql} "
                        "ORDER BY distance "
                        "LIMIT ? OFFSET ?"
                    )
                    rows = conn.execute(sql, [vector_blob, *params, limit, offset]).fetchall()
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
                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                vector_blob = sqlite_vec.serialize_float32(query_embedding)
                sql = (
                    f"SELECT m.*, distance(v.embedding, ?) AS distance "
                    f'FROM "{self._vector_table}" v '
                    f'JOIN "{self.collection_name}" m ON m.id = v.rowid '
                    f"{where_sql} "
                    "ORDER BY distance "
                    "LIMIT ?"
                )
                rows = conn.execute(sql, [vector_blob, *params, top_k]).fetchall()
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

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._initialized = False
