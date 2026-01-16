from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from itertools import combinations

from ..core.models import MemoryItem
from .serialization import normalize_entity_text

_DELETE_EDGE_BY_SOURCE_SQL = "DELETE FROM memory_edge WHERE source_id = ? AND kind = ?"



@dataclass(frozen=True)
class GraphMemoryHit:
    memory_id: str
    score: float
    depth: int
    matching_entities: list[dict[str, str]] | None = None


class SQLiteGraphIndex:
    def __init__(self, path: str) -> None:
        self.path = path
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    UNIQUE(name, type)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entity (
                    memory_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (memory_id, entity_id),
                    FOREIGN KEY (entity_id) REFERENCES entity(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_cooccurrence (
                    entity_id_a INTEGER NOT NULL,
                    entity_id_b INTEGER NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (entity_id_a, entity_id_b),
                    FOREIGN KEY (entity_id_a) REFERENCES entity(id) ON DELETE CASCADE,
                    FOREIGN KEY (entity_id_b) REFERENCES entity(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edge (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (source_id, target_id, kind)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_entity_entity ON memory_entity(entity_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_entity_memory ON memory_entity(memory_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_cooccurrence_a ON entity_cooccurrence(entity_id_a)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_cooccurrence_b ON entity_cooccurrence(entity_id_b)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_edge_source ON memory_edge(source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_edge_target ON memory_edge(target_id)"
            )
        self._initialized = True

    def close(self) -> None:
        self._initialized = False

    def reset(self) -> None:
        self.initialize()
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_entity")
            conn.execute("DELETE FROM entity_cooccurrence")
            conn.execute("DELETE FROM memory_edge")
            conn.execute("DELETE FROM entity")

    def upsert_memory(self, memory: MemoryItem) -> None:
        self.initialize()
        entities = self._normalize_entities(memory)
        mention_counts = self._resolve_mention_counts(memory)
        with self._connect() as conn:
            existing_ids = self._fetch_entity_ids_for_memory(conn, memory.memory_id)
            if existing_ids:
                self._adjust_cooccurrence(conn, existing_ids, delta=-1)
                conn.execute(
                    "DELETE FROM memory_entity WHERE memory_id = ?",
                    (memory.memory_id,),
                )

            entity_ids: list[int] = []
            for name, entity_type in entities:
                conn.execute(
                    """
                    INSERT INTO entity (name, type)
                    VALUES (?, ?)
                    ON CONFLICT(name, type) DO NOTHING
                    """,
                    (name, entity_type),
                )
                row = conn.execute(
                    "SELECT id FROM entity WHERE name = ? AND type = ?",
                    (name, entity_type),
                ).fetchone()
                if not row:
                    continue
                entity_id = int(row["id"])
                entity_ids.append(entity_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_entity (memory_id, entity_id, mention_count)
                    VALUES (?, ?, ?)
                    """,
                    (memory.memory_id, entity_id, mention_counts.get(name, 1)),
                )

            if entity_ids:
                self._adjust_cooccurrence(conn, entity_ids, delta=1)

            self._update_memory_edges(conn, memory)

    def delete_memory(self, memory_id: str) -> None:
        self.initialize()
        with self._connect() as conn:
            existing_ids = self._fetch_entity_ids_for_memory(conn, memory_id)
            if existing_ids:
                self._adjust_cooccurrence(conn, existing_ids, delta=-1)
            conn.execute("DELETE FROM memory_entity WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "DELETE FROM memory_edge WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            )
            conn.execute(
                """
                DELETE FROM entity
                WHERE id IN (
                    SELECT e.id
                    FROM entity e
                    LEFT JOIN memory_entity me ON me.entity_id = e.id
                    WHERE me.entity_id IS NULL
                )
                """
            )
            conn.execute("DELETE FROM entity_cooccurrence WHERE weight <= 0")

    def recall_by_entities(
        self,
        entities: dict[str, str] | list[str],
        *,
        limit: int = 10,
        hops: int = 2,
    ) -> list[GraphMemoryHit]:
        self.initialize()
        resolved = self._resolve_seed_entities(entities)
        if not resolved.entity_ids:
            return []
        with self._connect() as conn:
            direct_scores = self._fetch_direct_scores(conn, resolved.entity_ids)
            matching_entities = self._matching_entities_for_memories(conn, resolved.entity_ids)
            results = self._build_direct_results(direct_scores, matching_entities)

            if hops > 1:
                self._add_related_entity_results(
                    conn=conn,
                    resolved=resolved,
                    results=results,
                    limit=limit,
                )
                self._add_edge_neighbor_results(
                    conn=conn,
                    direct_scores=direct_scores,
                    results=results,
                    limit=limit,
                )

        hits = sorted(
            results.values(),
            key=lambda hit: (-hit.score, hit.depth, hit.memory_id),
        )
        return hits[:limit]

    def _fetch_direct_scores(
        self, conn: sqlite3.Connection, entity_ids: list[int]
    ) -> list[sqlite3.Row]:
        return conn.execute(
            """
            SELECT memory_id, SUM(mention_count) AS score
            FROM memory_entity
            WHERE entity_id IN ({placeholders})
            GROUP BY memory_id
            ORDER BY score DESC
            """.format(placeholders=",".join("?" * len(entity_ids))),
            entity_ids,
        ).fetchall()

    def _build_direct_results(
        self,
        direct_scores: list[sqlite3.Row],
        matching_entities: dict[str, list[dict[str, str]]],
    ) -> dict[str, GraphMemoryHit]:
        results: dict[str, GraphMemoryHit] = {}
        for row in direct_scores:
            memory_id = row["memory_id"]
            results[memory_id] = GraphMemoryHit(
                memory_id=memory_id,
                score=float(row["score"] or 0.0),
                depth=1,
                matching_entities=matching_entities.get(memory_id),
            )
        return results

    def _add_related_entity_results(
        self,
        *,
        conn: sqlite3.Connection,
        resolved: _ResolvedEntities,
        results: dict[str, GraphMemoryHit],
        limit: int,
    ) -> None:
        related_entities = self._related_entities(conn, resolved.entity_ids, limit)
        if not related_entities:
            return
        values_placeholders = ",".join("(?, ?)" for _ in related_entities)
        memory_scores = conn.execute(
            f"""
            WITH w(entity_id, weight) AS (
                VALUES {values_placeholders}
            )
            SELECT me.memory_id,
                   SUM(me.mention_count * w.weight) AS score
            FROM memory_entity me
            JOIN w ON w.entity_id = me.entity_id
            GROUP BY me.memory_id
            ORDER BY score DESC
            """,
            [val for row in related_entities for val in (row["entity_id"], row["weight"])],
        ).fetchall()

        for row in memory_scores:
            memory_id = row["memory_id"]
            if memory_id in results:
                continue
            results[memory_id] = GraphMemoryHit(
                memory_id=memory_id,
                score=float(row["score"] or 0.0) * 0.5,
                depth=2,
                matching_entities=None,
            )

    def _add_edge_neighbor_results(
        self,
        *,
        conn: sqlite3.Connection,
        direct_scores: list[sqlite3.Row],
        results: dict[str, GraphMemoryHit],
        limit: int,
    ) -> None:
        edge_seed_limit = min(max(limit * 5, 25), 200)
        direct_ids = [row["memory_id"] for row in direct_scores[:edge_seed_limit]]
        if not direct_ids:
            return
        edge_neighbors = conn.execute(
            """
            SELECT DISTINCT
                CASE
                    WHEN source_id IN ({placeholders}) THEN target_id
                    ELSE source_id
                END AS neighbor_id
            FROM memory_edge
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """.format(placeholders=",".join("?" * len(direct_ids))),
            direct_ids * 3,
        ).fetchall()
        for row in edge_neighbors:
            neighbor_id = row["neighbor_id"]
            if neighbor_id in results:
                continue
            results[neighbor_id] = GraphMemoryHit(
                memory_id=neighbor_id,
                score=0.5,
                depth=2,
                matching_entities=None,
            )

    def _fetch_entity_ids_for_memory(
        self, conn: sqlite3.Connection, memory_id: str
    ) -> list[int]:
        rows = conn.execute(
            "SELECT entity_id FROM memory_entity WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [int(row["entity_id"]) for row in rows]

    def _adjust_cooccurrence(
        self, conn: sqlite3.Connection, entity_ids: list[int], *, delta: int
    ) -> None:
        unique_ids = sorted(set(entity_ids))
        if len(unique_ids) < 2:
            return
        for left, right in combinations(unique_ids, 2):
            if delta > 0:
                conn.execute(
                    """
                    INSERT INTO entity_cooccurrence (entity_id_a, entity_id_b, weight)
                    VALUES (?, ?, ?)
                    ON CONFLICT(entity_id_a, entity_id_b)
                    DO UPDATE SET weight = weight + ?
                    """,
                    (left, right, delta, delta),
                )
            else:
                conn.execute(
                    """
                    UPDATE entity_cooccurrence
                    SET weight = weight + ?
                    WHERE entity_id_a = ? AND entity_id_b = ?
                    """,
                    (delta, left, right),
                )
        if delta < 0:
            conn.execute("DELETE FROM entity_cooccurrence WHERE weight <= 0")

    def _normalize_entities(self, memory: MemoryItem) -> list[tuple[str, str]]:
        entities: list[tuple[str, str]] = []
        for text, entity_type in (memory.entities or {}).items():
            if not text:
                continue
            normalized = normalize_entity_text(str(text))
            if not normalized:
                continue
            type_value = str(entity_type) if entity_type is not None else ""
            entities.append((normalized, type_value))
        return entities

    def _resolve_mention_counts(self, memory: MemoryItem) -> dict[str, int]:
        counts: dict[str, int] = {}
        for text, mentions in (memory.entity_mentions or {}).items():
            normalized = normalize_entity_text(str(text))
            if not normalized:
                continue
            count = len(mentions) if mentions else 1
            counts[normalized] = max(count, 1)
        return counts

    def _update_memory_edges(self, conn: sqlite3.Connection, memory: MemoryItem) -> None:
        conn.execute(
            _DELETE_EDGE_BY_SOURCE_SQL,
            (memory.memory_id, "association"),
        )
        conn.execute(
            _DELETE_EDGE_BY_SOURCE_SQL,
            (memory.memory_id, "conflict"),
        )
        conn.execute(
            _DELETE_EDGE_BY_SOURCE_SQL,
            (memory.memory_id, "temporal_next"),
        )
        conn.execute(
            "DELETE FROM memory_edge WHERE target_id = ? AND kind = ?",
            (memory.memory_id, "temporal_next"),
        )

        for related_id in memory.associations:
            if related_id and related_id != memory.memory_id:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_edge (source_id, target_id, kind, weight)
                    VALUES (?, ?, ?, 1)
                    """,
                    (memory.memory_id, related_id, "association"),
                )

        for conflict_id in memory.conflicts_with:
            if conflict_id and conflict_id != memory.memory_id:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_edge (source_id, target_id, kind, weight)
                    VALUES (?, ?, ?, 1)
                    """,
                    (memory.memory_id, conflict_id, "conflict"),
                )

        if memory.previous_event_id and memory.previous_event_id != memory.memory_id:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_edge (source_id, target_id, kind, weight)
                VALUES (?, ?, ?, 1)
                """,
                (memory.previous_event_id, memory.memory_id, "temporal_next"),
            )
        if memory.next_event_id and memory.next_event_id != memory.memory_id:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_edge (source_id, target_id, kind, weight)
                VALUES (?, ?, ?, 1)
                """,
                (memory.memory_id, memory.next_event_id, "temporal_next"),
            )

    def _matching_entities_for_memories(
        self, conn: sqlite3.Connection, entity_ids: list[int]
    ) -> dict[str, list[dict[str, str]]]:
        rows = conn.execute(
            """
            SELECT me.memory_id, e.name, e.type
            FROM memory_entity me
            JOIN entity e ON e.id = me.entity_id
            WHERE me.entity_id IN ({placeholders})
            """.format(placeholders=",".join("?" * len(entity_ids))),
            entity_ids,
        ).fetchall()
        matches: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            matches.setdefault(row["memory_id"], []).append(
                {"text": row["name"], "type": row["type"]}
            )
        return matches

    def _related_entities(
        self, conn: sqlite3.Connection, entity_ids: list[int], limit: int
    ) -> list[sqlite3.Row]:
        neighbor_limit = min(max(limit * 5, 25), 200)
        rows = conn.execute(
            """
            SELECT neighbor_id AS entity_id, SUM(weight) AS weight
            FROM (
                SELECT
                    CASE
                        WHEN entity_id_a IN ({placeholders}) THEN entity_id_b
                        ELSE entity_id_a
                    END AS neighbor_id,
                    weight
                FROM entity_cooccurrence
                WHERE entity_id_a IN ({placeholders}) OR entity_id_b IN ({placeholders})
            )
            WHERE neighbor_id NOT IN ({placeholders})
            GROUP BY neighbor_id
            ORDER BY weight DESC
            LIMIT ?
            """.format(
                placeholders=",".join("?" * len(entity_ids))
            ),
            entity_ids + entity_ids + entity_ids + entity_ids + [neighbor_limit],
        ).fetchall()
        return rows

    def _resolve_seed_entities(
        self, entities: dict[str, str] | list[str]
    ) -> _ResolvedEntities:
        typed, name_only = self._partition_seed_entities(entities)
        entity_rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            entity_rows.extend(self._fetch_name_only_entities(conn, name_only))
            entity_rows.extend(self._fetch_typed_entities(conn, typed))

        ids = sorted({int(row["id"]) for row in entity_rows})
        return _ResolvedEntities(entity_ids=ids, entities=entity_rows)

    def _partition_seed_entities(
        self, entities: dict[str, str] | list[str]
    ) -> tuple[list[tuple[str, str]], list[str]]:
        if isinstance(entities, dict):
            typed = []
            name_only = []
            for text, entity_type in entities.items():
                if not text:
                    continue
                normalized = normalize_entity_text(str(text))
                if not normalized:
                    continue
                if entity_type:
                    typed.append((normalized, str(entity_type)))
                else:
                    name_only.append(normalized)
            return typed, name_only

        typed = []
        name_only = [
            normalize_entity_text(str(text)) for text in entities if text and str(text).strip()
        ]
        return typed, name_only

    def _fetch_name_only_entities(
        self, conn: sqlite3.Connection, name_only: list[str]
    ) -> list[sqlite3.Row]:
        if not name_only:
            return []
        entity_rows: list[sqlite3.Row] = []
        rows = conn.execute(
            """
            SELECT id, name, type
            FROM entity
            WHERE name IN ({placeholders})
            """.format(placeholders=",".join("?" * len(name_only))),
            name_only,
        ).fetchall()
        entity_rows.extend(rows)
        for term in name_only:
            if len(term) < 4:
                continue
            rows = conn.execute(
                """
                SELECT id, name, type
                FROM entity
                WHERE name LIKE ?
                """,
                (f"%{term}%",),
            ).fetchall()
            entity_rows.extend(rows)
        return entity_rows

    def _fetch_typed_entities(
        self, conn: sqlite3.Connection, typed: list[tuple[str, str]]
    ) -> list[sqlite3.Row]:
        if not typed:
            return []
        entity_rows: list[sqlite3.Row] = []
        for name, entity_type in typed:
            rows = conn.execute(
                """
                SELECT id, name, type
                FROM entity
                WHERE name = ? AND type = ?
                """,
                (name, entity_type),
            ).fetchall()
            entity_rows.extend(rows)
            if len(name) < 4:
                continue
            rows = conn.execute(
                """
                SELECT id, name, type
                FROM entity
                WHERE name LIKE ? AND type = ?
                """,
                (f"%{name}%", entity_type),
            ).fetchall()
            entity_rows.extend(rows)
        return entity_rows


@dataclass(frozen=True)
class _ResolvedEntities:
    entity_ids: list[int]
    entities: list[sqlite3.Row]
