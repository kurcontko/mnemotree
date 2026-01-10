from __future__ import annotations

import sqlite3
from typing import Any

NEO4J_SCHEMA_STATEMENTS = (
    """
    CREATE CONSTRAINT memory_id IF NOT EXISTS
    FOR (m:MemoryItem) REQUIRE m.memory_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_text IF NOT EXISTS
    FOR (e:Entity) REQUIRE e.text IS UNIQUE
    """,
    """
    CREATE CONSTRAINT tag_name IF NOT EXISTS
    FOR (t:Tag) REQUIRE t.name IS UNIQUE
    """,
    """
    CREATE INDEX memory_timestamp IF NOT EXISTS
    FOR (m:MemoryItem) ON (m.timestamp)
    """,
    """
    CREATE INDEX memory_importance IF NOT EXISTS
    FOR (m:MemoryItem) ON (m.importance)
    """,
)


async def apply_neo4j_schema(tx: Any) -> None:
    for statement in NEO4J_SCHEMA_STATEMENTS:
        await tx.run(statement)


def create_sqlite_schema(
    conn: sqlite3.Connection,
    *,
    collection_name: str,
    meta_table: str,
) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{collection_name}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT UNIQUE NOT NULL,
            conversation_id TEXT,
            user_id TEXT,
            content TEXT NOT NULL,
            summary TEXT,
            author TEXT,
            memory_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            last_accessed TEXT,
            access_count INTEGER,
            access_history TEXT,
            importance REAL NOT NULL,
            decay_rate REAL,
            confidence REAL,
            fidelity REAL,
            emotional_valence REAL,
            emotional_arousal REAL,
            emotions TEXT,
            tags TEXT,
            source TEXT,
            credibility REAL,
            context TEXT,
            metadata TEXT,
            embedding TEXT,
            entities TEXT,
            entity_mentions TEXT,
            associations TEXT,
            linked_concepts TEXT,
            conflicts_with TEXT,
            previous_event_id TEXT,
            next_event_id TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{meta_table}" (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS "{collection_name}_memory_id_idx"
        ON "{collection_name}" (memory_id)
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS "{collection_name}_memory_type_idx"
        ON "{collection_name}" (memory_type)
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS "{collection_name}_timestamp_idx"
        ON "{collection_name}" (timestamp)
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS "{collection_name}_importance_idx"
        ON "{collection_name}" (importance)
        """
    )
    conn.commit()


def ensure_sqlite_vector_table(
    conn: sqlite3.Connection,
    *,
    vector_table: str,
    dimension: int,
) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS "{vector_table}"
        USING vec0(embedding float[{dimension}])
        """
    )
    conn.commit()
