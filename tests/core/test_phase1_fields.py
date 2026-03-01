"""Tests for Phase 1 data model additions: bi-temporal fields, contextual_intent, is_hot, new LinkTypes."""

from datetime import datetime, timezone

from mnemotree.core.models import LinkType, MemoryItem, MemoryType


class TestBiTemporalFields:
    def test_event_time_default_none(self) -> None:
        m = MemoryItem(content="test", memory_type=MemoryType.SEMANTIC, importance=0.5)
        assert m.event_time is None

    def test_event_time_set(self) -> None:
        evt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        m = MemoryItem(
            content="met Alice at conference",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            event_time=evt,
        )
        assert m.event_time == evt
        assert m.timestamp != evt  # storage time differs from event time

    def test_valid_from_until(self) -> None:
        vf = datetime(2024, 1, 1, tzinfo=timezone.utc)
        vu = datetime(2024, 12, 31, tzinfo=timezone.utc)
        m = MemoryItem(
            content="lease expires 2024",
            memory_type=MemoryType.SEMANTIC,
            importance=0.6,
            valid_from=vf,
            valid_until=vu,
        )
        assert m.valid_from == vf
        assert m.valid_until == vu

    def test_valid_until_none_means_still_valid(self) -> None:
        m = MemoryItem(
            content="email is bob@example.com",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert m.valid_until is None

    def test_observation_date(self) -> None:
        obs = datetime(2024, 3, 15, tzinfo=timezone.utc)
        m = MemoryItem(
            content="saw that the office moved",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            observation_date=obs,
        )
        assert m.observation_date == obs

    def test_referenced_date(self) -> None:
        ref = datetime(2023, 7, 4, tzinfo=timezone.utc)
        m = MemoryItem(
            content="Independence Day 2023",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            referenced_date=ref,
        )
        assert m.referenced_date == ref

    def test_temporal_offset(self) -> None:
        m = MemoryItem(
            content="two days ago I ran a marathon",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            temporal_offset="2 days ago",
        )
        assert m.temporal_offset == "2 days ago"


class TestContextualIntent:
    def test_default_none(self) -> None:
        m = MemoryItem(content="test", memory_type=MemoryType.SEMANTIC, importance=0.5)
        assert m.contextual_intent is None

    def test_set_intent(self) -> None:
        m = MemoryItem(
            content="how to cook pasta",
            memory_type=MemoryType.PROCEDURAL,
            importance=0.5,
            contextual_intent="procedural_instruction",
        )
        assert m.contextual_intent == "procedural_instruction"


class TestIsHot:
    def test_default_false(self) -> None:
        m = MemoryItem(content="test", memory_type=MemoryType.SEMANTIC, importance=0.5)
        assert m.is_hot is False

    def test_set_hot(self) -> None:
        m = MemoryItem(
            content="user prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            is_hot=True,
        )
        assert m.is_hot is True


class TestNewLinkTypes:
    def test_supersedes(self) -> None:
        assert LinkType.SUPERSEDES.value == "supersedes"

    def test_updates(self) -> None:
        assert LinkType.UPDATES.value == "updates"

    def test_sequence(self) -> None:
        assert LinkType.SEQUENCE.value == "sequence"

    def test_all_link_types_are_strings(self) -> None:
        for lt in LinkType:
            assert isinstance(lt.value, str)


class TestSQLiteSerialization:
    def test_roundtrip_new_fields(self) -> None:
        from mnemotree.store._records import sqlite_record_from_memory

        evt = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
        m = MemoryItem(
            content="test roundtrip",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            event_time=evt,
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 12, 31, tzinfo=timezone.utc),
            observation_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            referenced_date=datetime(2023, 7, 4, tzinfo=timezone.utc),
            temporal_offset="6 months ago",
            contextual_intent="factual_update",
            is_hot=True,
            embedding=[0.1, 0.2, 0.3],
        )
        record = sqlite_record_from_memory(m)

        assert record["event_time"] is not None
        assert record["valid_from"] is not None
        assert record["valid_until"] is not None
        assert record["observation_date"] is not None
        assert record["referenced_date"] is not None
        assert record["temporal_offset"] == "6 months ago"
        assert record["contextual_intent"] == "factual_update"
        assert record["is_hot"] == 1

    def test_serialization_none_fields(self) -> None:
        from mnemotree.store._records import sqlite_record_from_memory

        m = MemoryItem(
            content="minimal",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
        )
        record = sqlite_record_from_memory(m)

        assert record["event_time"] is None
        assert record["valid_from"] is None
        assert record["is_hot"] == 0


class TestSchemasMigration:
    def test_phase1_migration_idempotent(self) -> None:
        import sqlite3

        from mnemotree.store._schema import (
            create_sqlite_schema,
            migrate_sqlite_phase1_fields,
        )

        conn = sqlite3.connect(":memory:")
        create_sqlite_schema(conn, collection_name="test_mem", meta_table="test_meta")
        # Should be idempotent
        migrate_sqlite_phase1_fields(conn, "test_mem")
        migrate_sqlite_phase1_fields(conn, "test_mem")

        cursor = conn.execute('PRAGMA table_info("test_mem")')
        columns = {row[1] for row in cursor.fetchall()}
        assert "event_time" in columns
        assert "valid_from" in columns
        assert "valid_until" in columns
        assert "observation_date" in columns
        assert "referenced_date" in columns
        assert "temporal_offset" in columns
        assert "contextual_intent" in columns
        assert "is_hot" in columns
        conn.close()


class TestChromaSerializationNewFields:
    def test_roundtrip_new_fields(self) -> None:
        from mnemotree.store._records import (
            chroma_memory_from_record,
            chroma_metadata_from_memory,
        )

        m = MemoryItem(
            memory_id="chroma-test",
            content="test chroma roundtrip",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            event_time=datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc),
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 12, 31, tzinfo=timezone.utc),
            observation_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            referenced_date=datetime(2023, 7, 4, tzinfo=timezone.utc),
            temporal_offset="6 months ago",
            contextual_intent="factual_update",
            is_hot=True,
            embedding=[0.1, 0.2, 0.3],
        )
        metadata = chroma_metadata_from_memory(m)

        assert metadata["event_time"] != ""
        assert metadata["contextual_intent"] == "factual_update"
        assert metadata["is_hot"] == "1"
        assert metadata["temporal_offset"] == "6 months ago"

        restored = chroma_memory_from_record(
            memory_id="chroma-test",
            document="test chroma roundtrip",
            embedding=[0.1, 0.2, 0.3],
            metadata=metadata,
        )
        assert restored.contextual_intent == "factual_update"
        assert restored.is_hot is True
        assert restored.temporal_offset == "6 months ago"
        assert restored.event_time is not None

    def test_missing_fields_graceful(self) -> None:
        from mnemotree.store._records import chroma_memory_from_record

        # Simulate old ChromaDB collection without new fields
        metadata = {
            "memory_type": "semantic",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "importance": "0.5",
            "confidence": "1.0",
            "tags": "",
            "emotions": "",
            "source": "",
            "context": "",
            "last_accessed": "2024-01-01T00:00:00+00:00",
            "access_count": "0",
            "access_history": "[]",
            "entities": "{}",
        }
        restored = chroma_memory_from_record(
            memory_id="old-mem",
            document="old content",
            embedding=None,
            metadata=metadata,
        )
        assert restored.contextual_intent is None
        assert restored.is_hot is False
        assert restored.event_time is None
