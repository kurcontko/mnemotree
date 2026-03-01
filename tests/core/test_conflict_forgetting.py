"""Tests for conflict-forgetting features:

- ConflictDetector (core/conflict.py): cosine-threshold conflict detection + bi-temporal invalidation
- initial_stability (core/decay.py): importance-coupled initial stability
- invalidate_memory (store/sqlite_vec_store.py): bi-temporal soft-delete
- evict (core/memory.py): hybrid temporal+importance eviction
- spaced-rep boost (_update_access_metadata): stability growth on recall
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemotree.core.conflict import ConflictConfig, ConflictDetector
from mnemotree.core.decay import (
    MEMORY_TYPE_DEFAULTS,
    initial_stability,
)
from mnemotree.core.models import MemoryItem, MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    memory_id: str = "m1",
    content: str = "test content",
    importance: float = 0.5,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    embedding: list[float] | None = None,
    timestamp: datetime | None = None,
    valid_until: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        importance=importance,
        memory_type=memory_type,
        embedding=embedding or [1.0, 0.0, 0.0],
        timestamp=timestamp or datetime.now(timezone.utc),
        valid_until=valid_until,
    )


# ===========================================================================
# ConflictDetector tests
# ===========================================================================


class TestConflictConfig:
    def test_defaults(self) -> None:
        cfg = ConflictConfig()
        assert cfg.enable is False
        assert cfg.similarity_threshold == pytest.approx(0.92)
        assert cfg.candidate_limit == 10
        assert cfg.auto_invalidate is True
        assert cfg.create_contradicts_link is True

    def test_immutable(self) -> None:
        cfg = ConflictConfig()
        with pytest.raises(AttributeError):
            cfg.enable = True  # type: ignore[misc]


class TestConflictDetector:
    """Tests for core/conflict.py ConflictDetector (cosine-threshold)."""

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self) -> None:
        """When enable=False, no detection is performed."""
        detector = ConflictDetector(ConflictConfig(enable=False))
        result = await detector.detect_and_resolve(_make_memory(), MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_no_embedding_returns_empty(self) -> None:
        detector = ConflictDetector(ConflictConfig(enable=True))
        mem = _make_memory(embedding=None)
        # Force embedding to None after construction
        mem.embedding = None
        result = await detector.detect_and_resolve(mem, MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_store_without_vector_search_returns_empty(self) -> None:
        detector = ConflictDetector(ConflictConfig(enable=True))
        # Plain object doesn't implement SupportsVectorSearch
        result = await detector.detect_and_resolve(_make_memory(), object())
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_same_id(self) -> None:
        """Candidate with same ID as new memory is skipped."""
        store = AsyncMock()
        store.get_similar_memories = AsyncMock(return_value=[
            _make_memory(memory_id="m1", embedding=[1.0, 0.0, 0.0]),
        ])
        # Patch isinstance checks by making the store satisfy SupportsVectorSearch
        with patch("mnemotree.core.conflict.isinstance", side_effect=lambda obj, cls: True):
            detector = ConflictDetector(ConflictConfig(enable=True))
            # new_memory has same ID as candidate
            result = await detector.detect_and_resolve(
                _make_memory(memory_id="m1"), store
            )
        # candidate skipped because same ID → no invalidation
        assert result == []

    @pytest.mark.asyncio
    async def test_invalidates_older_similar_memory(self) -> None:
        """Full flow: older candidate with high cosine sim gets invalidated."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        embedding = [1.0, 0.0, 0.0]  # identical → cosine sim = 1.0

        old_mem = _make_memory(
            memory_id="old",
            content="Python is version 3.10",
            embedding=embedding,
            timestamp=old_time,
        )
        new_mem = _make_memory(
            memory_id="new",
            content="Python is version 3.12",
            embedding=embedding,
            timestamp=new_time,
        )

        # Build a mock store that satisfies the protocol checks
        store = AsyncMock()
        store.get_similar_memories = AsyncMock(return_value=[old_mem])
        store.invalidate_memory = AsyncMock(return_value=True)
        store.create_link = AsyncMock()

        detector = ConflictDetector(ConflictConfig(enable=True, similarity_threshold=0.9))

        # Patch isinstance to return True for all protocol checks
        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        with patch("mnemotree.core.conflict.isinstance", side_effect=patched_isinstance):
            result = await detector.detect_and_resolve(new_mem, store)

        assert result == ["old"]
        store.invalidate_memory.assert_called_once()
        store.create_link.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_already_invalidated_candidate(self) -> None:
        """Candidates with valid_until set are skipped."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        invalidated_time = datetime(2024, 3, 1, tzinfo=timezone.utc)

        embedding = [1.0, 0.0, 0.0]

        old_mem = _make_memory(
            memory_id="old",
            embedding=embedding,
            timestamp=old_time,
            valid_until=invalidated_time,  # already invalidated
        )
        new_mem = _make_memory(
            memory_id="new",
            embedding=embedding,
            timestamp=new_time,
        )

        store = AsyncMock()
        store.get_similar_memories = AsyncMock(return_value=[old_mem])

        detector = ConflictDetector(ConflictConfig(enable=True, similarity_threshold=0.9))

        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        with patch("mnemotree.core.conflict.isinstance", side_effect=patched_isinstance):
            result = await detector.detect_and_resolve(new_mem, store)

        assert result == []

    @pytest.mark.asyncio
    async def test_skips_newer_candidate(self) -> None:
        """If candidate is newer than new_memory, no invalidation."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        embedding = [1.0, 0.0, 0.0]

        # Candidate is newer than the "new" memory
        candidate = _make_memory(memory_id="candidate", embedding=embedding, timestamp=new_time)
        new_mem = _make_memory(memory_id="new", embedding=embedding, timestamp=old_time)

        store = AsyncMock()
        store.get_similar_memories = AsyncMock(return_value=[candidate])

        detector = ConflictDetector(ConflictConfig(enable=True, similarity_threshold=0.9))

        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        with patch("mnemotree.core.conflict.isinstance", side_effect=patched_isinstance):
            result = await detector.detect_and_resolve(new_mem, store)

        assert result == []

    @pytest.mark.asyncio
    async def test_below_threshold_not_invalidated(self) -> None:
        """Candidates with cosine sim below threshold are not invalidated."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        # Orthogonal embeddings → cosine sim = 0
        candidate = _make_memory(
            memory_id="old", embedding=[0.0, 1.0, 0.0], timestamp=old_time
        )
        new_mem = _make_memory(
            memory_id="new", embedding=[1.0, 0.0, 0.0], timestamp=new_time
        )

        store = AsyncMock()
        store.get_similar_memories = AsyncMock(return_value=[candidate])

        detector = ConflictDetector(ConflictConfig(enable=True, similarity_threshold=0.92))

        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        with patch("mnemotree.core.conflict.isinstance", side_effect=patched_isinstance):
            result = await detector.detect_and_resolve(new_mem, store)

        assert result == []

    @pytest.mark.asyncio
    async def test_auto_invalidate_disabled(self) -> None:
        """When auto_invalidate=False, no invalidation but link still created."""
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        embedding = [1.0, 0.0, 0.0]

        candidate = _make_memory(memory_id="old", embedding=embedding, timestamp=old_time)
        new_mem = _make_memory(memory_id="new", embedding=embedding, timestamp=new_time)

        store = AsyncMock()
        store.get_similar_memories = AsyncMock(return_value=[candidate])
        store.create_link = AsyncMock()

        detector = ConflictDetector(
            ConflictConfig(enable=True, similarity_threshold=0.9, auto_invalidate=False)
        )

        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        with patch("mnemotree.core.conflict.isinstance", side_effect=patched_isinstance):
            result = await detector.detect_and_resolve(new_mem, store)

        # No invalidation (auto_invalidate=False)
        assert result == []
        store.create_link.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_similar_memories_failure_handled(self) -> None:
        """Exception in get_similar_memories doesn't crash — returns empty."""
        store = AsyncMock()
        store.get_similar_memories = AsyncMock(side_effect=RuntimeError("DB error"))

        detector = ConflictDetector(ConflictConfig(enable=True))
        new_mem = _make_memory()

        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        with patch("mnemotree.core.conflict.isinstance", side_effect=patched_isinstance):
            result = await detector.detect_and_resolve(new_mem, store)

        assert result == []


# ===========================================================================
# initial_stability tests
# ===========================================================================


class TestInitialStability:
    def test_high_importance_doubles_base(self) -> None:
        base = MEMORY_TYPE_DEFAULTS[MemoryType.SEMANTIC].stability_seconds
        result = initial_stability(MemoryType.SEMANTIC, importance=0.9)
        assert result == pytest.approx(base * 2.0)

    def test_low_importance_halves_base(self) -> None:
        base = MEMORY_TYPE_DEFAULTS[MemoryType.SEMANTIC].stability_seconds
        result = initial_stability(MemoryType.SEMANTIC, importance=0.1)
        assert result == pytest.approx(base * 0.5)

    def test_mid_importance_uses_base(self) -> None:
        base = MEMORY_TYPE_DEFAULTS[MemoryType.SEMANTIC].stability_seconds
        result = initial_stability(MemoryType.SEMANTIC, importance=0.5)
        assert result == pytest.approx(base)

    def test_boundary_0_7_is_high(self) -> None:
        base = MEMORY_TYPE_DEFAULTS[MemoryType.EPISODIC].stability_seconds
        result = initial_stability(MemoryType.EPISODIC, importance=0.7)
        assert result == pytest.approx(base * 2.0)

    def test_boundary_0_3_is_mid(self) -> None:
        base = MEMORY_TYPE_DEFAULTS[MemoryType.EPISODIC].stability_seconds
        result = initial_stability(MemoryType.EPISODIC, importance=0.3)
        assert result == pytest.approx(base)

    def test_working_memory_has_short_stability(self) -> None:
        result = initial_stability(MemoryType.WORKING, importance=0.5)
        assert result == pytest.approx(3600.0)  # 1 hour base for working

    def test_procedural_has_long_stability(self) -> None:
        result = initial_stability(MemoryType.PROCEDURAL, importance=0.5)
        assert result == pytest.approx(60 * 86400.0)  # 60 days base

    def test_all_memory_types_have_defaults(self) -> None:
        for mt in MemoryType:
            result = initial_stability(mt, importance=0.5)
            assert result > 0


# ===========================================================================
# Spaced repetition boost (update_access with retrievability)
# ===========================================================================


class TestSpacedRepetitionBoost:
    def test_update_access_with_retrievability_grows_stability(self) -> None:
        """When retrievability is provided, stability_seconds should grow."""
        m = _make_memory(importance=0.5)
        m.stability_seconds = 86400.0  # 1 day

        old_stability = m.stability_seconds
        m.update_access(retrievability=0.7)

        assert m.stability_seconds > old_stability
        assert m.access_count == 1

    def test_update_access_without_retrievability_no_stability_change(self) -> None:
        """Without retrievability, stability is not modified."""
        m = _make_memory(importance=0.5)
        m.stability_seconds = 86400.0

        m.update_access(retrievability=None)

        assert m.stability_seconds == pytest.approx(86400.0)

    def test_update_access_no_stability_set(self) -> None:
        """If stability_seconds is None, update_access still works fine."""
        m = _make_memory(importance=0.5)
        m.stability_seconds = None

        m.update_access(retrievability=0.5)

        assert m.stability_seconds is None
        assert m.access_count == 1

    def test_low_retrievability_bigger_growth(self) -> None:
        """Lower retrievability = harder recall = bigger stability growth."""
        m1 = _make_memory(importance=0.5)
        m1.stability_seconds = 86400.0
        m1.update_access(retrievability=0.9)
        growth_easy = m1.stability_seconds - 86400.0

        m2 = _make_memory(importance=0.5)
        m2.stability_seconds = 86400.0
        m2.update_access(retrievability=0.3)
        growth_hard = m2.stability_seconds - 86400.0

        assert growth_hard > growth_easy


# ===========================================================================
# Eviction tests (using mock store)
# ===========================================================================


class TestEviction:
    def _build_mock_core(self, memories: list[MemoryItem]):
        """Build a MemoryCore with mocked store for eviction testing."""

        store = AsyncMock()
        store.list_memories = AsyncMock(return_value=memories)

        # Make isinstance checks work for SupportsMemoryListing
        original_isinstance = isinstance

        def patched_isinstance(obj, cls):
            if obj is store:
                return True
            return original_isinstance(obj, cls)

        return store, patched_isinstance

    @pytest.mark.asyncio
    async def test_evict_by_min_importance(self) -> None:
        """Memories below min_importance are evicted."""
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory(memory_id="low", importance=0.2, timestamp=now),
            _make_memory(memory_id="mid", importance=0.5, timestamp=now),
            _make_memory(memory_id="high", importance=0.9, timestamp=now),
        ]

        store, patched_isinstance = self._build_mock_core(memories)

        # Build a minimal MemoryCore-like object to call evict on
        from mnemotree.core.memory import MemoryCore

        core = MagicMock(spec=MemoryCore)
        core.store = store
        core.persistence = MagicMock()
        core.persistence.delete = AsyncMock()

        with patch("mnemotree.core.memory.isinstance", side_effect=patched_isinstance):
            result = await MemoryCore.evict(
                core, min_importance=0.4, dry_run=True
            )

        # low (0.2) is below min_importance (0.4) → evicted
        assert "low" in result
        # mid (0.5) is above min_importance (0.4) → not evicted
        assert "mid" not in result
        # high (0.9) is above importance_floor (0.7 default) → never evicted
        assert "high" not in result

    @pytest.mark.asyncio
    async def test_evict_dry_run_no_delete(self) -> None:
        """dry_run=True returns IDs but doesn't delete."""
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory(memory_id="low", importance=0.2, timestamp=now),
            _make_memory(memory_id="high", importance=0.9, timestamp=now),
        ]

        store, patched_isinstance = self._build_mock_core(memories)

        from mnemotree.core.memory import MemoryCore

        core = MagicMock(spec=MemoryCore)
        core.store = store
        core.persistence = MagicMock()
        core.persistence.delete = AsyncMock()

        with patch("mnemotree.core.memory.isinstance", side_effect=patched_isinstance):
            result = await MemoryCore.evict(core, min_importance=0.3, dry_run=True)

        assert "low" in result
        core.persistence.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_evict_importance_floor_protects_high(self) -> None:
        """Memories at or above importance_floor are never evicted."""
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory(memory_id="protected", importance=0.8, timestamp=now),
            _make_memory(memory_id="unprotected", importance=0.3, timestamp=now),
        ]

        store, patched_isinstance = self._build_mock_core(memories)

        from mnemotree.core.memory import MemoryCore

        core = MagicMock(spec=MemoryCore)
        core.store = store
        core.persistence = MagicMock()
        core.persistence.delete = AsyncMock()

        with patch("mnemotree.core.memory.isinstance", side_effect=patched_isinstance):
            result = await MemoryCore.evict(
                core, min_importance=0.5, importance_floor=0.7, dry_run=True
            )

        # protected (0.8) is above importance_floor (0.7) → never evicted
        assert "protected" not in result
        # unprotected (0.3) is below min_importance (0.5) → evicted
        assert "unprotected" in result

    @pytest.mark.asyncio
    async def test_evict_empty_store_returns_empty(self) -> None:
        """Evicting from empty store returns empty list."""
        store, patched_isinstance = self._build_mock_core([])

        from mnemotree.core.memory import MemoryCore

        core = MagicMock(spec=MemoryCore)
        core.store = store

        with patch("mnemotree.core.memory.isinstance", side_effect=patched_isinstance):
            result = await MemoryCore.evict(core, max_memories=5, dry_run=True)

        assert result == []

    @pytest.mark.asyncio
    async def test_evict_max_memories_caps_store(self) -> None:
        """Evict until store has at most max_memories."""
        now = datetime.now(timezone.utc)
        memories = [
            _make_memory(memory_id=f"m{i}", importance=0.1 * (i + 1), timestamp=now)
            for i in range(5)
        ]

        store, patched_isinstance = self._build_mock_core(memories)

        from mnemotree.core.memory import MemoryCore

        core = MagicMock(spec=MemoryCore)
        core.store = store
        core.persistence = MagicMock()
        core.persistence.delete = AsyncMock()

        with patch("mnemotree.core.memory.isinstance", side_effect=patched_isinstance):
            result = await MemoryCore.evict(
                core, max_memories=3, importance_floor=0.7, dry_run=True
            )

        # 5 memories, max 3, so 2 should be evicted (lowest importance first)
        assert len(result) == 2


# ===========================================================================
# SQLite store invalidation tests
# ===========================================================================


class TestSQLiteInvalidation:
    @pytest.fixture
    def _skip_if_no_sqlite_vec(self):
        pytest.importorskip("sqlite_vec")
        import sqlite3

        try:
            import sqlite_vec

            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.close()
        except (sqlite3.Error, Exception):
            pytest.skip("sqlite-vec extension not loadable")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_skip_if_no_sqlite_vec")
    async def test_invalidate_memory_sets_valid_until(self, tmp_path) -> None:
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

        db_path = tmp_path / "test.sqlite"
        store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=3)
        await store.initialize()

        try:
            mem = _make_memory(memory_id="to-invalidate")
            await store.store_memory(mem)

            result = await store.invalidate_memory("to-invalidate", reason="test")
            assert result is True

            # Fetch the memory and check valid_until is set
            retrieved = await store.get_memory("to-invalidate")
            assert retrieved is not None
            assert retrieved.valid_until is not None
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_skip_if_no_sqlite_vec")
    async def test_invalidate_nonexistent_returns_false(self, tmp_path) -> None:
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

        db_path = tmp_path / "test.sqlite"
        store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=3)
        await store.initialize()

        try:
            result = await store.invalidate_memory("nonexistent", reason="test")
            assert result is False
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_skip_if_no_sqlite_vec")
    async def test_list_memories_excludes_invalidated(self, tmp_path) -> None:
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

        db_path = tmp_path / "test.sqlite"
        store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=3)
        await store.initialize()

        try:
            m1 = _make_memory(memory_id="valid")
            m2 = _make_memory(memory_id="invalid")
            await store.store_memory(m1)
            await store.store_memory(m2)
            await store.invalidate_memory("invalid", reason="test")

            active = await store.list_memories(include_embeddings=False, include_invalidated=False)
            assert len(active) == 1
            assert active[0].memory_id == "valid"

            all_mems = await store.list_memories(include_embeddings=False, include_invalidated=True)
            assert len(all_mems) == 2
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_skip_if_no_sqlite_vec")
    async def test_store_memory_sets_valid_from(self, tmp_path) -> None:
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

        db_path = tmp_path / "test.sqlite"
        store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=3)
        await store.initialize()

        try:
            mem = _make_memory(memory_id="auto-valid-from")
            assert mem.valid_from is None  # not set before store

            await store.store_memory(mem)
            retrieved = await store.get_memory("auto-valid-from")
            assert retrieved is not None
            assert retrieved.valid_from is not None
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_skip_if_no_sqlite_vec")
    async def test_get_similar_memories_excludes_invalidated(self, tmp_path) -> None:
        from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

        db_path = tmp_path / "test.sqlite"
        store = SQLiteVecMemoryStore(db_path=db_path, embedding_dim=3)
        await store.initialize()

        try:
            m1 = _make_memory(memory_id="valid", embedding=[1.0, 0.0, 0.0])
            m2 = _make_memory(memory_id="invalid", embedding=[1.0, 0.0, 0.0])
            await store.store_memory(m1)
            await store.store_memory(m2)
            await store.invalidate_memory("invalid", reason="test")

            results = await store.get_similar_memories(
                query="test",
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
            )
            result_ids = [r.memory_id for r in results]
            assert "valid" in result_ids
            assert "invalid" not in result_ids
        finally:
            await store.close()
