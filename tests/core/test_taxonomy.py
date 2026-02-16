"""Tests for taxonomy module."""

from mnemotree.core.models import MemoryType
from mnemotree.core.taxonomy import (
    EPISODIC_TYPES,
    PROCEDURAL_TYPES,
    SEMANTIC_TYPES,
    is_episodic,
    is_procedural,
    is_semantic,
)


class TestTaxonomyClassification:
    """Tests for type classification functions."""

    def test_episodic_types_list(self):
        """EPISODIC_TYPES contains episodic and autobiographical."""
        assert MemoryType.EPISODIC in EPISODIC_TYPES
        assert MemoryType.AUTOBIOGRAPHICAL in EPISODIC_TYPES
        assert len(EPISODIC_TYPES) == 2

    def test_semantic_types_list(self):
        """SEMANTIC_TYPES contains only semantic."""
        assert MemoryType.SEMANTIC in SEMANTIC_TYPES
        assert len(SEMANTIC_TYPES) == 1

    def test_procedural_types_list(self):
        """PROCEDURAL_TYPES contains procedural, priming, and conditioning."""
        assert MemoryType.PROCEDURAL in PROCEDURAL_TYPES
        assert MemoryType.PRIMING in PROCEDURAL_TYPES
        assert MemoryType.CONDITIONING in PROCEDURAL_TYPES
        assert len(PROCEDURAL_TYPES) == 3

    def test_is_episodic_true(self):
        """is_episodic returns True for episodic types."""
        assert is_episodic(MemoryType.EPISODIC)
        assert is_episodic(MemoryType.AUTOBIOGRAPHICAL)

    def test_is_episodic_false(self):
        """is_episodic returns False for non-episodic types."""
        assert not is_episodic(MemoryType.SEMANTIC)
        assert not is_episodic(MemoryType.PROCEDURAL)
        assert not is_episodic(MemoryType.WORKING)

    def test_is_semantic_true(self):
        """is_semantic returns True for semantic type."""
        assert is_semantic(MemoryType.SEMANTIC)

    def test_is_semantic_false(self):
        """is_semantic returns False for non-semantic types."""
        assert not is_semantic(MemoryType.EPISODIC)
        assert not is_semantic(MemoryType.PROCEDURAL)

    def test_is_procedural_true(self):
        """is_procedural returns True for procedural types."""
        assert is_procedural(MemoryType.PROCEDURAL)
        assert is_procedural(MemoryType.PRIMING)
        assert is_procedural(MemoryType.CONDITIONING)

    def test_is_procedural_false(self):
        """is_procedural returns False for non-procedural types."""
        assert not is_procedural(MemoryType.EPISODIC)
        assert not is_procedural(MemoryType.SEMANTIC)


class TestMemoryTypeProperties:
    """Tests for MemoryType enum properties."""

    def test_episodic_property(self):
        """MemoryType.is_episodic works."""
        assert MemoryType.EPISODIC.is_episodic
        assert MemoryType.AUTOBIOGRAPHICAL.is_episodic
        assert not MemoryType.SEMANTIC.is_episodic

    def test_semantic_property(self):
        """MemoryType.is_semantic works."""
        assert MemoryType.SEMANTIC.is_semantic
        assert not MemoryType.EPISODIC.is_semantic
        assert not MemoryType.PROCEDURAL.is_semantic

    def test_procedural_property(self):
        """MemoryType.is_procedural works."""
        assert MemoryType.PROCEDURAL.is_procedural
        assert MemoryType.PRIMING.is_procedural
        assert not MemoryType.EPISODIC.is_procedural

    def test_category_property(self):
        """MemoryType.category returns correct category."""
        assert MemoryType.EPISODIC.category == "declarative"
        assert MemoryType.SEMANTIC.category == "declarative"
        assert MemoryType.PROCEDURAL.category == "non_declarative"
        assert MemoryType.WORKING.category == "working"
