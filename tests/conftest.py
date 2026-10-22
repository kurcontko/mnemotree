from datetime import UTC, datetime

import pytest


@pytest.fixture
def memory_item():
    from mnemotree.core.models import MemoryItem, MemoryType

    return MemoryItem(
        memory_id="test-id",
        content="Test content",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp=str(datetime.now(UTC)),
        embedding=[0.1] * 1536,  # Mock embedding
    )


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Use pytest's tmp_path for better isolation between tests"""
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    yield str(chroma_dir)
    # Cleanup is automatic with tmp_path
