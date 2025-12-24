
import pytest
import shutil
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mnemotree.core.models import MemoryItem, MemoryType

@pytest.fixture
def memory_item():
    return MemoryItem(
        memory_id="test-id",
        content="Test content",
        memory_type=MemoryType.SEMANTIC,
        importance=0.5,
        timestamp=str(datetime.now(timezone.utc)),
        embedding=[0.1] * 1536  # Mock embedding
    )

@pytest.fixture
def temp_chroma_dir():
    path = Path(".test_chroma")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    yield str(path)
    if path.exists():
        shutil.rmtree(path)
