"""Tests for the markdown export module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemotree.core.models import LinkType, MemoryItem, MemoryLink, MemoryType
from mnemotree.export.markdown import (
    ExportFilters,
    ExportOptions,
    _build_anchor_links,
    _build_wiki_links,
    _build_yaml_frontmatter,
    _filter_memories,
    _memory_title,
    _sanitize_filename,
    _slug,
    _sort_memories,
    render_single_file,
    render_vault,
)
from mnemotree.store.protocols import SupportsMemoryListing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(
    *,
    memory_id: str = "mem-001",
    content: str = "Python is a programming language.",
    summary: str | None = "About Python",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    importance: float = 0.8,
    tags: list[str] | None = None,
    entities: dict[str, str] | None = None,
    emotions: list[str] | None = None,
    timestamp: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        content=content,
        summary=summary,
        memory_type=memory_type,
        importance=importance,
        tags=tags or [],
        entities=entities or {},
        emotions=emotions or [],
        timestamp=timestamp or datetime(2026, 2, 15, 10, 30, tzinfo=timezone.utc),
    )


def _make_link(
    source_id: str = "mem-001",
    target_id: str = "mem-002",
    link_type: LinkType = LinkType.ELABORATES,
) -> MemoryLink:
    return MemoryLink(
        source_id=source_id,
        target_id=target_id,
        link_type=link_type,
    )


@pytest.fixture()
def sample_memories() -> list[MemoryItem]:
    return [
        _make_memory(
            memory_id="mem-001",
            content="Python is a programming language.",
            summary="About Python",
            tags=["project", "design"],
            entities={"Python": "language", "FastAPI": "framework"},
            importance=0.85,
            timestamp=datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc),
        ),
        _make_memory(
            memory_id="mem-002",
            content="Rust has zero-cost abstractions.",
            summary="About Rust",
            tags=["systems"],
            entities={"Rust": "language"},
            importance=0.7,
            timestamp=datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc),
        ),
        _make_memory(
            memory_id="mem-003",
            content="Docker containers share the host kernel.",
            summary="Docker basics",
            memory_type=MemoryType.PROCEDURAL,
            tags=["devops"],
            importance=0.5,
            timestamp=datetime(2026, 1, 10, 8, 0, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture()
def sample_links() -> dict[str, list[MemoryLink]]:
    return {
        "mem-001": [_make_link("mem-001", "mem-002", LinkType.ELABORATES)],
        "mem-002": [_make_link("mem-001", "mem-002", LinkType.ELABORATES)],
    }


# ---------------------------------------------------------------------------
# _memory_title
# ---------------------------------------------------------------------------


class TestMemoryTitle:
    def test_uses_summary(self) -> None:
        mem = _make_memory(summary="My Summary")
        assert _memory_title(mem) == "My Summary"

    def test_truncates_long_summary(self) -> None:
        mem = _make_memory(summary="A" * 100)
        title = _memory_title(mem)
        assert len(title) <= 80
        assert title.endswith("...")

    def test_falls_back_to_content(self) -> None:
        mem = _make_memory(summary=None, content="Some content here")
        assert _memory_title(mem) == "Some content here"

    def test_truncates_long_content(self) -> None:
        mem = _make_memory(summary=None, content="B" * 100)
        title = _memory_title(mem)
        assert len(title) <= 60
        assert title.endswith("...")

    def test_falls_back_to_memory_id(self) -> None:
        mem = _make_memory(summary=None, content="", memory_id="abcdef12-3456")
        assert _memory_title(mem) == "abcdef12"


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_basic(self) -> None:
        assert _sanitize_filename("Hello World", "id1") == "Hello-World"

    def test_removes_invalid_chars(self) -> None:
        result = _sanitize_filename('File: "test"?', "id1")
        assert ":" not in result
        assert '"' not in result
        assert "?" not in result

    def test_empty_falls_back_to_id(self) -> None:
        assert _sanitize_filename("???", "abcdef12") == "abcdef12"

    def test_truncates_long_names(self) -> None:
        result = _sanitize_filename("X" * 200, "id1")
        assert len(result) <= 120


# ---------------------------------------------------------------------------
# _slug
# ---------------------------------------------------------------------------


class TestSlug:
    def test_basic(self) -> None:
        assert _slug("About Python") == "about-python"

    def test_special_chars(self) -> None:
        assert _slug("Hello! World?") == "hello-world"


# ---------------------------------------------------------------------------
# _filter_memories
# ---------------------------------------------------------------------------


class TestFilterMemories:
    def test_no_filters(self, sample_memories: list[MemoryItem]) -> None:
        result = _filter_memories(sample_memories, None)
        assert len(result) == 3

    def test_filter_by_type(self, sample_memories: list[MemoryItem]) -> None:
        f = ExportFilters(memory_types=[MemoryType.SEMANTIC])
        result = _filter_memories(sample_memories, f)
        assert len(result) == 2
        assert all(m.memory_type == MemoryType.SEMANTIC for m in result)

    def test_filter_by_tags(self, sample_memories: list[MemoryItem]) -> None:
        f = ExportFilters(tags=["devops"])
        result = _filter_memories(sample_memories, f)
        assert len(result) == 1
        assert result[0].memory_id == "mem-003"

    def test_filter_by_importance_range(self, sample_memories: list[MemoryItem]) -> None:
        f = ExportFilters(min_importance=0.6, max_importance=0.9)
        result = _filter_memories(sample_memories, f)
        assert len(result) == 2  # mem-001 (0.85) and mem-002 (0.7)

    def test_filter_by_date_range(self, sample_memories: list[MemoryItem]) -> None:
        f = ExportFilters(
            since=datetime(2026, 2, 1, tzinfo=timezone.utc),
            until=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc),
        )
        result = _filter_memories(sample_memories, f)
        assert len(result) == 1
        assert result[0].memory_id == "mem-001"

    def test_combined_filters(self, sample_memories: list[MemoryItem]) -> None:
        f = ExportFilters(
            memory_types=[MemoryType.SEMANTIC],
            min_importance=0.8,
        )
        result = _filter_memories(sample_memories, f)
        assert len(result) == 1
        assert result[0].memory_id == "mem-001"

    def test_empty_result(self, sample_memories: list[MemoryItem]) -> None:
        f = ExportFilters(tags=["nonexistent"])
        result = _filter_memories(sample_memories, f)
        assert result == []


# ---------------------------------------------------------------------------
# _sort_memories
# ---------------------------------------------------------------------------


class TestSortMemories:
    def test_sort_by_timestamp_desc(self, sample_memories: list[MemoryItem]) -> None:
        result = _sort_memories(sample_memories, "timestamp", "desc")
        assert result[0].memory_id == "mem-002"
        assert result[-1].memory_id == "mem-003"

    def test_sort_by_importance_asc(self, sample_memories: list[MemoryItem]) -> None:
        result = _sort_memories(sample_memories, "importance", "asc")
        assert result[0].memory_id == "mem-003"
        assert result[-1].memory_id == "mem-001"

    def test_sort_by_type(self, sample_memories: list[MemoryItem]) -> None:
        result = _sort_memories(sample_memories, "memory_type", "asc")
        assert result[-1].memory_type == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# YAML Front Matter
# ---------------------------------------------------------------------------


class TestYamlFrontmatter:
    def test_basic_fields(self) -> None:
        mem = _make_memory(
            memory_id="abc123",
            tags=["project", "design"],
            entities={"Python": "language"},
            emotions=["curiosity"],
        )
        fm = _build_yaml_frontmatter(mem)
        assert fm.startswith("---")
        assert fm.endswith("---")
        assert 'memory_id: "abc123"' in fm
        assert "memory_type: semantic" in fm
        assert "importance: 0.8" in fm
        assert "tags: [project, design]" in fm
        assert "entities: {Python: language}" in fm
        assert "emotions: [curiosity]" in fm
        assert "created:" in fm

    def test_no_optional_fields(self) -> None:
        mem = _make_memory(tags=[], entities={}, emotions=[])
        fm = _build_yaml_frontmatter(mem)
        assert "tags:" not in fm
        assert "entities:" not in fm
        assert "emotions:" not in fm


# ---------------------------------------------------------------------------
# Wiki Links / Anchor Links
# ---------------------------------------------------------------------------


class TestLinks:
    def test_wiki_links(self) -> None:
        link = _make_link("mem-001", "mem-002", LinkType.ELABORATES)
        id_to_title = {"mem-001": "Title A", "mem-002": "Title B"}
        result = _build_wiki_links("mem-001", [link], id_to_title)
        assert "[[Title B]]" in result
        assert "**elaborates**" in result

    def test_anchor_links(self) -> None:
        link = _make_link("mem-001", "mem-002", LinkType.ELABORATES)
        id_to_title = {"mem-001": "Title A", "mem-002": "Title B"}
        result = _build_anchor_links("mem-001", [link], id_to_title)
        assert "[Title B](#title-b)" in result
        assert "**elaborates**" in result

    def test_empty_links(self) -> None:
        assert _build_wiki_links("mem-001", [], {}) == ""
        assert _build_anchor_links("mem-001", [], {}) == ""


# ---------------------------------------------------------------------------
# Single-file rendering
# ---------------------------------------------------------------------------


class TestRenderSingleFile:
    def test_header(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions()
        result = render_single_file(sample_memories, {}, opts)
        assert result.startswith("# Mnemotree Memory Export")
        assert f"Memories: {len(sample_memories)}" in result

    def test_toc(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions(include_toc=True)
        result = render_single_file(sample_memories, {}, opts)
        assert "## Table of Contents" in result
        assert "[About Python]" in result
        assert "[About Rust]" in result

    def test_no_toc(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions(include_toc=False)
        result = render_single_file(sample_memories, {}, opts)
        assert "## Table of Contents" not in result

    def test_sections(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions()
        result = render_single_file(sample_memories, {}, opts)
        assert "## About Python" in result
        assert "## About Rust" in result
        assert "**Type:** semantic" in result

    def test_with_links(
        self,
        sample_memories: list[MemoryItem],
        sample_links: dict[str, list[MemoryLink]],
    ) -> None:
        opts = ExportOptions(include_links=True)
        result = render_single_file(sample_memories, sample_links, opts)
        assert "### Links" in result
        assert "**elaborates**" in result

    def test_no_links(
        self,
        sample_memories: list[MemoryItem],
        sample_links: dict[str, list[MemoryLink]],
    ) -> None:
        opts = ExportOptions(include_links=False)
        result = render_single_file(sample_memories, sample_links, opts)
        assert "### Links" not in result

    def test_empty_export(self) -> None:
        opts = ExportOptions()
        result = render_single_file([], {}, opts)
        assert "# Mnemotree Memory Export" in result
        assert "Memories: 0" in result

    def test_tags_and_entities(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions()
        result = render_single_file(sample_memories, {}, opts)
        assert "#project" in result
        assert "Python (language)" in result


# ---------------------------------------------------------------------------
# Vault rendering
# ---------------------------------------------------------------------------


class TestRenderVault:
    def test_file_per_memory(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions(mode="vault")
        files = render_vault(sample_memories, {}, opts)
        # 3 memories + _index.md
        assert len(files) == 4
        assert "_index.md" in files

    def test_yaml_frontmatter_in_files(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions(mode="vault")
        files = render_vault(sample_memories, {}, opts)
        for name, content in files.items():
            if name == "_index.md":
                continue
            assert content.startswith("---"), f"{name} missing frontmatter"
            assert "memory_id:" in content

    def test_wiki_links_in_vault(
        self,
        sample_memories: list[MemoryItem],
        sample_links: dict[str, list[MemoryLink]],
    ) -> None:
        opts = ExportOptions(mode="vault", include_links=True)
        files = render_vault(sample_memories, sample_links, opts)
        # mem-001 should have a wiki-link to mem-002
        python_file = [v for k, v in files.items() if "Python" in k]
        assert python_file
        assert "[[About Rust]]" in python_file[0]

    def test_index_file(self, sample_memories: list[MemoryItem]) -> None:
        opts = ExportOptions(mode="vault")
        files = render_vault(sample_memories, {}, opts)
        index = files["_index.md"]
        assert "# Mnemotree Memory Index" in index
        assert "| Title | Type | Importance | Tags |" in index
        assert "[[About Python]]" in index

    def test_filename_collision(self) -> None:
        mems = [
            _make_memory(memory_id="id-aaa111", summary="Same Title"),
            _make_memory(memory_id="id-bbb222", summary="Same Title"),
        ]
        opts = ExportOptions(mode="vault")
        files = render_vault(mems, {}, opts)
        # Should be 2 memory files + _index.md = 3
        assert len(files) == 3
        filenames = [k for k in files if k != "_index.md"]
        assert len(set(filenames)) == 2  # no collision

    def test_empty_vault(self) -> None:
        opts = ExportOptions(mode="vault")
        files = render_vault([], {}, opts)
        assert len(files) == 1
        assert "_index.md" in files


# ---------------------------------------------------------------------------
# export_memories (async entry point)
# ---------------------------------------------------------------------------


class TestExportMemories:
    @pytest.mark.asyncio()
    async def test_single_file_write(
        self,
        sample_memories: list[MemoryItem],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        from mnemotree.export.markdown import export_memories

        store = _FakeListingStore(sample_memories)
        out = str(tmp_path / "export.md")  # type: ignore[operator]
        opts = ExportOptions(mode="single", output_path=out)
        path, count = await export_memories(store, opts)
        assert count == 3
        from pathlib import Path

        content = Path(path).read_text()
        assert "# Mnemotree Memory Export" in content

    @pytest.mark.asyncio()
    async def test_vault_write(
        self,
        sample_memories: list[MemoryItem],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        from pathlib import Path

        from mnemotree.export.markdown import export_memories

        store = _FakeListingStore(sample_memories)
        out = str(tmp_path / "vault")  # type: ignore[operator]
        opts = ExportOptions(mode="vault", output_path=out)
        path, count = await export_memories(store, opts)
        assert count == 3
        vault_dir = Path(path)
        assert vault_dir.is_dir()
        assert (vault_dir / "_index.md").exists()

    @pytest.mark.asyncio()
    async def test_unsupported_store(self) -> None:
        from mnemotree.export.markdown import export_memories

        with pytest.raises(NotImplementedError):
            await export_memories(object())

    @pytest.mark.asyncio()
    async def test_with_filters(
        self,
        sample_memories: list[MemoryItem],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        from mnemotree.export.markdown import export_memories

        store = _FakeListingStore(sample_memories)
        out = str(tmp_path / "filtered.md")  # type: ignore[operator]
        opts = ExportOptions(mode="single", output_path=out)
        filters = ExportFilters(memory_types=[MemoryType.PROCEDURAL])
        path, count = await export_memories(store, opts, filters)
        assert count == 1


class _FakeListingStore:
    """Minimal fake store implementing SupportsMemoryListing."""

    def __init__(self, memories: list[MemoryItem]) -> None:
        self._memories = memories

    async def list_memories(self, *, include_embeddings: bool = False) -> list[MemoryItem]:
        return list(self._memories)

    async def close(self) -> None:
        pass


SupportsMemoryListing.register(_FakeListingStore)  # type: ignore[attr-defined]
