"""Markdown export for Mnemotree memories.

Supports two modes:
- **single**: One consolidated markdown file with TOC and sections.
- **vault**: Per-memory files with YAML front matter and wiki-links (Obsidian-compatible).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mnemotree.core.models import MemoryItem, MemoryLink, MemoryType


@dataclass(frozen=True)
class ExportOptions:
    """Options controlling markdown export behaviour."""

    mode: Literal["single", "vault"] = "single"
    output_path: str | None = None
    include_links: bool = True
    include_toc: bool = True
    sort_by: Literal["timestamp", "importance", "memory_type"] = "timestamp"
    sort_order: Literal["asc", "desc"] = "desc"


@dataclass(frozen=True)
class ExportFilters:
    """Filters applied to memories before export."""

    memory_types: list[MemoryType] | None = None
    tags: list[str] | None = None
    min_importance: float | None = None
    max_importance: float | None = None
    since: datetime | None = None
    until: datetime | None = None


def _memory_title(memory: MemoryItem) -> str:
    """Derive a human-readable title from a memory."""
    if memory.summary:
        title = memory.summary.strip().split("\n")[0]
        if len(title) > 80:
            title = title[:77] + "..."
        return title
    if memory.content:
        first_line = memory.content.strip().split("\n")[0]
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        return first_line
    return memory.memory_id[:8]


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(title: str, memory_id: str) -> str:
    """Convert a title to a safe filename stem (no extension)."""
    name = _INVALID_FILENAME_CHARS.sub("", title).strip()
    name = re.sub(r"\s+", "-", name)
    if not name:
        name = memory_id[:8]
    if len(name) > 120:
        name = name[:120]
    return name


def _slug(title: str) -> str:
    """Build a markdown-anchor slug from a title."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def _format_datetime(dt: datetime | Any) -> str:
    """Format a datetime to ISO-8601 string."""
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return str(dt)


def _filter_memories(
    memories: list[MemoryItem],
    filters: ExportFilters | None,
) -> list[MemoryItem]:
    """Apply filters to a list of memories."""
    if filters is None:
        return list(memories)

    result = list(memories)

    if filters.memory_types:
        type_set = set(filters.memory_types)
        result = [m for m in result if m.memory_type in type_set]

    if filters.tags:
        tag_set = set(filters.tags)
        result = [m for m in result if tag_set.intersection(m.tags)]

    if filters.min_importance is not None:
        result = [m for m in result if m.importance >= filters.min_importance]

    if filters.max_importance is not None:
        result = [m for m in result if m.importance <= filters.max_importance]

    if filters.since is not None:
        since = filters.since
        result = [m for m in result if _to_utc(m.timestamp) >= since]

    if filters.until is not None:
        until = filters.until
        result = [m for m in result if _to_utc(m.timestamp) <= until]

    return result


def _to_utc(dt: datetime | Any) -> datetime:
    """Coerce a datetime to UTC-aware."""
    if not isinstance(dt, datetime):
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _sort_memories(
    memories: list[MemoryItem],
    sort_by: str,
    sort_order: str,
) -> list[MemoryItem]:
    """Sort memories according to options."""
    reverse = sort_order == "desc"

    def _key_importance(m: MemoryItem) -> float:
        return m.importance

    def _key_type(m: MemoryItem) -> str:
        return m.memory_type.value

    def _key_timestamp(m: MemoryItem) -> datetime:
        return _to_utc(m.timestamp)

    if sort_by == "importance":
        key = _key_importance
    elif sort_by == "memory_type":
        key = _key_type  # type: ignore[assignment]
    else:
        key = _key_timestamp  # type: ignore[assignment]

    return sorted(memories, key=key, reverse=reverse)


def _build_yaml_frontmatter(memory: MemoryItem) -> str:
    """Build YAML front matter block for vault mode."""
    lines = ["---"]
    lines.append(f'memory_id: "{memory.memory_id}"')
    lines.append(f"memory_type: {memory.memory_type.value}")
    lines.append(f"importance: {memory.importance}")

    if memory.tags:
        tag_list = ", ".join(memory.tags)
        lines.append(f"tags: [{tag_list}]")

    if memory.entities:
        entity_parts = ", ".join(
            f"{k}: {v}" for k, v in memory.entities.items()
        )
        lines.append(f"entities: {{{entity_parts}}}")

    if memory.emotions:
        emotion_list = ", ".join(memory.emotions)
        lines.append(f"emotions: [{emotion_list}]")

    lines.append(f"created: {_format_datetime(memory.timestamp)}")

    if memory.last_accessed:
        lines.append(f"last_accessed: {_format_datetime(memory.last_accessed)}")

    lines.append("---")
    return "\n".join(lines)


def _build_wiki_links(
    memory_id: str,
    links: list[MemoryLink],
    id_to_title: dict[str, str],
) -> str:
    """Build wiki-link references for vault mode."""
    if not links:
        return ""
    lines: list[str] = []
    for link in links:
        other_id = link.target_id if link.source_id == memory_id else link.source_id
        other_title = id_to_title.get(other_id, other_id[:8])
        lines.append(f"- **{link.link_type.value}** → [[{other_title}]]")
    if not lines:
        return ""
    return "\n## Links\n" + "\n".join(lines)


def _build_anchor_links(
    memory_id: str,
    links: list[MemoryLink],
    id_to_title: dict[str, str],
) -> str:
    """Build anchor-link references for single-file mode."""
    if not links:
        return ""
    lines: list[str] = []
    for link in links:
        other_id = link.target_id if link.source_id == memory_id else link.source_id
        other_title = id_to_title.get(other_id, other_id[:8])
        anchor = _slug(other_title)
        lines.append(f"- **{link.link_type.value}** → [{other_title}](#{anchor})")
    if not lines:
        return ""
    return "\n### Links\n" + "\n".join(lines)


def _memory_to_section(
    memory: MemoryItem,
    title: str,
    links: list[MemoryLink],
    id_to_title: dict[str, str],
    include_links: bool,
) -> str:
    """Render a single memory as a markdown section (for single-file mode)."""
    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append(
        f"**Type:** {memory.memory_type.value} | "
        f"**Importance:** {memory.importance} | "
        f"**Created:** {_format_datetime(memory.timestamp)}"
    )
    lines.append("")

    if memory.summary:
        lines.append(f"> {memory.summary}")
        lines.append("")

    if memory.content:
        lines.append(memory.content)
        lines.append("")

    if memory.tags:
        tag_str = ", ".join(f"#{tag}" for tag in memory.tags)
        lines.append(f"**Tags:** {tag_str}")

    if memory.entities:
        entity_str = ", ".join(
            f"{k} ({v})" for k, v in memory.entities.items()
        )
        lines.append(f"**Entities:** {entity_str}")

    if memory.emotions:
        lines.append(f"**Emotions:** {', '.join(memory.emotions)}")

    if include_links and links:
        lines.append(_build_anchor_links(memory.memory_id, links, id_to_title))

    return "\n".join(lines)


def _memory_to_file(
    memory: MemoryItem,
    title: str,
    links: list[MemoryLink],
    id_to_title: dict[str, str],
    include_links: bool,
) -> str:
    """Render a single memory as a full vault file."""
    parts: list[str] = []
    parts.append(_build_yaml_frontmatter(memory))
    parts.append("")
    parts.append(f"# {title}")
    parts.append("")

    if memory.summary:
        parts.append(f"> {memory.summary}")
        parts.append("")

    if memory.content:
        parts.append(memory.content)
        parts.append("")

    if memory.tags:
        tag_str = ", ".join(f"#{tag}" for tag in memory.tags)
        parts.append(f"**Tags:** {tag_str}")
        parts.append("")

    if memory.entities:
        entity_str = ", ".join(
            f"{k} ({v})" for k, v in memory.entities.items()
        )
        parts.append(f"**Entities:** {entity_str}")
        parts.append("")

    if include_links and links:
        parts.append(_build_wiki_links(memory.memory_id, links, id_to_title))

    return "\n".join(parts).rstrip() + "\n"


def render_single_file(
    memories: list[MemoryItem],
    links_by_id: dict[str, list[MemoryLink]],
    options: ExportOptions,
) -> str:
    """Render all memories into a single markdown document."""
    now = datetime.now(timezone.utc).isoformat()
    id_to_title: dict[str, str] = {m.memory_id: _memory_title(m) for m in memories}

    lines: list[str] = []
    lines.append("# Mnemotree Memory Export")
    lines.append(f"_Exported: {now} | Memories: {len(memories)}_")
    lines.append("")

    if options.include_toc and memories:
        lines.append("## Table of Contents")
        for memory in memories:
            title = id_to_title[memory.memory_id]
            anchor = _slug(title)
            lines.append(f"- [{title}](#{anchor})")
        lines.append("")
        lines.append("---")
        lines.append("")

    for memory in memories:
        title = id_to_title[memory.memory_id]
        mem_links = links_by_id.get(memory.memory_id, [])
        section = _memory_to_section(
            memory, title, mem_links, id_to_title, options.include_links
        )
        lines.append(section)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_vault(
    memories: list[MemoryItem],
    links_by_id: dict[str, list[MemoryLink]],
    options: ExportOptions,
) -> dict[str, str]:
    """Render memories as a dict of filename → content for vault mode.

    Returns a mapping where keys are relative file paths (e.g. ``Title.md``,
    ``_index.md``) and values are the file content strings.
    """
    id_to_title: dict[str, str] = {m.memory_id: _memory_title(m) for m in memories}
    used_filenames: dict[str, int] = {}
    files: dict[str, str] = {}

    for memory in memories:
        title = id_to_title[memory.memory_id]
        stem = _sanitize_filename(title, memory.memory_id)

        if stem in used_filenames:
            used_filenames[stem] += 1
            stem = f"{stem}-{memory.memory_id[:6]}"
        else:
            used_filenames[stem] = 1

        filename = f"{stem}.md"
        mem_links = links_by_id.get(memory.memory_id, [])
        content = _memory_to_file(
            memory, title, mem_links, id_to_title, options.include_links
        )
        files[filename] = content

    # Generate index file
    now = datetime.now(timezone.utc).isoformat()
    index_lines: list[str] = []
    index_lines.append("# Mnemotree Memory Index")
    index_lines.append(f"_Exported: {now} | Memories: {len(memories)}_")
    index_lines.append("")
    index_lines.append("| Title | Type | Importance | Tags |")
    index_lines.append("|-------|------|------------|------|")
    for memory in memories:
        title = id_to_title[memory.memory_id]
        tags = ", ".join(memory.tags) if memory.tags else ""
        index_lines.append(
            f"| [[{title}]] | {memory.memory_type.value} | "
            f"{memory.importance} | {tags} |"
        )
    index_lines.append("")
    files["_index.md"] = "\n".join(index_lines)

    return files


async def export_memories(
    store: Any,
    options: ExportOptions | None = None,
    filters: ExportFilters | None = None,
) -> tuple[str, int]:
    """Export memories from a store to markdown.

    Args:
        store: A memory store implementing ``SupportsMemoryListing``.
        options: Export options (defaults to single-file mode).
        filters: Optional filters to apply before export.

    Returns:
        Tuple of (output_path, count_of_exported_memories).

    Raises:
        NotImplementedError: If the store doesn't support list_memories.
    """
    from mnemotree.store.protocols import SupportsKnowledgeGraph, SupportsMemoryListing

    if options is None:
        options = ExportOptions()

    if not isinstance(store, SupportsMemoryListing):
        raise NotImplementedError(
            "Export requires a store that supports list_memories()."
        )

    memories = await store.list_memories(include_embeddings=False)
    memories = _filter_memories(memories, filters)
    memories = _sort_memories(memories, options.sort_by, options.sort_order)

    # Fetch links if the store supports knowledge graph operations
    links_by_id: dict[str, list[MemoryLink]] = {}
    if options.include_links and isinstance(store, SupportsKnowledgeGraph):
        for memory in memories:
            try:
                links = await store.get_links(memory.memory_id, direction="both")
                links_by_id[memory.memory_id] = links
            except Exception:
                pass

    if options.mode == "vault":
        output_path = options.output_path or "./memories"
        files = render_vault(memories, links_by_id, options)
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (out_dir / filename).write_text(content, encoding="utf-8")
        return str(out_dir), len(memories)
    else:
        output_path = options.output_path or "memories.md"
        content = render_single_file(memories, links_by_id, options)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return str(out), len(memories)
