from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemotree")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export", help="Export memories to markdown")
    export_parser.add_argument(
        "--mode",
        choices=["single", "vault"],
        default="single",
        help="Export mode: single file or vault directory (default: single)",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: memories.md or ./memories/)",
    )
    export_parser.add_argument(
        "--db",
        default=".mnemotree/mnemotree.sqlite",
        help="Path to SQLite database (default: .mnemotree/mnemotree.sqlite)",
    )
    export_parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Filter by tags",
    )
    export_parser.add_argument(
        "--memory-types",
        nargs="*",
        default=None,
        help="Filter by memory types (e.g. semantic episodic procedural)",
    )
    export_parser.add_argument(
        "--min-importance",
        type=float,
        default=None,
        help="Minimum importance threshold",
    )
    export_parser.add_argument(
        "--max-importance",
        type=float,
        default=None,
        help="Maximum importance threshold",
    )
    export_parser.add_argument(
        "--since",
        default=None,
        help="Only include memories after this ISO-8601 timestamp",
    )
    export_parser.add_argument(
        "--until",
        default=None,
        help="Only include memories before this ISO-8601 timestamp",
    )
    export_parser.add_argument(
        "--sort-by",
        choices=["timestamp", "importance", "memory_type"],
        default="timestamp",
        help="Sort field (default: timestamp)",
    )
    export_parser.add_argument(
        "--sort-order",
        choices=["asc", "desc"],
        default="desc",
        help="Sort order (default: desc)",
    )
    export_parser.add_argument(
        "--no-links",
        action="store_true",
        help="Exclude memory links from output",
    )
    export_parser.add_argument(
        "--no-toc",
        action="store_true",
        help="Exclude table of contents (single-file mode)",
    )

    return parser


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_memory_types(values: list[str] | None) -> list | None:
    if not values:
        return None
    from mnemotree.core.models import MemoryType

    result = []
    for v in values:
        normalized = v.strip().lower()
        for mt in MemoryType:
            if mt.value == normalized or mt.name.lower() == normalized:
                result.append(mt)
                break
    return result or None


async def _run_export(args: argparse.Namespace) -> int:
    from mnemotree.export.markdown import ExportFilters, ExportOptions, export_memories
    from mnemotree.store.sqlite_vec_store import SQLiteVecMemoryStore

    store = SQLiteVecMemoryStore(db_path=args.db)
    await store.initialize()

    try:
        options = ExportOptions(
            mode=args.mode,
            output_path=args.output,
            include_links=not args.no_links,
            include_toc=not args.no_toc,
            sort_by=args.sort_by,
            sort_order=args.sort_order,
        )
        filters = ExportFilters(
            memory_types=_parse_memory_types(args.memory_types),
            tags=args.tags if args.tags else None,
            min_importance=args.min_importance,
            max_importance=args.max_importance,
            since=_parse_iso_datetime(args.since),
            until=_parse_iso_datetime(args.until),
        )
        path, count = await export_memories(store, options, filters)
        print(f"Exported {count} memories to {path}")
        return 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        try:
            from importlib.metadata import PackageNotFoundError, version

            print(version("mnemotree"))
            return 0
        except PackageNotFoundError:
            print("mnemotree")
            return 0

    if args.command == "export":
        return asyncio.run(_run_export(args))

    # Show help when no command is provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
