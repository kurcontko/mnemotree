from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemotree")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.add_parser(
        "mcp",
        add_help=False,
        help="run the MCP stdio server (mnemotree mcp --help lists its options)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # Dispatched before parsing so every server flag, --help included, reaches the
    # server's own parser.
    if argv[:1] == ["mcp"]:
        from ..mcp.server import main as mcp

        # No default embedder: the library does not install a local model, so
        # keyword-only retrieval and an embedding config stay explicit choices.
        return mcp(argv[1:], prog="mnemotree mcp")

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

    # Show help when no command is provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
