from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemotree")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser


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

    # Show help when no command is provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
