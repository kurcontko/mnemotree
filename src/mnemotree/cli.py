from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemotree")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse uses SystemExit for parsing errors/help; return its code.
        code = exc.code if isinstance(exc.code, int) else 1
        return code

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
