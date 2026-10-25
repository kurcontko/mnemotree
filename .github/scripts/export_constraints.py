"""Export uv.lock's full resolution as a pip constraints file.

mnemotree's pins live in uv.lock, which pip cannot read. A release ships this
export next to its wheel so `pip install mnemotree==X -c constraints.txt`
reproduces exactly the resolution that release was tested against.

It is a release asset rather than a committed file on purpose: Dependabot
rewrites uv.lock but would not regenerate a committed copy, so every bump
would fail a drift check that protects nothing between releases.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

HEADER = "# Generated from uv.lock by .github/scripts/export_constraints.py; do not edit.\n"


def export() -> str:
    exported = subprocess.check_output(
        [
            "uv",
            "export",
            "--frozen",
            "--all-extras",
            "--all-groups",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
        ],
        text=True,
    )
    return HEADER + exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("constraints.txt"))
    parser.add_argument(
        "--check", action="store_true", help="fail if --output differs from uv.lock"
    )
    args = parser.parse_args()
    # Only ever read or write inside the checkout this runs from.
    output = args.output.resolve()
    if not output.is_relative_to(Path.cwd().resolve()):
        raise SystemExit(f"--output must be inside {Path.cwd()}, got {args.output}")
    content = export()
    if args.check:
        if not output.is_file() or output.read_text() != content:
            raise SystemExit(
                f"{args.output} drifted from uv.lock; run python .github/scripts/export_constraints.py"
            )
    else:
        output.write_text(content)


if __name__ == "__main__":
    main()
