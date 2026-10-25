#!/usr/bin/env python3
"""Fail when a workflow file contains a tab character.

A tab used as YAML indentation is a parse error, and GitHub reports it only
once the workflow is triggered — after the push that broke it. A tab *inside*
a `run:` block scalar is worse: it is valid YAML, so actionlint has nothing to
say, and it silently breaks the things shell scripts use tabs for (`<<-EOF`
heredoc stripping, `read` field splitting). Neither failure is worth
diagnosing twice, so no workflow file gets to contain one.
"""

from __future__ import annotations

import pathlib
import sys

ROOTS = (pathlib.Path(".github/workflows"), pathlib.Path(".github/actions"))


def main() -> int:
    bad: list[str] = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for suffix in ("*.yml", "*.yaml"):
            for path in sorted(root.rglob(suffix)):
                if b"\t" in path.read_bytes():
                    bad.append(str(path))

    if bad:
        print("Tab characters found in workflow file(s):")
        for filename in sorted(bad):
            print(f"- {filename}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
