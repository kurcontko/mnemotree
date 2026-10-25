"""Prepare reviewed release files; no network, git mutation, or publishing."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import tomllib
from pathlib import Path

VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:(a|b|rc)(0|[1-9][0-9]*))?"
PACKAGE = "mnemotree"
REPO = "https://github.com/kurcontko/mnemotree"
EMPTY_SECTIONS = "### Added\n\n### Changed\n\n### Fixed\n\n### Removed\n"


def version_key(value: str) -> tuple[int, ...]:
    match = re.fullmatch(VERSION, value)
    if not match:
        raise ValueError("Expected MAJOR.MINOR.PATCH optionally followed by aN, bN, or rcN")
    major, minor, patch, stage, number = match.groups()
    stage_rank = {"a": 0, "b": 1, "rc": 2, None: 3}[stage]
    return (int(major), int(minor), int(patch), stage_rank, int(number or 0))


def prepare(root: Path, version: str, date: str) -> dict[Path, str]:
    """Validate every edit before returning the complete replacement set."""
    version_key(version)
    dt.date.fromisoformat(date)
    paths = [root / p for p in ("pyproject.toml", "uv.lock", "CHANGELOG.md")]
    project, lock, changelog = (p.read_text() for p in paths)
    old = tomllib.loads(project)["project"]["version"]
    if version_key(version) <= version_key(old):
        raise ValueError("Release version must be newer than the current version")
    packages = [p for p in tomllib.loads(lock)["package"] if p["name"] == PACKAGE]
    if len(packages) != 1 or packages[0]["version"] != old:
        raise ValueError("Lockfile version disagrees with project metadata")
    if changelog.count("## [Unreleased]") != 1 or f"## [{version}]" in changelog:
        raise ValueError("Changelog needs one Unreleased section and no duplicate release")
    notes = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    notes = notes.split("\n[Unreleased]:", 1)[0]
    if not re.search(r"(?m)^- \S", notes):
        raise ValueError("Add user-facing Unreleased notes before preparing a release")
    project, count = re.subn(
        r'(?m)^version = "' + re.escape(old) + r'"$', f'version = "{version}"', project
    )
    if count != 1:
        raise ValueError("Expected one project version declaration")
    lock, count = re.subn(
        r'(name = "' + PACKAGE + r'"\nversion = ")' + re.escape(old) + '"',
        lambda m: m[1] + version + '"',
        lock,
    )
    if count != 1:
        raise ValueError("Expected one local package version in uv.lock")
    # Empty headings left in the released section are noise; drop them.
    released = re.sub(r"(?m)^### \w+\n+(?=### |\Z)", "", notes.strip() + "\n").strip()
    changelog = changelog.replace(
        "## [Unreleased]" + notes,
        f"## [Unreleased]\n\n{EMPTY_SECTIONS}\n## [{version}] — {date}\n\n{released}\n",
        1,
    )
    changelog = re.sub(
        r"(?m)^\[Unreleased\]:.*$", f"[Unreleased]: {REPO}/compare/v{version}...HEAD", changelog
    )
    if "[Unreleased]:" not in changelog:
        changelog = changelog.rstrip("\n") + f"\n\n[Unreleased]: {REPO}/compare/v{version}...HEAD\n"
    changelog += f"[{version}]: {REPO}/releases/tag/v{version}\n"
    return dict(zip(paths, (project, lock, changelog), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--date", default=dt.datetime.now(dt.UTC).date().isoformat())
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    changes = prepare(args.root, args.version, args.date)
    # The caller uses a disposable clean branch. Validation errors never write;
    # filesystem failures propagate and the workflow discards that branch.
    for path, content in changes.items():
        path.write_text(content)


if __name__ == "__main__":
    main()
