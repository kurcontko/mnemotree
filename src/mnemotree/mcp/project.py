"""One namespace per project directory, for clients that work in a repository."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def project_root(start: Path) -> Path:
    """The repository that ``start`` is in, or ``start`` itself outside one.

    A linked Git worktree resolves to its main working tree, so every worktree of a
    repository shares one memory.
    """
    start = start.resolve()
    for directory in (start, *start.parents):
        marker = directory / ".git"
        if marker.is_dir():
            return directory
        if marker.is_file():
            return _main_worktree(directory, marker)
    return start


def _main_worktree(directory: Path, marker: Path) -> Path:
    try:
        pointer = marker.read_text().strip()
        if not pointer.startswith("gitdir:"):
            return directory
        git_dir = (directory / pointer.removeprefix("gitdir:").strip()).resolve()
        # Only a linked worktree has a commondir file; a submodule is its own project.
        common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
    except OSError:
        return directory
    return common.parent if common.name == ".git" else directory


def project_namespace(start: Path | None = None) -> str:
    """A stable namespace for the project at ``start``.

    Without ``start`` it is the directory the client names in ``CLAUDE_PROJECT_DIR``,
    else the working directory. The name is the directory's, with a hash of its full
    path so two checkouts called ``api`` stay apart.
    """
    root = project_root(start or Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path.cwd()))
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    return f"{root.name or 'project'}-{suffix}"
