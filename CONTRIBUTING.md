# Contributing to mnemotree

Thank you for your interest in contributing to mnemotree!

## Development environment

```bash
git clone https://github.com/kurcontko/mnemotree.git
cd mnemotree
uv sync --extra dev --extra serve
make precommit-install
```

Without uv, create `.venv` and run `pip install -e ".[dev,serve]"`; the Makefile uses
the tools in `.venv`.

## Checks

Run these before opening a pull request; CI runs the same checks.

```bash
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy
make test        # pytest with coverage
make build       # wheel and source distribution
```

Tests exercise real FTS5 and sqlite-vec and make no paid model calls. Keep it that
way: use deterministic embedders and stubs for providers.

## Pull requests

1. Branch off `main`.
2. Add tests for new behavior and bug fixes.
3. Update [README.md](README.md) or [docs/GUIDE.md](docs/GUIDE.md) when behavior a
   user sees changes.
4. Open the pull request against `main` with a clear description, and reference any
   related issues.

## Reporting issues

Search existing issues first, then open a new one using the provided templates.
