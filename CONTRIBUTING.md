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

`tests/test_golden_retrieval.py` compares retrieval over a fixed corpus with the
snapshots in `tests/golden/retrieval/`. A refactor must leave them unchanged. A
deliberate retrieval change regenerates them, and the snapshot diff belongs in the
review:

```bash
MNEMOTREE_UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_golden_retrieval.py
```

## Pull requests

1. Branch off `main`.
2. Add tests for new behavior and bug fixes.
3. Update [README.md](README.md) or [docs/GUIDE.md](docs/GUIDE.md) when behavior a
   user sees changes.
4. Open the pull request against `main` with a clear description, and reference any
   related issues.

## Releasing

1. On `main`, set `version` in `pyproject.toml` and in
   `packages/mnemotree-mcp/pyproject.toml`, including its `mnemotree==` pin, and
   add the entry to [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md).
2. Tag the merged commit and push the tag:
   `git tag -a v1.2.3 -m v1.2.3 && git push origin v1.2.3`.

The [release workflow](.github/workflows/release.yml) checks that the tag is on
`main` and matches the version, runs full CI on the tagged commit, builds the
distributions and smoke-installs them. It then waits for approval in the `pypi`
environment, uploads to PyPI through trusted publishing, and attaches the files
to the GitHub release. To try it on a tag without uploading, run
`gh workflow run release.yml --ref v1.2.3`; add `-f publish=true` to upload.
If the upload fails partway, fix the cause and re-run the failed jobs: files
already on PyPI are skipped. A new package needs a pending publisher on PyPI
that matches the workflow exactly (repository, `release.yml`, environment
`pypi`); otherwise PyPI refuses to create the project.

## Reporting issues

Search existing issues first, then open a new one using the provided templates.
