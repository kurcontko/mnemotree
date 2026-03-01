# Mnemotree — Gemini CI Context

Mnemotree is a Python 3.10+ library providing biologically-inspired persistent memory for LLM agents.
It ships a `MemoryCore` API and an MCP server (`mnemotree-mcp`).

## Key Areas

- **Core API** — `src/mnemotree/core/` (`MemoryCore`, models, retrieval, scoring, decay)
- **Storage backends** — `src/mnemotree/store/` (ChromaDB, SQLite+sqlite-vec, Neo4j, Milvus)
- **MCP server** — `src/mnemotree/mcp/server.py`
- **CLI** — `src/mnemotree/cli.py`

## PR Review Guidelines

When reviewing pull requests, focus on:

1. **API stability** — Public surface lives in `MemoryCore`. Flag any breaking changes to
   `store()`, `recall()`, `analyze()`, or public model fields without a deprecation path.

2. **Backend abstraction** — Storage backends must implement the abstract base; check that new
   features are either backend-agnostic or gated behind capability checks.

3. **Memory correctness** — Verify that `MemoryItem` fields (`importance`, `stability_seconds`,
   `conflicts_with`, `valid_from`, `valid_until`) are handled consistently across all backends.

4. **Embedding safety** — Embedding calls must be nullable/optional; the library must work without
   an embedding model (lite tier).

5. **Decay / FSRS logic** — Changes to `decay.py` should preserve the power-law curve.
   `stability_seconds` must never be set to zero or negative.

6. **Type safety** — The project runs `mypy`; flag any `Any` escapes or missing annotations in
   public interfaces.

7. **Test coverage** — New features need unit tests under `tests/`. Backend-specific tests should
   be marked with the appropriate pytest fixture/marker.

8. **Security** — No credentials in code, no shell injection in CLI argument handling.

## Issue Triage Labels

Apply the following labels when triaging new issues:

| Label | When to apply |
|---|---|
| `bug` | Confirmed incorrect behaviour |
| `enhancement` | New feature or improvement request |
| `backend:sqlite` / `backend:chroma` / `backend:neo4j` / `backend:milvus` | Backend-specific issue |
| `mcp` | MCP server related |
| `api` | Public API (`MemoryCore`) related |
| `decay` | Forgetting / decay / FSRS logic |
| `docs` | Documentation only |
| `good first issue` | Self-contained, well-scoped, beginner-friendly |
| `needs-reproduction` | Missing steps to reproduce |

## Code Style

- Python 3.10+ syntax; use `match`/`case` where it improves clarity.
- Ruff for lint + format; Mypy strict on public interfaces.
- Prefer dataclasses / Pydantic models over bare dicts for structured data.
- Async-first for IO-bound operations; sync wrappers are acceptable in the CLI layer.
