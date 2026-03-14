#!/bin/bash
# Run after: gh auth login
set -e
git push -u origin fix/ci-cleanup
gh pr create --base main --title "v0.4.0: CI cleanup, remove langchain from core, make spacy optional" --body "$(cat <<'PREOF'
## Summary
- Fix CI: mypy errors, broken tests, missing `anyio` dep
- Apply `ruff format` to 34 files
- **Remove all langchain imports from core and experimental** — replaced with direct `LLMBackend.ainvoke()` and PydanticAI model strings
- **Make spacy optional** — moved from base deps to `ner_spacy` extra, base install drops ~200MB
- Harden CI: split lint/test jobs, Python 3.13, spacy tested on 3.12 only
- Consolidate duplicate `_protocols.py` into `core/protocols.py`
- Update README: hybrid retrieval, SQLite-vec as default store
- Add `py.typed` PEP 561 marker, `utils/__init__.py`
- Bump version to 0.4.0

## Base deps (now)
pydantic, numpy, scikit-learn, networkx, urllib3, filelock, sqlite-vec (~65MB)

Previously included spacy (~200MB+). Now install with `pip install mnemotree[ner_spacy]`.

## Test plan
- [x] 890 tests pass, 16 skipped (optional deps)
- [x] mypy clean (113 source files)
- [x] ruff lint + format clean
- [x] All non-optional modules import successfully

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PREOF
)"
echo "PR created!"
