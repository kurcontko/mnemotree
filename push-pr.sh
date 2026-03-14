#!/bin/bash
# Run after: gh auth login
set -e
git push -u origin fix/ci-cleanup
gh pr create --base main --title "v0.4.0: remove langchain from core, make spacy optional, CI overhaul" --body "$(cat <<'PREOF'
## Summary

See [CHANGELOG.md](CHANGELOG.md) for full details.

- **Remove all langchain from core + experimental** — adapters-only now
- **Make spacy optional** — base install ~65MB (was ~265MB)
- Fix CI: mypy errors, broken tests, stale patches
- Split CI into lint + test jobs, add Python 3.13
- Consolidate duplicate `_protocols.py`, add `py.typed`
- Add smoke tests, CHANGELOG, README updates
- Bump version to 0.4.0

## Breaking changes
- `spacy` moved to `ner_spacy` extra — install with `pip install mnemotree[ner_spacy]`
- `_protocols.py` removed — use `mnemotree.core.protocols` instead

## Test plan
- [x] 897 tests pass, 16 skipped (optional deps)
- [x] mypy clean (113 source files)
- [x] ruff lint + format clean
- [x] All non-optional modules import without spacy

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PREOF
)"
echo "PR created!"
