# Changelog

## 0.4.0

### Breaking Changes
- **spacy is no longer a base dependency.** Install with `pip install mnemotree[ner_spacy]` if you use SpacyNER or SpacyKeywordExtractor. The library gracefully disables NER when spacy is not installed.
- `_protocols.py` removed — import `EmbeddingModel` and `LLMBackend` from `mnemotree.core.protocols` instead.

### Changed
- Remove all langchain imports from core and experimental paths. LangChain is now only used in optional adapters/integrations (`mnemotree[integrations]`).
- Replace `langchain.prompts.PromptTemplate` + LCEL chains in consolidation and truth_maintenance with direct `LLMBackend.ainvoke()` calls.
- Replace `langchain_openai` fallbacks in `MemoryCore._resolve_embeddings` and `_resolve_analyzer_and_summarizer` with PydanticAI model strings.
- Base install reduced from ~265MB to ~65MB (pydantic, numpy, scikit-learn, networkx, sqlite-vec, urllib3, filelock).

### Added
- `py.typed` marker for PEP 561 compliance.
- `ner_spacy` optional extra in pyproject.toml.
- Smoke tests for public API imports.
- Python 3.13 in CI matrix.
- CI split into lint and test jobs; spacy tested on 3.12 only.
- `anyio` added to dev dependencies (required by pytest-asyncio).

### Fixed
- mypy errors: unused type-ignore comments, union-attr on Optional embeddings.
- Test failures from stale langchain_openai patches in test_configs.py.
- `test_multihop.py` now checks sqlite-vec extension loadability (not just importability).
- `NERResult` imported from `ner.base` instead of `ner.spacy` in enrichment pipeline.
- `.env.sample` uses `OPENAI_BASE_URL` (not deprecated `OPENAI_API_BASE`).
- `filterwarnings` in pyproject.toml to avoid CI breakage from dependency deprecation warnings.

## 0.3.0

Initial tracked release.
