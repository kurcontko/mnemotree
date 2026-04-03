# API Stability

This file defines the practical public surface for the copied `mnemotree` repo in this workspace.

## Stable public imports

Prefer importing from the package root:

```python
from mnemotree import (
    MemoryCore,
    MemoryCoreBuilder,
    MemoryMode,
    RememberOptions,
    RecallOptions,
    RecallFilters,
    LinkType,
    MemoryLink,
)
```

The root package in [`src/mnemotree/__init__.py`](../src/mnemotree/__init__.py) is the default compatibility contract.

## Stable package-level modules

These are reasonable public entrypoints for application code:

- `mnemotree`
- `mnemotree.store`
- `mnemotree.cli`
- `mnemotree.mcp.server`

Concrete stores exposed from `mnemotree.store` are part of the intended public surface:

- `ChromaMemoryStore`
- `Neo4jMemoryStore`
- `SQLiteVecMemoryStore`
- `MilvusMemoryStore`

## Use with caution

These modules are useful, but should be treated as advanced or semi-stable:

- `mnemotree.core`
- `mnemotree.configs`
- `mnemotree.integrations`
- `mnemotree.tools`
- `mnemotree.ner`
- `mnemotree.normalization`

They are acceptable for internal apps and benchmarks, but are a weaker compatibility contract than the package root.

## Experimental or internal

These should not be treated as stable application-facing APIs:

- `mnemotree.experimental`
- `mnemotree.analysis`
- `mnemotree.inference`
- `mnemotree.core._internal`
- direct imports from backend-specific implementation modules

Compatibility shims in `mnemotree.experimental` exist to keep older imports working, not to define the preferred long-term surface.

## Benchmark guidance

Benchmarks should depend on:

- package-root imports where possible
- explicit builder/config options
- documented store constructors

Benchmarks should avoid depending on:

- private `_internal` modules
- undocumented dataclass fields
- ad hoc imports from old repo copies
