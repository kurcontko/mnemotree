# Implementing Memory Taxonomy Without Cost

## Problem
Competitors emphasize explicit episodic/semantic/procedural taxonomy in 2026, but implementing this usually requires expensive LLM-based classification.

## Solution: Zero-Cost Taxonomy Layer

We leveraged **existing infrastructure** to create a competitive taxonomy system without adding any LLM/API costs:

### What Already Existed ✅
1. **MemoryType enum** with all major types (episodic, semantic, procedural, etc.)
2. **Type-specific decay curves** in `MEMORY_TYPE_DEFAULTS`
3. **RecallFilters** with `memory_types` parameter
4. **category property** on MemoryType enum

### What We Added (Zero Cost) 🆕

#### 1. **Taxonomy Module** (`src/mnemotree/core/taxonomy.py`)
- **Type groups**: `EPISODIC_TYPES`, `SEMANTIC_TYPES`, `PROCEDURAL_TYPES`
- **Helper functions**: `is_episodic()`, `is_semantic()`, `is_procedural()`
- **Convenience queries**:
  - `get_episodic_memories(memory, query)`
  - `get_semantic_memories(memory, query)`
  - `get_procedural_memories(memory, query)`
- **Optional mixin**: `TaxonomyQueryMixin` for advanced users

#### 2. **Enhanced MemoryType Properties** (`src/mnemotree/core/models.py`)
Added convenience properties:
```python
memory.memory_type.is_episodic   # → bool
memory.memory_type.is_semantic   # → bool
memory.memory_type.is_procedural # → bool
memory.memory_type.category      # → "declarative" | "non_declarative" | "working"
```

#### 3. **Documentation**
- **TAXONOMY.md**: Comprehensive guide with:
  - Cognitive science background
  - Type descriptions with examples
  - Query patterns
  - Decay characteristics
  - Competitive comparison table
- **examples/taxonomy_demo.py**: Working demo with no LLM calls

#### 4. **Public API Exports** (`src/mnemotree/core/__init__.py`)
All taxonomy helpers now accessible via:
```python
from mnemotree.core import (
    get_episodic_memories,
    get_semantic_memories,
    get_procedural_memories,
    is_episodic,
    EPISODIC_TYPES,
    # etc.
)
```

## Cost Analysis

| Feature | Mnemotree | Competitor Approach | Cost Difference |
|---------|-----------|---------------------|-----------------|
| **Type Classification** | Enum-based (free) | LLM-based ($0.002/memory) | **$0 vs $2/1000** |
| **Type-Specific Decay** | Configured defaults (free) | One-size-fits-all | **$0** |
| **Query Filtering** | Native RecallFilters (free) | Post-LLM filtering | **$0** |
| **Category Grouping** | Enum property (free) | LLM analysis | **$0** |

**Total Savings:** ~$2-5 per 1,000 memories

## Usage Examples

### Basic (Built-in Filters)
```python
from mnemotree.core import RecallFilters, MemoryType

# Get episodic memories
memories = await memory.recall(
    "yesterday's meeting",
    filters=RecallFilters(memory_types=[MemoryType.EPISODIC])
)
```

### Convenience Helpers
```python
from mnemotree.core import get_episodic_memories

# Simpler API
episodes = await get_episodic_memories(memory, "yesterday's meeting")
```

### Type Creation
```python
from mnemotree.core import MemoryType

# User explicitly chooses type (no LLM needed)
await memory.remember(
    "FastAPI uses Pydantic for validation",
    memory_type=MemoryType.SEMANTIC,
    importance=0.9,
)
```

### Type Checking
```python
# Runtime type checks
if memory_item.memory_type.is_semantic:
    # Handle factual knowledge differently
    pass
```

## Marketing Points

1. **"Research-Backed Taxonomy"**
   - Episodic/Semantic/Procedural distinction from cognitive science
   - FSRS-4.5 forgetting curves per type
   - Declarative vs non-declarative categories

2. **"Zero-Cost Classification"**
   - No LLM calls required
   - Enum-based type system
   - User or app chooses type explicitly

3. **"Type-Specific Decay"**
   - Procedural memories (skills) last 60 days
   - Semantic memories (facts) last 30 days
   - Episodic memories (events) last 7 days
   - Optimized for human forgetting curves

4. **"Native Query Support"**
   - Filter by type in all queries
   - Category-based retrieval
   - Composable with other filters

## Next Steps (Optional Enhancements)

### Low-Cost Ideas
1. **Type Inference Hints**: Heuristic-based suggestions (keywords, tense, etc.)
   - "contains step-by-step" → suggest PROCEDURAL
   - "I did X at Y time" → suggest EPISODIC
   - "X is Y" → suggest SEMANTIC
   - **Cost**: Zero (rule-based)

2. **Pre-configured Builders**:
   ```python
   memory = MemoryCoreBuilder.for_personal_assistant(store)  # Heavy on episodic
   memory = MemoryCoreBuilder.for_knowledge_base(store)      # Heavy on semantic
   ```
   - **Cost**: Zero (just builder presets)

3. **Type Statistics Dashboard**:
   - Show distribution by type/category
   - Decay visualizations per type
   - **Cost**: Zero (query existing data)

### Medium-Cost Ideas (Optional LLM)
4. **Auto-Classification (Opt-In)**:
   - For users who want convenience over cost
   - Fall back to heuristics if disabled
   - **Cost**: $0.001-0.002 per memory (optional)

## File Changes Summary

### Created
- `src/mnemotree/core/taxonomy.py` (185 lines)
- `TAXONOMY.md` (330 lines)
- `examples/taxonomy_demo.py` (185 lines)

### Modified
- `src/mnemotree/core/models.py` (+15 lines - added properties)
- `src/mnemotree/core/__init__.py` (+13 lines - exports)

**Total additions:** ~728 lines of code and docs
**External dependencies:** 0 new dependencies
**API costs:** $0 per memory

## Testing

Run the demo:
```bash
cd /Users/qrc/repos/mnemotree
python examples/taxonomy_demo.py
```

Expected output:
- Creates 9 memories (3 episodic, 3 semantic, 3 procedural)
- Demonstrates type-specific queries
- Shows decay characteristic differences
- Zero LLM API calls

## Conclusion

By **surfacing existing capabilities** and adding a **thin convenience layer**, we've created a competitive taxonomy system without adding costs. The classification happens at write-time (user/app chooses type), not read-time (expensive LLM inference), making it scalable and cost-effective.

**Competitive advantage**: Same features, zero marginal cost.
