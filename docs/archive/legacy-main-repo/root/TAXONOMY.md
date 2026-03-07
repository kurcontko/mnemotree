# Memory Type Taxonomy

Mnemotree implements a **research-backed memory taxonomy** inspired by cognitive science, distinguishing between episodic, semantic, and procedural memory systems with **type-specific decay curves** and optimized retrieval.

## 💰 Cost Comparison: Enum vs LLM Classification

**Why this matters:** Most competitors use LLM calls to classify memories. We use cognitive science + enums.

| Approach | Per Memory | Per 1K Memories | Per 1M Memories |
|----------|----------:|----------------:|----------------:|
| **LLM-Based (Competitors)** | $0.002 | ~$2-5 | ~$2,000-5,000 |
| **Enum-Based (Mnemotree)** | **$0.00** | **$0** | **$0** |

**Savings at scale:**
- 10K memories/day → Save ~$20-50/day = **$7,300-18,250/year**
- 100K memories/day → Save ~$200-500/day = **$73,000-182,500/year**

**How we do it:** User or application explicitly chooses the memory type at write-time (episodic, semantic, procedural). No inference needed.

---

## Overview

| Memory Type | Category | Stability | Decay Power | Use Case |
|------------|----------|-----------|-------------|----------|
| **Episodic** | Declarative | 7 days | 0.5 | Personal experiences, events |
| **Semantic** | Declarative | 30 days | 0.4 | Facts, general knowledge |
| **Procedural** | Non-declarative | 60 days | 0.35 | Skills, habits, how-to knowledge |
| **Autobiographical** | Declarative | 14 days | 0.45 | Life story, personal identity |
| **Prospective** | Declarative | 3 days | 0.6 | Future intentions, reminders |
| **Working** | Working | 1 hour | 0.7 | Short-term processing |

## Type Descriptions

### Declarative Memory (Explicit)

#### Episodic Memory
**Definition:** Personal experiences tied to specific times and places.

**Examples:**
- "I met Sarah at the coffee shop yesterday"
- "The team demo went well last Tuesday"
- "I had sushi for lunch at 12:30pm"

**Characteristics:**
- Context-dependent (who, what, when, where)
- Decays faster than semantic memory
- Enhanced by emotional content
- Strengthens with rehearsal

**Code:**
```python
from mnemotree.core.models import MemoryType

memory = MemoryItem(
    content="Met with the client at their office to discuss the Q1 roadmap",
    memory_type=MemoryType.EPISODIC,
    importance=0.8,
    timestamp=datetime.now(),
    emotional_valence=0.6,  # Positive experience
)
```

#### Semantic Memory
**Definition:** General facts and knowledge independent of personal experience.

**Examples:**
- "Python uses duck typing"
- "The capital of France is Paris"
- "REST APIs use HTTP methods like GET and POST"

**Characteristics:**
- Context-independent
- Slower decay than episodic
- High stability (30 days default)
- Reinforced through repeated exposure

**Code:**
```python
memory = MemoryItem(
    content="FastAPI uses Pydantic for request/response validation",
    memory_type=MemoryType.SEMANTIC,
    importance=0.9,
    tags=["python", "fastapi", "api"],
)
```

#### Autobiographical Memory
**Definition:** Personal life story and identity-defining experiences.

**Examples:**
- "I graduated from Stanford in 2018"
- "I prefer vim keybindings in my editor"
- "I'm working on the authentication microservice"

**Characteristics:**
- Combination of episodic and semantic
- Moderately stable (14 days)
- Often emotionally charged
- Central to identity

### Non-Declarative Memory (Implicit)

#### Procedural Memory
**Definition:** Skills, habits, and learned procedures.

**Examples:**
- "How to authenticate users with JWT"
- "Steps to deploy using Docker"
- "Git workflow: branch → commit → PR → merge"

**Characteristics:**
- Very stable (60 days)
- Slow decay (motor memory persists)
- Strengthens with practice
- Hard to verbalize but easy to execute

**Code:**
```python
memory = MemoryItem(
    content="To deploy: 1) Run tests 2) Build Docker image 3) Push to registry 4) Update k8s manifest",
    memory_type=MemoryType.PROCEDURAL,
    importance=0.95,
    tags=["deployment", "workflow", "docker"],
)
```

#### Priming & Conditioning
**Definition:** Implicit influence of prior exposure on behavior.

**Examples:**
- Recent code patterns influencing current implementation
- Associations between similar bugs
- Learned responses to error patterns

## Querying by Type

### Using Filters (Built-in)
```python
from mnemotree.core.memory import RecallFilters
from mnemotree.core.models import MemoryType

# Get all episodic memories about meetings
memories = await memory_core.recall(
    "client meetings",
    filters=RecallFilters(
        memory_types=[MemoryType.EPISODIC, MemoryType.AUTOBIOGRAPHICAL]
    )
)
```

### Using Taxonomy Helpers (Convenience)
```python
from mnemotree.core.taxonomy import (
    get_episodic_memories,
    get_semantic_memories,
    get_procedural_memories,
)

# Personal experiences
episodes = await get_episodic_memories(memory_core, "yesterday's standup")

# Factual knowledge
facts = await get_semantic_memories(memory_core, "Python async patterns")

# How-to knowledge
procedures = await get_procedural_memories(memory_core, "deployment steps")
```

### Using the Mixin (Advanced)
```python
from mnemotree.core.taxonomy import TaxonomyQueryMixin

# Monkey-patch MemoryCore (optional)
class EnhancedMemoryCore(MemoryCore, TaxonomyQueryMixin):
    pass

memory_core = EnhancedMemoryCore(...)
episodes = await memory_core.recall_episodic("team retrospective")
```

## Type-Specific Decay

Each memory type has **optimized decay parameters** based on cognitive science research (FSRS-4.5):

```python
from mnemotree.core.decay import MEMORY_TYPE_DEFAULTS

# View default decay configs
for mem_type, config in MEMORY_TYPE_DEFAULTS.items():
    print(f"{mem_type.value}: stability={config.stability_seconds/86400:.0f}d, power={config.decay_power}")
```

**Output:**
```
semantic: stability=30d, power=0.4
episodic: stability=7d, power=0.5
working: stability=0d, power=0.7
procedural: stability=60d, power=0.35
autobiographical: stability=14d, power=0.45
prospective: stability=3d, power=0.6
```

### Customizing Per-Type Decay
```python
# Enable per-type decay (uses MEMORY_TYPE_DEFAULTS)
memory_core = MemoryCoreBuilder(store) \
    .with_decay(per_type_decay=True) \
    .build()

# Or override defaults
from mnemotree.core.decay import MEMORY_TYPE_DEFAULTS, DecayConfig
from mnemotree.core.models import MemoryType

MEMORY_TYPE_DEFAULTS[MemoryType.SEMANTIC] = DecayConfig(
    stability_seconds=60 * 86400,  # 60 days for your knowledge base
    decay_power=0.3,
)
```

## Classification Helpers

```python
from mnemotree.core.models import MemoryType
from mnemotree.core.taxonomy import is_episodic, is_semantic, is_procedural

# Check memory type category
assert is_episodic(MemoryType.EPISODIC)
assert is_semantic(MemoryType.SEMANTIC)
assert is_procedural(MemoryType.PROCEDURAL)

# Using the enum directly
memory = MemoryItem(content="...", memory_type=MemoryType.EPISODIC)
assert memory.memory_type.is_episodic
assert memory.memory_type.category == "declarative"
```

## Best Practices

### 1. Choose the Right Type
- **Episodic**: Events with context (who, when, where)
- **Semantic**: Decontextualized facts
- **Procedural**: Multi-step processes, workflows

### 2. Leverage Type-Specific Decay
- Enable `per_type_decay=True` for automatic optimization
- Procedural memories last longer (skills persist)
- Working memory decays fastest (short-term cache)

### 3. Query by Category
- Use filters to separate "what happened" vs "what I know"
- Episodic for timelines, semantic for knowledge graphs

### 4. Combine Types
- Autobiographical = episodic + semantic (personal facts)
- Complex memories may span multiple types

## Competitive Advantages

| Feature | Mnemotree | Competitors |
|---------|-----------|-------------|
| **Explicit Taxonomy** | ✅ 6 types + categories | ⚠️ Generic tags |
| **Type-Specific Decay** | ✅ Research-backed FSRS | ❌ One-size-fits-all |
| **Zero-Cost Classification** | ✅ Enum-based | ⚠️ LLM-based (expensive) |
| **Query Filtering** | ✅ Built-in + helpers | ⚠️ Manual post-processing |
| **Cognitive Science Basis** | ✅ Declarative/Non-declarative | ❌ Ad-hoc categories |

## References

- **FSRS-4.5**: Forgetting curve model used for decay
- **Tulving, E. (1972)**: Episodic vs semantic memory distinction
- **Squire, L. R. (2004)**: Declarative/non-declarative taxonomy
- **Baddeley, A. (2000)**: Working memory model

---

**Next Steps:**
1. Read the [Quick Start](README.md#quick-start) to create memories
2. See [examples/taxonomy_demo.py](examples/taxonomy_demo.py) for code samples
3. Explore [Decay System](docs/decay.md) for advanced tuning
