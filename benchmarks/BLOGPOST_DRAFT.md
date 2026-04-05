# When Is Graph-Based Memory Worth It? An Honest Multi-Benchmark Analysis

*We built a graph-augmented memory system and benchmarked it against flat markdown files across 4 industry benchmarks. The results surprised us.*

---

## TL;DR

We compared three memory architectures across four benchmarks:
- **Mnemotree**: Semantic embeddings + NER entity graph + hybrid BM25/vector retrieval
- **Obsidian-style**: TF-IDF + wiki-link entity graph boost (novel baseline)
- **Markdown TF-IDF**: Plain keyword matching (simulating Claude Code's .md memory)

| Benchmark | Cases | Mnemotree | Obsidian | Markdown | Winner |
|---|---|---|---|---|---|
| LoCoMo (conversational) | 20 | **67.5%** | 65.0% | 57.5% | Mnemotree |
| LongMemEval (long-term) | 30 | 75.0% | 78.3% | **85.0%** | Markdown |
| MAB-CR (conflict resolution) | 80 | 11.9% | 11.9% | **13.8%** | Markdown |
| PersonaMem (preferences) | 6 | **75.0%** | 66.7% | 75.0% | Mnemotree/Markdown |

**The uncomfortable finding**: simple TF-IDF keyword matching is competitive or better on most tasks. Graph-based retrieval only provides clear value for temporal reasoning and multi-hop queries over long conversation histories.

---

## Motivation

Every memory system for LLM agents claims to be better than "just appending to a file." But better at what? And at what cost?

We built Mnemotree --- a graph-augmented memory system with semantic embeddings, named entity recognition, BM25+vector hybrid retrieval, and a knowledge graph with entity co-occurrence links. It's the kind of architecture that *should* crush a flat text file.

To test this honestly, we needed:
1. Multiple benchmarks testing different memory capabilities
2. Baselines that represent realistic alternatives (not strawmen)
3. Transparent reporting of where we lose, not just where we win

---

## The Three Contenders

### Mnemotree (Graph + Embeddings)
- **Ingestion**: Each memory gets a MiniLM-L6-v2 embedding + GLiNER NER entity extraction + BM25 index entry
- **Storage**: ChromaDB (vector) + SQLite (entity graph with co-occurrence links)
- **Retrieval**: Hybrid RRF fusion of BM25 keyword + cosine similarity, optional PPR graph traversal
- **Cost**: ~1 second per memory ingestion (CPU), ~500ms retrieval latency

### Obsidian-Style Wiki-Links (Novel Baseline)
We created a baseline that simulates an Obsidian vault:
- **Ingestion**: Heuristic entity extraction, `[[wiki-links]]` for each entity
- **Storage**: In-memory dict with entity-to-note backlink graph  
- **Retrieval**: TF-IDF + graph boost (notes sharing entities with the query get a 0.3x score boost)
- **Cost**: Instant ingestion, ~5ms retrieval

Nobody has benchmarked this architecture before. It represents a practical middle ground: some structure (entity links), no embedding cost.

### Markdown TF-IDF (Flat File Baseline)
Simulates how Claude Code's native `.md` memory works:
- **Ingestion**: Store text, tokenize for TF-IDF
- **Retrieval**: Rank all memories by TF-IDF cosine similarity, return top-k
- **Cost**: Instant everything

---

## The Benchmarks

### 1. LoCoMo --- Long-term Conversational Memory
**Source**: Snap Research (NeurIPS 2024)  
**What it tests**: Multi-turn conversational recall with 5 query types: single-hop, multi-hop, temporal, open-domain, adversarial  
**Our split**: 20 questions from a 419-turn conversation between two speakers

This is the benchmark where graph structure *should* shine --- long conversations with temporal dependencies and multi-hop fact chains.

### 2. LongMemEval --- Long-term Memory Evaluation  
**Source**: ICLR 2025  
**What it tests**: Information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention  
**Our split**: 30 questions stratified across 6 categories (5 per type)

Shorter sessions (avg 26 turns) testing diverse memory capabilities.

### 3. MemoryAgentBench --- Conflict Resolution
**Source**: ICLR 2026  
**What it tests**: Four agent memory competencies. We focused on Conflict Resolution (CR) --- detecting when facts have been updated and returning the current version.  
**Our split**: 80 multi-hop CR questions across 4 documents

The hardest benchmark. The original paper reports GPT-4o achieves <7% on multi-hop CR.

### 4. PersonaMem --- Personalized Preferences
**Source**: UPenn  
**What it tests**: Evolving user preferences across 180+ simulated personas with long chat histories  
**Our split**: 6 questions across different preference types

---

## The Bug That Changed Everything

Before any results: we found a **critical bug** in Mnemotree's normalization pipeline.

A coreference resolution module was silently activating despite being configured as disabled (a Heisenbug triggered by Python import order). It was replacing first-person pronouns with the speaker name:

```
Original:  "I recently set a personal best time of 25:50"
Stored:    "user recently set a personal best time of 25:50"
```

When the speaker was literally "user" (common in benchmark data), this garbled the content. The fix was a one-liner (`core.normalizer = None`), but the impact was massive:

| Category | Before Fix | After Fix |
|---|---|---|
| knowledge-update | **25%** | **100%** |
| Overall LongMemEval | 75% | 83.3% |

**Lesson**: Before benchmarking your fancy retrieval system, make sure your content isn't corrupted. The simplest bugs can dwarf any architectural advantage.

---

## Results Deep Dive

### Where Mnemotree Wins: Temporal Reasoning

On LoCoMo's temporal queries (10 cases), the results are clear:

| System | Temporal Accuracy |
|---|---|
| Mnemotree | **90%** |
| Obsidian | 75% |
| Markdown | 65% |

Semantic embeddings help here because temporal queries require finding memories that are *about* the same topic at *different* times. Keywords alone miss the semantic connection between "I ran a 5K in March" and "my race time improved."

### Where Markdown Wins: Short Sessions

On LongMemEval (avg 26 turns per case):

| System | Overall |
|---|---|
| Markdown | **85%** |
| Obsidian | 78.3% |
| Mnemotree | 75% |

With only 26 turns, the entire conversation easily fits in keyword-matching range. TF-IDF finds the relevant passages just as well as embeddings, without the ingestion overhead. The LLM generator then has the same context quality regardless of retrieval method.

### Where Everyone Struggles: Conflict Resolution

MAB-CR (80 multi-hop questions):

| System | Accuracy |
|---|---|
| Markdown | **13.8%** |
| Mnemotree | 11.9% |
| Obsidian | 11.9% |

All systems score below 14%. The MemoryAgentBench paper confirms this is fundamentally hard --- their evaluation of GPT-4o found <7% on multi-hop CR.

We tried several improvements:
1. **Temporal decay scoring** (favor newer memories): 0% -> 0% (no effect)
2. **Near-duplicate supersede filter**: 0% -> 0% (CR isn't about duplicates)
3. **Conflict-aware prompt + timestamp ordering**: 0% -> 10% (cheap win)
4. **PPR graph retrieval** (SQLiteVec + entity traversal): 0% -> **15%** (best result)

PPR helps because multi-hop CR questions chain entities: "What is the citizenship of the spouse of the author of X?" Each hop requires traversing an entity link --- exactly what PPR does.

---

## The Obsidian Baseline: A Practical Middle Ground

Our novel Obsidian-style baseline is interesting because it's *nearly free* but adds genuine structure:

| Benchmark | Obsidian vs Markdown |
|---|---|
| LoCoMo | +7.5pp (65% vs 57.5%) |
| LongMemEval | -6.7pp (78.3% vs 85%) |
| MAB-CR | -1.9pp (11.9% vs 13.8%) |
| PersonaMem | -8.3pp (66.7% vs 75%) |

Wiki-link entity graphs help on longer conversations (LoCoMo) but hurt on shorter ones --- the entity extraction adds noise without enough context to disambiguate. For production use, Obsidian-style linking is a reasonable upgrade from pure TF-IDF when conversation histories grow beyond ~100 turns.

---

## When Is Graph-Based Memory Worth It?

Based on our results across 4 benchmarks:

**Use graph-based memory (Mnemotree) when:**
- Conversation histories exceed ~200 turns
- Queries require temporal reasoning ("what changed between session 3 and 5?")
- Multi-hop entity chains matter ("who is the colleague of the person who...")
- You need the PPR traversal for conflict resolution

**Stick with TF-IDF (Markdown) when:**
- Sessions are short (<100 turns)
- Queries are factual/keyword-heavy
- Ingestion speed matters (real-time applications)
- You're optimizing for cost over accuracy

**Consider Obsidian-style wiki-links when:**
- You want structure without embedding cost
- Conversation histories are medium-length (100-500 turns)
- You need some entity-aware retrieval but can't afford NER + embeddings

---

## Cost Comparison

| | Mnemotree | Obsidian | Markdown |
|---|---|---|---|
| Ingestion | ~1s/turn (NER + embed) | Instant | Instant |
| Storage | Vector DB + SQLite graph | In-memory | In-memory |
| Retrieval | ~500ms | ~5ms | ~5ms |
| LLM tokens/query | ~2-3K (top-20) | ~2-3K (top-20) | ~2-3K (top-20) |

All systems are CPU-only. The main cost is ingestion time. LLM generation costs are identical since all systems send the same number of retrieved results to the LLM.

---

## What We'd Do Differently

1. **Larger sample sizes**: Our splits (10-80 cases) are noisy. A single wrong answer can swing results by 5-10%. Production evaluations need 200+ cases per category.

2. **Multiple LLM judge runs**: We used a single GPT-4.1 generation + GPT-4o-mini judge pass. Averaging 3-5 runs would reduce variance.

3. **Ablation studies**: Which component matters most --- BM25, embeddings, NER, or the graph? We tested the full stack vs baselines, but didn't isolate individual components.

4. **Ingestion-time conflict detection**: Mnemotree's `CONTRADICTS` link detection requires negation words + shared entities. For knowledge updates without explicit negation ("John has 3 cats" -> "John has 4 cats"), the heuristic misses.

---

## Reproduction

All code, benchmark runners, and result JSONs are at:
- Branch: `fix/cr-improvement` on [github.com/kurcontko/mnemotree](https://github.com/kurcontko/mnemotree)
- Runners: `benchmarks/run_locomo_comparison.py`, `run_longmemeval_comparison.py`, `run_mab_comparison.py`, `run_personamem_comparison.py`
- Baselines: `benchmarks/lib/obsidian_baseline.py`, `benchmarks/lib/markdown_baseline.py`
- Results: `benchmarks/results/v18_full_*.json`

---

## Conclusion

Graph-based memory isn't universally better than flat files. It's better at specific tasks --- temporal reasoning, multi-hop queries, and long conversation histories --- and worse at others. The engineering lesson: benchmark against the simplest possible baseline first. If TF-IDF solves your problem, you don't need embeddings.

The most impactful finding wasn't architectural --- it was a bug. A coreference normalizer silently corrupting stored content caused a 75 percentage point swing on knowledge-update queries. Before optimizing your retrieval pipeline, verify your data isn't corrupted.

For the memory systems community: we need more honest benchmarking. Cherry-picked results on favorable tasks don't help practitioners choose the right architecture. Multi-benchmark evaluations with transparent failure analysis do.
