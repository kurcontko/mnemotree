# Dataset Generation Pipelines for Small-Model Ingestion

**Goal**: Train two tiny models for CPU/MPS inference that genuinely improve memory quality.
**Teacher**: `gpt-oss-120b` on DGX Spark 128GB (free, local, unlimited).
**Upgrade path**: Qwen2.5-0.5B → Qwen3-4B for users with GPU.

---

## Target Models

| Role | Model | Size | Inference | Why |
|------|-------|------|-----------|-----|
| **Router** | DistilBERT-base | 66M | ~10ms CPU | Multi-label classification is a solved problem at this scale |
| **Extractor** | Qwen2.5-0.5B + LoRA | 500M + 5MB adapter | ~80ms CPU, ~20ms MPS | Smallest model that reliably outputs structured JSON |
| **Extractor (beefy)** | Qwen3-4B + LoRA | 4B + 10MB adapter | ~20ms GPU | Drop-in upgrade, same adapter architecture |

---

## What We're Training

### Router (DistilBERT)
**Input**: a conversation turn (1-3 sentences)
**Output**: multi-label `[episodic, semantic, procedural]` (sigmoid per head)

```
"I started working at Google last month" → [1, 1, 0]  (episodic event + semantic fact)
"You should always run tests before pushing" → [0, 0, 1]  (procedural)
"Hey what's up!" → [0, 0, 0]  (nothing worth remembering = skip ingestion)
```

The [0,0,0] case is critical — it's an **ingestion gate**. Most conversation turns are chitchat.
This alone saves cost by skipping enrichment on ~60-70% of turns.

### Extractor (Qwen2.5-0.5B)
**Input**: instruction prefix + conversation turn (with surrounding context window of ±2 turns)
**Output**: structured JSON per memory type

Three instruction prefixes, one model:
```
<|semantic|> "I started working at Google last month"
→ {"fact": "User works at Google", "subject": "User", "confidence": 0.95}

<|episodic|> "I started working at Google last month"
→ {"event": "Started working at Google", "when": "2026-01", "who": ["User"], "significance": "career_change"}

<|procedural|> "First you run pytest, then check coverage, then push"
→ {"procedure": "Code deployment workflow", "steps": ["Run pytest", "Check coverage", "Push"], "domain": "development"}
```

---

## Data Sources (ranked by priority)

### Tier 1: Free labels, no teacher needed

| Source | Turns | What you get | Effort |
|--------|-------|-------------|--------|
| **LOCOMO observations** | ~2,000 | Pre-extracted per-speaker facts with dia_id citations. Gold semantic extraction labels. | Parse only |
| **LOCOMO event_summary** | ~500 | Per-session event lists per speaker. Gold episodic labels. | Parse only |
| **LOCOMO session_summary** | ~190 | Narrative summaries. Useful for consolidation training later. | Parse only |
| **MSC PersonaSummary** | 130k train | Persona facts extracted across sessions. Gold semantic labels. | Download + filter |

### Tier 2: Teacher-labeled (gpt-oss-120b on DGX Spark, $0)

| Source | Raw turns | After labeling | What you get |
|--------|-----------|---------------|-------------|
| **LOCOMO raw turns** | ~6,000 | ~6,000 router + ~4,000 extractor | Full pipeline labels including [0,0,0] skip class |
| **MSC raw dialogues** | 237k | ~20,000 (subsample) | Diverse persona conversations, different style than LOCOMO |
| **Synthetic code convos** | generate ~2,000 | ~2,000 | Developer chats, code reviews, debugging sessions |
| **PersonaChat** | 20k convos | ~5,000 (subsample) | Short but dense persona facts |
| **LongMemEval histories** | ~500 sessions | ~3,000 turns | Multi-session reasoning, temporal updates |

### Tier 3: Augmentation (no teacher, mechanical transforms)

| Technique | Multiplier | What it does |
|-----------|-----------|-------------|
| **Paraphrase via back-translation** | 2x | Run through teacher: "Rephrase this naturally" |
| **Speaker name swapping** | 2x | Replace names to prevent overfitting to specific personas |
| **Temporal shifting** | 1.5x | Change dates/times to prevent memorizing specific dates |
| **Negation injection** | 1.3x | "I work at Google" → "I no longer work at Google" (for update detection) |

---

## Pipeline 1: LOCOMO Gold Extraction (no teacher needed)

```
scripts/data_gen/
  01_extract_locomo_gold.py
```

### What it does

Parses locomo10.json (and full LOCOMO when available) to extract pre-labeled training data
from the `observation`, `event_summary`, and raw conversation fields.

### Router labels from observations

```python
for sample in locomo_data:
    for session_key, obs in sample["observation"].items():
        for speaker, observations in obs.items():
            for obs_text, dia_id in observations:
                # Find the raw turn that produced this observation
                turn = find_turn_by_dia_id(sample["conversation"], dia_id)

                # Classify observation type heuristically:
                labels = classify_observation(obs_text)
                # - contains dates/events/actions → episodic=1
                # - contains facts/preferences/states → semantic=1
                # - contains instructions/workflows → procedural=1

                router_examples.append({
                    "text": turn["text"],
                    "speaker": turn["speaker"],
                    "labels": labels,  # [episodic, semantic, procedural]
                    "source": "locomo_gold"
                })
```

### Extractor labels from observations

The observation text IS the extraction target:

```python
# Observation: "Caroline works as a nurse at Memorial Hospital"
# Source turn: "Yeah so I've been at Memorial Hospital for 3 years now, nursing is tough but rewarding"
→ {
    "input": "Yeah so I've been at Memorial Hospital for 3 years now...",
    "type": "semantic",
    "output": {"fact": "Caroline works as a nurse at Memorial Hospital", "subject": "Caroline", "confidence": 0.95},
    "source": "locomo_gold"
}
```

### Negative examples (ingestion gate)

Equally important — turns that should NOT be remembered:

```python
# Find turns with NO observations pointing to them
all_cited_dia_ids = collect_all_cited_dia_ids(sample)
for turn in all_turns:
    if turn["dia_id"] not in all_cited_dia_ids:
        router_examples.append({
            "text": turn["text"],
            "labels": [0, 0, 0],  # skip
            "source": "locomo_gold_negative"
        })
```

### Expected yield from locomo10

| Type | Count |
|------|-------|
| Router positive (at least one label=1) | ~1,500-2,000 |
| Router negative ([0,0,0]) | ~3,000-4,000 |
| Semantic extractor pairs | ~1,500 |
| Episodic extractor pairs (from event_summary) | ~400-600 |
| Procedural extractor pairs | ~50-100 (rare in LOCOMO) |

### Heuristic type classifier for observations

```python
EPISODIC_SIGNALS = [
    r'\b(went|visited|attended|started|moved|traveled|met|celebrated)\b',
    r'\b(yesterday|last week|on \w+ \d+|in \d{4})\b',
    r'\b(happened|occurred|took place)\b',
]
SEMANTIC_SIGNALS = [
    r'\b(is a|works as|lives in|likes|prefers|allergic|favorite)\b',
    r'\b(has a|owns|studies|majored)\b',
    r'\b(always|usually|never|every)\b',
]
PROCEDURAL_SIGNALS = [
    r'\b(first.*then|step \d|you should|make sure to|the way to)\b',
    r'\b(recipe|instructions?|workflow|process|how to)\b',
]

def classify_observation(text: str) -> list[int]:
    e = int(any(re.search(p, text, re.I) for p in EPISODIC_SIGNALS))
    s = int(any(re.search(p, text, re.I) for p in SEMANTIC_SIGNALS))
    p = int(any(re.search(p, text, re.I) for p in PROCEDURAL_SIGNALS))
    # Default to semantic if no signal (facts are the most common observation type)
    if e == 0 and s == 0 and p == 0:
        s = 1
    return [e, s, p]
```

---

## Pipeline 2: Teacher Labeling via gpt-oss-120b

```
scripts/data_gen/
  02_teacher_label_router.py
  03_teacher_label_extractor.py
```

### Setup

```python
from openai import OpenAI

# DGX Spark - local, free, unlimited
client = OpenAI(
    base_url="http://localhost:30000/v1",  # or 192.168.0.206:30000
    api_key="none",
)
MODEL = "openai/gpt-oss-120b"
```

### Router labeling prompt

```python
ROUTER_SYSTEM = """You classify conversation turns for a memory system.
Given a turn from a conversation, output which memory types apply.

Rules:
- episodic: a personal experience, event, or something that happened at a specific time
- semantic: a fact, preference, opinion, trait, relationship, or piece of knowledge about someone
- procedural: a how-to, instruction, workflow, recipe, or process
- skip: chitchat, greetings, filler, or content with nothing worth remembering

A turn can have MULTIPLE types (e.g., "I started at Google last month" is both episodic AND semantic).
A turn can have ZERO types (skip).

Output ONLY valid JSON: {"episodic": bool, "semantic": bool, "procedural": bool}"""

ROUTER_USER = """Speaker: {speaker}
Context (previous 2 turns):
{context}

Turn to classify:
"{text}"
"""
```

### Extractor labeling prompts

Three separate prompts, one per type. Only called when router says the type applies.

```python
SEMANTIC_SYSTEM = """Extract the core fact(s) from this conversation turn.
Output JSON: {"fact": "...", "subject": "...", "confidence": 0.0-1.0}

Rules:
- "fact" should be a standalone statement that makes sense without context
- Replace pronouns with actual names using the speaker/context info
- "subject" is the person the fact is about
- "confidence" reflects how certain/permanent this fact is (preferences=0.7, jobs=0.9, etc.)
- Keep it concise — one fact per extraction. If multiple facts, return a JSON array."""

EPISODIC_SYSTEM = """Convert this conversation turn into an episodic memory.
Output JSON: {"event": "...", "when": "...", "who": [...], "significance": "..."}

Rules:
- "event" should be a clear description of what happened
- "when" should be an ISO-8601 date/time if inferrable, otherwise relative ("recently", "last week")
- "who" lists all people involved by name
- "significance" is a one-word category: career_change, travel, health, social, achievement, daily, relationship, education, loss"""

PROCEDURAL_SYSTEM = """Convert this conversation turn into a procedural memory.
Output JSON: {"procedure": "...", "steps": [...], "domain": "..."}

Rules:
- "procedure" is a short title for the process
- "steps" is an ordered list of action strings
- "domain" is the area: cooking, development, fitness, work, social, etc."""
```

### Batch processing with rate control

```python
import asyncio
from tqdm.asyncio import tqdm_asyncio

async def label_batch(turns: list[dict], prompt_type: str, max_concurrent: int = 32):
    """Label a batch of turns using gpt-oss-120b.

    DGX Spark with 128GB can handle high concurrency for a 120B model
    but we cap at 32 to avoid OOM on long sequences.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def label_one(turn):
        async with semaphore:
            system, user_template = get_prompts(prompt_type)
            context = format_context(turn.get("prev_turns", []))

            for attempt in range(3):  # retry on malformed JSON
                response = await aclient.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_template.format(
                            speaker=turn["speaker"],
                            context=context,
                            text=turn["text"],
                        )},
                    ],
                    temperature=0.1,  # low temp for consistent labels
                    max_tokens=256,
                )
                raw = response.choices[0].message.content
                parsed = try_parse_json(raw)
                if parsed is not None:
                    return {**turn, f"{prompt_type}_label": parsed}

            return {**turn, f"{prompt_type}_label": None, "label_error": True}

    results = await tqdm_asyncio.gather(*[label_one(t) for t in turns])
    return [r for r in results if r.get("label_error") is not True]
```

### Context window construction

The extractor sees the target turn + ±2 surrounding turns for coreference:

```python
def build_context_window(turns: list[dict], target_idx: int, window: int = 2) -> dict:
    """Build a context window around the target turn."""
    start = max(0, target_idx - window)
    end = min(len(turns), target_idx + window + 1)

    return {
        "text": turns[target_idx]["text"],
        "speaker": turns[target_idx]["speaker"],
        "dia_id": turns[target_idx].get("dia_id"),
        "prev_turns": [
            f"{t['speaker']}: {t['text']}"
            for t in turns[start:target_idx]
        ],
        "next_turns": [
            f"{t['speaker']}: {t['text']}"
            for t in turns[target_idx+1:end]
        ],
    }
```

---

## Pipeline 3: MSC (Multi-Session Chat) Harvesting

```
scripts/data_gen/
  04_harvest_msc.py
```

### Why MSC

- 237k training examples with built-in PersonaSummary task
- Multi-session: speakers return after hours/days, reference past conversations
- Already has persona fact annotations
- Different conversation style than LOCOMO → prevents overfitting

### What to extract

```python
from datasets import load_dataset

# PersonaSummary: already-extracted persona facts per session
msc_summary = load_dataset("nayohan/multi_session_chat", split="train")

for example in msc_summary:
    previous_sessions = example["previous"]     # list of past session texts
    current_session = example["current"]         # current session dialog
    persona_summary = example["persona_summary"] # extracted persona updates

    # persona_summary entries are gold SEMANTIC labels
    # current_session turns that produced them are the inputs

    # Use teacher to align which turns map to which persona facts
    # (MSC doesn't have turn-level citations like LOCOMO's dia_id)
```

### Teacher-assisted turn-fact alignment

Since MSC doesn't have turn-level evidence pointers, use the teacher to align:

```python
ALIGN_PROMPT = """Given these conversation turns and this persona fact,
identify which turn(s) most directly express or imply this fact.

Turns:
{numbered_turns}

Fact: "{persona_fact}"

Output JSON: {"turn_indices": [int], "confidence": float}"""
```

### Subsample strategy

MSC has 237k examples but we don't need all of them. Sample for diversity:

```python
# Target: ~10,000 MSC examples
# Strategy:
#   - 5,000 from PersonaSummary (turn → fact pairs)
#   - 3,000 negative examples (chitchat turns with no persona updates)
#   - 2,000 from multi-session context (turns referencing past sessions)
# Deduplicate by fact content similarity (embedding cosine > 0.92 = duplicate)
```

---

## Pipeline 4: Synthetic Code Conversations

```
scripts/data_gen/
  05_generate_code_convos.py
```

### Why synthetic code data

LOCOMO and MSC are personal conversations. Mnemotree also targets **developer workflows** —
code reviews, debugging sessions, architecture discussions, project planning. These have
different memory patterns:

| Personal conversation | Code conversation |
|----------------------|-------------------|
| "I work at Google" (semantic) | "The auth service uses JWT tokens" (semantic) |
| "I went to Tokyo last week" (episodic) | "We migrated to PostgreSQL yesterday" (episodic) |
| "First boil the water..." (procedural) | "To deploy: run tests, build image, push to registry" (procedural) |

### Generation templates

Use gpt-oss-120b to generate realistic code conversations from scenario seeds:

```python
CODE_SCENARIOS = [
    # Debugging sessions
    {"type": "debugging", "context": "Python FastAPI app, intermittent 500 errors on /api/users endpoint"},
    {"type": "debugging", "context": "React app, state not updating after API call in useEffect"},
    {"type": "debugging", "context": "Docker container OOM killed, Node.js memory leak"},

    # Code reviews
    {"type": "code_review", "context": "PR adding OAuth2 login flow, reviewer concerned about token storage"},
    {"type": "code_review", "context": "PR refactoring database layer from raw SQL to SQLAlchemy ORM"},

    # Architecture discussions
    {"type": "architecture", "context": "Choosing between Redis and Memcached for session caching"},
    {"type": "architecture", "context": "Microservices vs monolith for a new e-commerce platform"},

    # Project planning
    {"type": "planning", "context": "Sprint planning for authentication system overhaul"},
    {"type": "planning", "context": "Onboarding new developer, explaining codebase structure"},

    # Pair programming
    {"type": "pair_programming", "context": "Implementing binary search tree with deletion in Rust"},
    {"type": "pair_programming", "context": "Writing pytest fixtures for async database operations"},
]

GENERATE_PROMPT = """Generate a realistic multi-turn conversation between two developers.

Scenario: {scenario_type}
Context: {context}

Requirements:
- 8-15 turns, alternating speakers
- Include specific technical details (library names, error messages, code snippets)
- Naturally embed facts, decisions, and procedures in the conversation
- Include some chitchat/filler turns (greetings, "let me think...", "good point")
- Use realistic developer names

Output as JSON array: [{{"speaker": "...", "text": "..."}}, ...]"""
```

### Self-labeling (teacher labels its own output)

Since the teacher generates the conversations AND labels them, we do it in one shot:

```python
GENERATE_AND_LABEL_PROMPT = """Generate a realistic {turns}-turn developer conversation,
then label each turn for memory extraction.

Scenario: {scenario_type} — {context}

For each turn, output:
{{
  "speaker": "name",
  "text": "what they said",
  "memory_types": {{"episodic": bool, "semantic": bool, "procedural": bool}},
  "extractions": [
    {{"type": "semantic", "fact": "...", "subject": "...", "confidence": 0.9}},
    {{"type": "episodic", "event": "...", "when": "...", "who": [...]}},
    {{"type": "procedural", "procedure": "...", "steps": [...], "domain": "..."}}
  ]  // only include extractions matching true memory_types, empty array for skip turns
}}

Output a JSON array of these objects. Include 30-40% skip turns (greetings, filler)."""
```

### Volume target

```
50 scenario seeds × 3 variations each = 150 conversations
150 conversations × ~12 turns avg = ~1,800 turns
+ 200 additional scenarios from teacher brainstorming = ~2,400 more turns
Total: ~4,200 code-domain turns with labels
```

### Scenario seed expansion

Let the teacher brainstorm more scenarios from a few examples:

```python
EXPAND_PROMPT = """Here are 10 example developer conversation scenarios:
{examples}

Generate 50 MORE diverse scenarios in the same format. Cover:
- Different languages (Python, TypeScript, Rust, Go, Java, C++)
- Different domains (web, mobile, ML, infra, gamedev, embedded)
- Different activities (debugging, reviewing, architecting, deploying, onboarding)
- Different team dynamics (senior-junior, peer-peer, cross-team)

Output as a JSON array of {{"type": "...", "context": "..."}} objects."""
```

---

## Pipeline 5: Quality Filtering & Validation

```
scripts/data_gen/
  06_validate_and_filter.py
```

### JSON schema validation

```python
from pydantic import BaseModel, field_validator

class RouterLabel(BaseModel):
    text: str
    speaker: str | None = None
    labels: list[int]  # [episodic, semantic, procedural]
    source: str

    @field_validator("labels")
    def validate_labels(cls, v):
        assert len(v) == 3
        assert all(x in (0, 1) for x in v)
        return v

class SemanticExtraction(BaseModel):
    fact: str
    subject: str
    confidence: float

    @field_validator("confidence")
    def validate_confidence(cls, v):
        assert 0.0 <= v <= 1.0
        return v

class EpisodicExtraction(BaseModel):
    event: str
    when: str
    who: list[str]
    significance: str | None = None

class ProceduralExtraction(BaseModel):
    procedure: str
    steps: list[str]
    domain: str
```

### Deduplication

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def deduplicate(examples: list[dict], field: str = "text", threshold: float = 0.92):
    """Remove near-duplicate examples by embedding similarity."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [ex[field] for ex in examples]
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)

    # Greedy dedup: keep first occurrence, remove duplicates
    keep = []
    keep_embeddings = []
    for i, emb in enumerate(embeddings):
        if len(keep_embeddings) == 0:
            keep.append(i)
            keep_embeddings.append(emb)
            continue
        sims = cosine_similarity([emb], keep_embeddings)[0]
        if sims.max() < threshold:
            keep.append(i)
            keep_embeddings.append(emb)

    return [examples[i] for i in keep]
```

### Quality filters

```python
def quality_filter(examples: list[dict]) -> list[dict]:
    filtered = []
    for ex in examples:
        # Skip very short turns (< 4 words) — too ambiguous
        if len(ex["text"].split()) < 4:
            continue

        # Skip very long turns (> 200 words) — truncation issues for DistilBERT
        if len(ex["text"].split()) > 200:
            ex["text"] = " ".join(ex["text"].split()[:200])

        # For extractor: verify extraction is shorter than input
        # (extractions should compress, not expand)
        if "output" in ex:
            if len(str(ex["output"])) > len(ex["text"]) * 3:
                continue

        # For router: skip if label distribution is pathological
        # (all 3 types = 1 is suspicious for a single turn)
        if "labels" in ex and sum(ex["labels"]) == 3:
            continue  # almost never all three

        filtered.append(ex)
    return filtered
```

---

## Pipeline 6: Dataset Assembly & Splitting

```
scripts/data_gen/
  07_assemble_dataset.py
```

### Final dataset structure

```
data/training/
  router/
    train.jsonl      # ~8,000-12,000 examples
    val.jsonl        # ~1,500-2,000 examples
    test.jsonl       # ~1,500-2,000 examples
    label_stats.json # class distribution
  extractor/
    semantic_train.jsonl    # ~4,000-6,000 examples
    semantic_val.jsonl
    episodic_train.jsonl    # ~1,500-2,500 examples
    episodic_val.jsonl
    procedural_train.jsonl  # ~800-1,500 examples (smallest class)
    procedural_val.jsonl
    combined_train.jsonl    # all types, instruction-prefixed
    combined_val.jsonl
  metadata.json             # dataset stats, source breakdown, generation date
```

### Splitting strategy

**Conversation-level splits** — never leak turns from the same conversation across splits:

```python
from sklearn.model_selection import GroupShuffleSplit

def split_by_conversation(examples, test_size=0.15, val_size=0.15):
    """Split ensuring all turns from one conversation stay together."""
    conv_ids = [ex["conversation_id"] for ex in examples]

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    trainval_idx, test_idx = next(gss1.split(examples, groups=conv_ids))

    # Second split: train vs val
    trainval_conv_ids = [conv_ids[i] for i in trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=42)
    train_idx, val_idx = next(gss2.split(trainval_idx, groups=trainval_conv_ids))

    return (
        [examples[trainval_idx[i]] for i in train_idx],
        [examples[trainval_idx[i]] for i in val_idx],
        [examples[i] for i in test_idx],
    )
```

### Class balancing for router

The [0,0,0] skip class will dominate (~60%). Balance via undersampling:

```python
def balance_router_dataset(examples):
    """Balance skip vs non-skip, and within non-skip balance types."""
    positive = [ex for ex in examples if sum(ex["labels"]) > 0]
    negative = [ex for ex in examples if sum(ex["labels"]) == 0]

    # Keep all positives, undersample negatives to 1.2x positive count
    # (slight negative skew is realistic — most turns ARE chitchat)
    target_neg = int(len(positive) * 1.2)
    if len(negative) > target_neg:
        negative = random.sample(negative, target_neg)

    return positive + negative
```

### Extractor training format (instruction-prefixed)

```jsonl
{"messages": [{"role": "system", "content": "Extract semantic memory."}, {"role": "user", "content": "Context:\nAlice: Hey how's the new job?\n\nTurn:\nBob: It's great, I'm working at Stripe now on the payments team"}, {"role": "assistant", "content": "{\"fact\": \"Bob works at Stripe on the payments team\", \"subject\": \"Bob\", \"confidence\": 0.95}"}]}
{"messages": [{"role": "system", "content": "Extract episodic memory."}, {"role": "user", "content": "Context:\nAlice: Did you end up going?\n\nTurn:\nBob: Yeah I visited the new office in SF yesterday, it was amazing"}, {"role": "assistant", "content": "{\"event\": \"Bob visited Stripe's new SF office\", \"when\": \"yesterday\", \"who\": [\"Bob\"], \"significance\": \"daily\"}"}]}
```

---

## Pipeline 7: Augmentation

```
scripts/data_gen/
  08_augment.py
```

### Name swapping (mechanical, no teacher)

```python
import random

COMMON_NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
                "Avery", "Dakota", "Skyler", "Reese", "Finley", "Rowan", "Sage", "Kai"]

def swap_names(example: dict, original_names: list[str]) -> dict:
    """Replace speaker names with random alternatives."""
    new_names = random.sample(COMMON_NAMES, len(original_names))
    mapping = dict(zip(original_names, new_names))

    new_ex = example.copy()
    for old, new in mapping.items():
        new_ex["text"] = new_ex["text"].replace(old, new)
        if "output" in new_ex:
            new_ex["output"] = json.loads(
                json.dumps(new_ex["output"]).replace(old, new)
            )
        if new_ex.get("speaker") == old:
            new_ex["speaker"] = new

    new_ex["source"] = example["source"] + "_nameswap"
    return new_ex
```

### Paraphrasing via teacher (for extractor diversity)

```python
PARAPHRASE_PROMPT = """Rephrase this conversation turn naturally.
Keep the same meaning and information but change the wording.
The rephrased version should feel like a different person said it.

Original: "{text}"
Rephrased:"""
```

Only apply to extractor training data — the router needs to handle diverse phrasings
and paraphrasing the input is one way to teach that.

---

## Execution Order

```
Phase 1: No teacher needed (can start immediately)
  01_extract_locomo_gold.py         → ~5,000 examples
  04_harvest_msc.py (download only) → raw data ready

Phase 2: Teacher on DGX Spark (start vLLM, then run)
  02_teacher_label_router.py        → ~6,000 LOCOMO labels
  03_teacher_label_extractor.py     → ~4,000 LOCOMO extractions
  04_harvest_msc.py (label phase)   → ~10,000 MSC labels
  05_generate_code_convos.py        → ~4,200 code turns with labels

Phase 3: Post-processing (CPU, fast)
  06_validate_and_filter.py         → drop ~10-15% bad examples
  08_augment.py                     → ~1.5x multiplier
  07_assemble_dataset.py            → final splits

Expected final counts:
  Router:    ~15,000-20,000 train / ~2,500 val / ~2,500 test
  Extractor: ~10,000-14,000 train / ~1,500 val / ~1,500 test
```

### Time estimate on DGX Spark

```
gpt-oss-120b throughput: ~10-15 tok/s per request, 32 concurrent
Router labels: ~6,000 turns × ~100 tok/response ÷ (15 × 32) = ~12 min
Extractor labels: ~4,000 turns × ~150 tok/response ÷ (15 × 32) = ~12 min
MSC alignment: ~10,000 examples × ~200 tok ÷ (15 × 32) = ~70 min
Code generation: ~200 convos × ~2,000 tok ÷ (15 × 32) = ~14 min
Paraphrasing: ~5,000 × ~100 tok ÷ (15 × 32) = ~10 min

Total teacher time: ~2 hours
```

---

## Directory Layout

```
scripts/data_gen/
  config.py              # vLLM endpoint, model name, paths
  01_extract_locomo_gold.py
  02_teacher_label_router.py
  03_teacher_label_extractor.py
  04_harvest_msc.py
  05_generate_code_convos.py
  06_validate_and_filter.py
  07_assemble_dataset.py
  08_augment.py
  prompts/
    router.txt
    semantic_extractor.txt
    episodic_extractor.txt
    procedural_extractor.txt
    code_scenario_seeds.json
  utils/
    vllm_client.py       # async OpenAI wrapper with retry + rate limiting
    json_parser.py       # robust JSON extraction from LLM output
    locomo_parser.py     # LOCOMO format parsing helpers
    msc_loader.py        # MSC dataset loading + filtering
data/
  raw/                   # downloaded datasets (gitignored)
  labeled/               # teacher-labeled intermediate files
  training/              # final assembled datasets
    router/
    extractor/
    metadata.json
```

---

## What This Is NOT

This pipeline is designed to improve **real memory quality**, not benchmark scores:

- **Diverse sources** (LOCOMO + MSC + code + PersonaChat) prevent overfitting to one domain
- **The ingestion gate** ([0,0,0] skip) is the single biggest real-world improvement — most turns are noise
- **Code domain data** makes mnemotree useful for developer workflows, which no competitor targets
- **Name swapping + temporal shifting** prevent memorizing specific entities/dates
- **Conversation-level splits** ensure honest evaluation
- **Procedural memory** is underrepresented in all benchmarks but critical for real use (recipes, workflows, deployment steps)
