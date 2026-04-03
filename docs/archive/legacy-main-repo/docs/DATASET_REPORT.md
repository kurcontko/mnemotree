# Mnemotree Training Dataset Report

**Generated**: 2026-02-23
**Pipeline**: `scripts/data_gen/01_harvest_locomo.py` through `08_augment.py`
**Teacher model**: `gpt-oss-120b` (120B thinking model) via vLLM at `localhost:30000`

---

## 1. Overview

This dataset trains two small models for Mnemotree's memory ingestion pipeline:

| Model | Architecture | Task | Records |
|-------|-------------|------|---------|
| **Router** | DistilBERT 66M | Multi-label `[episodic, semantic, procedural]` classification + ingestion gate | 73,781 |
| **Extractor** | Qwen2.5-0.5B + LoRA | Structured JSON extraction per memory type | 43,783 |

**Total disk**: ~150 MB across labeled and training directories.

---

## 2. Pipeline Stages

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 01 | `01_harvest_locomo.py` | Gold labels from LoCoMo annotations | LoCoMo dataset | 5,707 router + 3,222 extractor |
| 02 | `02_label_router.py` | Teacher-labeled router data | LoCoMo conversations | 5,882 router records |
| 03 | `03_label_extractor.py` | Teacher-labeled extractor data | LoCoMo conversations | 6,612 extractor records |
| 04 | `04_harvest_msc.py` | Persona facts + chitchat negatives from MSC | Multi-Session Chat (HuggingFace) | 5,200 semantic + 5,000 negatives |
| 05 | `05_generate_code_convos.py` | Synthetic code/technical conversations | Teacher generation | 2,925 records |
| 06 | `06_validate_and_filter.py` | Schema validation, dedup, content filtering | All `*_labeled.jsonl` | 30,671 clean records (11.2% drop) |
| 07 | `07_assemble_dataset.py` | Train/val/test splits, format standardization | All `*_clean.jsonl` + augmented | 73,781 router + 43,783 extractor |
| 08 | `08_augment.py` | Name swapping + teacher paraphrasing | Clean records | 65,282 augmented records |

### Filtering (Script 06)

34,548 raw records reduced to 30,671 clean records (11.2% removed):

- Schema validation (required fields, correct types)
- Content filtering (minimum length, empty fields)
- Exact deduplication
- Embedding-based near-duplicate removal (all-MiniLM-L6-v2, cosine threshold)

### Augmentation (Script 08)

| Method | Records | Notes |
|--------|---------|-------|
| Name swapping | 61,342 | 2 random name variants per record (mechanical, no model) |
| Teacher paraphrasing | 3,940 | 30% of extractor records, 99% success rate via gpt-oss-120b |
| **Total augmented** | **65,282** | |

---

## 3. Data Sources

### Provenance Breakdown

| Source | Router | Extractor | Description |
|--------|--------|-----------|-------------|
| **MSC** | 30,898 | 16,183 | Multi-Session Chat persona facts + negatives |
| **LoCoMo Teacher** | 17,373 | 13,378 | gpt-oss-120b labeled LoCoMo conversations |
| **LoCoMo Gold** | 16,365 | 7,870 | Human-annotated LoCoMo ground truth |
| **Code Synthetic** | 9,145 | 5,428 | Teacher-generated code/technical conversations |
| **LoCoMo Events** | — | 924 | Event-specific episodic extraction |

### Original vs. Augmented

| Task | Original | Name-swapped | Paraphrased | Total |
|------|----------|-------------|-------------|-------|
| Router | 23,945 (32.5%) | 47,890 (64.9%) | 1,946 (2.6%) | 73,781 |
| Extractor | 13,281 (30.3%) | 26,562 (60.7%) | 3,940 (9.0%) | 43,783 |

---

## 4. Router Dataset

### Split Sizes

| Split | Records | Purpose |
|-------|---------|---------|
| Train | 49,706 | Model training |
| Val | 10,024 | Hyperparameter tuning / early stopping |
| Test | 14,051 | Final evaluation |

### Label Distribution (All Splits)

| Label | Count | % of Total |
|-------|-------|------------|
| Semantic | 37,504 | 50.8% |
| None (ingestion gate) | 28,713 | 38.9% |
| Episodic | 19,510 | 26.4% |
| Procedural | 5,384 | 7.3% |

> Percentages sum to >100% because labels are multi-label (a turn can be both episodic and semantic).

### Label Combinations (Train Split)

| Combination | Count | % |
|-------------|-------|---|
| `[0,0,0]` — none | 19,395 | 39.0% |
| `[0,1,0]` — semantic only | 13,414 | 27.0% |
| `[1,1,0]` — episodic + semantic | 10,725 | 21.6% |
| `[1,0,0]` — episodic only | 2,479 | 5.0% |
| `[0,0,1]` — procedural only | 2,475 | 5.0% |
| `[0,1,1]` — semantic + procedural | 1,090 | 2.2% |
| `[1,0,1]` — episodic + procedural | 128 | 0.3% |

The ~39% "none" class ensures the ingestion gate learns to filter chitchat. The dominant semantic class reflects that most memorable turns contain factual content.

### Input Text Statistics

| Metric | Words |
|--------|-------|
| Min | 3 |
| P5 | 7 |
| Median | 21 |
| Mean | 23.6 |
| P95 | 50 |
| Max | 173 |

### Record Schema

```json
{
  "text": "I started working at Google last month.",
  "speaker": "Alice",
  "context": [
    {"speaker": "Bob", "text": "How's the new job going?"},
    {"speaker": "Alice", "text": "Great, actually!"}
  ],
  "labels": [1, 1, 0]
}
```

- `text`: The conversation turn to classify
- `speaker`: Who said it
- `context`: Up to 2 preceding turns for context
- `labels`: `[episodic, semantic, procedural]` — multi-hot binary vector

---

## 5. Extractor Dataset

### Split Sizes

| Split | Records | Purpose |
|-------|---------|---------|
| Train | 32,368 | Model training |
| Val | 3,142 | Hyperparameter tuning / early stopping |
| Test | 8,273 | Final evaluation |

### Type Distribution (All Splits)

| Type | Count | % |
|------|-------|---|
| Semantic | 31,189 | 71.2% |
| Episodic | 11,163 | 25.5% |
| Procedural | 1,431 | 3.3% |

### Output Fields by Type

**Semantic** (31,189 records):
| Field | Presence | Description |
|-------|----------|-------------|
| `fact` | 100% | The extracted factual statement |
| `subject` | 100% | Who the fact is about |
| `confidence` | 100% | Model confidence score |

**Episodic** (11,163 records):
| Field | Presence | Description |
|-------|----------|-------------|
| `event` | 100% | What happened |
| `who` | 100% | Participants |
| `confidence` | 92% | Model confidence score |
| `when` | 63% | Temporal reference (when available) |

**Procedural** (1,431 records):
| Field | Presence | Description |
|-------|----------|-------------|
| `procedure` | 100% | The procedure or instruction |
| `subject` | 100% | What it applies to |
| `confidence` | 100% | Model confidence score |
| `frequency` | 33% | How often it's done (when available) |

### Input Text Statistics

| Metric | Words |
|--------|-------|
| Min | 4 |
| P5 | 11 |
| Median | 26 |
| Mean | 37.6 |
| P95 | 58 |
| Max | 1,051 |

### Record Schema

```json
{
  "input": "<|semantic|> I started working at Google last month.",
  "context": [
    {"speaker": "Bob", "text": "How's the new job going?"}
  ],
  "type": "semantic",
  "output": {
    "fact": "Works at Google",
    "subject": "Alice",
    "confidence": 0.92
  }
}
```

- `input`: Type prefix (`<|semantic|>`, `<|episodic|>`, `<|procedural|>`) + conversation turn
- `context`: Surrounding turns for disambiguation
- `type`: Which extraction type to perform
- `output`: Structured JSON — schema varies by type (see above)

---

## 6. Data Integrity

### Leakage Check

Conversation-level splitting ensures no data leakage across splits:

| Overlap | Count |
|---------|-------|
| Train-Val conversation overlap | **0** |
| Train-Test conversation overlap | **0** |
| Val-Test conversation overlap | **0** |

All splits are done at the `conversation_id` level — if any turn from a conversation appears in the train set, no turn from that same conversation appears in val or test.

### Quality Controls

1. **Gold standard anchor**: 8,929 records (5,707 router + 3,222 extractor) come from human-annotated LoCoMo ground truth
2. **Teacher model quality**: gpt-oss-120b (120B thinking model) provides high-quality soft labels
3. **Deduplication**: Exact + embedding-based near-duplicate removal prevents memorization of repeated patterns
4. **Content filtering**: Minimum length, schema validation, empty-field removal
5. **Diverse sources**: Four distinct data sources (LoCoMo, MSC, code synthetic, teacher-generated) reduce distribution bias

---

## 7. File Manifest

### Labeled Data (`data/labeled/`)

| File | Records | Source |
|------|---------|--------|
| `locomo_gold_router.jsonl` | 5,707 | Script 01 |
| `locomo_gold_extractor.jsonl` | 3,222 | Script 01 |
| `locomo_teacher_router.jsonl` | 5,882 | Script 02 |
| `locomo_teacher_extractor.jsonl` | 6,612 | Script 03 |
| `msc_semantic.jsonl` | 5,200 | Script 04 |
| `msc_negative.jsonl` | 5,000 | Script 04 |
| `code_convos_labeled.jsonl` | 2,925 | Script 05 |
| `*_clean.jsonl` (7 files) | 30,671 | Script 06 |
| `augmented_clean.jsonl` | 65,282 | Script 08 |

### Training Data (`data/training/`)

| File | Records |
|------|---------|
| `router/train.jsonl` | 49,706 |
| `router/val.jsonl` | 10,024 |
| `router/test.jsonl` | 14,051 |
| `extractor/train.jsonl` | 32,368 |
| `extractor/val.jsonl` | 3,142 |
| `extractor/test.jsonl` | 8,273 |
| `metadata.json` | Statistics |

---

## 8. Known Limitations

1. **Procedural class imbalance**: Only 7.3% of router labels and 3.3% of extractor records are procedural. Code synthetic data helps but doesn't fully close the gap. Consider targeted procedural data generation.

2. **Augmentation ratio**: Name-swapped records make up ~65% of the dataset. While this improves name generalization, monitor for overfitting to the swapping distribution.

3. **MSC domain**: Multi-Session Chat data skews toward casual persona-sharing conversations. Real-world usage may include more technical, professional, or domain-specific content.

4. **Episodic temporal coverage**: Only 63% of episodic extractor records have a `when` field. The model may underperform on temporal extraction for events without explicit time references.

5. **Single teacher model**: All teacher labels come from gpt-oss-120b. Cross-validation with a second teacher model would increase label reliability.
