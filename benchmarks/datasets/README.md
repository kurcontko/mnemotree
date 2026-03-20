# Benchmark Datasets

Download scripts and instructions for each benchmark dataset.

## LoCoMo (already available)

Data is in `../../EasyLocomo/data/`:
- `locomo10_evidence.json` — 10 conversations
- `locomo10_qa.json` — 1,986 questions

## LongMemEval

```bash
git clone https://github.com/xiaowu0162/LongMemEval
cp LongMemEval/data/longmemeval_oracle.json datasets/
cp LongMemEval/data/longmemeval_s_cleaned.json datasets/
cp LongMemEval/data/longmemeval_m_cleaned.json datasets/
```

## PersonaMem

```bash
# Download from HuggingFace
huggingface-cli download bowen-upenn/PersonaMem-v2 --local-dir datasets/PersonaMem

# Expected files in datasets/PersonaMem/:
#   shared_context_32k.jsonl
#   shared_context_128k.jsonl
#   benchmark.csv
```

## PrefEval

```bash
git clone https://github.com/amazon-science/PrefEval
# Copy data files to datasets/PrefEval/
cp PrefEval/data/prefeval_explicit.json datasets/PrefEval/
cp PrefEval/data/prefeval_implicit.json datasets/PrefEval/
cp PrefEval/data/prefeval_conflict.json datasets/PrefEval/
```

## HaluMem

```bash
# Download from HuggingFace
huggingface-cli download IAAR-Shanghai/HaluMem --local-dir datasets/HaluMem

# Expected files in datasets/HaluMem/:
#   halumem_extract.json
#   halumem_update.json
#   halumem_qa.json
#   halumem_medium.json (optional)
#   halumem_long.json (optional)
```

## MemoryAgentBench

```bash
# Download from HuggingFace
huggingface-cli download ai-hyz/MemoryAgentBench --local-dir datasets/MemoryAgentBench

# Expected files in datasets/MemoryAgentBench/:
#   memoryagentbench_ar.json
#   memoryagentbench_ttl.json
#   memoryagentbench_lru.json
#   memoryagentbench_cr.json
#   memoryagentbench_eventqa.json (optional)
#   memoryagentbench_factconsolidation.json (optional)
```
