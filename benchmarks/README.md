# Benchmark data

`data/` holds a small synthetic memory corpus and labelled queries:

- `memories.jsonl`: 70 first-person memories.
- `test_queries.jsonl`: queries with the memories each one should retrieve.
- `entity_queries.jsonl`: entity-centred queries over the same corpus.

The evaluation scripts that used this data ran on the removed legacy `MemoryCore`.
Evidence's retrieval tooling lives in `mnemotree-evidence` (`evaluate`) and in
`src/mnemotree/evidence/evaluation.py`.
