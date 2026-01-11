#!/bin/bash
# Ablation study: Compare NER and keyword extraction impact on retrieval

set -e

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$BENCHMARK_DIR/results"
CHROMA_DB="$BENCHMARK_DIR/.chroma"

echo "=========================================="
echo "Memory System Ablation Study"
echo "=========================================="
echo ""

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# 1. Baseline: No NER, No Keywords (RRF + BM25 + Vector)
echo "1/4 Running baseline (RRF + BM25 + vector; no NER, no keywords)..."
rm -rf "$CHROMA_DB"
python "$BENCHMARK_DIR/evaluate.py" \
    --store chroma \
    --mode lite \
    --retrieval-mode rrf \
    --enable-bm25 \
    --disable-ner \
    --disable-keywords \
    --output "$RESULTS_DIR/evaluation_rrf_bm25_baseline.json"
echo "✓ Baseline complete"
echo ""

# 2. NER only (RRF + BM25 + Vector)
echo "2/4 Running with NER only (RRF + BM25 + vector)..."
rm -rf "$CHROMA_DB"
python "$BENCHMARK_DIR/evaluate.py" \
    --store chroma \
    --mode lite \
    --retrieval-mode rrf \
    --enable-bm25 \
    --disable-keywords \
    --output "$RESULTS_DIR/evaluation_rrf_bm25_ner.json"
echo "✓ NER only complete"
echo ""

# 3. Keywords only (RRF + BM25 + Vector)
echo "3/4 Running with keywords only (RRF + BM25 + vector)..."
rm -rf "$CHROMA_DB"
python "$BENCHMARK_DIR/evaluate.py" \
    --store chroma \
    --mode lite \
    --retrieval-mode rrf \
    --enable-bm25 \
    --disable-ner \
    --enable-keywords \
    --output "$RESULTS_DIR/evaluation_rrf_bm25_keywords.json"
echo "✓ Keywords only complete"
echo ""

# 4. Both NER and Keywords (RRF + BM25 + Vector)
echo "4/4 Running with both NER and keywords (RRF + BM25 + vector)..."
rm -rf "$CHROMA_DB"
python "$BENCHMARK_DIR/evaluate.py" \
    --store chroma \
    --mode lite \
    --retrieval-mode rrf \
    --enable-bm25 \
    --enable-keywords \
    --output "$RESULTS_DIR/evaluation_rrf_bm25_both.json"
echo "✓ Both NER + keywords complete"
echo ""

# Display results comparison
echo "=========================================="
echo "Results Summary"
echo "=========================================="
python - <<'EOF'
import json
from pathlib import Path

results_dir = Path(__file__).parent / "results" if Path(__file__).parent.name == "benchmarks" else Path("benchmarks/results")

files = {
    'baseline (no NER/keywords)': results_dir / 'evaluation_rrf_bm25_baseline.json',
    'NER only': results_dir / 'evaluation_rrf_bm25_ner.json',
    'keywords only': results_dir / 'evaluation_rrf_bm25_keywords.json',
    'both NER + keywords': results_dir / 'evaluation_rrf_bm25_both.json'
}

print(f'\n{"Config":<25} {"P@1":<8} {"R@10":<8} {"NDCG@10":<10} {"MRR":<8}')
print('-' * 65)

baseline_metrics = None
for name, path in files.items():
    with open(path) as f:
        data = json.load(f)
        metrics = data['summary']['metrics']
        p1 = metrics['precision@k']['1'] * 100
        r10 = metrics['recall@k']['10'] * 100
        ndcg10 = metrics['ndcg@k']['10'] * 100
        mrr = metrics['mrr'] * 100
        
        if baseline_metrics is None:
            baseline_metrics = {'p1': p1, 'r10': r10, 'ndcg10': ndcg10, 'mrr': mrr}
            print(f'{name:<25} {p1:>6.1f}%  {r10:>6.1f}%  {ndcg10:>8.1f}%  {mrr:>6.1f}%')
        else:
            p1_diff = p1 - baseline_metrics['p1']
            ndcg10_diff = ndcg10 - baseline_metrics['ndcg10']
            mrr_diff = mrr - baseline_metrics['mrr']
            print(f'{name:<25} {p1:>6.1f}%  {r10:>6.1f}%  {ndcg10:>8.1f}%  {mrr:>6.1f}%  '
                  f'(Δ: P@1 {p1_diff:+.1f}%, NDCG {ndcg10_diff:+.1f}%, MRR {mrr_diff:+.1f}%)')

print('\n✓ All evaluations complete. Results saved in benchmarks/results/')
EOF

echo ""
echo "=========================================="
echo "Ablation study complete!"
echo "=========================================="
