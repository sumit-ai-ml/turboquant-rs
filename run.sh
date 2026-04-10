#!/bin/bash
# TurboQuant RS Benchmark — Full Pipeline
# Run each phase sequentially. Stop on first error.
set -e

echo "============================================"
echo "Phase 0: Sanity check (synthetic data)"
echo "============================================"
python sanity_check.py

echo ""
echo "============================================"
echo "Phase 1: Extract embeddings"
echo "============================================"
python extract.py --model all --dataset all

echo ""
echo "============================================"
echo "Phase 2: Validate Beta assumption"
echo "============================================"
python validate.py --model all --dataset all

echo ""
echo "============================================"
echo "Phase 3: Run benchmark"
echo "============================================"
python benchmark.py --model all --dataset all --method all

echo ""
echo "============================================"
echo "Phase 4: Analyze results"
echo "============================================"
python analyze.py

echo ""
echo "============================================"
echo "DONE. Results in results/, figures in figures/"
echo "============================================"
