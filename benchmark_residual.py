"""Benchmark TurboQuant + PQ residual hybrid on EuroSAT."""

import json
import time
import numpy as np
from pathlib import Path

from config import SEEDS, RECALL_K, TRAIN_SPLIT, EMBED_DIR, RESULTS_DIR
from quantize import TurboQuantResidualPQ, FP32Exact
from benchmark import split_train_eval

CONFIGS = [
    {"name": "tq2_pq32", "tq_bits": 2, "pq_m": 32},
    {"name": "tq2_pq64", "tq_bits": 2, "pq_m": 64},
    {"name": "tq3_pq32", "tq_bits": 3, "pq_m": 32},
    {"name": "tq2_pq96", "tq_bits": 2, "pq_m": 96},
]

all_results = []

for model_name in ['prithvi', 'remoteclip']:
    emb_path = EMBED_DIR / f'{model_name}_eurosat.npz'
    if not emb_path.exists():
        print(f'SKIP: {emb_path}')
        continue
    data = np.load(emb_path)
    embeddings = data['embeddings']
    d = embeddings.shape[1]
    print(f'\n{"="*60}')
    print(f'{model_name}/eurosat: {embeddings.shape}')
    print(f'{"="*60}')

    for cfg in CONFIGS:
        for seed in SEEDS:
            train_emb, eval_emb, _ = split_train_eval(embeddings, TRAIN_SPLIT, seed)
            n_queries = min(1000, len(eval_emb) // 10)
            queries = eval_emb[:n_queries]
            database = eval_emb[n_queries:]

            hybrid = TurboQuantResidualPQ(
                d=d, tq_bits=cfg['tq_bits'], pq_m=cfg['pq_m'],
                seed=seed, rotation_type='srht'
            )

            # Train PQ on residuals from full training set (80%)
            t0 = time.perf_counter()
            hybrid.train(train_emb)
            train_time = time.perf_counter() - t0

            # Encode the database (eval minus queries)
            t0 = time.perf_counter()
            hybrid.encode(database)
            encode_time = time.perf_counter() - t0

            # Ground truth
            fp32 = FP32Exact(d)
            gt = fp32.search(queries, database, max(RECALL_K))

            # Hybrid search
            t0 = time.perf_counter()
            approx = hybrid.search(queries, max(RECALL_K))
            search_time = time.perf_counter() - t0

            recall = {}
            for k in RECALL_K:
                gt_sets = [set(gt[i, :k]) for i in range(n_queries)]
                ap_sets = [set(approx[i, :k]) for i in range(n_queries)]
                recall[str(k)] = float(np.mean(
                    [len(g & a) / k for g, a in zip(gt_sets, ap_sets)]))

            bpv = hybrid.bytes_per_vector()
            r = {
                'model': model_name, 'dataset': 'eurosat',
                'method': f'turboquant_residual_{cfg["name"]}',
                'config': cfg['name'],
                'tq_bits': cfg['tq_bits'], 'pq_m': cfg['pq_m'],
                'actual_pq_m': hybrid.actual_pq_m,
                'seed': seed, 'd': d, 'n_vectors': len(embeddings),
                'recall': recall,
                'bytes_per_vector': bpv,
                'train_time_s': train_time,
                'encode_time_ms': encode_time / len(database) * 1000,
                'search_time_s': search_time,
                'queries_per_sec': n_queries / search_time,
            }
            all_results.append(r)
            print(f'  {cfg["name"]} seed={seed}: '
                  f'R@1={recall["1"]:.3f} R@10={recall["10"]:.3f} '
                  f'R@100={recall["100"]:.3f} '
                  f'({bpv:.0f} B/vec, train={train_time:.1f}s)')

# Save
out = RESULTS_DIR / 'residual_hybrid_results.json'
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nSaved to {out}')

# Summary
print(f'\n{"="*80}')
print('SUMMARY (mean over 5 seeds)')
print(f'{"="*80}')
from collections import defaultdict
grouped = defaultdict(list)
for r in all_results:
    key = (r['model'], r['config'])
    grouped[key].append(r)

print(f'{"Model":<12} {"Config":<12} {"TQ bits":<8} {"PQ m":<6} '
      f'{"B/vec":<8} {"R@1":<10} {"R@10":<10} {"R@100":<10} {"Train(s)":<10}')
print('-' * 90)
for key in sorted(grouped.keys()):
    runs = grouped[key]
    model, config = key
    r1 = np.mean([r['recall']['1'] for r in runs])
    r10 = np.mean([r['recall']['10'] for r in runs])
    r100 = np.mean([r['recall']['100'] for r in runs])
    bpv = runs[0]['bytes_per_vector']
    tt = np.mean([r['train_time_s'] for r in runs])
    tq_bits = runs[0]['tq_bits']
    pq_m = runs[0]['actual_pq_m']
    print(f'{model:<12} {config:<12} {tq_bits:<8} {pq_m:<6} '
          f'{bpv:<8.0f} {r1:<10.3f} {r10:<10.3f} {r100:<10.3f} {tt:<10.1f}')
