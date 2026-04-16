"""Benchmark SimHash, Uniform SQ, FlyHash, RandProj, RaBitQ on 4 additional models.

Fills in the missing cells of the full method × model matrix.
Saves R@10, Kendall's tau, and Pearson r to results/full_matrix.json.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau

from config import SEEDS, RECALL_K, TRAIN_SPLIT, EMBED_DIR, RESULTS_DIR
from quantize import (SimHashMultiBit, UniformScalarQuant,
                      FlyHash, RandProjQuant, RaBitQ)
from benchmark import split_train_eval

MODELS = ['dinov2', 'georsclip', 'ssl4eo', 'mae_base']
TOP_K = 1000


def _run_simhash(queries, database, d, bits, seed):
    h = SimHashMultiBit(d, bits, seed)
    qc = h.encode(queries).astype(np.float32)
    dc = h.encode(database).astype(np.float32)
    matches = qc @ dc.T + (1 - qc) @ (1 - dc).T
    sims = 2 * matches / h.k - 1
    return sims, h.bytes_per_vector()


def _run_uniform(queries, database, d, bits, seed):
    q = UniformScalarQuant(d, bits, seed, 'srht')
    qc = q.encode(queries)
    dc = q.encode(database)
    q_recon = q.decode(qc)
    db_recon = q.decode(dc)
    q_recon = q_recon / np.linalg.norm(q_recon, axis=1, keepdims=True)
    db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
    sims = q_recon @ db_recon.T
    return sims, q.bytes_per_vector()


def _run_flyhash(queries, database, d, bits, seed):
    h = FlyHash(d, bits, seed)
    qc = h.encode(queries).astype(np.float32)
    dc = h.encode(database).astype(np.float32)
    matches = qc @ dc.T + (1 - qc) @ (1 - dc).T
    sims = 2 * matches / h.m - 1
    return sims, h.bytes_per_vector()


def _run_randproj(queries, database, d, bits, seed):
    r = RandProjQuant(d, bits, seed)
    qc = r.encode(queries)
    dc = r.encode(database)
    q_recon = r.decode(qc)
    db_recon = r.decode(dc)
    q_recon = q_recon / np.linalg.norm(q_recon, axis=1, keepdims=True)
    db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
    sims = q_recon @ db_recon.T
    return sims, r.bytes_per_vector()


def _run_rabitq(queries, database, d, seed):
    rq = RaBitQ(d, seed)
    qc = rq.encode(queries).astype(np.float32)
    dc = rq.encode(database).astype(np.float32)
    matches = qc @ dc.T + (1 - qc) @ (1 - dc).T
    sims = 2 * matches / d - 1
    return sims, rq.bytes_per_vector()


def main():
    all_results = []
    for dataset in ['eurosat', 'bigearthnet']:
        for model_name in MODELS:
            emb_path = EMBED_DIR / f'{model_name}_{dataset}.npz'
            if not emb_path.exists():
                print(f'SKIP: {emb_path}')
                continue
            data = np.load(emb_path)
            embeddings = data['embeddings']
            d = embeddings.shape[1]
            print(f'\n{model_name}/{dataset}: {embeddings.shape}')

            for seed in SEEDS:
                train_emb, eval_emb, _ = split_train_eval(embeddings, TRAIN_SPLIT, seed)
                n_queries = min(1000, len(eval_emb) // 10)
                queries = eval_emb[:n_queries]
                database = eval_emb[n_queries:]

                fp32_sims = queries @ database.T
                gt_idx_r10 = np.argsort(-fp32_sims, axis=1)[:, :max(RECALL_K)]
                gt_idx_top_k = np.argsort(-fp32_sims, axis=1)[:, :TOP_K]
                n_rank = min(200, n_queries)

                method_runs = [
                    ('simhash_multi_4bit', 4, lambda: _run_simhash(queries, database, d, 4, seed)),
                    ('uniform_sq_4bit', 4, lambda: _run_uniform(queries, database, d, 4, seed)),
                    ('flyhash_4bit', 4, lambda: _run_flyhash(queries, database, d, 4, seed)),
                    ('randproj_quant_4bit', 4, lambda: _run_randproj(queries, database, d, 4, seed)),
                    ('rabitq', 1, lambda: _run_rabitq(queries, database, d, seed)),
                ]

                for method_name, bits, run_fn in method_runs:
                    sims, bpv = run_fn()
                    approx_idx = np.argsort(-sims, axis=1)[:, :max(RECALL_K)]
                    recall = {}
                    for k in RECALL_K:
                        gt_sets = [set(gt_idx_r10[i, :k]) for i in range(n_queries)]
                        ap_sets = [set(approx_idx[i, :k]) for i in range(n_queries)]
                        recall[str(k)] = float(np.mean(
                            [len(g & a) / k for g, a in zip(gt_sets, ap_sets)]))

                    taus, pearsons = [], []
                    for i in range(n_rank):
                        idx = gt_idx_top_k[i]
                        gt_scores = fp32_sims[i, idx]
                        ap_scores = sims[i, idx]
                        tau, _ = kendalltau(gt_scores, ap_scores)
                        if not np.isnan(tau):
                            taus.append(tau)
                        if np.std(ap_scores) > 1e-10:
                            r = np.corrcoef(gt_scores, ap_scores)[0, 1]
                            if not np.isnan(r):
                                pearsons.append(r)

                    all_results.append({
                        'model': model_name, 'dataset': dataset,
                        'method': method_name, 'bits': bits, 'seed': seed,
                        'd': d, 'recall': recall,
                        'bytes_per_vector': bpv,
                        'kendall_tau_mean': float(np.mean(taus)) if taus else 0,
                        'pearson_mean': float(np.mean(pearsons)) if pearsons else 0,
                    })

                print(f'  seed={seed}: {len(method_runs)} methods done')

    out = RESULTS_DIR / 'full_matrix.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved {len(all_results)} results to {out}')


if __name__ == '__main__':
    main()
