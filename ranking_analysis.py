"""Ranking quality analysis: Kendall's tau and similarity magnitude preservation."""

import json
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
from pathlib import Path

from config import SEEDS, TRAIN_SPLIT, EMBED_DIR, RESULTS_DIR
from quantize import TurboQuantMSE, ProductQuantization, BinaryHash, FP32Exact
from benchmark import split_train_eval

RESULTS_DIR.mkdir(exist_ok=True)
TOP_K = 1000


def _get_tq_sims(queries, database, d, bits, seed):
    """Get TQ similarity matrix (re-normalized cosine)."""
    tq = TurboQuantMSE(d, bits, seed, 'srht')
    qc = tq.encode(queries)
    dc = tq.encode(database)
    q_recon = tq.decode(qc)
    db_recon = tq.decode(dc)
    q_recon = q_recon / np.linalg.norm(q_recon, axis=1, keepdims=True)
    db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
    return q_recon @ db_recon.T


def _get_pq_sims(queries, database, train_emb, d, bits):
    """Get PQ similarity via reconstruction."""
    import faiss
    m = max(1, (bits * d) // 8)
    while m > 0 and d % m != 0:
        m -= 1
    m = max(1, m)
    index = faiss.IndexPQ(d, m, 8)
    index.train(train_emb.astype(np.float32))
    index.add(database.astype(np.float32))
    n_db = len(database)
    db_recon = np.zeros((n_db, d), dtype=np.float32)
    for i in range(n_db):
        db_recon[i] = index.reconstruct(i)
    db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
    return queries @ db_recon.T


def _get_bh_sims(queries, database):
    """Get binary hash similarity (scaled Hamming)."""
    bh = BinaryHash()
    qc = bh.encode(queries)
    dc = bh.encode(database)
    q_f = qc.astype(np.float32)
    d_f = dc.astype(np.float32)
    dim = qc.shape[1]
    matches = q_f @ d_f.T + (1 - q_f) @ (1 - d_f).T
    return 2 * matches / dim - 1


ALL_MODELS = {
    'dinov2':     {'d': 768, 'training': 'self-distillation'},
    'remoteclip': {'d': 512, 'training': 'contrastive'},
    'georsclip':  {'d': 768, 'training': 'contrastive'},
    'ssl4eo':     {'d': 768, 'training': 'MAE (RS)'},
    'mae_base':   {'d': 768, 'training': 'MAE'},
    'prithvi':    {'d': 768, 'training': 'MAE (RS)'},
}

all_results = []

for dataset in ['eurosat', 'bigearthnet']:
    for model_name, cfg in ALL_MODELS.items():
        emb_path = EMBED_DIR / f'{model_name}_{dataset}.npz'
        if not emb_path.exists():
            continue
        data = np.load(emb_path)
        embeddings = data['embeddings']
        d = embeddings.shape[1]
        print(f'\n{model_name}/{dataset}: {embeddings.shape}')

        for seed in SEEDS[:3]:  # 3 seeds for speed
            train_emb, eval_emb, _ = split_train_eval(embeddings, TRAIN_SPLIT, seed)
            n_q = min(200, len(eval_emb) // 10)  # 200 queries for tau speed
            queries = eval_emb[:n_q]
            database = eval_emb[n_q:]

            # Ground truth: FP32 similarities
            fp32_sims = queries @ database.T  # (n_q, n_db)
            gt_top_k_idx = np.argsort(-fp32_sims, axis=1)[:, :TOP_K]

            # For each method, compute ranking metrics on the top-1000 neighborhood
            methods_to_test = {
                'turboquant_mse_4bit': lambda: _get_tq_sims(queries, database, d, 4, seed),
                'turboquant_mse_2bit': lambda: _get_tq_sims(queries, database, d, 2, seed),
                'product_quant_4bit': lambda: _get_pq_sims(queries, database, train_emb, d, 4),
                'binary_hash': lambda: _get_bh_sims(queries, database),
            }

            for method_name, sim_fn in methods_to_test.items():
                approx_sims = sim_fn()  # (n_q, n_db)

                taus = []
                pearsons = []
                for i in range(n_q):
                    idx = gt_top_k_idx[i]  # top-1000 indices for this query
                    gt_scores = fp32_sims[i, idx]  # FP32 scores for top-1000
                    ap_scores = approx_sims[i, idx]  # quantized scores for same 1000

                    # Kendall's tau
                    tau, _ = kendalltau(gt_scores, ap_scores)
                    if not np.isnan(tau):
                        taus.append(tau)

                    # Pearson correlation (similarity magnitude)
                    if np.std(ap_scores) > 1e-10:
                        r = np.corrcoef(gt_scores, ap_scores)[0, 1]
                        if not np.isnan(r):
                            pearsons.append(r)

                result = {
                    'model': model_name, 'dataset': dataset,
                    'method': method_name, 'seed': seed,
                    'training': cfg['training'],
                    'kendall_tau_mean': float(np.mean(taus)) if taus else 0,
                    'kendall_tau_std': float(np.std(taus)) if taus else 0,
                    'pearson_mean': float(np.mean(pearsons)) if pearsons else 0,
                    'pearson_std': float(np.std(pearsons)) if pearsons else 0,
                    'n_queries': n_q,
                }
                all_results.append(result)

            print(f'  seed={seed} done')


# Run
if __name__ == '__main__':
    # Save raw results
    out = RESULTS_DIR / 'ranking_analysis.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {out}')

    # Aggregate and print
    print(f'\n{"="*100}')
    print('RANKING QUALITY ANALYSIS (top-1000 neighborhood)')
    print(f'{"="*100}')

    grouped = defaultdict(list)
    for r in all_results:
        key = (r['model'], r['dataset'], r['method'])
        grouped[key].append(r)

    for dataset in ['eurosat', 'bigearthnet']:
        print(f'\n--- {dataset.upper()} ---')
        print(f'{"Model":<12} {"Method":<25} {"Kendall tau":<15} {"Pearson r":<15}')
        print('-' * 70)
        for model in ['dinov2', 'remoteclip', 'georsclip', 'ssl4eo', 'mae_base', 'prithvi']:
            for method in ['turboquant_mse_4bit', 'product_quant_4bit', 'binary_hash']:
                key = (model, dataset, method)
                if key not in grouped:
                    continue
                runs = grouped[key]
                tau = np.mean([r['kendall_tau_mean'] for r in runs])
                pear = np.mean([r['pearson_mean'] for r in runs])
                print(f'{model:<12} {method:<25} {tau:<15.3f} {pear:<15.3f}')
            print()
