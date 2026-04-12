"""Phase 3: Run compression benchmark across all configurations."""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from config import (MODELS, DATASETS, METHODS, BITS, SEEDS, RECALL_K,
                    TRAIN_SPLIT, EMBED_DIR, RESULTS_DIR)
from quantize import (TurboQuantMSE, TurboQuantProd, TurboQuantAdaptive,
                      BinaryHash, ProductQuantization, FP32Exact,
                      UniformScalarQuant, SimHashMultiBit, RandProjQuant, FlyHash,
                      RaBitQ)
from utils import recall_at_k


def split_train_eval(embeddings: np.ndarray, train_ratio: float, seed: int):
    """Split embeddings into train (for PQ) and eval sets."""
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    indices = rng.permutation(n)
    split = int(n * train_ratio)
    train_idx, eval_idx = indices[:split], indices[split:]
    return embeddings[train_idx], embeddings[eval_idx], eval_idx


def run_single_config(model_name: str, dataset_name: str, method: str,
                      bits: int, seed: int, embeddings: np.ndarray) -> dict:
    """Run one experimental configuration. Returns metrics dict."""
    d = embeddings.shape[1]
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "method": method,
        "bits": bits,
        "seed": seed,
        "d": d,
        "n_vectors": len(embeddings),
    }

    # Split for PQ training
    train_emb, eval_emb, _ = split_train_eval(embeddings, TRAIN_SPLIT, seed)

    # Use subset of eval as queries (first 1000 or 10% whichever is smaller)
    n_queries = min(1000, len(eval_emb) // 10)
    queries = eval_emb[:n_queries]
    database = eval_emb[n_queries:]

    if method == "fp32_exact":
        fp32 = FP32Exact(d)
        t0 = time.perf_counter()
        gt_indices = fp32.search(queries, database, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): 1.0 for k in RECALL_K}  # exact is ground truth
        result["bytes_per_vector"] = fp32.bytes_per_vector()
        result["encode_time_ms"] = 0.0
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "binary_hash":
        bh = BinaryHash()

        t0 = time.perf_counter()
        q_codes = bh.encode(queries)
        db_codes = bh.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return bh.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)
        t0 = time.perf_counter()
        bh.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = bh.bytes_per_vector(d)
        result["encode_time_ms"] = encode_time / len(database) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "product_quant":
        pq = ProductQuantization(d, bits)

        t0 = time.perf_counter()
        pq.train(train_emb)
        pq.encode(database)
        encode_time = time.perf_counter() - t0

        # Search
        t0 = time.perf_counter()
        approx_indices = pq.search(queries, max(RECALL_K))
        search_time = time.perf_counter() - t0

        # Compute recall vs FP32 ground truth
        fp32 = FP32Exact(d)
        gt_indices = fp32.search(queries, database, max(RECALL_K))

        recall = {}
        for k in RECALL_K:
            gt_sets = [set(gt_indices[i, :k]) for i in range(n_queries)]
            ap_sets = [set(approx_indices[i, :k]) for i in range(n_queries)]
            recall[str(k)] = float(np.mean([len(g & a) / k for g, a in zip(gt_sets, ap_sets)]))

        result["recall"] = recall
        result["bytes_per_vector"] = pq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / len(database) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "turboquant_mse":
        tq = TurboQuantMSE(d, bits, seed, rotation_type="srht")

        t0 = time.perf_counter()
        q_codes = tq.encode(queries)
        db_codes = tq.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return tq.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)

        t0 = time.perf_counter()
        tq.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = tq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "turboquant_prod":
        # Prod is identical to MSE for cosine-similarity retrieval
        # (QJL correction is counterproductive — see quantize.py docstring)
        tq = TurboQuantProd(d, bits, seed, rotation_type="srht")

        t0 = time.perf_counter()
        q_codes = tq.encode(queries)
        db_codes = tq.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return tq.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)

        t0 = time.perf_counter()
        tq.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = tq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method in ("turboquant_ada", "turboquant_ada_w"):
        whiten = method == "turboquant_ada_w"
        tq = TurboQuantAdaptive(d, bits, seed, rotation_type="srht", whiten=whiten)

        # Train codebook on training split
        tq.train(train_emb)

        t0 = time.perf_counter()
        q_codes = tq.encode(queries)
        db_codes = tq.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return tq.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)

        t0 = time.perf_counter()
        tq.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = tq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "uniform_sq":
        usq = UniformScalarQuant(d, bits, seed, rotation_type="srht")

        t0 = time.perf_counter()
        q_codes = usq.encode(queries)
        db_codes = usq.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return usq.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)
        t0 = time.perf_counter()
        usq.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = usq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method in ("simhash_multi", "flyhash"):
        if method == "simhash_multi":
            hasher = SimHashMultiBit(d, bits, seed)
        else:
            hasher = FlyHash(d, bits, seed)

        t0 = time.perf_counter()
        q_codes = hasher.encode(queries)
        db_codes = hasher.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return hasher.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)
        t0 = time.perf_counter()
        hasher.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = hasher.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "randproj_quant":
        rpq = RandProjQuant(d, bits, seed)

        t0 = time.perf_counter()
        q_codes = rpq.encode(queries)
        db_codes = rpq.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return rpq.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)
        t0 = time.perf_counter()
        rpq.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = rpq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    elif method == "rabitq":
        rq = RaBitQ(d, seed)

        t0 = time.perf_counter()
        q_codes = rq.encode(queries)
        db_codes = rq.encode(database)
        encode_time = time.perf_counter() - t0

        def search_fn(qc, dbc, k):
            return rq.search(qc, dbc, k)

        recall = recall_at_k(queries, database, q_codes, db_codes, search_fn, RECALL_K)

        t0 = time.perf_counter()
        rq.search(q_codes, db_codes, max(RECALL_K))
        search_time = time.perf_counter() - t0

        result["recall"] = {str(k): v for k, v in recall.items()}
        result["bytes_per_vector"] = rq.bytes_per_vector()
        result["encode_time_ms"] = encode_time / (len(queries) + len(database)) * 1000
        result["search_time_s"] = search_time
        result["queries_per_sec"] = n_queries / search_time
        return result

    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Run TurboQuant RS benchmark")
    parser.add_argument("--model", choices=["prithvi", "remoteclip", "all"], default="all")
    parser.add_argument("--dataset", choices=["bigearthnet", "eurosat", "all"], default="all")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    parser.add_argument("--bits", type=int, nargs="+", default=BITS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    methods = METHODS if args.method == "all" else [args.method]

    all_results = []
    total_configs = (len(models) * len(datasets) * len(methods) *
                     len(args.bits) * len(args.seeds))
    done = 0

    for model_name in models:
        for dataset_name in datasets:
            # Load embeddings
            emb_path = EMBED_DIR / f"{model_name}_{dataset_name}.npz"
            if not emb_path.exists():
                print(f"SKIP: {emb_path} not found. Run extract.py first.")
                continue

            data = np.load(emb_path)
            embeddings = data["embeddings"]
            print(f"\nLoaded {model_name}/{dataset_name}: {embeddings.shape}")

            # Track PQ configs to skip duplicates (e.g. d=512: 2-bit and 3-bit
            # both map to m=128 because 512 has no divisor between 128 and 256)
            pq_seen = set()

            for method in methods:
                for bits in args.bits:
                    # Skip bits for methods that don't use it
                    if method in ("fp32_exact", "binary_hash", "rabitq") and bits != args.bits[0]:
                        continue

                    # Deduplicate PQ configs that map to identical (m, nbits)
                    if method == "product_quant":
                        pq = ProductQuantization(embeddings.shape[1], bits)
                        pq_key = (pq.m, pq.nbits_per_subspace)
                        if pq_key in pq_seen:
                            print(f"  SKIP: product_quant @ {bits}bit "
                                  f"(same as previous: m={pq.m}, nbits={pq.nbits_per_subspace})")
                            continue
                        pq_seen.add(pq_key)

                    for seed in args.seeds:
                        done += 1
                        print(f"  [{done}/{total_configs}] {method} @ {bits}bit, seed={seed}...",
                              end=" ", flush=True)

                        result = run_single_config(
                            model_name, dataset_name, method, bits, seed, embeddings
                        )
                        all_results.append(result)

                        r1 = result["recall"].get("1", result["recall"].get(1, "N/A"))
                        r10 = result["recall"].get("10", result["recall"].get(10, "N/A"))
                        print(f"R@1={r1:.3f} R@10={r10:.3f} "
                              f"({result['bytes_per_vector']:.1f} B/vec)")

    # Save all results
    out_path = RESULTS_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
