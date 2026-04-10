"""Phase 0: Validate TurboQuant implementation against text embeddings.

Reproduces a subset of the paper's results on a standard text embedding
model to confirm the implementation is correct before running RS experiments.
"""

import numpy as np
import time
from pathlib import Path

from config import RESULTS_DIR, BITS, RECALL_K
from quantize import TurboQuantMSE, BinaryHash, FP32Exact
from utils import l2_normalize, filter_zero_norm


def generate_text_embeddings(n: int = 10_000, d: int = 768, seed: int = 42) -> np.ndarray:
    """Generate synthetic embeddings that mimic text model outputs.

    For a real sanity check, replace this with actual BERT/OpenAI embeddings.
    The key property: high-dimensional vectors that, after normalization,
    should satisfy the Beta assumption when rotated.

    To use real embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = [...]  # load any text corpus
        embeddings = model.encode(sentences)
    """
    rng = np.random.RandomState(seed)
    # Simulate: Gaussian + some structure (clustered)
    n_clusters = 20
    centers = rng.randn(n_clusters, d).astype(np.float32)
    labels = rng.randint(0, n_clusters, size=n)
    noise = rng.randn(n, d).astype(np.float32) * 0.3
    embeddings = centers[labels] + noise

    # Normalize
    embeddings, _ = filter_zero_norm(embeddings)
    embeddings, norms = l2_normalize(embeddings)
    return embeddings


def run_sanity_check():
    """Verify TurboQuant works as expected on synthetic/text data."""
    print("=" * 60)
    print("SANITY CHECK: TurboQuant on synthetic text-like embeddings")
    print("=" * 60)

    d = 768
    embeddings = generate_text_embeddings(n=10_000, d=d)
    print(f"Generated {embeddings.shape[0]} embeddings, d={d}")

    n_queries = 200
    queries = embeddings[:n_queries]
    database = embeddings[n_queries:]

    # Ground truth
    fp32 = FP32Exact(d)
    gt_indices = fp32.search(queries, database, max(RECALL_K))

    print(f"\n{'Method':<25} {'Bits':<6} {'R@1':<10} {'R@10':<10} {'R@100':<10} {'ms/vec':<10}")
    print("-" * 75)

    # Binary hash baseline
    bh = BinaryHash()
    q_codes = bh.encode(queries)
    db_codes = bh.encode(database)
    approx = bh.search(q_codes, db_codes, max(RECALL_K))
    for k in RECALL_K:
        gt_sets = [set(gt_indices[i, :k]) for i in range(n_queries)]
        ap_sets = [set(approx[i, :k]) for i in range(n_queries)]
        recall = np.mean([len(g & a) / k for g, a in zip(gt_sets, ap_sets)])
        if k == 1:
            r1 = recall
        elif k == 10:
            r10 = recall
        else:
            r100 = recall
    print(f"{'binary_hash':<25} {'-':<6} {r1:<10.3f} {r10:<10.3f} {r100:<10.3f} {'N/A':<10}")

    # TurboQuant at each bit width
    for bits in BITS:
        tq = TurboQuantMSE(d, bits, seed=42, rotation_type="srht")

        t0 = time.perf_counter()
        q_codes = tq.encode(queries)
        db_codes = tq.encode(database)
        encode_time = (time.perf_counter() - t0) / (n_queries + len(database)) * 1000

        approx = tq.search(q_codes, db_codes, max(RECALL_K))

        recalls = {}
        for k in RECALL_K:
            gt_sets = [set(gt_indices[i, :k]) for i in range(n_queries)]
            ap_sets = [set(approx[i, :k]) for i in range(n_queries)]
            recalls[k] = np.mean([len(g & a) / k for g, a in zip(gt_sets, ap_sets)])

        print(f"{'turboquant_mse':<25} {bits:<6} "
              f"{recalls[1]:<10.3f} {recalls[10]:<10.3f} {recalls[100]:<10.3f} "
              f"{encode_time:<10.3f}")

    # Expected behavior:
    # - 4 bits: recall@10 should be > 0.95
    # - 3 bits: recall@10 should be > 0.85
    # - 2 bits: recall@10 should be > 0.60
    # If these thresholds aren't met, the implementation likely has a bug.

    print("\nExpected (approximate, from paper):")
    print("  4 bits: R@10 > 0.95")
    print("  3 bits: R@10 > 0.85")
    print("  2 bits: R@10 > 0.60")
    print("\nIf results are significantly below these, check the implementation.")


if __name__ == "__main__":
    run_sanity_check()
