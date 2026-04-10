"""Phase 0: Validate TurboQuant implementation against text embeddings.

Reproduces a subset of the paper's results on a standard text embedding
model to confirm the implementation is correct before running RS experiments.
"""

import numpy as np
import time
from pathlib import Path

from config import RESULTS_DIR, BITS, RECALL_K
from quantize import TurboQuantMSE, BinaryHash, FP32Exact
from utils import l2_normalize, filter_zero_norm, SRHTRotation, make_rotation, apply_rotation, random_orthogonal


def run_benchmark(embeddings: np.ndarray, label: str):
    """Run recall benchmark on a given embedding set."""
    d = embeddings.shape[1]
    n_queries = min(200, len(embeddings) // 10)
    queries = embeddings[:n_queries]
    database = embeddings[n_queries:]

    # Ground truth
    fp32 = FP32Exact(d)
    gt_indices = fp32.search(queries, database, max(RECALL_K))

    # Check neighbor gap (how hard is this dataset?)
    sims = queries[:10] @ database.T
    sorted_sims = np.sort(-sims, axis=1)
    gap_1_2 = np.mean(sorted_sims[:, 1] - sorted_sims[:, 0])
    gap_10_11 = np.mean(sorted_sims[:, 10] - sorted_sims[:, 9])
    print(f"  Neighbor gap: 1st-2nd={-gap_1_2:.6f}, 10th-11th={-gap_10_11:.6f}")

    print(f"\n  {'Method':<22} {'Bits':<6} {'R@1':<10} {'R@10':<10} {'R@100':<10} {'MSE/dim':<12}")
    print(f"  {'-'*72}")

    # Binary hash baseline
    bh = BinaryHash()
    q_codes = bh.encode(queries)
    db_codes = bh.encode(database)
    approx = bh.search(q_codes, db_codes, max(RECALL_K))
    recalls = {}
    for k in RECALL_K:
        gt_sets = [set(gt_indices[i, :k]) for i in range(n_queries)]
        ap_sets = [set(approx[i, :k]) for i in range(n_queries)]
        recalls[k] = np.mean([len(g & a) / k for g, a in zip(gt_sets, ap_sets)])
    print(f"  {'binary_hash':<22} {'-':<6} "
          f"{recalls[1]:<10.3f} {recalls[10]:<10.3f} {recalls[100]:<10.3f}")

    # TurboQuant
    for rot_type in ["dense"]:
        for bits in BITS:
            tq = TurboQuantMSE(d, bits, seed=42, rotation_type=rot_type)

            q_codes = tq.encode(queries)
            db_codes = tq.encode(database)

            # Reconstruction quality
            recon_q = tq.decode(q_codes)
            mse_per_dim = np.mean((queries - recon_q) ** 2)

            # Inner product preservation (the metric that actually matters)
            orig_sims = queries[:20] @ database[:100].T
            recon_sims = tq.decode(q_codes[:20]) @ tq.decode(db_codes[:100]).T
            ip_corr = np.corrcoef(orig_sims.ravel(), recon_sims.ravel())[0, 1]

            approx = tq.search(q_codes, db_codes, max(RECALL_K))
            recalls = {}
            for k in RECALL_K:
                gt_sets = [set(gt_indices[i, :k]) for i in range(n_queries)]
                ap_sets = [set(approx[i, :k]) for i in range(n_queries)]
                recalls[k] = np.mean([len(g & a) / k for g, a in zip(gt_sets, ap_sets)])

            name = f"tq_mse_{rot_type}"
            print(f"  {name:<22} {bits:<6} "
                  f"{recalls[1]:<10.3f} {recalls[10]:<10.3f} {recalls[100]:<10.3f} "
                  f"{mse_per_dim:<12.6f} ip_corr={ip_corr:.4f}")


def run_sanity_check():
    """Verify TurboQuant works as expected on synthetic data."""
    rng = np.random.RandomState(42)

    # ===== TEST 1: Isotropic random vectors (easiest case) =====
    print("=" * 70)
    print("TEST 1: Isotropic random vectors (d=768, n=5000)")
    print("  Best case for TurboQuant — uniform on the sphere.")
    print("=" * 70)
    iso = rng.randn(5000, 768).astype(np.float32)
    iso = iso / np.linalg.norm(iso, axis=1, keepdims=True)
    run_benchmark(iso, "isotropic")

    # ===== TEST 2: Mildly clustered (realistic) =====
    print(f"\n{'=' * 70}")
    print("TEST 2: 50 clusters, high noise (d=768, n=5000)")
    print("  Simulates real embeddings with spread-out clusters.")
    print("=" * 70)
    centers = rng.randn(50, 768).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 5
    labels = rng.randint(0, 50, size=5000)
    clustered = centers[labels] + rng.randn(5000, 768).astype(np.float32) * 1.0
    clustered = clustered / np.linalg.norm(clustered, axis=1, keepdims=True)
    run_benchmark(clustered, "clustered_mild")

    # ===== TEST 3: Tightly clustered (hard case) =====
    print(f"\n{'=' * 70}")
    print("TEST 3: 20 clusters, low noise (d=768, n=10000)")
    print("  Hard case — many near-equidistant neighbors within clusters.")
    print("=" * 70)
    centers2 = rng.randn(20, 768).astype(np.float32)
    labels2 = rng.randint(0, 20, size=10000)
    tight = centers2[labels2] + rng.randn(10000, 768).astype(np.float32) * 0.3
    tight = tight / np.linalg.norm(tight, axis=1, keepdims=True)
    run_benchmark(tight, "clustered_tight")

    # ===== Rotation quality check =====
    print(f"\n{'=' * 70}")
    print("ROTATION QUALITY CHECK")
    print("=" * 70)
    test_vecs = rng.randn(100, 768).astype(np.float32)
    test_vecs = test_vecs / np.linalg.norm(test_vecs, axis=1, keepdims=True)

    for rot_type in ["dense", "srht_512"]:
        if rot_type == "dense":
            R = random_orthogonal(768, 42)
            rotated = test_vecs @ R.T
            restored = rotated @ R
        else:
            # SRHT only works for power-of-2; test with d=512
            test_512 = rng.randn(100, 512).astype(np.float32)
            test_512 = test_512 / np.linalg.norm(test_512, axis=1, keepdims=True)
            R = SRHTRotation(512, 42)
            rotated = R.forward(test_512)
            restored = R.inverse(rotated)
            test_vecs = test_512  # use 512-d for this check

        # Check inner product preservation
        orig_dots = test_vecs[:10] @ test_vecs[10:20].T
        rot_dots = rotated[:10] @ rotated[10:20].T
        dot_err = np.max(np.abs(orig_dots - rot_dots))

        # Check round-trip
        roundtrip_err = np.max(np.abs(test_vecs - restored))

        # Check norm preservation
        orig_norms = np.linalg.norm(test_vecs, axis=1)
        rot_norms = np.linalg.norm(rotated, axis=1)
        norm_err = np.max(np.abs(orig_norms - rot_norms))

        print(f"  {rot_type:<8}: dot_err={dot_err:.6f}  roundtrip_err={roundtrip_err:.6f}  norm_err={norm_err:.6f}")

    # ===== Codebook check =====
    print(f"\n{'=' * 70}")
    print("CODEBOOK CHECK (d=768, 4 bits)")
    print("=" * 70)
    tq = TurboQuantMSE(768, 4, seed=42, rotation_type="dense")
    bounds_01 = (tq.boundaries + 1.0) / 2.0
    cents_01 = (tq.centroids + 1.0) / 2.0
    print(f"  Boundaries (in [0,1]): min={bounds_01[1]:.4f} max={bounds_01[-2]:.4f}")
    print(f"  Centroids  (in [0,1]): min={cents_01[0]:.4f} max={cents_01[-1]:.4f}")
    print(f"  Useful range: [{bounds_01[1]:.4f}, {bounds_01[-2]:.4f}]")

    # Check: what fraction of rotated data falls in the useful range?
    R = random_orthogonal(768, 42)
    rotated = iso @ R.T
    shifted = (rotated + 1.0) / 2.0
    in_range = np.mean((shifted >= bounds_01[1]) & (shifted <= bounds_01[-2]))
    print(f"  Fraction of rotated data in useful range: {in_range:.4f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("  If R@10 is high for Test 1 (isotropic) but low for Test 3 (tight):")
    print("    -> Code works. Tight clusters have near-equidistant neighbors.")
    print("  If R@10 is low even for Test 1:")
    print("    -> Codebook or quantization bug. Check codebook range vs data range.")
    print("  If rotation checks show large errors:")
    print("    -> Rotation implementation bug.")


if __name__ == "__main__":
    run_sanity_check()
