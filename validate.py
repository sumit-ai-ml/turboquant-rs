"""Phase 2: Validate Beta(d/2, d/2) assumption on RS embeddings."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from pathlib import Path

from config import MODELS, DATASETS, EMBED_DIR, RESULTS_DIR, FIGURES_DIR, SEEDS
from utils import random_orthogonal, srht_matrix, apply_rotation, verify_rotation
from utils import beta_ks_test, coordinate_independence_check


def validate_beta_assumption(embeddings: np.ndarray, d: int, seed: int,
                             rotation_type: str = "srht") -> dict:
    """Run full Beta assumption validation on one embedding set + seed."""
    # Build rotation matrix
    if rotation_type == "srht":
        R = srht_matrix(d, seed)
    else:
        R = random_orthogonal(d, seed)

    assert verify_rotation(R), f"Rotation matrix not orthogonal (seed={seed})"

    # Rotate
    rotated = apply_rotation(embeddings, R)

    # KS test on sampled coordinates (test 50 coordinates, not all d)
    rng = np.random.RandomState(seed)
    coord_indices = rng.choice(d, size=min(50, d), replace=False)

    ks_results = []
    for idx in coord_indices:
        result = beta_ks_test(rotated[:, idx], d)
        result["coordinate"] = int(idx)
        ks_results.append(result)

    # Aggregate KS statistics
    d_stats = [r["D_statistic"] for r in ks_results]

    # Independence check
    indep = coordinate_independence_check(rotated)

    return {
        "seed": seed,
        "rotation_type": rotation_type,
        "d": d,
        "n_vectors": len(embeddings),
        "ks_mean_D": float(np.mean(d_stats)),
        "ks_max_D": float(np.max(d_stats)),
        "ks_min_D": float(np.min(d_stats)),
        "ks_std_D": float(np.std(d_stats)),
        "fit_quality": categorize_fit(np.mean(d_stats)),
        "independence": indep,
        "per_coordinate": ks_results,
    }


def categorize_fit(mean_d: float) -> str:
    if mean_d < 0.01:
        return "EXCELLENT"
    elif mean_d < 0.02:
        return "GOOD"
    elif mean_d < 0.05:
        return "MODERATE"
    else:
        return "POOR"


def plot_qq(embeddings: np.ndarray, d: int, seed: int, model_name: str,
            dataset_name: str, n_coords: int = 6):
    """QQ plots of rotated coordinates vs Beta(d/2, d/2)."""
    R = srht_matrix(d, seed)
    rotated = apply_rotation(embeddings, R)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    rng = np.random.RandomState(seed)
    coord_indices = rng.choice(d, size=n_coords, replace=False)

    a, b = d / 2.0, d / 2.0
    theoretical_quantiles = np.linspace(0.001, 0.999, 500)
    theoretical_values = beta_dist.ppf(theoretical_quantiles, a, b)
    # Shift back to [-1, 1]
    theoretical_values = theoretical_values * 2 - 1

    for ax, idx in zip(axes.flat, coord_indices):
        data = np.sort(rotated[:, idx])
        # Subsample for plotting
        sample_indices = np.linspace(0, len(data) - 1, 500, dtype=int)
        sample = data[sample_indices]

        ax.scatter(theoretical_values, sample, s=1, alpha=0.5)
        lims = [min(theoretical_values.min(), sample.min()),
                max(theoretical_values.max(), sample.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Beta(d/2, d/2) quantiles")
        ax.set_ylabel("Observed quantiles")
        ax.set_title(f"Coord {idx}")

    fig.suptitle(f"QQ Plot: {model_name}/{dataset_name} (d={d}, seed={seed})")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"qq_{model_name}_{dataset_name}_seed{seed}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved QQ plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Beta assumption")
    parser.add_argument("--model", choices=["prithvi", "remoteclip", "all"], default="all")
    parser.add_argument("--dataset", choices=["bigearthnet", "eurosat", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_results = []

    for model_name in models:
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Validating Beta assumption: {model_name} / {dataset_name}")
            print(f"{'='*60}")

            # Load embeddings
            emb_path = EMBED_DIR / f"{model_name}_{dataset_name}.npz"
            if not emb_path.exists():
                print(f"  SKIP: {emb_path} not found. Run extract.py first.")
                continue

            data = np.load(emb_path)
            embeddings = data["embeddings"]
            d = embeddings.shape[1]
            print(f"  Loaded {embeddings.shape[0]} embeddings, d={d}")

            for seed in args.seeds:
                print(f"\n  Seed {seed}:")

                # Run validation with both rotation types
                for rot_type in ["srht", "dense"]:
                    result = validate_beta_assumption(embeddings, d, seed, rot_type)
                    result["model"] = model_name
                    result["dataset"] = dataset_name
                    all_results.append(result)

                    fit = result["fit_quality"]
                    mean_d = result["ks_mean_D"]
                    max_d = result["ks_max_D"]
                    indep = result["independence"]["mean_abs_correlation"]
                    print(f"    [{rot_type:>5}] KS D: {mean_d:.4f} (max {max_d:.4f}) "
                          f"| Fit: {fit} | Independence: {indep:.4f}")

                # QQ plot (SRHT only, first seed)
                if seed == args.seeds[0]:
                    plot_qq(embeddings, d, seed, model_name, dataset_name)

    # Save results
    out_path = RESULTS_DIR / "beta_validation.json"
    # Strip per-coordinate details for summary file
    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k != "per_coordinate"}
        summary.append(s)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("BETA ASSUMPTION VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Dataset':<14} {'Rotation':<8} {'Mean D':<10} {'Max D':<10} {'Fit':<12} {'Indep':<8}")
    print("-" * 80)
    for r in summary:
        print(f"{r['model']:<12} {r['dataset']:<14} {r['rotation_type']:<8} "
              f"{r['ks_mean_D']:<10.4f} {r['ks_max_D']:<10.4f} {r['fit_quality']:<12} "
              f"{r['independence']['mean_abs_correlation']:<8.4f}")


if __name__ == "__main__":
    main()
