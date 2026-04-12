"""Phase 4: Analyze results, produce tables and figures."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, RECALL_K, BITS, METHODS


def load_results() -> list[dict]:
    with open(RESULTS_DIR / "benchmark_results.json") as f:
        return json.load(f)


def aggregate_over_seeds(results: list[dict]) -> dict:
    """Group by (model, dataset, method, bits) and compute mean +/- std across seeds."""
    groups = defaultdict(list)
    for r in results:
        key = (r["model"], r["dataset"], r["method"], r["bits"])
        groups[key].append(r)

    aggregated = {}
    for key, runs in groups.items():
        agg = {
            "model": key[0],
            "dataset": key[1],
            "method": key[2],
            "bits": key[3],
            "n_seeds": len(runs),
            "bytes_per_vector": runs[0]["bytes_per_vector"],
        }
        for k in RECALL_K:
            vals = [r["recall"][str(k)] for r in runs]
            agg[f"recall@{k}_mean"] = float(np.mean(vals))
            agg[f"recall@{k}_std"] = float(np.std(vals))

        speed_vals = [r["queries_per_sec"] for r in runs]
        agg["qps_mean"] = float(np.mean(speed_vals))
        agg["qps_std"] = float(np.std(speed_vals))

        encode_vals = [r["encode_time_ms"] for r in runs]
        agg["encode_ms_mean"] = float(np.mean(encode_vals))

        aggregated[key] = agg

    return aggregated


def print_results_table(aggregated: dict):
    """Print formatted results table."""
    print(f"\n{'='*110}")
    print("TURBOQUANT RS BENCHMARK RESULTS (mean +/- std across 5 seeds)")
    print(f"{'='*110}")

    for model in ["prithvi", "remoteclip"]:
        for dataset in ["bigearthnet", "eurosat"]:
            print(f"\n--- {model.upper()} / {dataset.upper()} ---")
            print(f"{'Method':<20} {'Bits':<6} {'B/vec':<8} "
                  f"{'R@1':<16} {'R@10':<16} {'R@100':<16} {'QPS':<10}")
            print("-" * 110)

            for method in METHODS:
                for bits in BITS:
                    key = (model, dataset, method, bits)
                    if key not in aggregated:
                        # fp32/binary don't vary by bits
                        if method in ("fp32_exact", "binary_hash"):
                            key = (model, dataset, method, BITS[0])
                            if key not in aggregated:
                                continue
                        else:
                            continue

                    a = aggregated[key]
                    r1 = f"{a['recall@1_mean']:.3f}+/-{a['recall@1_std']:.3f}"
                    r10 = f"{a['recall@10_mean']:.3f}+/-{a['recall@10_std']:.3f}"
                    r100 = f"{a['recall@100_mean']:.3f}+/-{a['recall@100_std']:.3f}"
                    qps = f"{a['qps_mean']:.0f}"
                    bpv = f"{a['bytes_per_vector']:.1f}"

                    bits_str = "-" if method in ("fp32_exact", "binary_hash") else str(bits)
                    print(f"{method:<20} {bits_str:<6} {bpv:<8} {r1:<16} {r10:<16} {r100:<16} {qps:<10}")

                    # Only print once for methods that don't vary by bits
                    if method in ("fp32_exact", "binary_hash"):
                        break


def plot_recall_vs_bits(aggregated: dict):
    """Plot recall@10 vs bits for each model/dataset combination."""
    # Only plot combos that have data
    all_combos = [
        ("prithvi", "bigearthnet"),
        ("prithvi", "eurosat"),
        ("remoteclip", "bigearthnet"),
        ("remoteclip", "eurosat"),
    ]
    combos = [(m, d) for m, d in all_combos
              if any((m, d, method, bits) in aggregated
                     for method in METHODS for bits in BITS)]
    n = len(combos)
    if n == 0:
        print("No data to plot.")
        return
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    axes = axes.flat

    quantized_methods = ["turboquant_mse", "turboquant_ada", "product_quant",
                         "uniform_sq", "simhash_multi", "randproj_quant", "flyhash"]
    colors = {"turboquant_mse": "#2196F3", "turboquant_ada": "#9C27B0",
              "product_quant": "#FF9800", "uniform_sq": "#607D8B",
              "simhash_multi": "#E91E63", "randproj_quant": "#795548",
              "flyhash": "#009688"}
    labels = {"turboquant_mse": "TQ MSE (Beta)", "turboquant_ada": "TQ Adaptive",
              "product_quant": "Product Quant", "uniform_sq": "Uniform SQ",
              "simhash_multi": "SimHash Multi", "randproj_quant": "RandProj+Q",
              "flyhash": "FlyHash"}

    for ax, (model, dataset) in zip(axes, combos):
        # Plot FP32 exact as horizontal line
        fp32_key = (model, dataset, "fp32_exact", BITS[0])
        if fp32_key in aggregated:
            ax.axhline(y=aggregated[fp32_key]["recall@10_mean"],
                       color="black", linestyle="--", alpha=0.5, label="FP32 Exact")

        # Plot binary hash as horizontal line
        bh_key = (model, dataset, "binary_hash", BITS[0])
        if bh_key in aggregated:
            ax.axhline(y=aggregated[bh_key]["recall@10_mean"],
                       color="red", linestyle=":", alpha=0.5, label="Binary Hash")

        # Plot quantized methods
        for method in quantized_methods:
            x, y, yerr = [], [], []
            for bits in BITS:
                key = (model, dataset, method, bits)
                if key in aggregated:
                    x.append(bits)
                    y.append(aggregated[key]["recall@10_mean"])
                    yerr.append(aggregated[key]["recall@10_std"])

            if x:
                ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4,
                            color=colors[method], label=labels[method])

        ax.set_xlabel("Bits per dimension")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{model} / {dataset}")
        ax.set_xticks(BITS)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "recall_vs_bits.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_compression_vs_recall(aggregated: dict):
    """Plot bytes/vector vs recall@10 (Pareto frontier)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model in zip(axes, ["prithvi", "remoteclip"]):
        for dataset, marker in [("eurosat", "o"), ("bigearthnet", "s")]:
            for method in METHODS:
                for bits in BITS:
                    key = (model, dataset, method, bits)
                    if key not in aggregated:
                        if method in ("fp32_exact", "binary_hash"):
                            key = (model, dataset, method, BITS[0])
                        if key not in aggregated:
                            continue

                    a = aggregated[key]
                    color = {"fp32_exact": "black", "binary_hash": "red",
                             "product_quant": "#FF9800", "turboquant_mse": "#2196F3",
                             "turboquant_ada": "#9C27B0", "uniform_sq": "#607D8B",
                             "simhash_multi": "#E91E63", "randproj_quant": "#795548",
                             "flyhash": "#009688"}.get(method, "gray")

                    ax.scatter(a["bytes_per_vector"], a["recall@10_mean"],
                               c=color, marker=marker, s=60, alpha=0.8)

                    if method in ("fp32_exact", "binary_hash"):
                        break

        ax.set_xlabel("Bytes per vector")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{model}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "compression_vs_recall.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def scaling_projection(aggregated: dict):
    """Project memory and search time to 10M and 100M vectors."""
    print(f"\n{'='*80}")
    print("SCALING PROJECTIONS")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Bits':<6} {'1M':<12} {'10M':<12} {'100M':<12}")
    print("-" * 80)

    for method in METHODS:
        for bits in BITS:
            # Use first available config for bytes_per_vector
            for model in ["prithvi", "remoteclip"]:
                key = None
                for ds in ["eurosat", "bigearthnet"]:
                    key = (model, ds, method, bits)
                    if key in aggregated:
                        break
                    key = (model, ds, method, BITS[0])
                    if key in aggregated:
                        break
                    key = None
                if key and key in aggregated:
                    bpv = aggregated[key]["bytes_per_vector"]
                    bits_str = "-" if method in ("fp32_exact", "binary_hash") else str(bits)
                    mem_1m = bpv * 1e6 / 1e9
                    mem_10m = bpv * 1e7 / 1e9
                    mem_100m = bpv * 1e8 / 1e9
                    print(f"{method:<20} {bits_str:<6} "
                          f"{mem_1m:.2f} GB    {mem_10m:.2f} GB    {mem_100m:.1f} GB")
                    break

            if method in ("fp32_exact", "binary_hash"):
                break


def practitioner_recommendation(aggregated: dict):
    """Generate the 'what should RS practitioners use' recommendation."""
    print(f"\n{'='*80}")
    print("PRACTITIONER RECOMMENDATION")
    print(f"{'='*80}")

    # Find best method at each bit width by average recall@10
    for bits in BITS:
        best_method = None
        best_recall = -1

        for method in ["turboquant_mse", "turboquant_ada", "product_quant"]:
            recalls = []
            for model in ["prithvi", "remoteclip"]:
                for dataset in ["bigearthnet", "eurosat"]:
                    key = (model, dataset, method, bits)
                    if key in aggregated:
                        recalls.append(aggregated[key]["recall@10_mean"])

            if recalls:
                avg = np.mean(recalls)
                if avg > best_recall:
                    best_recall = avg
                    best_method = method

        if best_method:
            print(f"  At {bits} bits/dim: {best_method} (avg recall@10 = {best_recall:.3f})")

    print("\n  Summary: [Fill in based on actual results]")
    print("  - If Beta assumption holds (D < 0.02): TurboQuant is the winner.")
    print("    Zero training, better compression, comparable quality.")
    print("  - If Beta assumption fails: PQ remains the safe choice,")
    print("    but requires training data and is domain-specific.")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    aggregated = aggregate_over_seeds(results)

    print_results_table(aggregated)
    plot_recall_vs_bits(aggregated)
    plot_compression_vs_recall(aggregated)
    scaling_projection(aggregated)
    practitioner_recommendation(aggregated)

    # Save aggregated results
    agg_list = list(aggregated.values())
    with open(RESULTS_DIR / "aggregated_results.json", "w") as f:
        json.dump(agg_list, f, indent=2)
    print(f"\nAggregated results saved to {RESULTS_DIR / 'aggregated_results.json'}")


if __name__ == "__main__":
    main()
