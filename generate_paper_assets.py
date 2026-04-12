"""Generate all figures, tables, and data exports for the paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# Load and aggregate
# =============================================================================

with open(RESULTS_DIR / 'benchmark_results.json') as f:
    results = json.load(f)
with open(RESULTS_DIR / 'rabitq_results.json') as f:
    results.extend(json.load(f))

grouped = defaultdict(list)
for r in results:
    key = (r['model'], r['dataset'], r['method'], r['bits'])
    grouped[key].append(r)

agg = {}
for key, runs in grouped.items():
    a = {'model': key[0], 'dataset': key[1], 'method': key[2], 'bits': key[3],
         'n_seeds': len(runs), 'bytes_per_vector': runs[0]['bytes_per_vector']}
    for k in [1, 10, 100]:
        vals = [r['recall'][str(k)] for r in runs]
        a[f'r{k}_mean'] = np.mean(vals)
        a[f'r{k}_std'] = np.std(vals)
    if 'queries_per_sec' in runs[0]:
        a['qps_mean'] = np.mean([r['queries_per_sec'] for r in runs])
    agg[key] = a


# =============================================================================
# 1. CSV export (all results)
# =============================================================================

print('1. CSV export...')
with open(RESULTS_DIR / 'all_results.csv', 'w') as f:
    f.write('model,dataset,method,bits,bytes_per_vector,'
            'r1_mean,r1_std,r10_mean,r10_std,r100_mean,r100_std,training\n')
    for key in sorted(agg.keys()):
        a = agg[key]
        train = 'yes' if a['method'] in ('product_quant', 'turboquant_ada') else 'no'
        f.write(f"{a['model']},{a['dataset']},{a['method']},{a['bits']},"
                f"{a['bytes_per_vector']:.0f},"
                f"{a['r1_mean']:.4f},{a['r1_std']:.4f},"
                f"{a['r10_mean']:.4f},{a['r10_std']:.4f},"
                f"{a['r100_mean']:.4f},{a['r100_std']:.4f},{train}\n")


# =============================================================================
# 2. LaTeX tables
# =============================================================================

print('2. LaTeX tables...')
with open(RESULTS_DIR / 'table_main.tex', 'w') as f:
    for model in ['prithvi', 'remoteclip']:
        for dataset in ['eurosat', 'bigearthnet']:
            d = 768 if model == 'prithvi' else 512
            n = '16K' if dataset == 'eurosat' else '269K'
            title = f"{model.capitalize()} / {dataset.capitalize()} (d={d}, n={n})"
            f.write(f"% {title}\n")
            f.write("\\begin{tabular}{llccccc}\n")
            f.write("\\toprule\n")
            f.write("Method & Bits & B/vec & R@1 & R@10 & R@100 & Train \\\\\n")
            f.write("\\midrule\n")

            entries = []
            for key, a in agg.items():
                if a['model'] == model and a['dataset'] == dataset:
                    entries.append(a)
            entries.sort(key=lambda a: -a['r10_mean'])

            for a in entries:
                name = a['method'].replace('_', '\\_')
                if a['method'] in ('fp32_exact', 'binary_hash', 'rabitq'):
                    bits = '-'
                else:
                    bits = str(a['bits'])
                train = 'Yes' if a['method'] in ('product_quant', 'turboquant_ada') else 'No'
                bpv = f"{a['bytes_per_vector']:.0f}"
                r1 = f"{a['r1_mean']:.3f}"
                r10 = f"\\textbf{{{a['r10_mean']:.3f}}}" if a['method'] == 'turboquant_mse' and a['bits'] == 4 else f"{a['r10_mean']:.3f}"
                r100 = f"{a['r100_mean']:.3f}"
                f.write(f"{name} & {bits} & {bpv} & {r1} & {r10} & {r100} & {train} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n\n")


# =============================================================================
# 3. Figure 1: Recall@10 vs Bits (main figure, 2x2 grid)
# =============================================================================

print('3. Recall vs Bits (2x2)...')
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
plt.rcParams.update({'font.size': 11})

BITS = [2, 3, 4]
combos = [('prithvi', 'eurosat'), ('prithvi', 'bigearthnet'),
          ('remoteclip', 'eurosat'), ('remoteclip', 'bigearthnet')]

methods_plot = ['turboquant_mse', 'turboquant_ada', 'product_quant',
                'simhash_multi', 'uniform_sq', 'flyhash']
colors = {'turboquant_mse': '#2196F3', 'turboquant_ada': '#9C27B0',
          'product_quant': '#FF9800', 'simhash_multi': '#E91E63',
          'uniform_sq': '#607D8B', 'flyhash': '#009688'}
markers = {'turboquant_mse': 'o', 'turboquant_ada': 's', 'product_quant': 'D',
           'simhash_multi': '^', 'uniform_sq': 'v', 'flyhash': 'P'}
labels = {'turboquant_mse': 'TQ MSE (ours)', 'turboquant_ada': 'TQ Adaptive',
          'product_quant': 'Product Quant', 'simhash_multi': 'SimHash Multi',
          'uniform_sq': 'Uniform SQ', 'flyhash': 'FlyHash'}

for ax, (model, dataset) in zip(axes.flat, combos):
    d = 768 if model == 'prithvi' else 512
    n = '16K' if dataset == 'eurosat' else '269K'

    # FP32 line
    fp32_key = (model, dataset, 'fp32_exact', 2)
    if fp32_key in agg:
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.4, label='FP32 Exact')

    # Binary hash line
    bh_key = (model, dataset, 'binary_hash', 2)
    if bh_key in agg:
        ax.axhline(y=agg[bh_key]['r10_mean'], color='red', linestyle=':', alpha=0.5, label='Binary Hash')

    # RaBitQ point
    rq_key = (model, dataset, 'rabitq', 1)
    if rq_key in agg:
        ax.scatter([1], [agg[rq_key]['r10_mean']], c='#4CAF50', marker='*', s=120, zorder=5, label='RaBitQ')

    for method in methods_plot:
        x, y, yerr = [], [], []
        for bits in BITS:
            key = (model, dataset, method, bits)
            if key in agg:
                x.append(bits)
                y.append(agg[key]['r10_mean'])
                yerr.append(agg[key]['r10_std'])
        if x:
            ax.errorbar(x, y, yerr=yerr, marker=markers[method], capsize=3,
                        color=colors[method], label=labels[method], linewidth=1.5, markersize=6)

    ax.set_xlabel('Bits per dimension')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'{model.capitalize()} / {dataset.capitalize()} (d={d}, n={n})')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

# Single legend at bottom
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIGURES_DIR / 'fig1_recall_vs_bits.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig1_recall_vs_bits.pdf', bbox_inches='tight')
plt.close(fig)


# =============================================================================
# 4. Figure 2: Compression ratio vs Recall@10 (Pareto)
# =============================================================================

print('4. Compression vs Recall (Pareto)...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, dataset in zip(axes, ['eurosat', 'bigearthnet']):
    n = '16K' if dataset == 'eurosat' else '269K'
    for model, marker in [('prithvi', 'o'), ('remoteclip', 's')]:
        d = 768 if model == 'prithvi' else 512
        fp32_bpv = d * 4
        for key, a in agg.items():
            if a['model'] != model or a['dataset'] != dataset:
                continue
            method = a['method']
            color = colors.get(method, {'fp32_exact': 'black', 'binary_hash': 'red',
                                         'rabitq': '#4CAF50', 'randproj_quant': '#795548'}.get(method, 'gray'))
            ratio = fp32_bpv / a['bytes_per_vector']
            ax.scatter(ratio, a['r10_mean'], c=color, marker=marker, s=50, alpha=0.8)

    ax.set_xlabel('Compression ratio (FP32 / compressed)')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'{dataset.capitalize()} (n={n})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.2)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Prithvi', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='RemoteCLIP', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', label='TQ MSE', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', label='PQ', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E91E63', label='SimHash', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Binary Hash', markersize=8),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIGURES_DIR / 'fig2_pareto.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig2_pareto.pdf', bbox_inches='tight')
plt.close(fig)


# =============================================================================
# 5. Figure 3: Training-free method comparison (bar chart)
# =============================================================================

print('5. Training-free comparison bar chart...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

tf_methods = ['turboquant_mse', 'simhash_multi', 'rabitq', 'binary_hash',
              'uniform_sq', 'flyhash', 'randproj_quant']
tf_labels = ['TQ MSE\n(4-bit)', 'SimHash\n(4-bit)', 'RaBitQ\n(1-bit)', 'BinHash\n(1-bit)',
             'Uniform SQ\n(4-bit)', 'FlyHash\n(4-bit)', 'RandProj\n(4-bit)']
tf_colors = ['#2196F3', '#E91E63', '#4CAF50', 'red', '#607D8B', '#009688', '#795548']

for ax, dataset in zip(axes, ['eurosat', 'bigearthnet']):
    n = '16K' if dataset == 'eurosat' else '269K'
    x = np.arange(len(tf_methods))
    width = 0.35

    prithvi_vals = []
    rclip_vals = []
    for method in tf_methods:
        bits = 1 if method in ('binary_hash', 'rabitq') else 4
        pk = ('prithvi', dataset, method, bits)
        rk = ('remoteclip', dataset, method, bits)
        prithvi_vals.append(agg[pk]['r10_mean'] if pk in agg else 0)
        rclip_vals.append(agg[rk]['r10_mean'] if rk in agg else 0)

    bars1 = ax.bar(x - width/2, prithvi_vals, width, label='Prithvi (d=768)',
                   color=tf_colors, alpha=0.5, edgecolor=tf_colors, linewidth=1.5)
    bars2 = ax.bar(x + width/2, rclip_vals, width, label='RemoteCLIP (d=512)',
                   color=tf_colors, alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Recall@10')
    ax.set_title(f'Training-Free Methods — {dataset.capitalize()} (n={n})')
    ax.set_xticks(x)
    ax.set_xticklabels(tf_labels, fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig3_training_free_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig3_training_free_comparison.pdf', bbox_inches='tight')
plt.close(fig)


# =============================================================================
# 6. Figure 4: Scaling — EuroSAT vs BigEarthNet
# =============================================================================

print('6. Scaling comparison...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

scale_methods = ['product_quant', 'turboquant_mse', 'simhash_multi', 'binary_hash']
scale_colors = ['#FF9800', '#2196F3', '#E91E63', 'red']
scale_labels = ['Product Quant', 'TQ MSE', 'SimHash Multi', 'Binary Hash']

for ax, model in zip(axes, ['prithvi', 'remoteclip']):
    d = 768 if model == 'prithvi' else 512
    datasets = ['eurosat', 'bigearthnet']
    x_pos = np.arange(len(datasets))
    width = 0.18

    for i, (method, color, label) in enumerate(zip(scale_methods, scale_colors, scale_labels)):
        vals = []
        for ds in datasets:
            bits = 4 if method not in ('binary_hash',) else 2
            key = (model, ds, method, bits)
            vals.append(agg[key]['r10_mean'] if key in agg else 0)
        ax.bar(x_pos + i * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel('Recall@10')
    ax.set_title(f'{model.capitalize()} (d={d})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['EuroSAT\n(16K)', 'BigEarthNet\n(269K)'])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig4_scaling.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig4_scaling.pdf', bbox_inches='tight')
plt.close(fig)


# =============================================================================
# 7. Figure 5: Codebook ablation (Beta vs Uniform vs Adaptive)
# =============================================================================

print('7. Codebook ablation...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ablation_methods = ['turboquant_mse', 'turboquant_ada', 'uniform_sq']
ablation_colors = ['#2196F3', '#9C27B0', '#607D8B']
ablation_labels = ['TQ MSE (Beta codebook)', 'TQ Adaptive (empirical)', 'Uniform SQ (no codebook)']
BITS = [2, 3, 4]

for ax, (model, dataset) in zip(axes, [('prithvi', 'bigearthnet'), ('remoteclip', 'bigearthnet')]):
    d = 768 if model == 'prithvi' else 512
    for method, color, label in zip(ablation_methods, ablation_colors, ablation_labels):
        x, y, yerr = [], [], []
        for bits in BITS:
            key = (model, dataset, method, bits)
            if key in agg:
                x.append(bits)
                y.append(agg[key]['r10_mean'])
                yerr.append(agg[key]['r10_std'])
        if x:
            ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, color=color,
                        label=label, linewidth=2, markersize=7)

    ax.set_xlabel('Bits per dimension')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'{model.capitalize()} / BigEarthNet (d={d}, n=269K)')
    ax.set_xticks(BITS)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig5_codebook_ablation.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig5_codebook_ablation.pdf', bbox_inches='tight')
plt.close(fig)


# =============================================================================
# 8. Summary table for paper abstract
# =============================================================================

print('\n8. Paper summary numbers:')
print('=' * 80)
for model in ['prithvi', 'remoteclip']:
    for dataset in ['eurosat', 'bigearthnet']:
        d = 768 if model == 'prithvi' else 512
        n = '16K' if dataset == 'eurosat' else '269K'
        pq4 = agg.get((model, dataset, 'product_quant', 4), {}).get('r10_mean', 0)
        tq4 = agg.get((model, dataset, 'turboquant_mse', 4), {}).get('r10_mean', 0)
        bh = agg.get((model, dataset, 'binary_hash', 2), {}).get('r10_mean', 0)
        gap_closed = (tq4 - bh) / (pq4 - bh) * 100 if (pq4 - bh) > 0 else 0
        print(f'{model}/{dataset} (d={d}, n={n}):  '
              f'PQ={pq4:.3f}  TQ_MSE={tq4:.3f}  BinHash={bh:.3f}  '
              f'Gap closed: {gap_closed:.0f}%')

# =============================================================================
# 9. List all generated files
# =============================================================================

print('\n' + '=' * 80)
print('GENERATED FILES:')
print('=' * 80)
for p in sorted(FIGURES_DIR.glob('*')):
    print(f'  {p} ({p.stat().st_size / 1024:.0f} KB)')
for p in sorted(RESULTS_DIR.glob('*')):
    print(f'  {p} ({p.stat().st_size / 1024:.0f} KB)')
print('\nDone.')
