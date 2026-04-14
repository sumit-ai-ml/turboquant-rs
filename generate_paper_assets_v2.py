"""Generate all figures, tables, and data exports for the 6-model paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from pathlib import Path

FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# =============================================================================
# Load all results
# =============================================================================

# EuroSAT 6-model
with open(RESULTS_DIR / 'six_model_results.json') as f:
    euro_6 = json.load(f)

# BigEarthNet 6-model
with open(RESULTS_DIR / 'six_model_ben_results.json') as f:
    ben_6 = json.load(f)

# Full benchmark (original 2 models, all methods)
with open(RESULTS_DIR / 'benchmark_results.json') as f:
    bench = json.load(f)

# RaBitQ
with open(RESULTS_DIR / 'rabitq_results.json') as f:
    rabitq = json.load(f)

# Aggregate benchmark results
def agg_bench(results):
    grouped = defaultdict(list)
    for r in results:
        key = (r['model'], r['dataset'], r['method'], r['bits'])
        grouped[key].append(r)
    out = {}
    for key, runs in grouped.items():
        r10_vals = [r['recall']['10'] for r in runs]
        out[key] = {
            'r10_mean': np.mean(r10_vals),
            'r10_std': np.std(r10_vals),
            'bpv': runs[0]['bytes_per_vector'],
        }
    return out

bench_agg = agg_bench(bench + rabitq)

# Model metadata
MODEL_META = {
    'dinov2':     {'training': 'Self-distillation', 'color': '#4CAF50', 'marker': 'D', 'group': 'contrastive'},
    'remoteclip': {'training': 'Contrastive (CLIP)', 'color': '#2196F3', 'marker': 'o', 'group': 'contrastive'},
    'georsclip':  {'training': 'Contrastive (CLIP)', 'color': '#03A9F4', 'marker': 's', 'group': 'contrastive'},
    'ssl4eo':     {'training': 'MAE (RS)', 'color': '#FF9800', 'marker': '^', 'group': 'mae'},
    'mae_base':   {'training': 'MAE (ImageNet)', 'color': '#F44336', 'marker': 'v', 'group': 'mae'},
    'prithvi':    {'training': 'MAE (RS)', 'color': '#9C27B0', 'marker': 'P', 'group': 'mae'},
}

# =============================================================================
# Figure 1: The Money Plot — Coord Correlation vs TQ R@10 (scatter)
# =============================================================================

print('Fig 1: Coord correlation vs TQ R@10...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (data, dataset_label) in zip(axes, [(euro_6, 'EuroSAT (16K)'), (ben_6, 'BigEarthNet (269K)')]):
    corrs = [r['coord_corr'] for r in data]
    tqs = [r['tq_r10'] for r in data]
    r_val = np.corrcoef(corrs, tqs)[0, 1]

    for r in data:
        meta = MODEL_META[r['model']]
        ax.scatter(r['coord_corr'], r['tq_r10'], c=meta['color'], marker=meta['marker'],
                   s=120, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(r['model'], (r['coord_corr'], r['tq_r10']),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)

    # Regression line
    z = np.polyfit(corrs, tqs, 1)
    x_line = np.linspace(min(corrs) - 0.05, max(corrs) + 0.05, 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Coordinate Correlation (lower = more isotropic)')
    ax.set_ylabel('TurboQuant R@10 (4-bit)')
    ax.set_title(f'{dataset_label}\nr = {r_val:.3f}')
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.15)

# Legend
from matplotlib.lines import Line2D
legend_els = []
for model in ['dinov2', 'remoteclip', 'georsclip', 'ssl4eo', 'mae_base', 'prithvi']:
    m = MODEL_META[model]
    legend_els.append(Line2D([0], [0], marker=m['marker'], color='w',
                             markerfacecolor=m['color'], markersize=8,
                             markeredgecolor='black', markeredgewidth=0.5,
                             label=f"{model} ({m['training']})"))
fig.legend(handles=legend_els, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.08))
plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(FIGURES_DIR / 'fig1_correlation_vs_recall.png', dpi=300)
fig.savefig(FIGURES_DIR / 'fig1_correlation_vs_recall.pdf')
plt.close(fig)


# =============================================================================
# Figure 2: R@10 vs Bits for all 6 models (2x1 grid)
# =============================================================================

print('Fig 2: R@10 vs bits per model...')

# Need per-bit results. Run quick computation for the 4 new models
# For now, use the data we have (4-bit only for new models, full for original 2)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

BITS = [2, 3, 4]

for ax, dataset in zip(axes, ['eurosat', 'bigearthnet']):
    n_label = '16K' if dataset == 'eurosat' else '269K'

    # Original 2 models have full bit-sweep data
    for model in ['prithvi', 'remoteclip']:
        meta = MODEL_META[model]
        x, y, yerr = [], [], []
        for bits in BITS:
            key = (model, dataset, 'turboquant_mse', bits)
            if key in bench_agg:
                x.append(bits)
                y.append(bench_agg[key]['r10_mean'])
                yerr.append(bench_agg[key]['r10_std'])
        if x:
            ax.errorbar(x, y, yerr=yerr, marker=meta['marker'], capsize=3,
                        color=meta['color'], label=model, linewidth=1.5, markersize=7)

    # PQ reference
    for model in ['prithvi', 'remoteclip']:
        x, y = [], []
        for bits in BITS:
            key = (model, dataset, 'product_quant', bits)
            if key in bench_agg:
                x.append(bits)
                y.append(bench_agg[key]['r10_mean'])
        if x:
            ax.plot(x, y, '--', color='gray', alpha=0.4, linewidth=1)

    # Binary hash baselines
    for model in ['prithvi', 'remoteclip']:
        key = (model, dataset, 'binary_hash', 2)
        if key in bench_agg:
            ax.axhline(y=bench_agg[key]['r10_mean'], color=MODEL_META[model]['color'],
                       linestyle=':', alpha=0.3, linewidth=1)

    ax.set_xlabel('Bits per dimension')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'TQ MSE — {dataset.capitalize()} (n={n_label})')
    ax.set_xticks(BITS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig2_recall_vs_bits.png', dpi=300)
fig.savefig(FIGURES_DIR / 'fig2_recall_vs_bits.pdf')
plt.close(fig)


# =============================================================================
# Figure 3: Training-free method comparison (grouped bar, BigEarthNet)
# =============================================================================

print('Fig 3: Training-free methods comparison...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

methods = ['turboquant_mse', 'simhash_multi', 'uniform_sq', 'flyhash', 'randproj_quant', 'rabitq', 'binary_hash']
method_labels = ['TQ MSE\n(4-bit)', 'SimHash\n(4-bit)', 'Uniform SQ\n(4-bit)', 'FlyHash\n(4-bit)',
                 'RandProj\n(4-bit)', 'RaBitQ\n(1-bit)', 'BinHash\n(1-bit)']
bar_colors = ['#2196F3', '#E91E63', '#607D8B', '#009688', '#795548', '#4CAF50', '#F44336']

for ax, dataset in zip(axes, ['eurosat', 'bigearthnet']):
    n_label = '16K' if dataset == 'eurosat' else '269K'
    x = np.arange(len(methods))
    width = 0.35

    prithvi_vals, rclip_vals = [], []
    for method in methods:
        bits = 1 if method in ('binary_hash', 'rabitq') else 4
        pk = ('prithvi', dataset, method, bits)
        rk = ('remoteclip', dataset, method, bits)
        prithvi_vals.append(bench_agg[pk]['r10_mean'] if pk in bench_agg else 0)
        rclip_vals.append(bench_agg[rk]['r10_mean'] if rk in bench_agg else 0)

    ax.bar(x - width/2, prithvi_vals, width, label='Prithvi (MAE, d=768)',
           color=bar_colors, alpha=0.5, edgecolor=bar_colors, linewidth=1.5)
    ax.bar(x + width/2, rclip_vals, width, label='RemoteCLIP (CLIP, d=512)',
           color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Recall@10')
    ax.set_title(f'Training-Free Methods — {dataset.capitalize()} (n={n_label})')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.15, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig3_training_free_methods.png', dpi=300)
fig.savefig(FIGURES_DIR / 'fig3_training_free_methods.pdf')
plt.close(fig)


# =============================================================================
# Figure 4: Codebook ablation (Beta vs Adaptive vs Uniform)
# =============================================================================

print('Fig 4: Codebook ablation...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ablation = ['turboquant_mse', 'turboquant_ada', 'uniform_sq']
abl_colors = ['#2196F3', '#9C27B0', '#607D8B']
abl_labels = ['TQ MSE (Beta codebook)', 'TQ Adaptive (empirical)', 'Uniform SQ (no codebook)']

for ax, (model, dataset) in zip(axes, [('prithvi', 'bigearthnet'), ('remoteclip', 'bigearthnet')]):
    d = 768 if model == 'prithvi' else 512
    for method, color, label in zip(ablation, abl_colors, abl_labels):
        x, y, yerr = [], [], []
        for bits in BITS:
            key = (model, dataset, method, bits)
            if key in bench_agg:
                x.append(bits)
                y.append(bench_agg[key]['r10_mean'])
                yerr.append(bench_agg[key]['r10_std'])
        if x:
            ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, color=color,
                        label=label, linewidth=2, markersize=7)

    ax.set_xlabel('Bits per dimension')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'{model.capitalize()} / BigEarthNet (d={d}, n=269K)')
    ax.set_xticks(BITS)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig4_codebook_ablation.png', dpi=300)
fig.savefig(FIGURES_DIR / 'fig4_codebook_ablation.pdf')
plt.close(fig)


# =============================================================================
# Figure 5: 6-model bar chart — BigEarthNet R@10
# =============================================================================

print('Fig 5: 6-model bar chart...')
fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

models_sorted = sorted(ben_6, key=lambda r: -r['tq_r10'])
model_names = [r['model'] for r in models_sorted]
tq_vals = [r['tq_r10'] for r in models_sorted]
pq_vals = [r['pq_r10'] for r in models_sorted]
bh_vals = [r['bh_r10'] for r in models_sorted]
colors = [MODEL_META[m]['color'] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

bars_tq = ax.bar(x - width, tq_vals, width, label='TQ MSE (4-bit, no training)',
                 color=colors, edgecolor='black', linewidth=0.5)
bars_pq = ax.bar(x, pq_vals, width, label='PQ (4-bit, trained)',
                 color='#FFD54F', edgecolor='black', linewidth=0.5)
bars_bh = ax.bar(x + width, bh_vals, width, label='Binary Hash (1-bit)',
                 color='#BDBDBD', edgecolor='black', linewidth=0.5)

# Annotate gap closed
for i, r in enumerate(models_sorted):
    gap = r['gap_closed']
    ax.annotate(f'{gap:.0f}%', xy=(x[i] - width, tq_vals[i]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold')

# Training type annotations
for i, m in enumerate(model_names):
    meta = MODEL_META[m]
    ax.annotate(meta['training'], xy=(x[i], -0.05),
                ha='center', fontsize=7, color='gray', style='italic')

ax.set_ylabel('Recall@10')
ax.set_title('BigEarthNet (269K vectors) — 6 Foundation Models')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.15, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig5_six_model_bars.png', dpi=300)
fig.savefig(FIGURES_DIR / 'fig5_six_model_bars.pdf')
plt.close(fig)


# =============================================================================
# Figure 6: Scaling — EuroSAT vs BigEarthNet for all 6 models
# =============================================================================

print('Fig 6: Scaling across datasets...')
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for r_euro in euro_6:
    model = r_euro['model']
    r_ben = next((r for r in ben_6 if r['model'] == model), None)
    if r_ben is None:
        continue
    meta = MODEL_META[model]
    ax.plot([r_euro['tq_r10'], r_ben['tq_r10']], [0, 1],
            color=meta['color'], marker=meta['marker'], markersize=10,
            linewidth=2, markeredgecolor='black', markeredgewidth=0.5)
    ax.annotate(model, (r_euro['tq_r10'], 0), textcoords="offset points",
                xytext=(-5, -15), fontsize=8, ha='center')

ax.set_yticks([0, 1])
ax.set_yticklabels(['EuroSAT (16K)', 'BigEarthNet (269K)'])
ax.set_xlabel('TurboQuant R@10 (4-bit)')
ax.set_title('Scaling: How Much Does R@10 Drop?')
ax.set_xlim(0.5, 1.0)
ax.grid(True, alpha=0.15, axis='x')
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig6_scaling.png', dpi=300)
fig.savefig(FIGURES_DIR / 'fig6_scaling.pdf')
plt.close(fig)


# =============================================================================
# LaTeX Tables
# =============================================================================

print('LaTeX tables...')
with open(RESULTS_DIR / 'table_6model.tex', 'w') as f:
    for data, dataset, n_label in [(euro_6, 'EuroSAT', '16K'), (ben_6, 'BigEarthNet', '269K')]:
        f.write(f"% {dataset} ({n_label} vectors), 4-bit R@10\n")
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Training & Coord Corr & TQ MSE & PQ & BinHash & Gap \\\\\n")
        f.write("\\midrule\n")
        for r in sorted(data, key=lambda x: -x['tq_r10']):
            name = r['model'].replace('_', '\\_')
            training = r['training'].replace('_', '\\_')
            gap = f"{r['gap_closed']:.0f}\\%"
            f.write(f"{name} & {training} & {r['coord_corr']:.3f} & "
                    f"{r['tq_r10']:.3f} & {r['pq_r10']:.3f} & "
                    f"{r['bh_r10']:.3f} & {gap} \\\\\n")
        corrs = [r['coord_corr'] for r in data]
        tqs = [r['tq_r10'] for r in data]
        r_val = np.corrcoef(corrs, tqs)[0, 1]
        f.write("\\bottomrule\n")
        f.write(f"\\multicolumn{{7}}{{l}}{{Pearson $r$ = {r_val:.3f}}} \\\\\n")
        f.write("\\end{tabular}\n\n")

    # Full method comparison table (BigEarthNet, original 2 models)
    f.write("% All methods, BigEarthNet, 4-bit R@10\n")
    f.write("\\begin{tabular}{llcccc}\n")
    f.write("\\toprule\n")
    f.write("Method & Bits & Prithvi R@10 & RemoteCLIP R@10 & B/vec & Training \\\\\n")
    f.write("\\midrule\n")

    method_order = [
        ('fp32\\_exact', 'fp32_exact', 2, '-', 'No'),
        ('Product Quant', 'product_quant', 4, '4', 'Yes'),
        ('\\textbf{TQ MSE}', 'turboquant_mse', 4, '4', '\\textbf{No}'),
        ('TQ Adaptive', 'turboquant_ada', 4, '4', 'Yes'),
        ('SimHash Multi', 'simhash_multi', 4, '4', 'No'),
        ('Uniform SQ', 'uniform_sq', 4, '4', 'No'),
        ('FlyHash', 'flyhash', 4, '4', 'No'),
        ('RaBitQ', 'rabitq', 1, '1', 'No'),
        ('Binary Hash', 'binary_hash', 2, '-', 'No'),
    ]

    for label, method, bits, bits_str, train in method_order:
        pk = ('prithvi', 'bigearthnet', method, bits)
        rk = ('remoteclip', 'bigearthnet', method, bits)
        p_r10 = f"{bench_agg[pk]['r10_mean']:.3f}" if pk in bench_agg else '---'
        r_r10 = f"{bench_agg[rk]['r10_mean']:.3f}" if rk in bench_agg else '---'
        bpv_p = f"{bench_agg[pk]['bpv']:.0f}" if pk in bench_agg else '---'
        bpv_r = f"{bench_agg[rk]['bpv']:.0f}" if rk in bench_agg else '---'
        bpv = f"{bpv_p}/{bpv_r}"
        f.write(f"{label} & {bits_str} & {p_r10} & {r_r10} & {bpv} & {train} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")


# =============================================================================
# CSV export
# =============================================================================

print('CSV export...')
with open(RESULTS_DIR / 'paper_results.csv', 'w') as f:
    f.write('dataset,model,training,d,coord_corr,ks_d,tq_r10,pq_r10,bh_r10,gap_closed\n')
    for data, dataset in [(euro_6, 'eurosat'), (ben_6, 'bigearthnet')]:
        for r in sorted(data, key=lambda x: -x['tq_r10']):
            f.write(f"{dataset},{r['model']},{r['training']},{r['d']},"
                    f"{r['coord_corr']:.4f},{r.get('ks_d', 0):.4f},"
                    f"{r['tq_r10']:.4f},{r['pq_r10']:.4f},{r['bh_r10']:.4f},"
                    f"{r['gap_closed']:.1f}\n")


# =============================================================================
# Summary
# =============================================================================

print('\n' + '=' * 70)
print('PAPER ASSETS GENERATED')
print('=' * 70)
print('\nFigures:')
for p in sorted(FIGURES_DIR.glob('fig*')):
    print(f'  {p.name} ({p.stat().st_size / 1024:.0f} KB)')
print('\nTables:')
for p in sorted(RESULTS_DIR.glob('table_*')):
    print(f'  {p.name} ({p.stat().st_size / 1024:.0f} KB)')
print('\nData:')
for p in sorted(RESULTS_DIR.glob('paper_*')):
    print(f'  {p.name} ({p.stat().st_size / 1024:.0f} KB)')

print('\nKey numbers for abstract:')
for data, label in [(euro_6, 'EuroSAT'), (ben_6, 'BigEarthNet')]:
    corrs = [r['coord_corr'] for r in data]
    tqs = [r['tq_r10'] for r in data]
    r_val = np.corrcoef(corrs, tqs)[0, 1]
    best = max(data, key=lambda x: x['tq_r10'])
    worst = min(data, key=lambda x: x['tq_r10'])
    print(f'  {label}: r={r_val:.3f}, best={best["model"]} ({best["tq_r10"]:.3f}), '
          f'worst={worst["model"]} ({worst["tq_r10"]:.3f})')
