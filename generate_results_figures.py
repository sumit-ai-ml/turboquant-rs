"""Generate all paper-ready figures for the Results section."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from collections import defaultdict
from pathlib import Path

FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')

# Paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.2,
})

COLORS = {
    'dinov2': '#2E7D32', 'remoteclip': '#1565C0', 'georsclip': '#0277BD',
    'ssl4eo': '#E65100', 'mae_base': '#C62828', 'prithvi': '#6A1B9A',
}
MARKERS = {
    'dinov2': 'D', 'remoteclip': 'o', 'georsclip': 's',
    'ssl4eo': '^', 'mae_base': 'v', 'prithvi': 'P',
}
LABELS = {
    'dinov2': 'DINOv2', 'remoteclip': 'RemoteCLIP', 'georsclip': 'GeoRSCLIP',
    'ssl4eo': 'SSL4EO', 'mae_base': 'MAE-base', 'prithvi': 'Prithvi',
}

# Load data
with open(RESULTS_DIR / 'six_model_results.json') as f:
    euro_6 = json.load(f)
with open(RESULTS_DIR / 'six_model_ben_results.json') as f:
    ben_6 = json.load(f)
with open(RESULTS_DIR / 'benchmark_results.json') as f:
    bench = json.load(f)
with open(RESULTS_DIR / 'rabitq_results.json') as f:
    bench.extend(json.load(f))
with open(RESULTS_DIR / 'ranking_analysis.json') as f:
    ranking = json.load(f)

# Aggregate benchmark
def agg(results):
    g = defaultdict(list)
    for r in results:
        g[(r['model'], r['dataset'], r['method'], r['bits'])].append(r)
    out = {}
    for k, runs in g.items():
        vals = [r['recall']['10'] for r in runs]
        out[k] = {'r10': np.mean(vals), 'r10_std': np.std(vals), 'bpv': runs[0]['bytes_per_vector']}
    return out

ba = agg(bench)

# Aggregate ranking
def agg_ranking(results):
    g = defaultdict(list)
    for r in results:
        g[(r['model'], r['dataset'], r['method'])].append(r)
    out = {}
    for k, runs in g.items():
        out[k] = {
            'tau': np.mean([r['kendall_tau_mean'] for r in runs]),
            'pearson': np.mean([r['pearson_mean'] for r in runs]),
        }
    return out

ra = agg_ranking(ranking)


# =========================================================================
# Fig R1: 6-model bar chart — BigEarthNet (TQ vs PQ vs BinHash)
# =========================================================================
print('Fig R1: 6-model grouped bars...')
fig, ax = plt.subplots(1, 1, figsize=(7.16, 3.0))

models_sorted = sorted(ben_6, key=lambda r: -r['tq_r10'])
names = [r['model'] for r in models_sorted]
tq = [r['tq_r10'] for r in models_sorted]
pq = [r['pq_r10'] for r in models_sorted]
bh = [r['bh_r10'] for r in models_sorted]
colors = [COLORS[n] for n in names]

x = np.arange(len(names))
w = 0.24

ax.bar(x - w, tq, w, color=colors, edgecolor='black', linewidth=0.4, label='TQ MSE (4-bit, no training)')
ax.bar(x, pq, w, color='#FFD54F', edgecolor='black', linewidth=0.4, label='PQ (4-bit, trained)')
ax.bar(x + w, bh, w, color='#BDBDBD', edgecolor='black', linewidth=0.4, label='Binary Hash (1-bit)')

# Gap closed labels
for i, r in enumerate(models_sorted):
    ax.text(x[i] - w, tq[i] + 0.01, f'{r["gap_closed"]:.0f}%',
            ha='center', fontsize=7, fontweight='bold', color=colors[i])

TRAINING_SHORT = {'dinov2': 'self-distill.', 'remoteclip': 'contrastive',
                   'georsclip': 'contrastive', 'ssl4eo': 'MAE (RS)',
                   'mae_base': 'MAE', 'prithvi': 'MAE (RS)'}

ax.set_ylabel('Recall@10')
ax.set_title('BigEarthNet (269K vectors)', fontweight='bold')
ax.set_xticks(x)
tick_labels = [f'{LABELS[n]}\n({TRAINING_SHORT[n]})' for n in names]
ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha='right')
ax.set_ylim(0, 1.08)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#CCCCCC')
ax.tick_params(direction='in')
ax.grid(True, alpha=0.08, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'results_6model_bars.png', dpi=300)
fig.savefig(FIGURES_DIR / 'results_6model_bars.pdf')
plt.close(fig)


# =========================================================================
# Fig R2: All 9 methods bar chart (Prithvi + RemoteCLIP, BigEarthNet)
# =========================================================================
print('Fig R2: All methods comparison...')
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

methods = [
    ('product_quant', 4, 'PQ', '#FFD54F'),
    ('turboquant_mse', 4, 'TQ MSE', '#2196F3'),
    ('turboquant_ada', 4, 'TQ Ada', '#9C27B0'),
    ('simhash_multi', 4, 'SimHash', '#E91E63'),
    ('uniform_sq', 4, 'Unif. SQ', '#607D8B'),
    ('flyhash', 4, 'FlyHash', '#009688'),
    ('randproj_quant', 4, 'RandProj', '#795548'),
    ('rabitq', 1, 'RaBitQ', '#4CAF50'),
    ('binary_hash', 2, 'BinHash', '#F44336'),
]

for ax, (model, model_label) in zip(axes, [('prithvi', 'Prithvi (d=768)'), ('remoteclip', 'RemoteCLIP (d=512)')]):
    vals = []
    labels = []
    colors_bar = []
    for method, bits, label, color in methods:
        key = (model, 'bigearthnet', method, bits)
        if key in ba:
            vals.append(ba[key]['r10'])
            labels.append(label)
            colors_bar.append(color)

    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=colors_bar, edgecolor='black', linewidth=0.4, height=0.7)

    # Value labels
    for i, v in enumerate(vals):
        ax.text(v + 0.01, y[i], f'{v:.3f}', va='center', fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('R@10')
    ax.set_title(model_label, fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.08)
    ax.tick_params(direction='in')
    ax.grid(True, alpha=0.08, axis='x')
    ax.invert_yaxis()

plt.tight_layout(w_pad=2.0)
fig.savefig(FIGURES_DIR / 'results_all_methods.png', dpi=300)
fig.savefig(FIGURES_DIR / 'results_all_methods.pdf')
plt.close(fig)


# =========================================================================
# Fig R3: Ranking quality (Kendall tau) — grouped bar
# =========================================================================
print('Fig R3: Ranking quality...')
fig, ax = plt.subplots(1, 1, figsize=(7.16, 3.0))

models_order = ['dinov2', 'remoteclip', 'georsclip', 'ssl4eo', 'mae_base', 'prithvi']
x = np.arange(len(models_order))
w = 0.25

tq_tau = [ra.get((m, 'bigearthnet', 'turboquant_mse_4bit'), {}).get('tau', 0) for m in models_order]
pq_tau = [ra.get((m, 'bigearthnet', 'product_quant_4bit'), {}).get('tau', 0) for m in models_order]
bh_tau = [ra.get((m, 'bigearthnet', 'binary_hash'), {}).get('tau', 0) for m in models_order]

ax.bar(x - w, tq_tau, w, color=[COLORS[m] for m in models_order],
       edgecolor='black', linewidth=0.4, label='TQ MSE (4-bit)')
ax.bar(x, pq_tau, w, color='#FFD54F', edgecolor='black', linewidth=0.4, label='PQ (4-bit)')
ax.bar(x + w, bh_tau, w, color='#BDBDBD', edgecolor='black', linewidth=0.4, label='Binary Hash')

ax.set_ylabel("Kendall's $\\tau$")
ax.set_title('Ranking Quality within Top-1000 Neighborhood (BigEarthNet)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in models_order], fontsize=8)
ax.set_ylim(0, 1.08)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#CCCCCC')
ax.tick_params(direction='in')
ax.grid(True, alpha=0.08, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'results_ranking_quality.png', dpi=300)
fig.savefig(FIGURES_DIR / 'results_ranking_quality.pdf')
plt.close(fig)


# =========================================================================
# Fig R4: Codebook ablation
# =========================================================================
print('Fig R4: Codebook ablation...')
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

abl_methods = [
    ('turboquant_mse', 'TQ MSE (Beta)', '#2196F3'),
    ('turboquant_ada', 'TQ Adaptive (empirical)', '#9C27B0'),
    ('uniform_sq', 'Uniform SQ (no codebook)', '#607D8B'),
]
BITS = [2, 3, 4]

for ax, (model, title) in zip(axes, [('prithvi', 'Prithvi / BigEarthNet'), ('remoteclip', 'RemoteCLIP / BigEarthNet')]):
    for method, label, color in abl_methods:
        x_vals, y_vals, yerr = [], [], []
        for bits in BITS:
            key = (model, 'bigearthnet', method, bits)
            if key in ba:
                x_vals.append(bits)
                y_vals.append(ba[key]['r10'])
                yerr.append(ba[key]['r10_std'])
        if x_vals:
            ax.errorbar(x_vals, y_vals, yerr=yerr, marker='o', capsize=2,
                        color=color, label=label, linewidth=1.2, markersize=5)

    ax.set_xlabel('Bits per dimension')
    ax.set_ylabel('R@10')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticks(BITS)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, framealpha=0.95, edgecolor='#CCCCCC')
    ax.tick_params(direction='in')
    ax.grid(True, alpha=0.08)

plt.tight_layout(w_pad=2.0)
fig.savefig(FIGURES_DIR / 'results_codebook_ablation.png', dpi=300)
fig.savefig(FIGURES_DIR / 'results_codebook_ablation.pdf')
plt.close(fig)


# =========================================================================
# Fig R5: Scaling (EuroSAT vs BigEarthNet) — slope chart
# =========================================================================
print('Fig R5: Scaling slope chart...')
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))

for r_e in sorted(euro_6, key=lambda r: -r['tq_r10']):
    m = r_e['model']
    r_b = next((r for r in ben_6 if r['model'] == m), None)
    if r_b is None:
        continue
    ax.plot([0, 1], [r_e['tq_r10'], r_b['tq_r10']],
            color=COLORS[m], marker=MARKERS[m], markersize=7,
            linewidth=1.5, markeredgecolor='black', markeredgewidth=0.4)
    # Labels on right side
    ax.text(1.05, r_b['tq_r10'], f"{LABELS[m]} ({r_e['tq_r10']:.2f} $\\to$ {r_b['tq_r10']:.2f})",
            fontsize=6.5, va='center', color=COLORS[m])

ax.set_xticks([0, 1])
ax.set_xticklabels(['EuroSAT\n(16K)', 'BigEarthNet\n(269K)'], fontsize=8)
ax.set_ylabel('TQ MSE R@10 (4-bit)')
ax.set_title('Scaling Behavior', fontsize=10, fontweight='bold')
ax.set_xlim(-0.1, 1.6)
ax.set_ylim(0.5, 1.0)
ax.tick_params(direction='in')
ax.grid(True, alpha=0.08, axis='y')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'results_scaling.png', dpi=300)
fig.savefig(FIGURES_DIR / 'results_scaling.pdf')
plt.close(fig)


# =========================================================================
# Fig R6: Bit-width scaling (Prithvi + RemoteCLIP, both datasets)
# =========================================================================
print('Fig R6: Bit-width scaling...')
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

for ax, dataset in zip(axes, ['eurosat', 'bigearthnet']):
    n = '16K' if dataset == 'eurosat' else '269K'
    for model in ['prithvi', 'remoteclip']:
        # TQ MSE
        x_vals, y_vals, yerr = [], [], []
        for bits in BITS:
            key = (model, dataset, 'turboquant_mse', bits)
            if key in ba:
                x_vals.append(bits)
                y_vals.append(ba[key]['r10'])
                yerr.append(ba[key]['r10_std'])
        if x_vals:
            ax.errorbar(x_vals, y_vals, yerr=yerr, marker=MARKERS[model], capsize=2,
                        color=COLORS[model], label=f'{LABELS[model]} (TQ)',
                        linewidth=1.2, markersize=5)

        # PQ reference (dashed)
        x_vals, y_vals = [], []
        for bits in BITS:
            key = (model, dataset, 'product_quant', bits)
            if key in ba:
                x_vals.append(bits)
                y_vals.append(ba[key]['r10'])
        if x_vals:
            ax.plot(x_vals, y_vals, '--', color=COLORS[model], alpha=0.4,
                    linewidth=0.8, label=f'{LABELS[model]} (PQ)')

    ax.set_xlabel('Bits per dimension')
    ax.set_ylabel('R@10')
    ax.set_title(f'{dataset.capitalize()} (n={n})', fontsize=9, fontweight='bold')
    ax.set_xticks(BITS)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=6.5, framealpha=0.95, edgecolor='#CCCCCC', ncol=2)
    ax.tick_params(direction='in')
    ax.grid(True, alpha=0.08)

plt.tight_layout(w_pad=2.0)
fig.savefig(FIGURES_DIR / 'results_bitwidth.png', dpi=300)
fig.savefig(FIGURES_DIR / 'results_bitwidth.pdf')
plt.close(fig)


print('\nAll results figures generated.')
for p in sorted(FIGURES_DIR.glob('results_*.png')):
    print(f'  {p.name} ({p.stat().st_size / 1024:.0f} KB)')
