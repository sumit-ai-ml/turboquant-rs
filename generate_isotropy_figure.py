"""Generate paper-ready isotropy illustration figures from actual embedding data."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, beta as beta_dist
from pathlib import Path

from config import EMBED_DIR
from utils import random_orthogonal

FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# Paper-ready matplotlib defaults
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'axes.spines.top': True,
    'axes.spines.right': True,
})

# Consistent color scheme
COLORS = {
    'dinov2': '#2E7D32',
    'remoteclip': '#1565C0',
    'georsclip': '#0277BD',
    'ssl4eo': '#E65100',
    'mae_base': '#C62828',
    'prithvi': '#6A1B9A',
}

MARKERS = {
    'dinov2': 'D',
    'remoteclip': 'o',
    'georsclip': 's',
    'ssl4eo': '^',
    'mae_base': 'v',
    'prithvi': 'P',
}


def get_rotated_coords(model_name, dataset='eurosat', seed=42, n=3000):
    """Load embeddings, rotate, return two coordinate columns."""
    emb_path = EMBED_DIR / f'{model_name}_{dataset}.npz'
    data = np.load(emb_path)
    embeddings = data['embeddings'][:n]
    d = embeddings.shape[1]
    R = random_orthogonal(d, seed)
    rotated = embeddings @ R.T
    return rotated[:, 0], rotated[:, 1], d


def get_beta_boundaries(d, bits=4):
    """Get Beta(d/2,d/2) codebook boundaries."""
    n_levels = 2 ** bits
    a, b = d / 2.0, d / 2.0
    boundaries_01 = beta_dist.ppf(np.linspace(0, 1, n_levels + 1), a, b)
    boundaries_01[0] = 0.0
    boundaries_01[-1] = 1.0
    return boundaries_01 * 2 - 1


# =============================================================================
# Figure 1: Two-panel isotropy illustration (main paper figure)
# IEEE two-column width: ~7.16 inches. Single column: ~3.5 inches.
# =============================================================================

print('Fig 1: Two-panel isotropy illustration...')

fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

panels = [
    ('dinov2', '(a) DINOv2 (self-distillation)', COLORS['dinov2'], 0.253, 0.900),
    ('prithvi', '(b) Prithvi (MAE)', COLORS['prithvi'], 0.663, 0.572),
]

lim = 0.058

for ax, (model_name, title, color, corr, r10) in zip(axes, panels):
    x, y, d = get_rotated_coords(model_name, 'bigearthnet', seed=42, n=4000)

    # White background
    ax.set_facecolor('white')

    # Scatter with low alpha
    ax.scatter(x, y, c=color, alpha=0.08, s=3, edgecolors='none', rasterized=True)

    # Density contours (clean, thin)
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=0.3)
        xg = np.linspace(-lim, lim, 120)
        yg = np.linspace(-lim, lim, 120)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=5, colors=color, alpha=0.6, linewidths=0.6)
    except Exception:
        pass

    # Quantization grid (Beta codebook)
    boundaries = get_beta_boundaries(d, bits=4)
    vis = boundaries[(boundaries > -lim) & (boundaries < lim)]
    for bv in vis:
        ax.axhline(y=bv, color='#BF360C', linestyle=':', linewidth=0.35, alpha=0.5)
        ax.axvline(x=bv, color='#BF360C', linestyle=':', linewidth=0.35, alpha=0.5)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    # Clean ticks
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.tick_params(direction='in', which='both')

    # Labels
    ax.set_xlabel('Rotated coordinate $i$')
    ax.set_ylabel('Rotated coordinate $j$')

    # Title
    ax.set_title(title, fontsize=10, pad=6)

    # Stats box
    r10_color = '#2E7D32' if r10 > 0.85 else '#C62828'
    stats_text = f'$\\rho$ = {corr:.3f}\nR@10 = {r10:.3f}'
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, fontsize=8,
            va='top', ha='right', color=r10_color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=r10_color, alpha=0.95, linewidth=0.6))

# Shared caption
fig.text(0.5, -0.04,
         'Pairwise rotated coordinates from BigEarthNet embeddings (n=4000). '
         'Dotted orange lines: 4-bit Beta codebook boundaries.\n'
         'Isotropic embeddings (a) produce round clouds where the codebook partitions data evenly. '
         'Anisotropic embeddings (b) produce elongated clouds\n'
         'where most codebook cells are empty, degrading retrieval quality.',
         ha='center', fontsize=7.5, color='#444444', linespacing=1.4)

plt.tight_layout(rect=[0, 0.08, 1, 1], w_pad=2.0)
fig.savefig(FIGURES_DIR / 'fig_isotropy_illustration.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig_isotropy_illustration.pdf', bbox_inches='tight')
plt.close(fig)
print('  Done.')


# =============================================================================
# Figure 2: 6-model grid (2x3)
# =============================================================================

print('Fig 2: 6-model isotropy grid...')

fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8))

all_models = [
    ('dinov2', 'DINOv2\n(self-distill.)', COLORS['dinov2'], 0.253, 0.900),
    ('remoteclip', 'RemoteCLIP\n(contrastive)', COLORS['remoteclip'], 0.215, 0.878),
    ('georsclip', 'GeoRSCLIP\n(contrastive)', COLORS['georsclip'], 0.247, 0.830),
    ('ssl4eo', 'SSL4EO\n(MAE, RS)', COLORS['ssl4eo'], 0.345, 0.770),
    ('mae_base', 'MAE-base\n(MAE)', COLORS['mae_base'], 0.521, 0.737),
    ('prithvi', 'Prithvi\n(MAE, RS)', COLORS['prithvi'], 0.663, 0.572),
]

lim = 0.058

for ax, (model_name, label, color, corr, r10) in zip(axes.flat, all_models):
    for dataset in ['bigearthnet', 'eurosat']:
        emb_path = EMBED_DIR / f'{model_name}_{dataset}.npz'
        if emb_path.exists():
            x, y, d = get_rotated_coords(model_name, dataset, seed=42, n=3000)
            break

    ax.set_facecolor('white')
    ax.scatter(x, y, c=color, alpha=0.08, s=2, edgecolors='none', rasterized=True)

    # Contours
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=0.3)
        xg = np.linspace(-lim, lim, 80)
        yg = np.linspace(-lim, lim, 80)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=4, colors=color, alpha=0.5, linewidths=0.5)
    except Exception:
        pass

    # Beta grid
    boundaries = get_beta_boundaries(d, bits=4)
    vis = boundaries[(boundaries > -lim) & (boundaries < lim)]
    for bv in vis:
        ax.axhline(y=bv, color='#BF360C', linestyle=':', linewidth=0.3, alpha=0.4)
        ax.axvline(x=bv, color='#BF360C', linestyle=':', linewidth=0.3, alpha=0.4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.04))
    ax.tick_params(direction='in', labelsize=6)

    # Minimal axis labels (only on edges)
    if ax in axes[-1, :]:
        ax.set_xlabel('Coord. $i$', fontsize=7)
    else:
        ax.set_xticklabels([])
    if ax in axes[:, 0]:
        ax.set_ylabel('Coord. $j$', fontsize=7)
    else:
        ax.set_yticklabels([])

    # Title
    ax.set_title(label, fontsize=8, pad=4, linespacing=1.1)

    # R@10 badge
    r10_color = '#2E7D32' if r10 > 0.85 else ('#E65100' if r10 > 0.75 else '#C62828')
    badge_bg = '#E8F5E9' if r10 > 0.85 else ('#FFF3E0' if r10 > 0.75 else '#FFEBEE')
    ax.text(0.96, 0.96, f'R@10={r10:.2f}',
            transform=ax.transAxes, fontsize=6.5, fontweight='bold',
            color=r10_color, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=badge_bg,
                      edgecolor=r10_color, alpha=0.95, linewidth=0.4))

    # Corr value
    ax.text(0.04, 0.96, f'$\\rho$={corr:.2f}',
            transform=ax.transAxes, fontsize=6.5, color='#555555',
            va='top', ha='left')

fig.text(0.5, -0.01,
         'Top: low coordinate correlation (isotropic) with high retrieval recall. '
         'Bottom: high correlation (anisotropic) with degraded recall.',
         ha='center', fontsize=7.5, color='#444444')

plt.tight_layout(rect=[0, 0.02, 1, 1], h_pad=1.0, w_pad=0.8)
fig.savefig(FIGURES_DIR / 'fig_isotropy_6model.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig_isotropy_6model.pdf', bbox_inches='tight')
plt.close(fig)
print('  Done.')


# =============================================================================
# Figure 3: Correlation vs Recall scatter (single panel)
# =============================================================================

print('Fig 3: Correlation vs recall scatter...')

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))

ben_models = [
    ('dinov2', 'DINOv2', 0.253, 0.900),
    ('remoteclip', 'RemoteCLIP', 0.215, 0.878),
    ('georsclip', 'GeoRSCLIP', 0.247, 0.830),
    ('ssl4eo', 'SSL4EO', 0.345, 0.770),
    ('mae_base', 'MAE-base', 0.521, 0.737),
    ('prithvi', 'Prithvi', 0.663, 0.572),
]

corrs = [m[2] for m in ben_models]
tqs = [m[3] for m in ben_models]

# Regression line
z = np.polyfit(corrs, tqs, 1)
x_line = np.linspace(0.12, 0.72, 100)
ax.plot(x_line, np.polyval(z, x_line), color='#999999', linestyle='--',
        linewidth=0.8, zorder=1)

# Subtle shaded regions
ax.axhspan(0.85, 1.0, color='#E8F5E9', alpha=0.25, zorder=0)
ax.axhspan(0.0, 0.65, color='#FFEBEE', alpha=0.25, zorder=0)

# Points
for name, label, corr, r10 in ben_models:
    ax.scatter(corr, r10, c=COLORS[name], marker=MARKERS[name], s=70, zorder=5,
               edgecolors='black', linewidth=0.4)

# Labels with manual offsets to avoid overlap
offsets = {
    'DINOv2': (0.012, 0.012),
    'RemoteCLIP': (-0.01, -0.028),
    'GeoRSCLIP': (0.012, -0.002),
    'SSL4EO': (0.012, -0.008),
    'MAE-base': (0.012, -0.008),
    'Prithvi': (-0.01, -0.028),
}
for name, label, corr, r10 in ben_models:
    dx, dy = offsets[label]
    ax.annotate(label, (corr + dx, r10 + dy), fontsize=7, color='#333333')

# r value
r_val = np.corrcoef(corrs, tqs)[0, 1]
ax.text(0.97, 0.03, f'$r$ = {r_val:.3f}',
        transform=ax.transAxes, fontsize=9, fontweight='bold',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='black', linewidth=0.8))

ax.set_xlabel('Coordinate correlation after rotation\n(lower = more isotropic)')
ax.set_ylabel('TurboQuant R@10 (4-bit)')
ax.set_title('BigEarthNet (269K vectors)', fontsize=10, fontweight='bold', pad=8)
ax.set_xlim(0.12, 0.72)
ax.set_ylim(0.52, 0.95)
ax.tick_params(direction='in')
ax.grid(True, alpha=0.08, linewidth=0.3)

# Legend
legend_els = []
for name, label, corr, r10 in ben_models:
    legend_els.append(Line2D([0], [0], marker=MARKERS[name], color='w',
                             markerfacecolor=COLORS[name], markersize=5,
                             markeredgecolor='black', markeredgewidth=0.3,
                             label=label))
ax.legend(handles=legend_els, loc='upper right', fontsize=6.5,
          framealpha=0.9, edgecolor='#CCCCCC', handletextpad=0.3,
          borderpad=0.3, labelspacing=0.3)

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig_corr_vs_recall_final.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig_corr_vs_recall_final.pdf', bbox_inches='tight')
plt.close(fig)
print('  Done.')


print('\nAll paper-ready figures generated.')
