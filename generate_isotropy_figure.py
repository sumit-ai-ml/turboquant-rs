"""Generate the isotropy illustration figure from actual embedding data."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy.stats import beta as beta_dist

from config import EMBED_DIR
from utils import random_orthogonal

FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)


def get_rotated_coords(model_name, dataset='eurosat', seed=42, n=2000):
    """Load embeddings, rotate, return two coordinate columns."""
    emb_path = EMBED_DIR / f'{model_name}_{dataset}.npz'
    data = np.load(emb_path)
    embeddings = data['embeddings'][:n]
    d = embeddings.shape[1]
    R = random_orthogonal(d, seed)
    rotated = embeddings @ R.T
    return rotated[:, 0], rotated[:, 1], d


def get_beta_boundaries(d, bits=4):
    """Get Beta(d/2,d/2) codebook boundaries in the rotated coordinate space."""
    n_levels = 2 ** bits
    a, b = d / 2.0, d / 2.0
    boundaries_01 = beta_dist.ppf(np.linspace(0, 1, n_levels + 1), a, b)
    boundaries_01[0] = 0.0
    boundaries_01[-1] = 1.0
    # Shift from [0,1] to [-1,1]
    boundaries = boundaries_01 * 2 - 1
    return boundaries


# =============================================================================
# Figure 4: Isotropy Illustration (the money figure)
# =============================================================================

print('Generating isotropy illustration from real data...')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

models = [
    ('dinov2', 'DINOv2 (self-distillation)', '#2196F3', 0.253, 0.900),
    ('prithvi', 'Prithvi (MAE)', '#F44336', 0.663, 0.572),
]

for ax, (model_name, label, color, corr, r10) in zip(axes, models):
    # Get real rotated coordinates
    x_coords, y_coords, d = get_rotated_coords(model_name, 'bigearthnet', seed=42, n=3000)

    # Background
    ax.set_facecolor('#FAFAFA')

    # Scatter plot of actual rotated coordinates
    ax.scatter(x_coords, y_coords, c=color, alpha=0.15, s=6, edgecolors='none', rasterized=True)

    # Density contours
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([x_coords, y_coords])
        kde = gaussian_kde(xy)
        xgrid = np.linspace(x_coords.min(), x_coords.max(), 100)
        ygrid = np.linspace(y_coords.min(), y_coords.max(), 100)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=4, colors=color, alpha=0.4, linewidths=0.8)
    except Exception:
        pass

    # Quantization grid (Beta codebook boundaries)
    boundaries = get_beta_boundaries(d, bits=4)
    # Only show boundaries that fall within data range
    data_range = max(abs(x_coords).max(), abs(y_coords).max()) * 1.3
    visible = boundaries[(boundaries > -data_range) & (boundaries < data_range)]
    for b_val in visible:
        ax.axhline(y=b_val, color='#FF9800', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=b_val, color='#FF9800', linestyle='--', linewidth=0.5, alpha=0.5)

    # Axis limits (same for both panels)
    lim = 0.065
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    # Grid
    ax.grid(True, alpha=0.1, linewidth=0.5)

    # Title with stats
    ax.set_title(f'{label}\ncorr = {corr:.3f}  |  R@10 = {r10:.3f}',
                 fontsize=11, pad=10)

    # Labels
    ax.set_xlabel('Rotated coordinate i', fontsize=9, color='#666666')
    ax.set_ylabel('Rotated coordinate j', fontsize=9, color='#666666')

    # Tick styling
    ax.tick_params(labelsize=8, colors='#999999')

    # Annotation
    if corr < 0.3:
        ax.annotate('Independent\nErrors cancel',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=8, color='#2E7D32', fontweight='bold',
                     va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                               edgecolor='#4CAF50', alpha=0.9))
    else:
        ax.annotate('Correlated\nErrors accumulate',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=8, color='#C62828', fontweight='bold',
                     va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                               edgecolor='#F44336', alpha=0.9))

# Shared subtitle
fig.text(0.5, -0.02,
         'Same quantization algorithm. Same number of bits. The difference is the embedding geometry.',
         ha='center', fontsize=10, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIGURES_DIR / 'fig_isotropy_illustration.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig_isotropy_illustration.pdf', bbox_inches='tight')
plt.close(fig)
print(f'  Saved fig_isotropy_illustration.png/pdf')


# =============================================================================
# Figure 4b: All 6 models in a 2x3 grid
# =============================================================================

print('Generating 6-model isotropy grid...')

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

all_models = [
    ('dinov2', 'DINOv2\n(self-distillation)', '#4CAF50', 0.253, 0.900),
    ('remoteclip', 'RemoteCLIP\n(contrastive)', '#2196F3', 0.215, 0.878),
    ('georsclip', 'GeoRSCLIP\n(contrastive)', '#03A9F4', 0.247, 0.830),
    ('ssl4eo', 'SSL4EO\n(MAE, RS)', '#FF9800', 0.345, 0.770),
    ('mae_base', 'MAE-base\n(MAE, ImageNet)', '#F44336', 0.521, 0.737),
    ('prithvi', 'Prithvi\n(MAE, RS)', '#9C27B0', 0.663, 0.572),
]

lim = 0.065

for ax, (model_name, label, color, corr, r10) in zip(axes.flat, all_models):
    # Try bigearthnet first, fall back to eurosat
    for dataset in ['bigearthnet', 'eurosat']:
        emb_path = EMBED_DIR / f'{model_name}_{dataset}.npz'
        if emb_path.exists():
            x_coords, y_coords, d = get_rotated_coords(model_name, dataset, seed=42, n=3000)
            break

    ax.set_facecolor('#FAFAFA')
    ax.scatter(x_coords, y_coords, c=color, alpha=0.12, s=4, edgecolors='none', rasterized=True)

    # Contours
    try:
        xy = np.vstack([x_coords, y_coords])
        kde = gaussian_kde(xy)
        xgrid = np.linspace(-lim, lim, 80)
        ygrid = np.linspace(-lim, lim, 80)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=3, colors=color, alpha=0.5, linewidths=0.8)
    except Exception:
        pass

    # Beta grid
    boundaries = get_beta_boundaries(d, bits=4)
    visible = boundaries[(boundaries > -lim) & (boundaries < lim)]
    for b_val in visible:
        ax.axhline(y=b_val, color='#FF9800', linestyle='--', linewidth=0.4, alpha=0.4)
        ax.axvline(x=b_val, color='#FF9800', linestyle='--', linewidth=0.4, alpha=0.4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.08, linewidth=0.5)
    ax.tick_params(labelsize=7, colors='#AAAAAA')

    # Title with R@10 color-coded
    r10_color = '#2E7D32' if r10 > 0.85 else ('#FF8F00' if r10 > 0.75 else '#C62828')
    ax.set_title(f'{label}\ncorr={corr:.3f} | R@10={r10:.3f}',
                 fontsize=9, pad=8, color='#333333')

    # R@10 badge
    badge_color = '#E8F5E9' if r10 > 0.85 else ('#FFF3E0' if r10 > 0.75 else '#FFEBEE')
    badge_text_color = '#2E7D32' if r10 > 0.85 else ('#E65100' if r10 > 0.75 else '#C62828')
    ax.annotate(f'R@10={r10:.2f}', xy=(0.97, 0.95), xycoords='axes fraction',
                fontsize=8, fontweight='bold', color=badge_text_color,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=badge_color,
                          edgecolor=badge_text_color, alpha=0.9, linewidth=0.5))

fig.text(0.5, -0.01,
         'Top row: low correlation (isotropic, round clouds) → high recall. '
         'Bottom row: high correlation (anisotropic, elongated clouds) → lower recall.',
         ha='center', fontsize=10, color='#555555')

plt.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(FIGURES_DIR / 'fig_isotropy_6model.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig_isotropy_6model.pdf', bbox_inches='tight')
plt.close(fig)
print(f'  Saved fig_isotropy_6model.png/pdf')


# =============================================================================
# Figure: Correlation vs Recall scatter (improved version)
# =============================================================================

print('Generating improved correlation vs recall scatter...')

fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

ben_models = [
    ('dinov2', 'DINOv2', 0.253, 0.900, '#4CAF50', 'D'),
    ('remoteclip', 'RemoteCLIP', 0.215, 0.878, '#2196F3', 'o'),
    ('georsclip', 'GeoRSCLIP', 0.247, 0.830, '#03A9F4', 's'),
    ('ssl4eo', 'SSL4EO', 0.345, 0.770, '#FF9800', '^'),
    ('mae_base', 'MAE-base', 0.521, 0.737, '#F44336', 'v'),
    ('prithvi', 'Prithvi', 0.663, 0.572, '#9C27B0', 'P'),
]

corrs = [m[2] for m in ben_models]
tqs = [m[3] for m in ben_models]

# Regression line
z = np.polyfit(corrs, tqs, 1)
x_line = np.linspace(0.1, 0.75, 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.2, linewidth=1.5)

# Shaded regions
ax.axhspan(0.85, 1.0, color='#E8F5E9', alpha=0.3, label='_nolegend_')
ax.axhspan(0.0, 0.7, color='#FFEBEE', alpha=0.3, label='_nolegend_')

for _, name, corr, r10, color, marker in ben_models:
    ax.scatter(corr, r10, c=color, marker=marker, s=150, zorder=5,
               edgecolors='black', linewidth=0.8)
    offset_x, offset_y = 0.015, -0.015
    if name == 'RemoteCLIP':
        offset_x, offset_y = -0.06, 0.02
    if name == 'GeoRSCLIP':
        offset_x, offset_y = 0.015, 0.02
    ax.annotate(name, (corr + offset_x, r10 + offset_y),
                fontsize=9, fontweight='bold', color='#333333')

# r value
r_val = np.corrcoef(corrs, tqs)[0, 1]
ax.text(0.97, 0.03, f'Pearson r = {r_val:.3f}',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#333333', linewidth=1.5))

ax.set_xlabel('Coordinate Correlation After Rotation\n(lower = more isotropic)', fontsize=11)
ax.set_ylabel('TurboQuant R@10 (4-bit)', fontsize=11)
ax.set_title('BigEarthNet (269K vectors)', fontsize=13, fontweight='bold')
ax.set_xlim(0.1, 0.75)
ax.set_ylim(0.5, 0.95)
ax.grid(True, alpha=0.1)

# Region labels
ax.text(0.12, 0.91, 'Compress well', fontsize=8, color='#2E7D32', style='italic')
ax.text(0.12, 0.55, 'Compress poorly', fontsize=8, color='#C62828', style='italic')

plt.tight_layout()
fig.savefig(FIGURES_DIR / 'fig_corr_vs_recall_final.png', dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'fig_corr_vs_recall_final.pdf', bbox_inches='tight')
plt.close(fig)
print(f'  Saved fig_corr_vs_recall_final.png/pdf')


print('\nAll isotropy figures generated.')
