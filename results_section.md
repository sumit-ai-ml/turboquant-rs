# 4. Results

We organize results around four comparison axes. Section 4.1 provides comprehensive tables for both datasets. Section 4.2 compares methods to isolate what each design choice contributes. Section 4.3 compares datasets to show how scale affects quantization. Section 4.4 compares models and develops the isotropy story that underlies all other findings.

## 4.1 Comprehensive Results

Two tables (one per dataset) cover the full method $\times$ model matrix. Rows are grouped: trained methods (PQ, TQ Adaptive), the TQ family and its direct ablation (TQ MSE, SimHash Multi, Uniform SQ, RaBitQ, Binary Hash), and other training-free methods (FlyHash, RandProj Quant). Columns are grouped: contrastive and self-distillation models (DINOv2, RemoteCLIP, GeoRSCLIP), then MAE models (SSL4EO, MAE-base, Prithvi). The TQ MSE row is our headline method and is shown in bold. All values are R@10 averaged over 5 seeds.

The FP32 Exact row is 1.000 by definition and not shown. B/vec is the storage cost at d=768 (RemoteCLIP uses d=512, so its actual bytes are ~33% smaller for the rotation-based methods: 260 vs 388; 128 vs 192 for binary methods).

### Table 1: EuroSAT (16K vectors) R@10 at 4 bits (1 bit for RaBitQ and Binary Hash)

| Method | DINOv2 | RemoteCLIP | GeoRSCLIP | SSL4EO | MAE-base | Prithvi | Train | B/vec |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|:-:|:----:|
| PQ | 0.960 | 0.961 | 0.965 | 0.968 | 0.953 | 0.961 | yes | 384 |
| TQ Adaptive | --- | 0.912 | --- | --- | --- | 0.782 | yes | 388 |
| **TQ MSE** | **0.943** | **0.911** | **0.882** | **0.834** | **0.859** | **0.779** | **no** | **388** |
| SimHash Multi | 0.792 | 0.751 | 0.708 | 0.743 | 0.744 | 0.702 | no | 384 |
| Uniform SQ | 0.660 | 0.549 | 0.499 | 0.544 | 0.558 | 0.502 | no | 388 |
| RaBitQ | 0.659 | 0.567 | 0.501 | 0.544 | 0.558 | 0.502 | no | 96 |
| Binary Hash | 0.654 | 0.607 | 0.576 | 0.609 | 0.179 | 0.451 | no | 96 |
| FlyHash | 0.592 | 0.545 | 0.497 | 0.468 | 0.209 | 0.468 | no | 384 |
| RandProj Quant | 0.724 | 0.718 | 0.671 | 0.518 | 0.562 | 0.394 | no | 384 |

### Table 2: BigEarthNet (269K vectors) R@10 at 4 bits (1 bit for RaBitQ and Binary Hash)

| Method | DINOv2 | RemoteCLIP | GeoRSCLIP | SSL4EO | MAE-base | Prithvi | Train | B/vec |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|:-:|:----:|
| PQ | 0.947 | 0.944 | 0.950 | 0.955 | 0.935 | 0.925 | yes | 384 |
| TQ Adaptive | --- | 0.887 | --- | --- | --- | 0.584 | yes | 388 |
| **TQ MSE** | **0.900** | **0.878** | **0.830** | **0.770** | **0.737** | **0.572** | **no** | **388** |
| SimHash Multi | 0.688 | 0.648 | 0.601 | 0.651 | 0.573 | 0.481 | no | 384 |
| Uniform SQ | 0.479 | 0.399 | 0.349 | 0.422 | 0.338 | 0.255 | no | 388 |
| RaBitQ | 0.479 | 0.418 | 0.348 | 0.422 | 0.337 | 0.256 | no | 96 |
| Binary Hash | 0.483 | 0.473 | 0.447 | 0.468 | 0.128 | 0.273 | no | 96 |
| FlyHash | 0.432 | 0.409 | 0.340 | 0.354 | 0.152 | 0.207 | no | 384 |
| RandProj Quant | 0.595 | 0.619 | 0.505 | 0.336 | 0.251 | 0.073 | no | 384 |

The headline BigEarthNet result: DINOv2 with TQ MSE achieves R@10 = 0.900, Kendall's $\tau$ = 0.884, and Pearson $r$ = 0.989, with no training data (Table 10 below).

The cells marked `---` for TQ Adaptive on DINOv2, GeoRSCLIP, SSL4EO, and MAE-base indicate the empirical Lloyd-Max codebook was not run on those models. TQ Adaptive tracks TQ MSE within 1-2 points on Prithvi and RemoteCLIP (where both were measured), and we expect similar behavior on the other four models.

### Observations from the comprehensive matrix

Three patterns emerge across both datasets.

First, every method's recall is ordered consistently across models: DINOv2 best, Prithvi worst, with contrastive/distillation models above MAE models. This ordering holds for PQ (weakly), for TQ MSE (strongly), and for every training-free method. The model ordering is the same regardless of quantizer. Section 4.4 develops this as the isotropy story.

Second, Uniform SQ and RaBitQ give nearly identical R@10 at 4 bits. This is a quantization artifact: rotated unit-norm coordinates at d=768 concentrate within $\pm$0.036, so a uniform grid on [-1, 1] collapses almost all values into the two central bins (effectively 2-level, sign-like quantization). Uniform SQ at 4 bits is degenerate at high d; more bits do not help. The Beta codebook in TQ MSE avoids this by concentrating levels in the data-occupied range.

Third, RandProj Quant has high variance across models. It performs reasonably on isotropic models (R@10 = 0.72 on DINOv2/EuroSAT) but collapses on Prithvi (R@10 = 0.07 on BigEarthNet). The random projection quality is highly sensitive to the covariance structure of the input.

## 4.2 Method Comparison

We compare methods pairwise to isolate what each design choice contributes. Each contrast fixes all design choices except one.

### Rotation effect (Binary Hash vs RaBitQ)

Both operate at 1 bit per dimension. Both use Hamming distance. The only difference: RaBitQ applies a random orthogonal rotation before taking sign bits. Table 5 compares them.

### Table 5: Rotation effect at 1 bit (BigEarthNet)

| Model | Binary Hash R@10 | RaBitQ R@10 | Rotation gain |
|-------|:----:|:----:|:----:|
| DINOv2 | 0.483 | 0.479 | -0.004 |
| RemoteCLIP | 0.473 | 0.418 | -0.055 |
| GeoRSCLIP | 0.447 | 0.348 | -0.099 |
| SSL4EO | 0.468 | 0.422 | -0.046 |
| MAE-base | 0.128 | 0.337 | +0.209 |
| Prithvi | 0.273 | 0.256 | -0.017 |
| Mean | --- | --- | -0.002 |

Rotation at 1 bit is inconsistent. On 5 of 6 models it hurts or has negligible effect. On MAE-base it helps dramatically (+21 points), because MAE-base's raw coordinates are basis-aligned with the training objective's output heads and produce poor binary codes without rotation. The mean effect across all 6 models is close to zero.

### Codebook quality effect (Uniform SQ vs TQ MSE)

Both apply the same random rotation. Both use 4 bits per dimension. The difference: Uniform SQ uses a uniform grid on [-1, 1]; TQ MSE uses a Beta(d/2, d/2) optimal codebook. Table 6 shows the result.

### Table 6: Codebook quality at 4 bits (BigEarthNet)

| Model | Uniform SQ R@10 | TQ MSE R@10 | Codebook gain | Ratio |
|-------|:----:|:----:|:----:|:----:|
| DINOv2 | 0.479 | 0.900 | +0.421 | 1.88x |
| RemoteCLIP | 0.399 | 0.878 | +0.479 | 2.20x |
| GeoRSCLIP | 0.349 | 0.830 | +0.481 | 2.38x |
| SSL4EO | 0.422 | 0.770 | +0.348 | 1.83x |
| MAE-base | 0.338 | 0.737 | +0.399 | 2.18x |
| Prithvi | 0.255 | 0.572 | +0.317 | 2.24x |
| Mean | --- | --- | +0.408 | 2.12x |

The Beta codebook more than doubles recall across all 6 models, mean gain +0.408. Rotated unit-norm coordinates at d=768 concentrate within approximately $\pm$0.036. A uniform grid on [-1, 1] places 96% of bins on empty space and effectively degenerates to 2-level (sign-like) quantization. The Beta codebook puts all levels in the data-occupied range.

### Training data effect (TQ MSE vs TQ Adaptive)

Both use the same rotation. Both use a Lloyd-Max codebook. The difference: TQ MSE uses the theoretical Beta distribution; TQ Adaptive fits the codebook to the empirical distribution of rotated coordinates from a training set.

### Table 7: Training the codebook (BigEarthNet, 4-bit)

| Model | TQ MSE R@10 | TQ Adaptive R@10 | Training gain |
|-------|:----:|:----:|:----:|
| Prithvi | 0.572 | 0.584 | +0.012 |
| RemoteCLIP | 0.878 | 0.887 | +0.009 |

Training the codebook adds 1 to 2 percentage points. The theoretical Beta codebook already captures most of what an empirical codebook would. The codebook assumption is not the bottleneck.

### Subspace structure effect (TQ MSE vs PQ)

Both use codebooks. Both operate at 4 bits per dimension of budget. The difference: TQ quantizes each coordinate independently; PQ partitions the vector into m subspaces and learns joint codebooks over each subspace via k-means.

### Table 8: Subspace structure (BigEarthNet, 4-bit)

| Model | TQ MSE R@10 | PQ R@10 | Subspace gain |
|-------|:----:|:----:|:----:|
| DINOv2 | 0.900 | 0.947 | +0.047 |
| RemoteCLIP | 0.878 | 0.944 | +0.066 |
| GeoRSCLIP | 0.830 | 0.950 | +0.120 |
| SSL4EO | 0.770 | 0.955 | +0.185 |
| MAE-base | 0.737 | 0.935 | +0.198 |
| Prithvi | 0.572 | 0.925 | +0.353 |

The subspace gain scales with anisotropy. For DINOv2 (isotropic), PQ is only 5 points ahead. For Prithvi (anisotropic), PQ is 35 points ahead. PQ learns per-subspace joint structure that captures coordinate correlations within each subspace; TQ's per-coordinate independence assumption does not.

### Ranking of design contributions

Combining the four contrasts on BigEarthNet, ordered by mean gain across tested models:

| Design choice | Mean gain | Range | Models tested |
|---------------|:----:|:----:|:----:|
| Beta codebook (TQ MSE over Uniform SQ) | +0.408 | +0.32 to +0.48 | All 6 |
| Subspace structure (PQ over TQ MSE) | +0.162 | +0.05 to +0.35 | All 6 |
| Empirical codebook (TQ Adaptive over TQ MSE) | +0.011 | +0.01 to +0.01 | Prithvi, RemoteCLIP |
| Rotation alone at 1 bit (RaBitQ over Binary Hash) | -0.002 | -0.10 to +0.21 | All 6 |

![Figure 6: All 9 methods on Prithvi and RemoteCLIP](figures/fig6_all_methods.png)
*Figure 6: All 9 methods on BigEarthNet for Prithvi and RemoteCLIP. TQ MSE (blue) is the best training-free method on both models.*

![Figure 7: Codebook ablation](figures/fig7_codebook_ablation.png)
*Figure 7: Codebook ablation across bit-widths. Beta (blue) gives 2.2x over Uniform (gray). Adaptive (purple) adds only +1% over Beta.*

The mechanistic picture: the largest gain comes from using a codebook matched to the narrow data range that rotated coordinates occupy (the Beta codebook vs uniform grid contrast). The next largest gain comes from learning joint structure across correlated coordinates within subspaces (what PQ does on top of TQ). Training the codebook on real data adds little. Rotation alone at 1 bit is inconsistent.

A caveat on this ranking. The Beta codebook gain is measured against Uniform SQ, which is a weak baseline because a uniform grid on [-1, 1] wastes most bins on empty space. The subspace structure gain is measured against TQ MSE, which already has the Beta codebook advantage. So the Beta codebook number reflects what happens when you go from no useful codebook to a data-matched codebook, while the subspace structure number reflects what happens when you go from a good scalar quantizer to a trained vector quantizer. They are not directly comparable design decisions but they quantify what each mechanism is worth relative to its natural alternative.

## 4.3 Dataset Comparison

EuroSAT has 16K vectors across 10 land-use classes. BigEarthNet has 269K vectors across 43 multi-label classes. BigEarthNet is 17x larger and the classes overlap more, so neighbor rankings are harder to preserve under quantization.

### Table 9: R@10 drop when scaling from EuroSAT to BigEarthNet (4-bit)

| Model | EuroSAT TQ | BigEarthNet TQ | TQ drop | EuroSAT PQ | BigEarthNet PQ | PQ drop |
|-------|:----:|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | 0.943 | 0.900 | -0.043 | 0.960 | 0.947 | -0.013 |
| RemoteCLIP | 0.911 | 0.878 | -0.033 | 0.961 | 0.944 | -0.017 |
| GeoRSCLIP | 0.882 | 0.830 | -0.052 | 0.965 | 0.950 | -0.015 |
| SSL4EO | 0.834 | 0.770 | -0.064 | 0.968 | 0.955 | -0.013 |
| MAE-base | 0.859 | 0.737 | -0.122 | 0.953 | 0.935 | -0.018 |
| Prithvi | 0.779 | 0.572 | -0.207 | 0.961 | 0.925 | -0.036 |

![Figure 8: Scaling slope chart](figures/fig8_scaling.png)
*Figure 8: TQ MSE R@10 scaling from EuroSAT (16K) to BigEarthNet (269K). Isotropic models (green, blue) degrade 3-5 points. Anisotropic models (red, purple) degrade 12-21 points.*

Three observations:

Isotropic models degrade gracefully. DINOv2, RemoteCLIP, and GeoRSCLIP drop by 3 to 5 points.

Anisotropic models degrade steeply. Prithvi drops by 21 points (0.779 to 0.572). MAE-base drops by 12 points.

PQ degrades uniformly and slightly across all models, losing 1 to 4 points regardless of model isotropy. Learned per-subspace codebooks generalize well from 16K training vectors to the 269K test set.

The isotropy gap widens with scale. On EuroSAT the DINOv2 vs Prithvi gap is 16 points (0.943 vs 0.779). On BigEarthNet it is 33 points (0.900 vs 0.572). Anisotropy hurts more when more near-neighbors need to be disambiguated.

### Implications for archive-scale deployment

Global Sentinel-2 archives contain tens of millions of patches. Extrapolating from the 16K to 269K trend:

For isotropic models (DINOv2, RemoteCLIP, GeoRSCLIP), TQ MSE at 4 bits is a viable training-free solution at archive scale. Recall is expected to remain above 0.80 at 10M patches.

For anisotropic models (Prithvi, MAE-base, SSL4EO), training-free TQ MSE is unlikely to scale. Options are to use PQ (which requires training a codebook on a representative sample), to increase the bit budget at the cost of more storage, or to use a different foundation model.

Our data does not measure behavior above 269K. The 10M estimate is an extrapolation.

## 4.4 Model Comparison

This subsection develops the isotropy story that underlies all previous findings. We measure embedding geometry, connect it to quantization recall, and provide a practical diagnostic for practitioners.

### Table 10: Embedding isotropy by model (BigEarthNet)

Mean absolute pairwise correlation between rotated coordinates ($\rho$), averaged over 5 rotation seeds, $\pm$ one standard deviation. Participation ratio (PR) is the rotation-invariant measure $(\sum_i \lambda_i)^2 / (d \sum_i \lambda_i^2)$ of the covariance eigenvalues.[^fn1]

| Model | Training | d | $\rho$ (mean $\pm$ std) | PR | KS D |
|-------|----------|---|:----:|:----:|:----:|
| RemoteCLIP | Contrastive (CLIP) | 512 | 0.206 $\pm$ 0.008 | 0.0205 | 0.551 |
| GeoRSCLIP | Contrastive (CLIP) | 768 | 0.229 $\pm$ 0.019 | 0.0100 | 0.849 |
| DINOv2 | Self-distillation | 768 | 0.260 $\pm$ 0.015 | 0.0078 | 0.450 |
| SSL4EO | MAE (RS) | 768 | 0.318 $\pm$ 0.020 | 0.0068 | 0.846 |
| MAE-base | MAE (ImageNet) | 768 | 0.524 $\pm$ 0.005 | 0.0020 | 0.360 |
| Prithvi | MAE (RS) | 768 | 0.673 $\pm$ 0.057 | 0.0016 | 0.730 |

[^fn1]: Mean absolute pairwise correlation is not a rotation invariant, so we report it over multiple random rotation seeds. Strictly, we measure the correlation structure that TurboQuant's own random rotation exposes. The rotation-invariant participation ratio PR gives the same model ordering (lower $\rho$ corresponds to higher PR), but a weaker predictive relationship: Pearson $r$(PR, TQ R@10) $= 0.703$ on BigEarthNet, versus $r(\rho$, TQ R@10$) = -0.951$. We use $\rho$ because it directly measures the quantity that governs quantization error accumulation: the correlation between coordinates after the specific rotation the quantizer applies. PR measures the eigenvalue spread of the covariance, which is a weaker proxy for what TurboQuant actually sees.

KS D does not cleanly separate models. MAE-base has the lowest KS D (0.360) despite being an MAE model. Coordinate correlation and PR both give the same ordering: contrastive and self-distillation models at the low-$\rho$/high-PR end, MAE models at the high-$\rho$/low-PR end.

### Figure 1: Isotropy illustration

![Figure 1: Isotropy illustration](figures/fig1_isotropy_illustration.png)
*Figure 1: Pairwise rotated coordinates from BigEarthNet embeddings. Left: DINOv2 produces a round scatter cloud with coordinates that are approximately independent. Right: Prithvi produces an elongated cloud with coordinates that remain correlated even after rotation. Orange dotted lines show 4-bit Beta codebook boundaries.*

### The correlation between isotropy and recall

### Figure 3: Correlation vs recall

![Figure 3: Correlation vs recall](figures/fig3_corr_vs_recall.png)
*Figure 3: Coordinate correlation versus TQ MSE R@10 on BigEarthNet. Pearson r = -0.951 across 6 models. The dashed line is a linear fit. Shaded regions indicate the compress-well (R@10 > 0.85, green) and compress-poorly (R@10 < 0.65, red) zones.*

The Pearson correlation between $\rho$ and R@10 is $r = -0.851$ on EuroSAT and $r = -0.951$ on BigEarthNet. KS D gives $r = -0.507$. Coordinate correlation is a much better predictor.

### Figure 4: 6-model bars

![Figure 4: 6-model bars](figures/fig4_6model_bars.png)
*Figure 4: Recall@10 on BigEarthNet for all six models. Green: TQ MSE (no training). Yellow: PQ (trained). Gray: Binary Hash (1 bit). Percentages above TQ bars show the gap closed between Binary Hash and PQ.*

### Figure 5: Ranking quality

![Figure 5: Ranking quality](figures/fig5_ranking_quality.png)
*Figure 5: Kendall's $\tau$ within the top-1000 neighborhood. The same ordering as R@10: DINOv2 $\tau$=0.88, Prithvi $\tau$=0.64. PQ (yellow) maintains $\tau$ > 0.93 for all models.*

### Group patterns and outliers

The contrastive and self-distillation models cluster at low $\rho$ (0.21 to 0.26 on BigEarthNet). The MAE models cluster at high $\rho$ (0.32 to 0.67).

SSL4EO is an interesting case: it is an MAE model trained on remote sensing, but has $\rho = 0.318$, closer to the contrastive group than to Prithvi. The large SSL4EO pretraining corpus (250K global locations, Sentinel-1 plus Sentinel-2) may contribute to the moderate isotropy. Its TQ MSE R@10 (0.770) is correspondingly higher than Prithvi's (0.572).

MAE-base (general-domain MAE on ImageNet) has $\rho = 0.524$, sitting between SSL4EO and Prithvi. Its TQ MSE R@10 (0.737) follows the pattern.

Prithvi is the most anisotropic model tested. Its training mix (multi-temporal Sentinel-2 over one region, 6-band MAE) produces the most correlated rotated coordinates and the lowest training-free compression recall.

### Practitioner diagnostic

Coordinate correlation can be computed on a sample of 5,000 embeddings in seconds using any random orthogonal rotation. We suggest the following rule:

| $\rho$ range | Recommendation |
|--------------|---------------|
| $\rho < 0.30$ | Use TurboQuant MSE. Expected R@10 gap to PQ: under 7 points. |
| $0.30 \leq \rho < 0.50$ | Test both. TQ loses 12 to 18 points to PQ at this range. |
| $\rho \geq 0.50$ | Use PQ if training data is available. TQ loses over 20 points. |

These thresholds are calibrated on BigEarthNet (269K vectors). At smaller scales the gap narrows; at larger scales we expect it to widen.
