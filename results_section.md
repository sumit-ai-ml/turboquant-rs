# 4. Results

## 4.1 Embedding Isotropy Varies by Training Objective

Before evaluating quantization, we measure the geometry of each model's embeddings. Table 1 reports coordinate correlation and KS D statistic after random orthogonal rotation.

**Table 1: Embedding isotropy by model (BigEarthNet)**

| Model | Training | d | Coord Corr ($\rho$) | KS D |
|-------|----------|---|:----:|:----:|
| RemoteCLIP | Contrastive (CLIP) | 512 | 0.215 | 0.551 |
| GeoRSCLIP | Contrastive (CLIP) | 768 | 0.247 | 0.849 |
| DINOv2 | Self-distillation | 768 | 0.253 | 0.450 |
| SSL4EO | MAE (RS) | 768 | 0.345 | 0.846 |
| MAE-base | MAE (ImageNet) | 768 | 0.521 | 0.360 |
| Prithvi | MAE (RS) | 768 | 0.663 | 0.730 |

The three contrastive/self-distillation models (RemoteCLIP, GeoRSCLIP, DINOv2) all have coordinate correlation below 0.26. The three MAE models (SSL4EO, MAE-base, Prithvi) all have correlation above 0.34, with Prithvi highest at 0.66. This matches the theoretical expectation: contrastive and self-distillation losses push embeddings toward uniform distribution on the sphere (Wang & Isola, 2020), while MAE reconstruction has no such pressure.

Note that the KS D statistic does not cleanly separate the two groups. MAE-base has the lowest KS D (0.360) despite being an MAE model. GeoRSCLIP has the highest KS D (0.849) despite being contrastive. KS D measures distributional fit to the theoretical Beta, which depends on factors beyond isotropy (e.g., dimension, tail behavior). Coordinate correlation is a more direct measure of coordinate independence, which is the property TurboQuant actually relies on.

## 4.2 Quantization Recall Depends on Isotropy

Table 2 presents the main result: TurboQuant MSE Recall@10 at 4 bits across all 6 models and both datasets.

**Table 2: TurboQuant MSE R@10 (4-bit) with PQ and binary hash baselines**

*EuroSAT (16K vectors):*

| Model | Training | $\rho$ | TQ MSE | PQ | BinHash | Gap Closed |
|-------|----------|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | Self-distillation | 0.132 | 0.943 | 0.960 | 0.654 | 95% |
| RemoteCLIP | Contrastive | 0.205 | 0.911 | 0.961 | 0.607 | 86% |
| GeoRSCLIP | Contrastive | 0.190 | 0.882 | 0.965 | 0.576 | 79% |
| MAE-base | MAE | 0.510 | 0.859 | 0.953 | 0.179 | 88% |
| SSL4EO | MAE (RS) | 0.293 | 0.834 | 0.968 | 0.609 | 62% |
| Prithvi | MAE (RS) | 0.629 | 0.779 | 0.961 | 0.451 | 64% |

*BigEarthNet (269K vectors):*

| Model | Training | $\rho$ | TQ MSE | PQ | BinHash | Gap Closed |
|-------|----------|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | Self-distillation | 0.253 | 0.900 | 0.947 | 0.483 | 90% |
| RemoteCLIP | Contrastive | 0.215 | 0.878 | 0.944 | 0.473 | 86% |
| GeoRSCLIP | Contrastive | 0.247 | 0.830 | 0.950 | 0.447 | 76% |
| SSL4EO | MAE (RS) | 0.345 | 0.770 | 0.955 | 0.468 | 62% |
| MAE-base | MAE | 0.521 | 0.737 | 0.935 | 0.128 | 76% |
| Prithvi | MAE (RS) | 0.663 | 0.572 | 0.925 | 0.273 | 46% |

The Pearson correlation between coordinate correlation and TQ R@10 is r = -0.851 on EuroSAT and r = -0.951 on BigEarthNet (Figure 3). Coordinate correlation is a much stronger predictor than KS D (r = -0.507 on BigEarthNet).

Three observations:

First, the top three models by TQ recall are the contrastive/self-distillation models on both datasets. DINOv2 achieves R@10 = 0.90 on BigEarthNet at 4 bits with no training, closing 90% of the gap to trained PQ.

Second, the correlation strengthens at scale. On BigEarthNet (269K vectors), r = -0.951 versus r = -0.851 on EuroSAT (16K vectors). With more database vectors to distinguish between, coordinate independence matters more.

Third, PQ recall is nearly model-independent (0.925-0.968 across all models). PQ learns per-subspace codebooks that adapt to the data geometry, so it handles anisotropy well. TurboQuant uses a fixed codebook that assumes coordinate independence, which is why its recall tracks isotropy.

## 4.3 Ranking Quality Follows the Same Pattern

Recall@k measures whether the right items are retrieved. Kendall's tau and Pearson r measure whether the ranking and similarity structure within the top-1000 neighborhood are preserved. Table 3 reports both.

**Table 3: Ranking quality within top-1000 neighborhood (BigEarthNet, 4-bit)**

| Model | $\rho$ | TQ MSE | | PQ | | BinHash | |
|-------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| | | $\tau$ | Pearson | $\tau$ | Pearson | $\tau$ | Pearson |
| DINOv2 | 0.253 | 0.884 | 0.989 | 0.945 | 0.998 | 0.481 | 0.706 |
| RemoteCLIP | 0.215 | 0.860 | 0.984 | 0.936 | 0.997 | 0.461 | 0.686 |
| GeoRSCLIP | 0.247 | 0.811 | 0.968 | 0.944 | 0.998 | 0.457 | 0.680 |
| SSL4EO | 0.345 | 0.795 | 0.952 | 0.959 | 0.999 | 0.503 | 0.717 |
| MAE-base | 0.521 | 0.744 | 0.919 | 0.930 | 0.996 | 0.253 | 0.408 |
| Prithvi | 0.663 | 0.641 | 0.830 | 0.943 | 0.997 | 0.381 | 0.550 |

TurboQuant on DINOv2 achieves tau = 0.884 and Pearson r = 0.989. This means that within the top-1000 retrieval neighborhood, 94% of pairwise rankings are preserved, and the similarity magnitude structure is almost perfectly maintained. On Prithvi, tau drops to 0.641 and Pearson r to 0.830.

The Pearson r numbers are particularly informative. A value of 0.989 means that TQ-compressed similarities are a near-perfect linear rescaling of the true similarities. The neighbor that is 5% more similar in FP32 is still approximately 5% more similar after compression. A value of 0.830 means the relationship is still positive but noisy, with the fine-grained distance structure partially scrambled.

PQ achieves tau > 0.93 and Pearson r > 0.996 across all models, confirming that learned codebooks handle anisotropy well.

## 4.4 TurboQuant Is the Best Training-Free Method

Table 4 compares all 9 methods on BigEarthNet for Prithvi and RemoteCLIP.

**Table 4: All methods, BigEarthNet, 4-bit R@10 (1-bit for binary methods)**

| Method | Bits | Prithvi | RemoteCLIP | B/vec | Training? |
|--------|:----:|:----:|:----:|:----:|:-:|
| FP32 Exact | - | 1.000 | 1.000 | 3072 / 2048 | - |
| Product Quant | 4 | 0.925 | 0.944 | 384 / 256 | Yes |
| TQ Adaptive | 4 | 0.584 | 0.887 | 388 / 260 | Yes |
| **TQ MSE** | **4** | **0.572** | **0.878** | **388 / 260** | **No** |
| SimHash Multi | 4 | 0.481 | 0.648 | 384 / 256 | No |
| Uniform SQ | 4 | 0.255 | 0.399 | 388 / 260 | No |
| FlyHash | 4 | 0.207 | 0.409 | 384 / 256 | No |
| RandProj Quant | 4 | 0.073 | 0.619 | 384 / 256 | No |
| RaBitQ | 1 | 0.256 | 0.418 | 96 / 64 | No |
| Binary Hash | 1 | 0.273 | 0.473 | 96 / 64 | No |

TurboQuant MSE outperforms all other training-free methods on both models. The margin over SimHash (the runner-up) is 9 percentage points on Prithvi and 23 points on RemoteCLIP.

Among the 1-bit methods, RaBitQ (which adds a random rotation before binarization) slightly outperforms binary hash on some configurations but not consistently. The rotation helps less than expected at 1 bit because too much information is lost in binarization for the rotation to recover.

Random Projection + Quantization performs poorly on Prithvi (R@10 = 0.073) with very high variance across seeds, suggesting the random projection quality varies wildly at d=768.

## 4.5 The Codebook Matters, But Not in the Way You'd Expect

TurboQuant uses a Beta(d/2, d/2) codebook precomputed from the theoretical distribution. Two ablations test whether this matters:

**Table 5: Codebook ablation (BigEarthNet, 4-bit R@10)**

| Method | Codebook | Prithvi | RemoteCLIP | Training? |
|--------|----------|:----:|:----:|:-:|
| TQ MSE | Beta (theoretical) | 0.572 | 0.878 | No |
| TQ Adaptive | Empirical (trained) | 0.584 | 0.887 | Yes |
| Uniform SQ | Uniform [-1, 1] | 0.255 | 0.399 | No |

Two findings. First, the Beta codebook provides a 2.2x recall improvement over uniform quantization on both models. This is because rotated unit-norm coordinates concentrate within a narrow range (approximately $\pm 0.036$ for d=768, corresponding to 3 standard deviations of $\mathcal{N}(0, 1/d)$). A uniform grid on [-1, 1] places 96% of its bins on empty space. The Beta codebook concentrates all quantization levels in the data-occupied range.

Second, replacing the theoretical Beta codebook with one trained on the actual data distribution gives only +1-2% R@10. The codebook assumption is not the bottleneck. Correlated quantization error from non-independent coordinates is the dominant source of quality loss, and a better codebook cannot fix correlated noise.

A related observation: Uniform SQ is insensitive to bit-width. At d=768, R@10 is 0.256, 0.256, and 0.255 at 2, 3, and 4 bits respectively on Prithvi/BigEarthNet. More bits simply subdivide the empty tails of the [-1, 1] range without improving quantization of the narrow central region where data actually lives.

## 4.6 Scaling from 16K to 269K Vectors

All methods degrade when the database grows from 16K (EuroSAT) to 269K (BigEarthNet). More database vectors means more near-neighbors to distinguish, making the quantization task harder.

**Table 6: Scaling behavior (TQ MSE 4-bit R@10)**

| Model | EuroSAT (16K) | BigEarthNet (269K) | Drop |
|-------|:----:|:----:|:----:|
| DINOv2 | 0.943 | 0.900 | -0.043 |
| RemoteCLIP | 0.911 | 0.878 | -0.033 |
| GeoRSCLIP | 0.882 | 0.830 | -0.052 |
| SSL4EO | 0.834 | 0.770 | -0.064 |
| MAE-base | 0.859 | 0.737 | -0.122 |
| Prithvi | 0.779 | 0.572 | -0.207 |

Isotropic models (DINOv2, RemoteCLIP) degrade by 3-5 percentage points. Anisotropic models (MAE-base, Prithvi) degrade by 12-21 points. The isotropy gap widens at scale.

For comparison, PQ degrades by only 1-4 points across all models (0.961 to 0.925 for Prithvi, 0.961 to 0.944 for RemoteCLIP), confirming that learned codebooks handle scaling better.

## 4.7 Bit-Width Scaling

Table 7 shows how recall varies with bit-width for the two original models across both datasets.

**Table 7: R@10 at 2, 3, 4 bits (TQ MSE)**

*EuroSAT:*

| Method | Prithvi | | | RemoteCLIP | | |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|
| | 2-bit | 3-bit | 4-bit | 2-bit | 3-bit | 4-bit |
| TQ MSE | 0.636 | 0.711 | 0.779 | 0.715 | 0.834 | 0.911 |
| PQ | 0.901 | 0.930 | 0.961 | 0.884 | 0.884 | 0.961 |

*BigEarthNet:*

| Method | Prithvi | | | RemoteCLIP | | |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|
| | 2-bit | 3-bit | 4-bit | 2-bit | 3-bit | 4-bit |
| TQ MSE | 0.382 | 0.485 | 0.572 | 0.613 | 0.778 | 0.878 |
| PQ | 0.834 | 0.875 | 0.925 | 0.835 | 0.835 | 0.944 |

TQ recall increases monotonically with bits for all model-dataset combinations, as expected. The per-bit gain is larger for RemoteCLIP (isotropic) than for Prithvi (anisotropic). Going from 2-bit to 4-bit on BigEarthNet, RemoteCLIP gains 26.5 points (0.613 to 0.878) while Prithvi gains 19 points (0.382 to 0.572). Extra bits help more when the codebook assumption is closer to correct.
