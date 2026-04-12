# TurboQuant for Remote Sensing: Training-Free Embedding Quantization for Scalable Retrieval

## Abstract

Large-scale remote sensing (RS) retrieval systems built on foundation model embeddings face a storage bottleneck: a 590K-patch archive at 768 dimensions requires 1.7 GB in FP32. We evaluate TurboQuant, a training-free scalar quantization method based on random orthogonal rotation and an analytically optimal codebook, for compressing RS embeddings from two foundation models — Prithvi-EO (ViT-MAE, d=768) and RemoteCLIP (CLIP ViT-B-32, d=512) — on EuroSAT (16K patches) and BigEarthNet (269K patches). At 4 bits per dimension, TurboQuant achieves Recall@10 of 0.878 on RemoteCLIP/BigEarthNet and 0.572 on Prithvi/BigEarthNet, closing 86% and 46% of the gap between binary hashing and trained Product Quantization respectively — without any training data or codebook fitting. We benchmark TurboQuant against seven alternative methods and find that its performance depends critically on the isotropy of the embedding space: contrastive models (RemoteCLIP) produce near-isotropic embeddings that closely match TurboQuant's distributional assumptions, while reconstruction-based models (Prithvi) produce anisotropic embeddings that degrade quantization quality. These results establish TurboQuant as a practical, zero-cost compression baseline for RS embedding retrieval and highlight embedding geometry as a key factor in quantization performance.

## 1. Introduction

Foundation models for remote sensing — including Prithvi-EO (Jakubik et al., 2023) and RemoteCLIP (Liu et al., 2023) — produce high-dimensional embeddings that enable large-scale content-based image retrieval (CBIR) over satellite archives. As these archives grow to millions of patches (e.g., BigEarthNet with 590K Sentinel-2 tiles), storing FP32 embeddings becomes a significant cost: 590K vectors at d=768 occupy 1.7 GB, and this scales linearly with archive size.

Vector quantization compresses embeddings to a few bits per dimension, reducing storage by 8-16x while preserving approximate nearest neighbor retrieval quality. Product Quantization (PQ) is the standard approach, but it requires training codebooks on a representative data sample — a process that is dataset-specific and must be repeated when the embedding model changes.

TurboQuant (Guo et al., 2024) offers a training-free alternative. It exploits a theoretical result: after applying a random orthogonal rotation to a unit-norm vector, each coordinate follows a Beta(d/2, d/2) distribution. This known distribution allows precomputing an MSE-optimal Lloyd-Max codebook analytically — no training data needed. The method consists of three steps: (1) L2-normalize and store the norm, (2) rotate via a random orthogonal matrix, and (3) quantize each rotated coordinate using the precomputed codebook.

We evaluate whether this theoretical elegance translates to practical gains on real RS embeddings, where the distributional assumptions may not hold. We benchmark TurboQuant against seven alternative methods across two foundation models and two datasets, and find that performance depends critically on the geometry of the embedding space.

## 2. Methods

### 2.1 TurboQuant MSE

Given a unit-norm embedding x in R^d, TurboQuant applies a random orthogonal rotation P and quantizes each coordinate of Px independently using a scalar codebook optimized for the Beta(d/2, d/2) distribution via Lloyd-Max iteration. The codebook requires no training data — it is computed once from the known distribution and reused for all vectors. At b bits per dimension, each vector is stored as d*b/8 + 4 bytes (quantized codes plus the FP32 norm).

During search, quantized codes are decoded to centroids, inverse-rotated, re-normalized, and compared via cosine similarity. The re-normalization step is critical: without it, quantization shrinkage (decoded vectors having norms < 1) introduces a systematic bias that inverts the recall-vs-bits relationship (Section 4.1).

### 2.2 Baselines

We compare against seven alternative methods:

**Training-free:**
- **Binary Hash**: sign(x), Hamming distance. The simplest possible baseline. 1 bit/dim.
- **RaBitQ** (Gao & Long, 2024): Random rotation + binarization + Hamming distance. Same as binary hash but with a random orthogonal rotation before binarization. 1 bit/dim.
- **SimHash Multi-bit**: k independent random hyperplanes, 1 bit each. Total hash bits = b*d.
- **Uniform Scalar Quantization**: Same rotation as TurboQuant, but with a uniform grid on [-1, 1] instead of the Beta-optimal codebook.
- **FlyHash**: Bio-inspired sparse random expansion (6 connections per output neuron) + winner-take-all activation.
- **Random Projection + Quantization**: Project to m < d dimensions via Gaussian matrix, then uniformly quantize to 8 bits per projected dimension.

**Requires training:**
- **Product Quantization (PQ)**: FAISS IndexPQ with 8 bits per subspace. Trained on an 80% split of the data.
- **TurboQuant Adaptive**: Same as TurboQuant MSE but trains the Lloyd-Max codebook on the empirical distribution of rotated coordinates rather than the theoretical Beta.

### 2.3 Models and Datasets

**Prithvi-EO-1.0-100M** (IBM/NASA): A Vision Transformer trained as a Masked Autoencoder (MAE) on multi-temporal Sentinel-2 imagery. Embed dim = 768. The MAE training objective reconstructs masked patches, producing embeddings optimized for spatial reconstruction rather than discriminative retrieval.

**RemoteCLIP ViT-B-32**: A CLIP model fine-tuned on remote sensing image-text pairs using contrastive learning. Embed dim = 512. The contrastive objective explicitly maximizes cosine similarity between matched image-text pairs while minimizing it for unmatched pairs, producing embeddings distributed more uniformly on the unit sphere.

**EuroSAT**: 27,000 Sentinel-2 patches across 10 land-use classes. We use the train split (16,200 patches after torchgeo loading).

**BigEarthNet-S2**: 590,326 Sentinel-2 patches with 43 multi-label classes. We use the train split (269,695 patches).

### 2.4 Evaluation Protocol

For each configuration, we split embeddings 80/20 into train and evaluation sets. From the evaluation set, we use the first min(1000, 10%) vectors as queries and the rest as the database. Ground truth is computed via exact FP32 cosine similarity. We report Recall@k (k = 1, 10, 100), defined as the fraction of true top-k neighbors found in the approximate top-k. All experiments use 5 random seeds (42, 123, 456, 789, 1024); we report mean and standard deviation.

## 3. Results

### 3.1 Main Results

Table 1 shows Recall@10 at 4 bits per dimension across all configurations.

**EuroSAT (16K vectors):**

| Method | Prithvi R@10 | RemoteCLIP R@10 | Training? |
|--------|-------------|----------------|-----------|
| FP32 Exact | 1.000 | 1.000 | - |
| Product Quant | 0.961 +/- 0.002 | 0.961 +/- 0.002 | Yes |
| **TurboQuant MSE** | **0.779 +/- 0.007** | **0.911 +/- 0.006** | **No** |
| TurboQuant Adaptive | 0.782 +/- 0.007 | 0.912 +/- 0.002 | Yes |
| SimHash Multi-bit | 0.702 +/- 0.016 | 0.751 +/- 0.009 | No |
| Uniform SQ | 0.502 +/- 0.012 | 0.549 +/- 0.005 | No |
| FlyHash | 0.468 +/- 0.005 | 0.545 +/- 0.007 | No |
| RaBitQ (1-bit) | 0.502 +/- 0.010 | 0.567 +/- 0.011 | No |
| Binary Hash (1-bit) | 0.451 +/- 0.011 | 0.607 +/- 0.005 | No |

**BigEarthNet (269K vectors):**

| Method | Prithvi R@10 | RemoteCLIP R@10 | Training? |
|--------|-------------|----------------|-----------|
| FP32 Exact | 1.000 | 1.000 | - |
| Product Quant | 0.925 +/- 0.002 | 0.944 +/- 0.002 | Yes |
| **TurboQuant MSE** | **0.572 +/- 0.002** | **0.878 +/- 0.002** | **No** |
| TurboQuant Adaptive | 0.584 +/- 0.012 | 0.887 +/- 0.001 | Yes |
| SimHash Multi-bit | 0.481 +/- 0.010 | 0.648 +/- 0.006 | No |
| Uniform SQ | 0.255 +/- 0.010 | 0.399 +/- 0.008 | No |
| FlyHash | 0.207 +/- 0.012 | 0.409 +/- 0.010 | No |
| RaBitQ (1-bit) | 0.256 +/- 0.010 | 0.418 +/- 0.005 | No |
| Binary Hash (1-bit) | 0.273 +/- 0.005 | 0.473 +/- 0.003 | No |

Two patterns are immediately visible:

1. **TurboQuant MSE is the best training-free method** across all configurations, with a significant margin over SimHash (the second best).

2. **Performance is dramatically model-dependent**: TurboQuant closes 86% of the gap between binary hashing and PQ on RemoteCLIP/BigEarthNet, but only 46% on Prithvi/BigEarthNet. This gap — 0.878 vs 0.572 in R@10 — is the central finding of this paper.

### 3.2 The Isotropy Gap: Why RemoteCLIP Works and Prithvi Doesn't

TurboQuant's theoretical guarantee relies on a key assumption: after random rotation, each coordinate of a unit-norm vector follows Beta(d/2, d/2) independently. This holds exactly when embeddings are uniformly distributed on the unit sphere (isotropic). Real embeddings deviate from this assumption, and the degree of deviation predicts quantization quality.

We quantify this using two metrics from our Beta validation phase:

| Model | KS D statistic | Coordinate Independence |
|-------|---------------|------------------------|
| Prithvi (d=768) | 0.594 | 0.577 |
| RemoteCLIP (d=512) | 0.405 | 0.179 |

**RemoteCLIP embeddings are significantly more isotropic than Prithvi embeddings.** The KS D statistic (lower = better fit to Beta) is 31% lower for RemoteCLIP, and the mean absolute pairwise correlation between rotated coordinates is 3.2x lower (0.179 vs 0.577). This means RemoteCLIP's rotated coordinates are closer to independent and identically distributed — exactly the setting where TurboQuant's codebook is optimal.

**The training objective explains the difference.** RemoteCLIP uses contrastive learning (InfoNCE loss), which maximizes cosine similarity between matched image-text pairs while pushing unmatched pairs apart. This objective has been shown to produce embeddings that are more uniformly distributed on the unit hypersphere — a phenomenon known as the "uniformity" property of contrastive representations (Wang & Isola, 2020). The more uniform the distribution, the better TurboQuant's Beta assumption holds.

Prithvi, by contrast, uses a Masked Autoencoder (MAE) objective that reconstructs masked patches. This produces embeddings optimized for spatial reconstruction rather than discriminative retrieval. The resulting embedding space has strong directional structure — certain coordinates carry disproportionate variance, leading to high inter-coordinate correlation (0.577) that the rotation cannot fully remove. When TurboQuant quantizes these non-independent coordinates with a codebook designed for independent Beta-distributed values, the quantization error accumulates in a correlated fashion, degrading retrieval quality.

### 3.3 The Codebook Matters More Than the Data

A natural question is whether TurboQuant's performance gap on Prithvi is due to the fixed Beta codebook being suboptimal for non-isotropic data. We test this with TurboQuant Adaptive, which trains the Lloyd-Max codebook on the actual rotated coordinate distribution.

| Method | Prithvi/BEN R@10 | RemoteCLIP/BEN R@10 |
|--------|-----------------|-------------------|
| TQ MSE (Beta codebook) | 0.572 | 0.878 |
| TQ Adaptive (empirical) | 0.584 | 0.887 |
| Delta | +0.012 | +0.009 |

The adaptive codebook provides only marginal improvement (+1-2% R@10), confirming that **the codebook mismatch is not the primary source of error**. The correlated quantization error from non-independent coordinates is the dominant factor. A better codebook cannot fix correlated noise.

### 3.4 Codebook Ablation: Beta vs Uniform

To measure the value of TurboQuant's analytically optimal codebook, we compare against Uniform Scalar Quantization — same rotation, but with a uniform grid on [-1, 1].

| Method | Prithvi/BEN R@10 | RemoteCLIP/BEN R@10 |
|--------|-----------------|-------------------|
| TQ MSE (Beta codebook) | 0.572 | 0.878 |
| Uniform SQ | 0.255 | 0.399 |

The Beta codebook provides a 2.2x improvement on both models. Notably, Uniform SQ is **insensitive to bit-width**: at d=768, R@10 is 0.256, 0.256, and 0.255 at 2, 3, and 4 bits respectively. This occurs because rotated coordinates of unit-norm vectors concentrate within +/-0.036 (3 standard deviations of N(0, 1/768)), so a uniform grid on [-1, 1] places nearly all values in a single central bin regardless of granularity. The Beta(d/2, d/2) codebook avoids this by concentrating all quantization levels in the narrow data-occupied range.

### 3.5 Scaling: EuroSAT to BigEarthNet

All methods degrade when scaling from 16K to 269K vectors, as expected — more database vectors means more potential confounders at similar distances. The degradation is not uniform across methods:

| Method | Prithvi EuroSAT | Prithvi BEN | Delta |
|--------|----------------|-------------|-------|
| PQ (4-bit) | 0.961 | 0.925 | -0.036 |
| TQ MSE (4-bit) | 0.779 | 0.572 | -0.207 |
| Binary Hash | 0.451 | 0.273 | -0.178 |

| Method | RemoteCLIP EuroSAT | RemoteCLIP BEN | Delta |
|--------|-------------------|----------------|-------|
| PQ (4-bit) | 0.961 | 0.944 | -0.017 |
| TQ MSE (4-bit) | 0.911 | 0.878 | -0.033 |
| Binary Hash | 0.607 | 0.473 | -0.134 |

TurboQuant on RemoteCLIP degrades gracefully (-0.033), comparable to PQ (-0.017). On Prithvi, the degradation is much steeper (-0.207), again reflecting the harder quantization problem posed by anisotropic embeddings. PQ's subspace-adapted codebooks handle this better because they learn the per-subspace variance structure.

### 3.6 RaBitQ and the Value of Rotation

RaBitQ applies a random orthogonal rotation before binarization, then uses Hamming distance — the same pipeline as binary hash but with rotation. Comparing the two isolates the value of rotation at 1 bit per dimension:

| Setting | RaBitQ R@10 | Binary Hash R@10 | Rotation helps? |
|---------|------------|-----------------|----------------|
| Prithvi/EuroSAT | 0.502 | 0.451 | Yes (+11%) |
| Prithvi/BigEarthNet | 0.256 | 0.273 | No (-6%) |
| RemoteCLIP/EuroSAT | 0.567 | 0.607 | No (-7%) |
| RemoteCLIP/BigEarthNet | 0.418 | 0.473 | No (-12%) |

Rotation helps only on Prithvi/EuroSAT. On RemoteCLIP — where embeddings are already near-isotropic — rotation is unnecessary and the extra computational cost (dense matrix multiply) provides no benefit. On BigEarthNet at larger scale, even for Prithvi, the rotation's information-spreading effect is insufficient to overcome the increased retrieval difficulty.

## 4. Discussion

### 4.1 Implementation Lessons

**Re-normalization after decoding is critical.** Quantized-then-decoded vectors have norms below 1 due to quantization shrinkage. At 2-bit, norms average ~0.94; at 4-bit, ~0.998. Without re-normalization, the tiny per-vector norm variation at higher bit-widths becomes the dominant signal in inner product ranking, producing an inverted recall-vs-bits relationship where more bits give worse recall. This bug affected our initial implementation and would affect any reproduction — we recommend always normalizing decoded vectors before cosine similarity search.

**QJL residual correction is counterproductive for retrieval.** TurboQuant's "Prod" variant adds a Quantized Johnson-Lindenstrauss (QJL) correction term to the inner product estimate. We implemented this and found it catastrophically degrades recall (e.g., R@10 drops from 0.636 to 0.423 on Prithvi/EuroSAT at 2-bit). The sign sketch variance overwhelms the correction signal for cosine similarity retrieval, consistent with findings from multiple independent implementations. The QJL estimator is designed for raw inner product estimation with theoretical unbiasedness guarantees, not for ranking quality in retrieval.

**Uniform quantization is useless at high dimension.** At d=768, rotated unit-norm coordinates concentrate within +/-0.036. A uniform grid on [-1, 1] wastes all but one bin on empty space, making the quantizer insensitive to bit-width. Any rotation-based scalar quantizer for high-dimensional unit-norm vectors must adapt its grid to the data-occupied range.

### 4.2 Practical Recommendations

**For contrastive RS models (RemoteCLIP and similar):** TurboQuant MSE at 4 bits is an excellent choice, achieving R@10 > 0.87 on BigEarthNet with zero training. The 7% gap to PQ (0.944) may be acceptable given the simplicity and generality of the approach.

**For reconstruction-based RS models (Prithvi and similar):** TurboQuant MSE at 4 bits provides moderate quality (R@10 = 0.572 on BigEarthNet). If higher recall is needed, PQ (0.925) is recommended despite the training requirement. Alternatively, applying a whitening step or fine-tuning the model with a contrastive objective could improve isotropy and thus quantization quality.

**For maximum compression:** Binary hashing at 1 bit per dimension provides a 32x compression ratio with R@10 of 0.27-0.47 depending on the model. RaBitQ's rotation provides inconsistent benefits at this budget.

### 4.3 Limitations

1. We evaluate only brute-force search. In practice, quantization is combined with indexing structures (IVF, HNSW) that may interact differently with different quantizers.

2. Our evaluation uses cosine similarity on L2-normalized embeddings. Inner product search on unnormalized embeddings would produce different relative rankings.

3. BigEarthNet extraction used only the train split (269K of 590K patches) due to partial extraction. Results on the full dataset may differ slightly.

4. We evaluate only two RS foundation models. The isotropy hypothesis should be validated on additional models (e.g., SatMAE, Scale-MAE, GFM).

## 5. Conclusion

TurboQuant provides a practical, training-free baseline for compressing remote sensing foundation model embeddings. Its effectiveness depends on the isotropy of the embedding space: contrastive models like RemoteCLIP produce near-isotropic embeddings where TurboQuant closes 86% of the gap to trained Product Quantization, while reconstruction-based models like Prithvi produce anisotropic embeddings where only 46% of the gap is closed.

The key insight is that **the training objective of the foundation model determines quantization quality as much as the quantizer design itself.** Contrastive learning produces representations that are naturally suited to rotation-based scalar quantization; MAE-based reconstruction does not. This suggests that RS practitioners choosing between foundation models for retrieval workloads should consider not only task accuracy but also embedding geometry and its downstream implications for storage efficiency.

All code, results, and paper assets are available at: https://github.com/sumit-ai-ml/turboquant-rs

## References

- Guo, R., et al. "TurboQuant: Online Vector Quantization with Near-Optimal Distortion." 2024.
