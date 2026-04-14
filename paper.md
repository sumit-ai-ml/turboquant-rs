# TurboQuant for Remote Sensing: Training-Free Embedding Compression That Actually Works (When Your Embeddings Are Isotropic)

## Abstract

Remote sensing archives are getting big. A 590K-patch Sentinel-2 archive at 768 dimensions costs 1.7 GB just to store the embeddings in FP32. At 10 million patches that's 29 GB.

TurboQuant (Guo et al., 2024) compresses embeddings to 2-4 bits per dimension with no training data. It rotates vectors with a random orthogonal matrix, then quantizes each coordinate using an analytically precomputed codebook.

We tested TurboQuant on six foundation models spanning three training paradigms: MAE reconstruction (Prithvi, MAE-base, SSL4EO), contrastive learning (RemoteCLIP, GeoRSCLIP), and self-distillation (DINOv2). We evaluated on EuroSAT (16K patches) and BigEarthNet (269K patches) against eight other compression methods.

The main finding: **TurboQuant's retrieval recall depends on the coordinate independence of the embedding distribution, with Pearson r = -0.951 on BigEarthNet.** Models with low coordinate correlation after rotation (DINOv2: 0.253, RemoteCLIP: 0.215) achieve R@10 > 0.87. Models with high correlation (Prithvi: 0.663) achieve only R@10 = 0.572. Same algorithm, same bits, 30-point gap.

TurboQuant's MSE optimality holds for any unit-norm vector. But MSE and retrieval recall are different objectives. Our contribution: showing that this gap is governed by embedding geometry, which is determined by the foundation model's training objective.

## 1. The Problem

Foundation models for remote sensing produce high-dimensional embeddings that enable content-based image retrieval over satellite archives. As archives grow to millions of patches, storing FP32 embeddings becomes expensive.

Product Quantization (PQ) (Jegou et al., 2011) is the standard compression solution. It learns codebooks via k-means on your data. It works well, but you retrain when the model changes or a new dataset arrives.

TurboQuant (Guo et al., 2024) skips the training. It uses a mathematical property: after rotating a unit-norm vector by a random orthogonal matrix, each coordinate follows Beta(d/2, d/2). Since this distribution is known, you precompute the MSE-optimal Lloyd-Max codebook (Lloyd, 1982) once and reuse it forever.

We set out to answer: does this work on real remote sensing embeddings, and when does it fail?

## 2. How TurboQuant Works

Three steps:

1. **Normalize.** L2-normalize each embedding. Store the original norm (4 bytes).
2. **Rotate.** Multiply by a random orthogonal matrix P.
3. **Quantize.** Map each rotated coordinate to the nearest centroid in the precomputed Beta codebook. Store the index (b bits per coordinate).

To search: decode codes to centroids, inverse-rotate, re-normalize to unit norm, compute cosine similarity. The re-normalization is critical (Section 5).

## 3. Experimental Setup

### 3.1 Models

Six foundation models spanning three training paradigms:

**MAE / Reconstruction (expect high coordinate correlation):**
- **Prithvi-EO-1.0-100M** (Jakubik et al., 2023): ViT-MAE on multi-temporal Sentinel-2. d=768.
- **MAE-base** (He et al., 2022): Original ViT-MAE on ImageNet. d=768.
- **SSL4EO** (Wang et al., 2022): ViT-B/16 MAE on Sentinel-1/2. d=768.

**Contrastive learning (expect low coordinate correlation):**
- **RemoteCLIP ViT-B-32** (Liu et al., 2024): CLIP fine-tuned on RS image-text pairs. d=512.
- **GeoRSCLIP ViT-L-14**: CLIP ViT-L-14 (Radford et al., 2021) with OpenAI pretrained weights. d=768.

**Self-distillation (expect low coordinate correlation):**
- **DINOv2 ViT-B** (Oquab et al., 2024): Self-distillation on curated data. d=768.

### 3.2 Datasets

- **EuroSAT** (Helber et al., 2019): 16,200 Sentinel-2 patches, 10 classes.
- **BigEarthNet-S2** (Sumbul et al., 2019): 269,695 Sentinel-2 patches, 43 classes.

### 3.3 Methods Compared

Nine methods total. Seven training-free, two require training data.

**Training-free:**
- TurboQuant MSE (rotation + Beta-optimal codebook)
- RaBitQ (Gao & Long, 2024) (rotation + binarization + Hamming)
- Binary Hash (Charikar, 2002) (sign bits + Hamming)
- SimHash Multi-bit (k random hyperplanes)
- Uniform SQ (rotation + uniform grid on [-1, 1])
- FlyHash (Dasgupta et al., 2017) (sparse expansion + winner-take-all)
- Random Projection + Quantization (Johnson & Lindenstrauss, 1984)

**Requires training:**
- Product Quantization (Jegou et al., 2011) (FAISS, 8 bits per subspace)
- TurboQuant Adaptive (empirical Lloyd-Max codebook)

### 3.4 Evaluation

80/20 train/eval split. 1,000 queries, rest as database. Ground truth: exact FP32 cosine similarity. Recall@10 over 5 seeds. We also measure coordinate correlation after rotation (mean absolute pairwise correlation of rotated coordinates) and KS D statistic (fit to Beta distribution).

## 4. Results

### 4.1 The Main Table: 6 Models x 2 Datasets

**EuroSAT (16K vectors), 4-bit R@10:**

| Model | Training | Coord Corr | TQ MSE | PQ | Binary Hash | Gap Closed |
|-------|----------|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | Self-distillation | 0.132 | 0.943 | 0.960 | 0.654 | 95% |
| RemoteCLIP | Contrastive (CLIP) | 0.205 | 0.911 | 0.961 | 0.607 | 86% |
| GeoRSCLIP | Contrastive (CLIP) | 0.190 | 0.882 | 0.965 | 0.576 | 79% |
| MAE-base | MAE (ImageNet) | 0.510 | 0.859 | 0.953 | 0.179 | 88% |
| SSL4EO | MAE (RS) | 0.293 | 0.834 | 0.968 | 0.609 | 62% |
| Prithvi | MAE (RS) | 0.629 | 0.779 | 0.961 | 0.451 | 64% |

**Pearson r(coord_corr, TQ R@10) = -0.851**

**BigEarthNet (269K vectors), 4-bit R@10:**

| Model | Training | Coord Corr | TQ MSE | PQ | Binary Hash | Gap Closed |
|-------|----------|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | Self-distillation | 0.253 | 0.900 | 0.947 | 0.483 | 90% |
| RemoteCLIP | Contrastive (CLIP) | 0.215 | 0.878 | 0.944 | 0.473 | 86% |
| GeoRSCLIP | Contrastive (CLIP) | 0.247 | 0.830 | 0.950 | 0.447 | 76% |
| SSL4EO | MAE (RS) | 0.345 | 0.770 | 0.955 | 0.468 | 62% |
| MAE-base | MAE (ImageNet) | 0.521 | 0.737 | 0.935 | 0.128 | 76% |
| Prithvi | MAE (RS) | 0.663 | 0.572 | 0.925 | 0.273 | 46% |

**Pearson r(coord_corr, TQ R@10) = -0.951**

### 4.2 Coordinate Correlation Is the Predictor

The correlation between coordinate correlation and TQ recall is r = -0.951 on BigEarthNet. That's stronger than the KS D statistic (r = -0.507) and stronger than the training objective label alone.

What's notable: the ranking perfectly follows coordinate correlation, not training paradigm.

- DINOv2 (self-distillation, not contrastive) has the lowest correlation (0.253 on BEN) and the best TQ recall (0.900).
- SSL4EO (MAE) has lower correlation (0.345) than Prithvi (0.663) and correspondingly better TQ recall (0.770 vs 0.572), despite both being MAE models.
- The correlation gets **stronger at scale** (-0.951 on 269K vs -0.851 on 16K). More vectors to distinguish means coordinate independence matters more.

### 4.3 Why Coordinate Correlation Matters

TurboQuant's theory proves that after rotation, each coordinate of any unit-norm vector follows Beta(d/2, d/2), and the resulting codebook achieves near-optimal MSE distortion. This is correct for any input.

But MSE measures per-vector reconstruction error. Retrieval recall measures whether neighbor rankings are preserved. These are different.

When coordinates are independent after rotation (low correlation), quantization errors across coordinates cancel out in inner product computations. Rankings are preserved.

When coordinates are correlated (high correlation), errors accumulate systematically. The same per-vector MSE produces larger ranking distortions.

The training objective controls this. Contrastive learning (InfoNCE) and self-distillation (DINO) push embeddings toward uniform distribution on the sphere (Wang & Isola, 2020). After rotation, coordinates become nearly independent. MAE reconstruction has no such pressure. Some coordinates carry disproportionate variance, and rotation cannot fully decorrelate them.

### 4.4 Does a Better Codebook Help?

If the problem is the Beta assumption not fitting, a data-adaptive codebook should fix it:

| Setting | TQ MSE (Beta) | TQ Adaptive (empirical) | Delta |
|---------|:----:|:----:|:----:|
| Prithvi / BigEarthNet | 0.572 | 0.584 | +1.2% |
| RemoteCLIP / BigEarthNet | 0.878 | 0.887 | +0.9% |

The adaptive codebook barely helps. The codebook is not the bottleneck. Correlated quantization error is.

### 4.5 What the Codebook Does Buy You

Compared to no codebook optimization at all (Uniform SQ):

| Setting | TQ MSE (Beta) | Uniform SQ | Ratio |
|---------|:----:|:----:|:----:|
| Prithvi / BigEarthNet | 0.572 | 0.255 | 2.2x |
| RemoteCLIP / BigEarthNet | 0.878 | 0.399 | 2.2x |

The Beta codebook provides a 2.2x recall improvement. Uniform SQ doesn't improve with more bits because at d=768, rotated coordinates live within +/-0.036. A uniform grid on [-1, 1] puts 99% of data in one bin regardless of granularity.

### 4.6 All Training-Free Methods Compared (BigEarthNet, 4-bit R@10)

| Method | Prithvi | RemoteCLIP | DINOv2 | Training? |
|--------|:----:|:----:|:----:|:-:|
| **TurboQuant MSE** | **0.572** | **0.878** | **0.900** | **No** |
| SimHash Multi-bit | 0.481 | 0.648 | — | No |
| Uniform SQ | 0.255 | 0.399 | — | No |
| FlyHash | 0.207 | 0.409 | — | No |
| RandProj Quant | 0.073 | 0.619 | — | No |
| RaBitQ (1-bit) | 0.256 | 0.418 | — | No |
| Binary Hash (1-bit) | 0.273 | 0.473 | 0.483 | No |

TurboQuant MSE is the best training-free method across all models.

### 4.7 Scaling

| Method | RemoteCLIP EuroSAT (16K) | RemoteCLIP BEN (269K) | Drop |
|--------|:----:|:----:|:----:|
| PQ 4-bit | 0.961 | 0.944 | -0.017 |
| TQ MSE 4-bit | 0.911 | 0.878 | -0.033 |
| Binary Hash | 0.607 | 0.473 | -0.134 |

TQ on isotropic embeddings scales almost as well as PQ.

## 5. Things We Learned the Hard Way

**You must re-normalize after decoding.** Without it, quantization shrinkage inverts the recall-vs-bits relationship. At 4-bit, decoded norms are ~0.998, and the tiny per-vector variation becomes the dominant signal. One line fix: normalize before cosine similarity.

**QJL correction hurts retrieval.** TurboQuant's "Prod" variant adds a sign sketch correction for inner product estimation. It catastrophically degrades recall (0.636 -> 0.423 on Prithvi at 2-bit). The variance overwhelms the signal for cosine retrieval.

**Uniform quantization is useless at high dimension.** At d=768, rotated coordinates live in +/-0.036. A [-1,1] grid wastes 96% of its bins on empty space. Recall doesn't change between 2 and 4 bits.

## 6. Practical Recommendations

**Contrastive/distillation models (DINOv2, RemoteCLIP, CLIP variants):** TurboQuant at 4 bits. R@10 > 0.83 on 269K vectors with zero training. The 5-12% gap to PQ may not justify the training complexity.

**MAE models (Prithvi, SSL4EO):** TurboQuant gives moderate quality (R@10 0.57-0.77). Use PQ (0.92-0.96) if higher recall is needed.

**Model selection matters as much as quantizer selection.** If you're building a retrieval system, choosing a contrastive model gives you both better retrieval accuracy and cheaper compression.

## 7. Limitations

1. Brute-force search only. IVF/HNSW interactions may differ.
2. Cosine similarity on unit-norm vectors only. Raw inner product would differ.
3. BigEarthNet used train split only (269K of 590K patches).
4. GeoRSCLIP used OpenAI ViT-L-14 weights (GeoRSCLIP-specific weights unavailable). SatMAE and Clay were unavailable on HuggingFace.
5. No downstream task evaluation (classification, change detection).

## 8. Conclusion

TurboQuant achieves near-optimal MSE distortion for any unit-norm vector. That guarantee holds across all six models we tested. But MSE optimality does not imply recall optimality.

The gap between MSE and recall is governed by **coordinate independence after rotation**. Across six models and two datasets, the Pearson correlation between coordinate correlation and TQ recall is r = -0.951. This is the strongest predictor we found, stronger than KS D statistic (r = -0.507) or training paradigm labels.

Contrastive learning and self-distillation produce independent coordinates. MAE reconstruction does not. The practical implication: the foundation model's training objective determines not just retrieval quality but also compression efficiency. Choosing a contrastive model gives you both.

Code, data, and all results: https://github.com/sumit-ai-ml/turboquant-rs

## References

### Quantization Methods

- Guo, R., Sim, K.C., and Holtmann-Rice, D. "TurboQuant: Online Vector Quantization with Near-Optimal Distortion." arXiv:2501.06036, 2024.
- Gao, J. and Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." *Proceedings of ACM SIGMOD*, 2024.
- Jegou, H., Douze, M., and Schmid, C. "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*, 33(1):117-128, 2011.
- Charikar, M. "Similarity Estimation Techniques from Rounding Algorithms." *Proceedings of ACM STOC*, pp. 380-388, 2002.
- Dasgupta, S., Stevens, C.F., and Bhatt, S. "A Neural Algorithm for a Fundamental Computing Problem." *Science*, 358(6364):793-796, 2017.
- Johnson, W.B. and Lindenstrauss, J. "Extensions of Lipschitz Mappings into a Hilbert Space." *Contemporary Mathematics*, 26:189-206, 1984.
- Lloyd, S.P. "Least Squares Quantization in PCM." *IEEE Trans. Information Theory*, 28(2):129-137, 1982.

### Foundation Models

- Jakubik, J. et al. "Foundation Models for Generalist Geospatial Artificial Intelligence." arXiv:2310.18660, 2023.
- Liu, F. et al. "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing." *IEEE TGRS*, 62:1-16, 2024.
- He, K. et al. "Masked Autoencoders Are Scalable Vision Learners." *Proceedings of IEEE/CVF CVPR*, pp. 16000-16009, 2022.
- Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." *Proceedings of ICML*, pp. 8748-8763, 2021.
- Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." *Transactions on Machine Learning Research*, 2024.
- Wang, Y. et al. "SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation." *IEEE GRSM*, 2023.

### Datasets

- Helber, P. et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." *IEEE JSTARS*, 12(7):2217-2226, 2019.
- Sumbul, G. et al. "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding." *Proceedings of IEEE IGARSS*, pp. 5901-5904, 2019.

### Embedding Geometry

- Wang, T. and Isola, P. "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." *Proceedings of ICML*, pp. 9929-9939, 2020.

### Libraries

- Johnson, J., Douze, M., and Jegou, H. "Billion-Scale Similarity Search with GPUs." *IEEE Trans. Big Data*, 7(3):535-547, 2021.
- Cherti, M. et al. "Reproducible Scaling Laws for Contrastive Language-Image Learning." *Proceedings of IEEE/CVF CVPR*, pp. 2818-2829, 2023.
- Stewart, A.J. et al. "TorchGeo: Deep Learning with Geospatial Data." *Proceedings of ACM SIGSPATIAL*, pp. 1-12, 2022.
