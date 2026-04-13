# TurboQuant for Remote Sensing: Training-Free Embedding Compression That Actually Works (When Your Embeddings Are Isotropic)

## Abstract

Remote sensing archives are getting big. A 590K-patch Sentinel-2 archive at 768 dimensions costs 1.7 GB just to store the embeddings in FP32. That scales linearly. At 10 million patches you're looking at 29 GB of floating point numbers.

TurboQuant (Guo et al., 2024) compresses these embeddings to 2-4 bits per dimension with no training data and no codebook fitting. It works by rotating vectors with a random orthogonal matrix, then quantizing each coordinate using a codebook that's computed analytically from the known Beta(d/2, d/2) distribution of rotated unit-norm coordinates.

We tested TurboQuant on embeddings from two remote sensing foundation models (Prithvi-EO and RemoteCLIP) across two datasets (EuroSAT and BigEarthNet) and compared it against seven other compression methods.

The main finding: **TurboQuant works great on some embeddings and poorly on others, and the difference comes down to one thing: how isotropic the embeddings are.**

On RemoteCLIP (trained with contrastive learning, naturally isotropic), TurboQuant at 4 bits gets Recall@10 = 0.878 on BigEarthNet. That closes 86% of the gap between naive binary hashing and trained Product Quantization, without any training.

On Prithvi (trained with masked autoencoding, anisotropic), the same method gets R@10 = 0.572. Only 46% of the gap. Same algorithm, same bit budget, very different outcome.

The embedding model you choose matters more than the quantizer you choose. Contrastive learning produces embeddings that are ready for cheap compression. Reconstruction-based training does not.

## 1. The Problem

Foundation models for remote sensing produce high-dimensional embeddings. Prithvi-EO gives you 768-dimensional vectors. RemoteCLIP gives you 512-dimensional vectors. These embeddings enable content-based image retrieval: given a query patch, find the most similar patches in your archive.

The storage problem is straightforward. Each FP32 vector at d=768 takes 3,072 bytes. BigEarthNet has 590K patches. That's 1.7 GB just for the embeddings. Scale to a national-level archive with millions of patches and you're spending real money on storage.

Product Quantization (PQ) (Jegou et al., 2011) is the standard solution. It divides each vector into subspaces, learns a codebook per subspace via k-means, and stores compact codes. PQ gets excellent recall. The catch: you need to train those codebooks on your data. When you change the embedding model, you retrain. When you add a new dataset, you retrain. At scale, this retraining cost is real.

TurboQuant (Guo et al., 2024) skips the training entirely. It uses a mathematical property of random rotations: after rotating a unit-norm vector by a random orthogonal matrix, each coordinate follows a Beta(d/2, d/2) distribution. Since this distribution is known analytically, you can precompute the MSE-optimal Lloyd-Max codebook (Lloyd, 1982) once and reuse it forever. No training data. No k-means. No retraining when the model changes.

The question we set out to answer: does this actually work on real remote sensing embeddings?

## 2. How TurboQuant Works

Three steps:

1. **Normalize.** L2-normalize each embedding to unit norm. Store the original norm separately (4 bytes, FP32).

2. **Rotate.** Multiply by a random orthogonal matrix P. This spreads information evenly across coordinates. For unit-norm vectors on the sphere, each rotated coordinate follows Beta(d/2, d/2) when the data is isotropic.

3. **Quantize.** Map each rotated coordinate to the nearest centroid in the precomputed codebook. Store the index (b bits per coordinate). At 4 bits, that's d/2 bytes for the codes plus 4 bytes for the norm.

To search: decode the codes back to centroids, inverse-rotate, re-normalize to unit norm, and compute cosine similarity.

The re-normalization step is important. We'll come back to that in Section 4.

## 3. What We Compared

We tested nine methods total. Seven are training-free, two need training data.

**Training-free methods (no data needed):**

| Method | How it works | Bits/dim |
|--------|-------------|----------|
| TurboQuant MSE | Rotation + Beta-optimal codebook | 2, 3, 4 |
| RaBitQ (Gao & Long, 2024) | Rotation + sign bits + Hamming distance | 1 |
| Binary Hash (Charikar, 2002) | sign(x) + Hamming distance | 1 |
| SimHash Multi-bit | k random hyperplanes, 1 bit each | 2, 3, 4 |
| Uniform SQ | Rotation + uniform grid on [-1, 1] | 2, 3, 4 |
| FlyHash (Dasgupta et al., 2017) | Sparse random expansion + winner-take-all | 2, 3, 4 |
| RandProj Quant (Johnson & Lindenstrauss, 1984) | Random projection to lower dim + 8-bit quantization | 2, 3, 4 |

**Methods that need training data:**

| Method | How it works | Bits/dim |
|--------|-------------|----------|
| Product Quantization (Jegou et al., 2011) | Learned subspace codebooks (FAISS) | 2, 3, 4 |
| TurboQuant Adaptive | Same rotation, but codebook trained on actual data distribution | 2, 3, 4 |

**Models:**

- **Prithvi-EO-1.0-100M** (Jakubik et al., 2023): ViT-MAE trained to reconstruct masked Sentinel-2 patches. Embed dim = 768. The MAE objective (He et al., 2022) optimizes for pixel-level reconstruction, not for discriminative retrieval.

- **RemoteCLIP ViT-B-32** (Liu et al., 2024): CLIP model (Radford et al., 2021) fine-tuned on remote sensing image-text pairs. Embed dim = 512. The contrastive objective pushes embeddings toward uniform distribution on the sphere.

**Datasets:**

- **EuroSAT** (Helber et al., 2019): 16,200 Sentinel-2 patches across 10 land-use classes.
- **BigEarthNet-S2** (Sumbul et al., 2019): 269,695 Sentinel-2 patches with 43 multi-label classes.

**Evaluation:** For each experiment, we split 80/20 into train/eval. From the eval set, we take 1,000 queries and use the rest as the database. Ground truth is exact FP32 cosine similarity. We report Recall@10 (fraction of true top-10 neighbors found in approximate top-10). All results averaged over 5 random seeds with standard deviations.

## 4. Results

### 4.1 The Main Table

Recall@10 at 4 bits per dimension (except 1-bit methods):

**EuroSAT (16K vectors):**

| Method | Prithvi (d=768) | RemoteCLIP (d=512) | Training? |
|--------|:-:|:-:|:-:|
| FP32 Exact | 1.000 | 1.000 | - |
| Product Quant | 0.961 | 0.961 | Yes |
| **TurboQuant MSE** | **0.779** | **0.911** | **No** |
| TurboQuant Adaptive | 0.782 | 0.912 | Yes |
| SimHash Multi-bit | 0.702 | 0.751 | No |
| Uniform SQ | 0.502 | 0.549 | No |
| FlyHash | 0.468 | 0.545 | No |
| RaBitQ (1-bit) | 0.502 | 0.567 | No |
| Binary Hash (1-bit) | 0.451 | 0.607 | No |

**BigEarthNet (269K vectors):**

| Method | Prithvi (d=768) | RemoteCLIP (d=512) | Training? |
|--------|:-:|:-:|:-:|
| FP32 Exact | 1.000 | 1.000 | - |
| Product Quant | 0.925 | 0.944 | Yes |
| **TurboQuant MSE** | **0.572** | **0.878** | **No** |
| TurboQuant Adaptive | 0.584 | 0.887 | Yes |
| SimHash Multi-bit | 0.481 | 0.648 | No |
| Uniform SQ | 0.255 | 0.399 | No |
| FlyHash | 0.207 | 0.409 | No |
| RaBitQ (1-bit) | 0.256 | 0.418 | No |
| Binary Hash (1-bit) | 0.273 | 0.473 | No |

Two things jump out:

**TurboQuant is the best training-free method across the board.** It beats SimHash (the runner-up) by 9-23 percentage points depending on the setting.

**But the gap between Prithvi and RemoteCLIP is enormous.** On BigEarthNet, TurboQuant gets 0.878 on RemoteCLIP but only 0.572 on Prithvi. Same algorithm. Same number of bits. The difference is 30 percentage points.

### 4.2 Why: The Isotropy Story

TurboQuant assumes that after rotation, each coordinate is independently Beta-distributed. This assumption holds when embeddings are spread uniformly on the unit sphere (isotropic). It fails when embeddings cluster along certain directions (anisotropic).

We measured this directly using the Kolmogorov-Smirnov test on rotated coordinates:

| Model | KS D (lower = more isotropic) | Coordinate correlation |
|-------|:----:|:----:|
| RemoteCLIP | 0.405 | 0.179 |
| Prithvi | 0.594 | 0.577 |

RemoteCLIP embeddings are much more isotropic. The KS statistic is 32% lower. The correlation between rotated coordinates is 3.2x lower.

**Why the difference? The training objective.**

RemoteCLIP uses contrastive learning. The InfoNCE loss pulls matched pairs together and pushes unmatched pairs apart in cosine similarity space. Wang & Isola (2020) showed this produces embeddings with a "uniformity" property: they spread out on the sphere. More uniform means more isotropic. More isotropic means TurboQuant's Beta assumption is closer to correct.

Prithvi uses masked autoencoding. The MAE objective reconstructs pixel patches from partial observations. This doesn't care about distributing embeddings uniformly. The result is an embedding space with strong directional structure. Some coordinates carry much more variance than others. The random rotation helps, but it can't fully decorrelate a fundamentally anisotropic distribution.

When TurboQuant quantizes correlated coordinates with a codebook designed for independent ones, the errors add up in the same direction instead of canceling out. That's why recall drops.

### 4.3 Does a Better Codebook Help?

If the problem is the Beta assumption not fitting, would a data-adaptive codebook fix it?

We tested this with TurboQuant Adaptive, which trains the Lloyd-Max codebook on the actual distribution of rotated coordinates instead of the theoretical Beta.

| Setting | TQ MSE (Beta) | TQ Adaptive (empirical) | Improvement |
|---------|:----:|:----:|:----:|
| Prithvi / BigEarthNet | 0.572 | 0.584 | +1.2% |
| RemoteCLIP / BigEarthNet | 0.878 | 0.887 | +0.9% |

The adaptive codebook barely helps. The codebook was never the bottleneck. The real problem is correlated quantization error from non-independent coordinates. A better codebook can't fix correlated noise.

### 4.4 What the Codebook Does Buy You

The codebook does matter a lot compared to no codebook at all. We tested Uniform Scalar Quantization: same rotation, but a naive uniform grid on [-1, 1].

| Setting | TQ MSE (Beta) | Uniform SQ | Ratio |
|---------|:----:|:----:|:----:|
| Prithvi / BigEarthNet | 0.572 | 0.255 | 2.2x |
| RemoteCLIP / BigEarthNet | 0.878 | 0.399 | 2.2x |

The Beta codebook gives a 2.2x recall improvement on both models.

There's a fun detail here: Uniform SQ doesn't improve with more bits. At d=768, the recall is 0.256, 0.256, and 0.255 at 2, 3, and 4 bits. This happens because rotated unit-norm coordinates live in a tiny range around zero (about +/-0.036 for d=768). A uniform grid on [-1, 1] puts 99%+ of the data into one central bin. More bins just subdivide the empty tails. The Beta codebook avoids this by placing all quantization levels inside the narrow range where data actually exists.

### 4.5 How Well Does It Scale?

Everything degrades with more vectors. More database vectors means more near-neighbors to distinguish. But the degradation rate varies:

| Method | RemoteCLIP EuroSAT (16K) | RemoteCLIP BEN (269K) | Drop |
|--------|:----:|:----:|:----:|
| PQ 4-bit | 0.961 | 0.944 | -0.017 |
| TQ MSE 4-bit | 0.911 | 0.878 | -0.033 |
| Binary Hash | 0.607 | 0.473 | -0.134 |

| Method | Prithvi EuroSAT (16K) | Prithvi BEN (269K) | Drop |
|--------|:----:|:----:|:----:|
| PQ 4-bit | 0.961 | 0.925 | -0.036 |
| TQ MSE 4-bit | 0.779 | 0.572 | -0.207 |
| Binary Hash | 0.451 | 0.273 | -0.178 |

TurboQuant on RemoteCLIP scales almost as well as PQ (3.3% drop vs 1.7%). On Prithvi the drop is steep (20.7%). The isotropy gap widens at scale.

### 4.6 RaBitQ: Does Rotation Alone Help?

RaBitQ (Gao & Long, 2024) is binary hash plus a random rotation. Same Hamming distance search, but on rotated sign bits instead of raw sign bits. Comparing the two tells us what rotation is worth at 1 bit per dimension:

| Setting | RaBitQ | Binary Hash | Rotation helps? |
|---------|:----:|:----:|:-:|
| Prithvi / EuroSAT | 0.502 | 0.451 | Yes (+11%) |
| Prithvi / BigEarthNet | 0.256 | 0.273 | No (-6%) |
| RemoteCLIP / EuroSAT | 0.567 | 0.607 | No (-7%) |
| RemoteCLIP / BigEarthNet | 0.418 | 0.473 | No (-12%) |

Rotation helps only on Prithvi/EuroSAT (small dataset, anisotropic embeddings). On RemoteCLIP, embeddings are already well-spread across coordinates, so rotation adds nothing. At BigEarthNet scale, even for Prithvi, rotation doesn't help. At 1 bit, you lose too much information for the rotation to make a difference.

### 4.7 The Gap Closed

How much of the distance between "free and bad" (binary hash) and "trained and good" (PQ) does TurboQuant close?

| Setting | Binary Hash | TQ MSE 4-bit | PQ 4-bit | Gap Closed |
|---------|:----:|:----:|:----:|:----:|
| RemoteCLIP / EuroSAT | 0.607 | 0.911 | 0.961 | **86%** |
| RemoteCLIP / BigEarthNet | 0.473 | 0.878 | 0.944 | **86%** |
| Prithvi / EuroSAT | 0.451 | 0.779 | 0.961 | **64%** |
| Prithvi / BigEarthNet | 0.273 | 0.572 | 0.925 | **46%** |

On RemoteCLIP, TurboQuant closes 86% of the gap. On Prithvi, 46-64%. The difference is isotropy.

## 5. Things We Learned the Hard Way

### You must re-normalize after decoding.

Decoded vectors have norms less than 1. Quantization always shrinks things slightly. At 2-bit, average norm is ~0.94. At 4-bit, ~0.998.

If you skip re-normalization and compute inner products on unnormalized decoded vectors, something weird happens: more bits gives worse recall. We saw this in our initial implementation. R@10 went 0.456 (2-bit) -> 0.314 (3-bit) -> 0.248 (4-bit) on Prithvi. Completely inverted.

Why? At 2-bit, all vectors shrink a lot and roughly equally, so rankings are preserved. At 4-bit, vectors shrink very little but the tiny per-vector shrinkage variation becomes the dominant signal in inner product scores, overpowering the actual directional similarity.

The fix is one line: normalize decoded vectors before computing cosine similarity.

### The QJL correction makes things worse.

TurboQuant's paper describes a "Prod" variant that adds a Quantized Johnson-Lindenstrauss correction term to improve inner product estimation. We implemented it. It was catastrophic. R@10 dropped from 0.636 to 0.423 on Prithvi at 2-bit.

The sign sketch variance overwhelms the correction signal for cosine similarity retrieval. Multiple independent implementations of TurboQuant reached the same conclusion. The QJL correction is designed for theoretical unbiasedness guarantees on raw inner products, not for retrieval ranking quality.

### Uniform quantization is useless at high dimension.

A uniform grid on [-1, 1] wastes almost every bin. At d=768, rotated coordinates live within +/-0.036. That's 3.6% of the [-1, 1] range. The other 96.4% is empty bins. Adding more bits just subdivides empty space. Recall doesn't change at all between 2 and 4 bits.

## 6. What This Means in Practice

**If you use a contrastive RS model (RemoteCLIP or similar):** TurboQuant at 4 bits is a strong default. R@10 > 0.87 on 269K vectors with zero training. The 7% gap to PQ might not be worth the training complexity.

**If you use an MAE-based RS model (Prithvi or similar):** TurboQuant at 4 bits gives moderate quality (R@10 = 0.57). For better recall, either use PQ (0.92, requires training) or consider whether your retrieval workload would be better served by a contrastive model in the first place.

**If you want maximum compression:** Binary hashing at 1 bit gives 32x compression with R@10 of 0.27-0.47 depending on the model. It's crude but it's fast and it's free.

**The real takeaway:** If you're building a retrieval system over RS embeddings, the choice of foundation model affects your storage costs as much as the choice of compression algorithm. Contrastive models (CLIP-based) are not just better for retrieval accuracy. They also produce embeddings that compress better with cheap, training-free methods. That's a compounding advantage.

## 7. Limitations

1. We only tested brute-force search. In practice, you'd combine quantization with an index (IVF, HNSW). The quantizer-index interaction may differ across methods.

2. We only tested cosine similarity on unit-norm vectors. Inner product search on raw vectors would change the rankings.

3. BigEarthNet used only the train split (269K of 590K patches). Full dataset results may differ slightly.

4. We tested two RS foundation models. The isotropy hypothesis needs validation on others (SatMAE, Scale-MAE, GFM, etc.).

5. We did not evaluate downstream task accuracy (classification, change detection). Recall@k measures retrieval quality, not task performance.

## 8. Conclusion

TurboQuant is a good training-free compressor for remote sensing embeddings. How good depends on whether your embeddings are isotropic.

Contrastive models like RemoteCLIP produce isotropic embeddings where TurboQuant closes 86% of the gap to trained PQ. Reconstruction models like Prithvi produce anisotropic embeddings where only 46% is closed.

The training objective of the embedding model shapes the geometry of the embedding space. That geometry determines how well cheap compression works. If you want embeddings that are both good for retrieval and cheap to store, contrastive learning gives you both.

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

### Datasets

- Helber, P. et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." *IEEE JSTARS*, 12(7):2217-2226, 2019.

- Sumbul, G. et al. "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding." *Proceedings of IEEE IGARSS*, pp. 5901-5904, 2019.

### Embedding Geometry

- Wang, T. and Isola, P. "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." *Proceedings of ICML*, pp. 9929-9939, 2020.

### Libraries

- Johnson, J., Douze, M., and Jegou, H. "Billion-Scale Similarity Search with GPUs." *IEEE Trans. Big Data*, 7(3):535-547, 2021.

- Cherti, M. et al. "Reproducible Scaling Laws for Contrastive Language-Image Learning." *Proceedings of IEEE/CVF CVPR*, pp. 2818-2829, 2023.

- Stewart, A.J. et al. "TorchGeo: Deep Learning with Geospatial Data." *Proceedings of ACM SIGSPATIAL*, pp. 1-12, 2022.
