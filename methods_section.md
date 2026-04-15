# Methods

## 2.1 Problem Formulation

We have a database of N satellite image patches, each encoded by a foundation model into a d-dimensional embedding vector. Given a query patch, the goal is to find the k most similar patches in the database using cosine similarity.

All embeddings are L2-normalized to unit norm before quantization. The original norms are stored separately (4 bytes per vector in FP32). This means all similarity computations reduce to inner products on the unit sphere.

The storage cost in FP32 is N x d x 4 bytes. For BigEarthNet (N=269K, d=768), that is 790 MB. For a national archive with 10 million patches, it would be 29 GB. Quantization to b bits per dimension reduces this to N x d x b/8 bytes, an 8-16x reduction at 2-4 bits.

The question is how much retrieval quality degrades at each compression level.

## 2.2 Quantization Methods

We evaluate nine methods. Seven require no training data. Two learn codebooks from a data sample.

### 2.2.1 TurboQuant MSE (training-free)

TurboQuant (Guo et al., 2024) exploits a property of random rotations on the unit sphere. For any unit-norm vector x in R^d, applying a random orthogonal rotation P produces coordinates that each follow a Beta(d/2, d/2) distribution. Since this distribution is known analytically, a Lloyd-Max optimal codebook (Lloyd, 1982) can be precomputed once and reused for all vectors, regardless of the data.

The encoding procedure:
1. Apply a random orthogonal matrix P to the input vector: x' = Px
2. Quantize each coordinate of x' independently using the precomputed codebook
3. Store the resulting b-bit index per coordinate

The decoding procedure:
1. Map each code back to the corresponding codebook centroid
2. Apply the inverse rotation P^T
3. Re-normalize the result to unit norm

The re-normalization in step 3 is important. Quantization introduces shrinkage: decoded vectors have norms slightly less than 1. Without re-normalization, this shrinkage creates a systematic bias in inner product rankings that worsens at higher bit-widths (Section 5).

The codebook is computed via 100 iterations of Lloyd-Max on the Beta(d/2, d/2) probability density function using numerical integration. This converges in under a second and needs to be done only once per dimension d.

Storage: b x d / 8 + 4 bytes per vector (codes + FP32 norm).

### 2.2.2 Product Quantization (requires training)

Product Quantization (PQ) (Jegou et al., 2011) splits each d-dimensional vector into m equal subspaces of d/m dimensions each, then learns a separate codebook of 256 centroids (8 bits) per subspace using k-means. Each sub-vector is replaced by the index of its nearest centroid.

We use the FAISS implementation (Johnson et al., 2021). The number of subspaces m is chosen so the total storage matches the TurboQuant budget at each bit-width. PQ codebooks are trained on an 80% split of the data.

Storage: m bytes per vector (1 byte per subspace at 8 bits).

### 2.2.3 TurboQuant Adaptive (requires training)

Same rotation and encoding procedure as TurboQuant MSE, but the codebook is trained on the actual empirical distribution of rotated coordinates from a training set, rather than the theoretical Beta distribution. This isolates the effect of the codebook assumption: if adaptive significantly outperforms MSE, the Beta assumption is the bottleneck. If not, the bottleneck is elsewhere.

### 2.2.4 Binary Hash / SimHash (training-free)

The simplest baseline. Each coordinate is replaced by its sign bit: 1 if positive, 0 if negative (Charikar, 2002). Search uses Hamming distance between binary codes, which approximates cosine similarity.

Storage: d/8 bytes per vector (1 bit per dimension).

### 2.2.5 RaBitQ (training-free)

RaBitQ (Gao & Long, 2024) applies a random orthogonal rotation before binarization, then searches via Hamming distance on the rotated codes. The rotation spreads information evenly across coordinates before taking sign bits. For unit-norm cosine retrieval, this is the entire algorithm. The correction factor from the original paper (designed for theoretical unbiasedness guarantees) does not improve retrieval recall and is omitted.

Storage: d/8 bytes per vector (same as binary hash).

### 2.2.6 SimHash Multi-bit (training-free)

An extension of binary hash to multiple bits. Instead of using the d coordinate axes, we generate k = b x d independent random unit vectors (hyperplanes) and compute the sign of the inner product with each. This produces k hash bits per vector. Search uses Hamming distance.

Storage: k/8 = b x d / 8 bytes per vector.

### 2.2.7 Uniform Scalar Quantization (training-free)

Same rotation as TurboQuant, but the codebook is a uniform grid on [-1, 1] instead of the Beta-optimal codebook. This serves as an ablation: it isolates the contribution of the Beta codebook from the contribution of the rotation.

Storage: b x d / 8 + 4 bytes per vector.

### 2.2.8 FlyHash (training-free)

Inspired by the fruit fly olfactory circuit (Dasgupta et al., 2017). Each of k = b x d output neurons connects to 6 randomly chosen input dimensions via a sparse binary matrix. After computing the activations, only the top 5% (winner-take-all) are kept as 1-bits, the rest become 0. Search uses Hamming distance.

Storage: k/8 bytes per vector.

### 2.2.9 Random Projection + Quantization (training-free)

Project from d dimensions to m = b x d / 8 dimensions via a random Gaussian matrix (Johnson & Lindenstrauss, 1984), then uniformly quantize each projected coordinate to 8 bits. This trades dimensionality reduction for higher per-coordinate precision.

Storage: m bytes per vector.

## 2.3 Evaluation Metrics

### 2.3.1 Recall@k

The standard retrieval metric. For each query, Recall@k is the fraction of the true top-k nearest neighbors (computed from exact FP32 cosine similarity) that appear in the approximate top-k returned by the quantized search. We report Recall@10 as the primary metric, with Recall@1 and Recall@100 as secondary metrics.

### 2.3.2 Kendall's Tau (ranking preservation)

Recall@k measures whether the right items are found but not whether their relative ordering is preserved. Kendall's tau measures rank correlation. For each query, we take the ground-truth top-1000 neighbors, re-rank them using the quantized similarity scores, and compute Kendall's tau between the two orderings. Values range from -1 (perfectly inverted) to +1 (perfectly preserved). A tau of 0.88 means that for 94% of randomly chosen pairs within the top-1000, the quantizer agrees with the exact ranking on which is more similar.

### 2.3.3 Pearson Correlation (magnitude preservation)

A quantizer can preserve ranking (high tau) while collapsing the similarity scale. If true similarities [0.95, 0.90, 0.85] map to [0.70, 0.69, 0.68], the rank is preserved (tau = 1.0) but the distance structure is destroyed. Pearson correlation between FP32 and quantized inner products for the top-1000 neighbors captures whether the quantizer preserves not just the order but the relative spacing between neighbors.

### 2.3.4 Coordinate Correlation

To measure the isotropy of an embedding distribution, we apply a random orthogonal rotation and compute the mean absolute Pearson correlation between 50 randomly sampled pairs of rotated coordinates. For truly isotropic (uniform on the sphere) embeddings, this would be near zero. Higher values indicate anisotropy that the rotation cannot remove. We also report the Kolmogorov-Smirnov D statistic against the Beta(d/2, d/2) distribution as a secondary isotropy measure.

## 2.4 Experimental Protocol

For each (model, dataset, method, bit-width, seed) configuration:

1. **Split.** 80% of embeddings for training (used only by PQ and TurboQuant Adaptive), 20% for evaluation.
2. **Queries and database.** From the evaluation set, the first min(1000, 10%) vectors serve as queries, the rest as the database.
3. **Ground truth.** Exact FP32 cosine similarity (inner product on unit-norm vectors) between all queries and all database vectors.
4. **Encoding.** Both queries and database are encoded with the quantization method.
5. **Search.** Approximate nearest neighbors are retrieved from the quantized representations.
6. **Metrics.** Recall@k (k = 1, 10, 100), Kendall's tau, and Pearson r are computed against ground truth.

All experiments use 5 random seeds (42, 123, 456, 789, 1024) for the train/eval split, the random rotation matrix, and any random initialization. We report mean and standard deviation across seeds.

For the ranking analysis (Kendall's tau, Pearson r), we use 3 seeds and 200 queries per seed for computational efficiency, since Kendall's tau is O(k^2) per query for k=1000.

Bit-widths evaluated: 2, 3, and 4 bits per dimension for multi-bit methods. Binary methods (Binary Hash, RaBitQ) operate at 1 bit per dimension only.
