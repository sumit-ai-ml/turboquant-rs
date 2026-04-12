"""TurboQuant + baseline quantization implementations."""

import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional

from utils import SRHTRotation, random_orthogonal, make_rotation, apply_rotation, pca_whiten


# =============================================================================
# TurboQuant MSE: rotation + Lloyd-Max optimal scalar quantization
# =============================================================================

class TurboQuantMSE:
    """TurboQuant with MSE-optimal scalar quantization.

    Steps:
    1. Input already L2-normalized (norms stored separately)
    2. Rotate via SRHT (or dense orthogonal)
    3. Per-coordinate scalar quantization using Beta(d/2,d/2) codebook
    """

    def __init__(self, d: int, bits: int, seed: int, rotation_type: str = "srht"):
        self.d = d
        self.bits = bits
        self.seed = seed
        self.n_levels = 2 ** bits

        # Build rotation (auto-falls back to dense for non-power-of-2 d)
        self.rotation = make_rotation(d, seed, rotation_type)

        # Pre-compute optimal codebook for Beta(d/2, d/2) on [0, 1]
        # then shift to [-1, 1]
        self.boundaries, self.centroids = self._build_codebook(d, bits)

    def _build_codebook(self, d: int, bits: int):
        """Build Lloyd-Max optimal codebook for Beta(d/2, d/2).

        No training data needed. The distribution is known analytically.
        """
        n_levels = 2 ** bits
        a, b = d / 2.0, d / 2.0

        # Initialize with uniform quantiles
        boundaries = beta_dist.ppf(np.linspace(0, 1, n_levels + 1), a, b)
        boundaries[0] = 0.0
        boundaries[-1] = 1.0

        # Lloyd-Max iteration
        for _ in range(100):
            # Centroids: conditional expectation within each bin
            centroids = np.zeros(n_levels)
            for i in range(n_levels):
                lo, hi = boundaries[i], boundaries[i + 1]
                # E[X | lo < X < hi] for Beta distribution
                # Use numerical integration
                x = np.linspace(lo + 1e-10, hi - 1e-10, 1000)
                pdf = beta_dist.pdf(x, a, b)
                mass = np.trapezoid(pdf, x)
                if mass > 1e-15:
                    centroids[i] = np.trapezoid(x * pdf, x) / mass
                else:
                    centroids[i] = (lo + hi) / 2

            # Boundaries: midpoints of adjacent centroids
            new_boundaries = np.zeros(n_levels + 1)
            new_boundaries[0] = 0.0
            new_boundaries[-1] = 1.0
            for i in range(1, n_levels):
                new_boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

            if np.allclose(boundaries, new_boundaries, atol=1e-10):
                break
            boundaries = new_boundaries

        # Shift to [-1, 1]
        boundaries = boundaries * 2 - 1
        centroids = centroids * 2 - 1

        return boundaries.astype(np.float32), centroids.astype(np.float32)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress: rotate then quantize each coordinate."""
        rotated = apply_rotation(embeddings, self.rotation)

        # Shift to [0, 1] for quantization
        shifted = (rotated + 1.0) / 2.0

        # Map to codebook indices via boundaries (shifted back to [0,1])
        boundaries_01 = (self.boundaries + 1.0) / 2.0
        codes = np.digitize(shifted, boundaries_01[1:-1]).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decompress: map codes to centroids, inverse-rotate."""
        reconstructed = self.centroids[codes]
        # Inverse rotation
        if isinstance(self.rotation, SRHTRotation):
            return self.rotation.inverse(reconstructed)
        return reconstructed @ self.rotation  # R^-1 = R^T, so x @ R = x @ R^{-T}

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        """Approximate nearest neighbor search via cosine similarity.

        Decoded vectors have quantization shrinkage (norms < 1), so we
        re-normalize before computing inner products. The ground truth
        uses unit-norm vectors, so cosine similarity is the correct metric.
        """
        query_recon = self.decode(query_codes)
        db_recon = self.decode(db_codes)
        # Re-normalize to remove quantization shrinkage bias
        query_recon = query_recon / np.linalg.norm(query_recon, axis=1, keepdims=True)
        db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
        sims = query_recon @ db_recon.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        """Actual bytes per vector (codes + norm storage)."""
        code_bits = self.d * self.bits
        norm_bytes = 4  # FP32 for the stored norm
        return code_bits / 8 + norm_bytes


# =============================================================================
# TurboQuant Adaptive: data-driven Lloyd-Max codebook
# =============================================================================

class TurboQuantAdaptive:
    """TurboQuant with data-adaptive codebook.

    Instead of assuming Beta(d/2,d/2), trains the Lloyd-Max codebook on the
    actual distribution of rotated coordinates from a training set.
    Optional PCA whitening before the random rotation to better isotropize.
    """

    def __init__(self, d: int, bits: int, seed: int, rotation_type: str = "srht",
                 whiten: bool = False):
        self.d = d
        self.bits = bits
        self.seed = seed
        self.n_levels = 2 ** bits
        self.whiten = whiten
        self.rotation = make_rotation(d, seed, rotation_type)
        self.pca_params = None
        self.boundaries = None
        self.centroids = None

    def train(self, embeddings: np.ndarray):
        """Train codebook on actual rotated coordinate distribution."""
        if self.whiten:
            self.pca_params = pca_whiten(embeddings)
            whitened = (embeddings - self.pca_params["mean"]) @ self.pca_params["transform"]
            # Re-normalize after whitening
            norms = np.linalg.norm(whitened, axis=1, keepdims=True)
            whitened = whitened / np.clip(norms, 1e-8, None)
            rotated = apply_rotation(whitened, self.rotation)
        else:
            rotated = apply_rotation(embeddings, self.rotation)

        # Collect all coordinate values as the empirical distribution
        flat = rotated.ravel()

        # Lloyd-Max on empirical data via histogram approximation
        self.boundaries, self.centroids = self._lloyd_max_empirical(flat)

    def _lloyd_max_empirical(self, data: np.ndarray):
        """Lloyd-Max optimal codebook from empirical data."""
        n_levels = self.n_levels

        # Initialize with uniform quantiles of the data
        percentiles = np.linspace(0, 100, n_levels + 1)
        boundaries = np.percentile(data, percentiles)
        boundaries[0] = data.min() - 1e-6
        boundaries[-1] = data.max() + 1e-6

        for _ in range(100):
            # Centroids: mean of data in each bin
            centroids = np.zeros(n_levels)
            for i in range(n_levels):
                mask = (data >= boundaries[i]) & (data < boundaries[i + 1])
                if i == n_levels - 1:
                    mask = (data >= boundaries[i]) & (data <= boundaries[i + 1])
                if mask.sum() > 0:
                    centroids[i] = data[mask].mean()
                else:
                    centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2

            # Boundaries: midpoints of adjacent centroids
            new_boundaries = np.zeros(n_levels + 1)
            new_boundaries[0] = data.min() - 1e-6
            new_boundaries[-1] = data.max() + 1e-6
            for i in range(1, n_levels):
                new_boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

            if np.allclose(boundaries, new_boundaries, atol=1e-10):
                break
            boundaries = new_boundaries

        return boundaries.astype(np.float32), centroids.astype(np.float32)

    def _apply_pre_rotation(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA whitening (if enabled) then rotation."""
        if self.whiten and self.pca_params is not None:
            whitened = (embeddings - self.pca_params["mean"]) @ self.pca_params["transform"]
            norms = np.linalg.norm(whitened, axis=1, keepdims=True)
            whitened = whitened / np.clip(norms, 1e-8, None)
            return apply_rotation(whitened, self.rotation)
        return apply_rotation(embeddings, self.rotation)

    def _inverse_rotation(self, reconstructed: np.ndarray) -> np.ndarray:
        """Inverse rotation then inverse PCA whitening."""
        if isinstance(self.rotation, SRHTRotation):
            unrotated = self.rotation.inverse(reconstructed)
        else:
            unrotated = reconstructed @ self.rotation
        if self.whiten and self.pca_params is not None:
            return unrotated @ self.pca_params["inverse_transform"] + self.pca_params["mean"]
        return unrotated

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress: whiten + rotate then quantize each coordinate."""
        rotated = self._apply_pre_rotation(embeddings)
        codes = np.digitize(rotated, self.boundaries[1:-1]).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decompress: map codes to centroids, inverse-rotate."""
        reconstructed = self.centroids[codes]
        return self._inverse_rotation(reconstructed)

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        """Approximate nearest neighbor search via cosine similarity."""
        query_recon = self.decode(query_codes)
        db_recon = self.decode(db_codes)
        query_recon = query_recon / np.linalg.norm(query_recon, axis=1, keepdims=True)
        db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
        sims = query_recon @ db_recon.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        code_bits = self.d * self.bits
        norm_bytes = 4
        return code_bits / 8 + norm_bytes


# =============================================================================
# TurboQuant Prod: MSE-only (QJL found counterproductive for retrieval)
# =============================================================================

class TurboQuantProd(TurboQuantMSE):
    """TurboQuant Prod variant — identical to MSE for embedding retrieval.

    The QJL residual correction (paper Section 4) is designed for raw inner
    product estimation. For cosine-similarity retrieval on unit-norm embeddings,
    the sign sketch variance dominates the correction signal, degrading recall.
    This matches findings from multiple independent implementations
    (back2matching/turboquant, cksac/turboquant-model).

    Kept as an alias so benchmark code doesn't break. Results will match MSE.
    """
    pass


# =============================================================================
# Baselines
# =============================================================================

class BinaryHash:
    """Binary hashing: sign(x) -> Hamming distance."""

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert to binary codes."""
        return (embeddings > 0).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Binary codes can't reconstruct, return as float for API compat."""
        return codes.astype(np.float32) * 2 - 1

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        """Hamming distance search."""
        # XOR + popcount via sum of differences
        n_q = len(query_codes)
        indices = np.zeros((n_q, k), dtype=np.int64)
        for i in range(n_q):
            hamming = np.sum(query_codes[i] != db_codes, axis=1)
            indices[i] = np.argsort(hamming)[:k]
        return indices

    def bytes_per_vector(self, d: int) -> float:
        return d / 8  # 1 bit per dimension


class ProductQuantization:
    """FAISS Product Quantization wrapper.

    Target budget: bits * d total bits for the code, matching TurboQuant's budget.
    FAISS PQ splits d into m subspaces with nbits per subspace.
    Total storage = m * nbits / 8 bytes. We set nbits=8 (standard) and choose
    m to approximate the target budget: m = bits * d / 8.
    """

    def __init__(self, d: int, bits: int):
        import faiss
        self.d = d
        self.bits = bits
        self.nbits_per_subspace = 8
        # Target: total bits = bits * d, so m = bits * d / 8 (since each subspace uses 8 bits)
        target_m = max(1, (bits * d) // (self.nbits_per_subspace))
        # m must divide d
        self.m = target_m
        while self.m > 0 and d % self.m != 0:
            self.m -= 1
        if self.m < 1:
            self.m = 1
        self.index = faiss.IndexPQ(d, self.m, self.nbits_per_subspace)
        self.trained = False

    def train(self, embeddings: np.ndarray):
        """Train PQ codebooks on training split."""
        self.index.train(embeddings.astype(np.float32))
        self.trained = True

    def encode(self, embeddings: np.ndarray):
        """Encode after training."""
        assert self.trained, "Must call train() first"
        self.index.add(embeddings.astype(np.float32))
        return None  # FAISS manages internally

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Search the FAISS index."""
        _, indices = self.index.search(queries.astype(np.float32), k)
        return indices

    def bytes_per_vector(self) -> float:
        return self.m * self.nbits_per_subspace / 8


class TurboQuantResidualPQ:
    """Hybrid: TurboQuant (training-free) + PQ on residuals (data-adaptive).

    TQ captures the isotropic component for free.
    PQ learns only the structured residual that TQ couldn't handle.

    Inner product decomposition: <q, x> = <q, x_hat> + <q, r>
    where x_hat = TQ reconstruction, r = x - x_hat.
    """

    def __init__(self, d: int, tq_bits: int, pq_m: int, seed: int,
                 rotation_type: str = "srht"):
        self.d = d
        self.tq_bits = tq_bits
        self.pq_m = pq_m
        self.seed = seed
        self.tq = TurboQuantMSE(d, tq_bits, seed, rotation_type)
        self.pq_index = None
        self.tq_reconstructed_db = None
        self.actual_pq_m = None

    def train(self, embeddings: np.ndarray):
        """Encode with TQ, compute residuals, train PQ on residuals."""
        import faiss

        codes_tq = self.tq.encode(embeddings)
        tq_recon = self.tq.decode(codes_tq)
        # Re-normalize TQ reconstruction (critical for cosine similarity)
        tq_recon = tq_recon / np.linalg.norm(tq_recon, axis=1, keepdims=True)
        residuals = (embeddings - tq_recon).astype(np.float32)

        # Find valid m that divides d
        m = self.pq_m
        while m > 0 and self.d % m != 0:
            m -= 1
        self.actual_pq_m = max(1, m)

        self.pq_index = faiss.IndexPQ(self.d, self.actual_pq_m, 8)
        self.pq_index.train(residuals)

    def encode(self, database: np.ndarray):
        """Encode database: TQ codes + residuals added to PQ index."""
        import faiss
        codes_tq = self.tq.encode(database)
        tq_recon = self.tq.decode(codes_tq)
        tq_recon = tq_recon / np.linalg.norm(tq_recon, axis=1, keepdims=True)
        residuals = (database - tq_recon).astype(np.float32)

        self.pq_index.add(residuals)
        self.tq_reconstructed_db = tq_recon
        return codes_tq

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Search: reconstruct full vector (TQ + PQ residual), then cosine."""
        # Reconstruct: x_approx = x_hat + r_hat
        n_db = self.pq_index.ntotal
        residual_recon = np.zeros((n_db, self.d), dtype=np.float32)
        for i in range(n_db):
            residual_recon[i] = self.pq_index.reconstruct(i)

        full_recon = self.tq_reconstructed_db + residual_recon
        # Re-normalize for cosine similarity
        full_recon = full_recon / np.linalg.norm(full_recon, axis=1, keepdims=True)

        sims = queries @ full_recon.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        tq_bytes = self.d * self.tq_bits / 8 + 4
        pq_bytes = self.actual_pq_m  # 1 byte per subspace (8 bits each)
        return tq_bytes + pq_bytes


class RaBitQ:
    """RaBitQ: random rotation + binarization for approximate nearest neighbor.

    From Gao & Long, SIGMOD 2024 (arXiv:2405.12497).

    For unit-norm cosine similarity retrieval, the core idea is simple:
    apply a random orthogonal rotation before binarization. This spreads
    information uniformly across coordinates before taking sign bits,
    unlike binary hash which binarizes raw (axis-aligned) coordinates.

    Search uses Hamming distance on the rotated codes, which estimates:
      <o, q> ≈ 1 - 2 * hamming(code_o, code_q) / d

    Training-free. Storage: D/8 bytes (binary code only).
    """

    def __init__(self, d: int, seed: int):
        self.d = d
        self.seed = seed
        self.P = random_orthogonal(d, seed)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode: rotate with random orthogonal P, then binarize."""
        rotated = embeddings @ self.P.T  # (N, d)
        return (rotated >= 0).astype(np.uint8)  # (N, d)

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int,
               **kwargs) -> np.ndarray:
        """Vectorized Hamming distance search on rotated binary codes.

        ip_estimate = 1 - 2 * hamming(code_q, code_db) / d
        Equivalent to: (2 * XNOR_popcount / d) - 1
        """
        q_float = query_codes.astype(np.float32)
        db_float = db_codes.astype(np.float32)
        # matches[i,j] = number of bit positions where codes agree
        matches = q_float @ db_float.T + (1 - q_float) @ (1 - db_float).T
        ip_estimates = 2 * matches / self.d - 1
        return np.argsort(-ip_estimates, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        return self.d / 8  # binary code only


class FP32Exact:
    """No compression. Brute-force exact search (upper bound)."""

    def __init__(self, d: int):
        self.d = d

    def search(self, queries: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
        sims = queries @ database.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        return self.d * 4  # FP32


# =============================================================================
# Training-free methods
# =============================================================================

class UniformScalarQuant:
    """Rotation + uniform scalar quantization. No codebook optimization.

    Same pipeline as TurboQuant MSE but uses a uniform grid on the data range
    instead of the Beta-optimal codebook. The simplest possible rotation-based
    quantizer — serves as an ablation to measure the value of the Beta codebook.
    """

    def __init__(self, d: int, bits: int, seed: int, rotation_type: str = "srht"):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.rotation = make_rotation(d, seed, rotation_type)
        # Uniform grid on [-1, 1] (covers unit-norm rotated coordinates)
        self.boundaries = np.linspace(-1, 1, self.n_levels + 1).astype(np.float32)
        self.centroids = ((self.boundaries[:-1] + self.boundaries[1:]) / 2).astype(np.float32)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        rotated = apply_rotation(embeddings, self.rotation)
        codes = np.digitize(rotated, self.boundaries[1:-1]).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        reconstructed = self.centroids[codes]
        if isinstance(self.rotation, SRHTRotation):
            return self.rotation.inverse(reconstructed)
        return reconstructed @ self.rotation

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        query_recon = self.decode(query_codes)
        db_recon = self.decode(db_codes)
        query_recon = query_recon / np.linalg.norm(query_recon, axis=1, keepdims=True)
        db_recon = db_recon / np.linalg.norm(db_recon, axis=1, keepdims=True)
        sims = query_recon @ db_recon.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        return self.d * self.bits / 8 + 4


class SimHashMultiBit:
    """Multi-bit SimHash: k independent random hyperplanes, 1 bit each.

    Standard SimHash uses sign(x) for d bits. This variant uses k random
    projections (k can be > d or < d) to decouple hash length from embedding dim.
    Bits parameter controls total bits: k = bits * d (same budget as TurboQuant).
    """

    def __init__(self, d: int, bits: int, seed: int):
        self.d = d
        self.bits = bits
        self.k = bits * d  # total hash bits = same storage as TQ
        rng = np.random.RandomState(seed)
        # Random hyperplanes, normalized
        self.hyperplanes = rng.randn(self.k, d).astype(np.float32)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        projections = embeddings @ self.hyperplanes.T
        return (projections > 0).astype(np.uint8)

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        n_q = len(query_codes)
        indices = np.zeros((n_q, k), dtype=np.int64)
        for i in range(n_q):
            hamming = np.sum(query_codes[i] != db_codes, axis=1)
            indices[i] = np.argsort(hamming)[:k]
        return indices

    def bytes_per_vector(self) -> float:
        return self.k / 8


class RandProjQuant:
    """Random Projection + Scalar Quantization.

    Project from d dimensions to m < d via random Gaussian matrix, then
    uniformly quantize each projected coordinate. Trades dimension reduction
    for more bits per coordinate. Total storage matches TurboQuant.
    """

    def __init__(self, d: int, bits: int, seed: int):
        self.d = d
        self.bits = bits
        # Choose m so total bits ≈ bits * d
        # Each projected coordinate gets 8 bits (uint8), so m = bits * d / 8
        self.m = max(1, bits * d // 8)
        self.n_levels = 256  # 8-bit quantization per projected dimension
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(d, self.m).astype(np.float32) / np.sqrt(self.m)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        projected = embeddings @ self.proj  # (N, m)
        # Per-batch min/max quantization to uint8
        self._min = projected.min(axis=0)
        self._scale = projected.max(axis=0) - self._min
        self._scale = np.where(self._scale < 1e-8, 1.0, self._scale)
        normalized = (projected - self._min) / self._scale
        return (normalized * 255).clip(0, 255).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return codes.astype(np.float32) / 255 * self._scale + self._min

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        # Reconstruct in projected space and use inner product
        q = self.decode(query_codes)
        db = self.decode(db_codes)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        db = db / np.linalg.norm(db, axis=1, keepdims=True)
        sims = q @ db.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        return self.m  # 1 byte per projected dimension


class FlyHash:
    """FlyHash: bio-inspired sparse random expansion + winner-take-all.

    Inspired by the fruit fly olfactory circuit (Dasgupta et al., Science 2017).
    1. Sparse random expansion: d -> m (m >> d) via sparse binary matrix
    2. Winner-take-all: keep top-k activations, zero the rest
    3. Binary encoding: nonzero = 1

    The hash length is m bits, and sparsity is controlled by the WTA ratio.
    """

    def __init__(self, d: int, bits: int, seed: int, expansion: int = 20, wta_ratio: float = 0.05):
        self.d = d
        self.bits = bits
        # Hash length = bits * d (same total storage as TurboQuant)
        self.m = bits * d
        self.wta_k = max(1, int(self.m * wta_ratio))
        rng = np.random.RandomState(seed)
        # Sparse random connections: each output neuron connects to ~6 inputs
        n_connections = 6
        self.connections = np.zeros((self.m, d), dtype=np.float32)
        for i in range(self.m):
            idx = rng.choice(d, size=n_connections, replace=False)
            self.connections[i, idx] = 1.0

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        # Sparse expansion
        activations = embeddings @ self.connections.T  # (N, m)
        # Winner-take-all: keep top-k per vector
        codes = np.zeros_like(activations, dtype=np.uint8)
        for i in range(len(activations)):
            top_k = np.argpartition(activations[i], -self.wta_k)[-self.wta_k:]
            codes[i, top_k] = 1
        return codes

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        n_q = len(query_codes)
        indices = np.zeros((n_q, k), dtype=np.int64)
        for i in range(n_q):
            # Hamming distance (or equivalently, overlap of active bits)
            hamming = np.sum(query_codes[i] != db_codes, axis=1)
            indices[i] = np.argsort(hamming)[:k]
        return indices

    def bytes_per_vector(self) -> float:
        return self.m / 8
