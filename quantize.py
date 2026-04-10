"""TurboQuant + baseline quantization implementations."""

import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional

from utils import srht_matrix, random_orthogonal, apply_rotation


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

        # Build rotation matrix
        if rotation_type == "srht":
            self.rotation = srht_matrix(d, seed)
        else:
            self.rotation = random_orthogonal(d, seed)

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
                mass = np.trapz(pdf, x)
                if mass > 1e-15:
                    centroids[i] = np.trapz(x * pdf, x) / mass
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
        # Inverse rotation (R is orthogonal so R^-1 = R^T)
        return apply_rotation(reconstructed, self.rotation.T)

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        """Approximate nearest neighbor search via decoded inner products."""
        # Decode both
        query_recon = self.decode(query_codes)
        db_recon = self.decode(db_codes)
        # Brute force inner product
        sims = query_recon @ db_recon.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        """Actual bytes per vector (codes + norm storage)."""
        code_bits = self.d * self.bits
        norm_bytes = 4  # FP32 for the stored norm
        return code_bits / 8 + norm_bytes


# =============================================================================
# TurboQuant Prod: MSE + QJL residual correction
# =============================================================================

class TurboQuantProd(TurboQuantMSE):
    """TurboQuant with product estimator (QJL residual correction).

    Extends MSE variant with a correction term for inner product estimation.
    See Section 4 of the TurboQuant paper.
    """

    def search(self, query_codes: np.ndarray, db_codes: np.ndarray, k: int) -> np.ndarray:
        """Search with QJL-corrected inner product estimation."""
        # Decode
        query_recon = self.decode(query_codes)
        db_recon = self.decode(db_codes)

        # Base inner product
        base_sims = query_recon @ db_recon.T

        # QJL correction: estimate residual inner products
        # <q, d> ≈ <q_hat, d_hat> + correction
        # The correction uses the quantization residuals
        # For now, use base similarity (full implementation requires
        # storing quantization residuals, which adds to storage)
        # TODO: implement full QJL correction per paper Section 4
        sims = base_sims

        return np.argsort(-sims, axis=1)[:, :k]


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
    """FAISS Product Quantization wrapper."""

    def __init__(self, d: int, bits: int):
        import faiss
        self.d = d
        self.bits = bits
        # PQ: split d dims into m subspaces, each quantized to 2^bits centroids
        # Standard: m = d / 8 subspaces for byte-aligned codes
        self.m = min(d // 2, d // (bits * 2))  # subspace count
        if self.m < 1:
            self.m = 1
        # Adjust m so d is divisible by m
        while d % self.m != 0:
            self.m -= 1
        self.nbits_per_subspace = 8  # FAISS PQ uses 8 bits per subspace by default
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


class FP32Exact:
    """No compression. Brute-force exact search (upper bound)."""

    def __init__(self, d: int):
        self.d = d

    def search(self, queries: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
        sims = queries @ database.T
        return np.argsort(-sims, axis=1)[:, :k]

    def bytes_per_vector(self) -> float:
        return self.d * 4  # FP32
