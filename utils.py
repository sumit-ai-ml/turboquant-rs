"""Shared utilities: normalization, filtering, rotation, metrics."""

import numpy as np
from scipy.linalg import hadamard
from scipy.stats import beta, kstest
from typing import Optional


# --- Embedding preprocessing ---

def filter_zero_norm(embeddings: np.ndarray, threshold: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Remove zero-norm vectors (blank/cloudy tiles). Returns (filtered, mask)."""
    norms = np.linalg.norm(embeddings, axis=1)
    mask = norms > threshold
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  Filtered {n_removed} zero-norm vectors ({n_removed/len(mask)*100:.2f}%)")
    return embeddings[mask], mask


def l2_normalize(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize to unit sphere. Returns (normalized, norms).

    Norms are stored separately for reconstruction (TurboQuant Step 1).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    # Sanity check
    check_norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(check_norms, 1.0, atol=1e-5), f"Normalization failed: max deviation {np.max(np.abs(check_norms - 1.0))}"
    return normalized, norms.squeeze()


# --- Rotation matrices ---

def random_orthogonal(d: int, seed: int) -> np.ndarray:
    """Dense random orthogonal matrix via QR decomposition. O(d^2) per vector."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    # Fix sign ambiguity to get uniform Haar measure
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)


class SRHTRotation:
    """Structured Randomized Hadamard Transform. O(d log d) per vector.

    Only works correctly when d is a power of 2. For other dimensions,
    use random_orthogonal() instead (see make_rotation helper).
    """

    def __init__(self, d: int, seed: int):
        self.d = d
        if d & (d - 1) != 0:
            raise ValueError(
                f"SRHTRotation requires d to be a power of 2, got d={d}. "
                f"Use make_rotation(d, seed, 'srht') which auto-falls back to dense."
            )

        rng = np.random.RandomState(seed)
        self.signs_d = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        self.signs_s = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        self.H = hadamard(d).astype(np.float32) / np.sqrt(d)

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Rotate embeddings. (N, d) -> (N, d)."""
        rotated = embeddings * self.signs_s[None, :]
        rotated = rotated @ self.H.T
        rotated = rotated * self.signs_d[None, :]
        return rotated

    def inverse(self, rotated: np.ndarray) -> np.ndarray:
        """Inverse rotation. (N, d) -> (N, d)."""
        inv = rotated * self.signs_d[None, :]
        inv = inv @ self.H.T  # H is symmetric and orthogonal
        inv = inv * self.signs_s[None, :]
        return inv


def make_rotation(d: int, seed: int, rotation_type: str = "srht"):
    """Build a rotation. Uses SRHT for power-of-2 d, dense orthogonal otherwise.

    Returns either an SRHTRotation object or a dense orthogonal np.ndarray.
    """
    is_power_of_2 = (d & (d - 1) == 0) and d > 0
    if rotation_type == "srht" and is_power_of_2:
        return SRHTRotation(d, seed)
    if rotation_type == "srht" and not is_power_of_2:
        # SRHT doesn't work for non-power-of-2; fall back to dense
        pass
    return random_orthogonal(d, seed)


def apply_rotation(embeddings: np.ndarray, rotation) -> np.ndarray:
    """Apply rotation to embeddings. Supports both matrix and SRHTRotation."""
    if isinstance(rotation, SRHTRotation):
        return rotation.forward(embeddings)
    # Dense orthogonal matrix: (N, d) @ (d, d).T -> (N, d)
    return embeddings @ rotation.T


def verify_rotation(rotation, atol: float = 1e-3) -> bool:
    """Verify rotation preserves inner products (orthogonality check)."""
    if isinstance(rotation, SRHTRotation):
        # Test with random vectors: inner products should be preserved
        rng = np.random.RandomState(999)
        test = rng.randn(100, rotation.d).astype(np.float32)
        test = test / np.linalg.norm(test, axis=1, keepdims=True)
        rotated = rotation.forward(test)
        orig_dots = test[:10] @ test[10:20].T
        rot_dots = rotated[:10] @ rotated[10:20].T
        return np.allclose(orig_dots, rot_dots, atol=atol)
    d = rotation.shape[0]
    product = rotation.T @ rotation
    return np.allclose(product, np.eye(d), atol=atol)


# --- Beta distribution validation ---

def beta_ks_test(rotated_coords: np.ndarray, d: int) -> dict:
    """KS test of rotated coordinates against Beta(d/2, d/2).

    TurboQuant assumes each coordinate of a rotated unit vector follows
    Beta(d/2, d/2) on [-1, 1], which is Beta(d/2, d/2) on [0, 1] after
    shifting: x_shifted = (x + 1) / 2.

    Returns dict with D statistic, p-value, and effect size interpretation.
    """
    # Shift from [-1, 1] to [0, 1] for Beta CDF
    shifted = (rotated_coords + 1.0) / 2.0
    shifted = np.clip(shifted, 1e-10, 1 - 1e-10)  # avoid exact 0/1

    a = d / 2.0
    b = d / 2.0
    D, p_value = kstest(shifted, beta(a, b).cdf)

    # Effect size interpretation (critical: large N always rejects)
    if D < 0.01:
        interpretation = "excellent_fit"
    elif D < 0.02:
        interpretation = "good_fit"
    elif D < 0.05:
        interpretation = "moderate_fit"
    else:
        interpretation = "poor_fit"

    return {
        "D_statistic": float(D),
        "p_value": float(p_value),
        "interpretation": interpretation,
        "n_samples": len(rotated_coords),
    }


def coordinate_independence_check(rotated: np.ndarray, n_pairs: int = 50) -> dict:
    """Check pairwise correlation of rotated coordinates (should be ~0)."""
    d = rotated.shape[1]
    rng = np.random.RandomState(0)
    pairs = rng.choice(d, size=(n_pairs, 2), replace=True)

    correlations = []
    for i, j in pairs:
        if i != j:
            corr = np.corrcoef(rotated[:, i], rotated[:, j])[0, 1]
            correlations.append(abs(corr))

    return {
        "mean_abs_correlation": float(np.mean(correlations)),
        "max_abs_correlation": float(np.max(correlations)),
        "n_pairs_checked": len(correlations),
    }


# --- Retrieval metrics ---

def recall_at_k(queries: np.ndarray, database: np.ndarray,
                queries_compressed, database_compressed,
                search_fn, k_values: list[int]) -> dict:
    """Compute recall@k for compressed search vs exact FP32 search.

    Args:
        queries: (N_q, d) FP32 query embeddings
        database: (N_db, d) FP32 database embeddings
        queries_compressed: compressed query representations
        database_compressed: compressed database representations
        search_fn: callable(queries_compressed, database_compressed, k) -> indices (N_q, k)
        k_values: list of k values to evaluate

    Returns:
        dict mapping k -> recall@k (float in [0, 1])
    """
    max_k = max(k_values)

    # Ground truth: exact FP32 inner product search
    sims = queries @ database.T  # (N_q, N_db)
    gt_indices = np.argsort(-sims, axis=1)[:, :max_k]

    # Approximate search
    approx_indices = search_fn(queries_compressed, database_compressed, max_k)

    results = {}
    for k in k_values:
        gt_set = [set(gt_indices[i, :k]) for i in range(len(queries))]
        ap_set = [set(approx_indices[i, :k]) for i in range(len(queries))]
        recall = np.mean([len(g & a) / k for g, a in zip(gt_set, ap_set)])
        results[k] = float(recall)

    return results


def measure_compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Compression ratio: original / compressed."""
    return original_bytes / compressed_bytes
