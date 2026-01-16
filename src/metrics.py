from typing import Tuple

import numpy as np
from sklearn.manifold import trustworthiness


# Maximum number of samples before switching to sampling-based evaluation
# For N samples, sklearn computes an N×N distance matrix
# At 10k samples, this is 10k × 10k × 4 bytes = 400MB (manageable)
MAX_SAMPLES_FOR_FULL_EVALUATION = 10000


def continuity(
    high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int = 5
) -> float:
    """
    Compute continuity metric between high-dimensional and low-dimensional embeddings.

    Continuity measures how well the local neighborhood structure is preserved when
    going from high-dimensional to low-dimensional space. It's the complement of
    trustworthiness in the reverse direction.

    Parameters
    ----------
    high_dim : np.ndarray, shape (n_samples, n_features_high)
        High-dimensional data
    low_dim : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional embedding
    n_neighbors : int, default=5
        Number of neighbors to consider

    Returns
    -------
    continuity : float
        Continuity score between 0 and 1, higher is better
    """
    # Compute trustworthiness in the reverse direction (low -> high)
    # This gives us continuity (high -> low)
    return trustworthiness(low_dim, high_dim, n_neighbors=n_neighbors)


def evaluate_embedding(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    n_neighbors: int = 5,
    max_samples: int = MAX_SAMPLES_FOR_FULL_EVALUATION,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Evaluate embedding quality using trustworthiness and continuity metrics.

    For large datasets (> max_samples), uses random sampling to estimate metrics
    without computing the full N×N distance matrix.

    Parameters
    ----------
    high_dim : np.ndarray, shape (n_samples, n_features_high)
        Original high-dimensional data
    low_dim : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional embedding
    n_neighbors : int, default=5
        Number of neighbors to consider for metrics
    max_samples : int, default=10000
        Maximum samples before using sampling-based evaluation
    random_state : int, default=42
        Random seed for reproducible sampling

    Returns
    -------
    trustworthiness_score : float
        Trustworthiness metric (0-1, higher is better)
    continuity_score : float
        Continuity metric (0-1, higher is better)
    """
    n_samples = high_dim.shape[0]

    if n_samples > max_samples:
        # Use sampling-based evaluation for large datasets
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, size=max_samples, replace=False)
        high_dim = high_dim[indices]
        low_dim = low_dim[indices]
        print(f"  (Using {max_samples} sampled points for metric evaluation)")

    trust_score = trustworthiness(high_dim, low_dim, n_neighbors=n_neighbors)
    cont_score = continuity(high_dim, low_dim, n_neighbors=n_neighbors)

    return trust_score, cont_score
