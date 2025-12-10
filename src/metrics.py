from typing import Tuple

import numpy as np
from sklearn.manifold import trustworthiness


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
    high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int = 5
) -> Tuple[float, float]:
    """
    Evaluate embedding quality using trustworthiness and continuity metrics.

    Parameters
    ----------
    high_dim : np.ndarray, shape (n_samples, n_features_high)
        Original high-dimensional data
    low_dim : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional embedding
    n_neighbors : int, default=5
        Number of neighbors to consider for metrics

    Returns
    -------
    trustworthiness_score : float
        Trustworthiness metric (0-1, higher is better)
    continuity_score : float
        Continuity metric (0-1, higher is better)
    """
    trust_score = trustworthiness(high_dim, low_dim, n_neighbors=n_neighbors)
    cont_score = continuity(high_dim, low_dim, n_neighbors=n_neighbors)

    return trust_score, cont_score
