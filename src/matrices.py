import math

import torch
from torch import Tensor


def compute_euclidean_distances_batched(
    X: Tensor, indices_i: Tensor, indices_j: Tensor
) -> Tensor:
    """
    Compute Euclidean distances for specified pairs on-the-fly.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points
    indices_i : Tensor, shape (batch_size,)
        First point indices
    indices_j : Tensor, shape (batch_size,)
        Second point indices

    Returns
    -------
    Tensor, shape (batch_size,)
        Euclidean distances for the specified pairs
    """
    points_i = X[indices_i]  # (batch_size, D)
    points_j = X[indices_j]  # (batch_size, D)
    diff = points_i - points_j
    return torch.sqrt((diff**2).sum(dim=1))


def normalize_data(X: Tensor, n_samples: int = 10000, verbose: bool = True) -> Tensor:
    """
    Normalize data so that mean pairwise distance equals 1.

    This normalization makes the data geometry-agnostic, allowing
    embeddings to use a fixed initialization scale regardless of the
    original data scale.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points
    n_samples : int
        Number of pairs to sample for estimation (default: 10000)
    verbose : bool
        Print statistics

    Returns
    -------
    Tensor
        - Normalized data tensor with mean pairwise distance = 1
    """
    n_points = X.shape[0]
    device = X.device

    # Sample random pairs
    n_samples = min(n_samples, n_points * (n_points - 1) // 2)
    indices_i = torch.randint(0, n_points, (n_samples,), device=device)
    indices_j = torch.randint(0, n_points, (n_samples,), device=device)

    # Ensure i != j
    mask = indices_i == indices_j
    while mask.any():
        indices_j[mask] = torch.randint(0, n_points, (int(mask.sum()),), device=device)
        mask = indices_i == indices_j

    # Compute distances for sampled pairs
    distances = compute_euclidean_distances_batched(X, indices_i, indices_j)

    mean_distance = distances.mean().item()

    if verbose:
        std_distance = distances.std().item()
        print(
            f"Original distance statistics: mean={mean_distance:.4f}, std={std_distance:.4f}"
        )

    # Normalize data so mean distance = 1
    X_normalized = X / mean_distance

    if verbose:
        print(f"Normalized data (divided by {mean_distance:.4f})")

    return X_normalized


def get_default_init_scale(embed_dim: int) -> float:
    """
    Get a principled initialization scale for normalized data.

    For data with mean pairwise distance = 1, this returns the standard
    deviation σ such that points initialized as N(0, σ²I) will have
    expected pairwise distance ≈ 1.

    For Gaussian-distributed points in d dimensions:
        E[||x - y||] ≈ σ * sqrt(2 * d)

    So to get E[||x - y||] = 1, we need σ = 1 / sqrt(2 * d).

    Parameters
    ----------
    embed_dim : int
        Target embedding dimension

    Returns
    -------
    float
        Initialization scale (standard deviation)
    """
    return 1.0 / math.sqrt(2 * embed_dim)
