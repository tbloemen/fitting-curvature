"""Pytest configuration for fitting-curvature tests."""

import sys
from pathlib import Path

import torch
from torch import Tensor

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_distance_matrix(X: Tensor) -> Tensor:
    """
    Compute pairwise Euclidean distance matrix efficiently using vectorized operations.

    This uses the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points where N is number of points and D is dimensionality

    Returns
    -------
    Tensor, shape (N, N)
        Pairwise distance matrix on the same device as X
    """
    # Compute squared norms for each point: ||x||^2
    # Shape: (N, 1)
    squared_norms = (X**2).sum(dim=1, keepdim=True)

    # Compute dot products: X @ X^T
    # Shape: (N, N)
    dot_products = torch.mm(X, X.t())

    # Use the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    # Broadcasting: (N, 1) + (1, N) - (N, N) = (N, N)
    squared_distances = squared_norms + squared_norms.t() - 2 * dot_products

    # Clamp to avoid negative values due to numerical errors
    squared_distances = torch.clamp(squared_distances, min=0.0)

    # Take square root to get Euclidean distances
    distances = torch.sqrt(squared_distances)

    return distances


def compute_loss(
    embedded_distances: Tensor,
    target_distances: Tensor,
    loss_type: str = "gu2019",
) -> Tensor:
    """
    Compute loss function for full distance matrices.

    Used in testing to evaluate embedding quality on all pairs.

    Parameters
    ----------
    embedded_distances : Tensor, shape (N, N)
        Pairwise distances in the embedded space
    target_distances : Tensor, shape (N, N)
        Target pairwise distances from original space
    loss_type : str
        Type of loss function: 'gu2019' for relative distortion or 'mse' for mean squared error

    Returns
    -------
    Tensor
        Scalar loss value
    """
    if loss_type == "gu2019":
        # Gu et al. (2019) relative distortion loss
        # L = sum((d_embedded / d_target - 1)^2)
        # Ignore diagonal (self-distances) and use only upper triangle to avoid double-counting
        mask = torch.triu(
            torch.ones_like(embedded_distances, dtype=torch.bool), diagonal=1
        )
        embedded_masked = embedded_distances[mask]
        target_masked = target_distances[mask]
        loss = torch.sum((embedded_masked / (target_masked + 1e-8) - 1) ** 2)
    elif loss_type == "mse":
        # Mean squared error (stress function)
        # Ignore diagonal and use only upper triangle
        mask = torch.triu(
            torch.ones_like(embedded_distances, dtype=torch.bool), diagonal=1
        )
        embedded_masked = embedded_distances[mask]
        target_masked = target_distances[mask]
        loss = torch.sum((embedded_masked - target_masked) ** 2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'gu2019' or 'mse'")

    return loss
