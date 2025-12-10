import torch
from torch import Tensor


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


def get_scale_and_n(
    distance_matrix: Tensor, embed_dim: int, verbose=True
) -> tuple[int, float]:
    n_points = distance_matrix.shape[0]

    # Compute statistics from the distance matrix to inform initialization
    # Exclude diagonal (zero distances) when computing statistics
    mask = ~torch.eye(n_points, dtype=torch.bool, device=distance_matrix.device)
    distances_no_diag = distance_matrix[mask]

    mean_distance = distances_no_diag.mean().item()
    std_distance = distances_no_diag.std().item()

    # Use a scale that produces reasonable initial distances in embedding space
    # We want initial random distances to be on the same order as target distances
    init_scale = (
        mean_distance
        / (2 * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))).item()
    )

    if verbose:
        print(f"Distance statistics: mean={mean_distance:.4f}, std={std_distance:.4f}")
        print(f"Initialization scale: {init_scale:.4f}")

    return n_points, init_scale
