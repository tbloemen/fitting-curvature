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


def get_init_scale_from_data(
    X: Tensor, embed_dim: int, n_samples: int = 10000, verbose: bool = True
) -> float:
    """
    Compute initialization scale from raw data by sampling pairs.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points
    embed_dim : int
        Target embedding dimension
    n_samples : int
        Number of pairs to sample for estimation (default: 10000)
    verbose : bool
        Print statistics

    Returns
    -------
    float
        Initialization scale
    """
    n_points = X.shape[0]
    device = X.device

    # Sample random pairs
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
    std_distance = distances.std().item()

    init_scale = (
        mean_distance
        / (2 * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))).item()
    )

    if verbose:
        print(
            f"Distance statistics (sampled): mean={mean_distance:.4f}, std={std_distance:.4f}"
        )
        print(f"Initialization scale: {init_scale:.4f}")

    return init_scale
