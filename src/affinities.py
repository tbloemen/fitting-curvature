"""
Affinity computation for t-SNE.

This module computes perplexity-based affinities for t-SNE embeddings.
Affinities are probability distributions that capture neighborhood relationships
in the high-dimensional space.

References:
- van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
"""

import torch
from torch import Tensor


def _binary_search_sigma(
    distances_row: Tensor,
    target_perplexity: float,
    tol: float = 1e-5,
    max_iterations: int = 50,
) -> float:
    """
    Binary search for sigma that achieves target perplexity for a single point.

    For point i, find sigma_i such that the perplexity of the conditional
    distribution p_j|i equals the target perplexity.

    Parameters
    ----------
    distances_row : Tensor, shape (k,)
        Squared distances from point i to its neighbors (excluding self)
    target_perplexity : float
        Desired perplexity value
    tol : float
        Tolerance for binary search convergence
    max_iterations : int
        Maximum number of binary search iterations

    Returns
    -------
    float
        Optimal sigma value for this point
    """
    target_entropy = torch.log(torch.tensor(target_perplexity))

    # Initial bounds for sigma
    sigma_min = 1e-10
    sigma_max = 1e4
    sigma = 1.0

    for _ in range(max_iterations):
        # Compute conditional probabilities: p_j|i = exp(-d_ij^2 / 2*sigma^2) / Z
        neg_sq_dist_scaled = -distances_row / (2.0 * sigma * sigma)
        # Subtract max for numerical stability
        neg_sq_dist_scaled = neg_sq_dist_scaled - neg_sq_dist_scaled.max()
        exp_vals = torch.exp(neg_sq_dist_scaled)
        sum_exp = exp_vals.sum()

        if sum_exp == 0:
            sum_exp = torch.tensor(1e-10)

        probs = exp_vals / sum_exp

        # Compute entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + eps))

        # Check convergence
        entropy_diff = entropy - target_entropy
        if torch.abs(entropy_diff) < tol:
            break

        # Binary search update
        if entropy_diff > 0:
            # Entropy too high -> decrease sigma
            sigma_max = sigma
        else:
            # Entropy too low -> increase sigma
            sigma_min = sigma

        sigma = (sigma_min + sigma_max) / 2.0

    return sigma


def compute_perplexity_affinities(
    data: Tensor,
    perplexity: float = 30.0,
    verbose: bool = False,
) -> Tensor:
    """
    Compute symmetric perplexity-based affinities for t-SNE.

    For each point i, computes conditional probabilities p_j|i using a Gaussian
    kernel with bandwidth sigma_i chosen to achieve the target perplexity.
    The affinities are then symmetrized: V_ij = (p_j|i + p_i|j) / (2*n).

    Parameters
    ----------
    data : Tensor, shape (n_points, n_features)
        Input data points
    perplexity : float
        Target perplexity (effective number of neighbors). Typical values: 5-50.
    verbose : bool
        Print progress information

    Returns
    -------
    Tensor, shape (n_points, n_points)
        Symmetric affinity matrix V where V_ij represents the joint probability
        of points i and j being neighbors. Sum of V equals 1.

    Notes
    -----
    The algorithm uses k-NN to limit computation where k = min(n-1, 3*perplexity+1).
    This is standard practice in t-SNE implementations for efficiency.
    """
    n_points = data.shape[0]
    device = data.device

    # Limit neighbors for efficiency
    k = min(n_points - 1, int(3 * perplexity + 1))

    if verbose:
        print(f"Computing affinities with perplexity={perplexity}, using k={k} neighbors")

    # Compute all pairwise squared distances
    # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    squared_norms = (data**2).sum(dim=1, keepdim=True)
    squared_distances = squared_norms + squared_norms.t() - 2 * torch.mm(data, data.t())
    squared_distances = torch.clamp(squared_distances, min=0.0)

    # Set diagonal to large value to exclude self
    squared_distances.fill_diagonal_(float("inf"))

    # Find k nearest neighbors for each point
    _, indices = torch.topk(squared_distances, k, dim=1, largest=False)

    # Compute conditional probabilities for each point
    P_conditional = torch.zeros(n_points, n_points, device=device)

    for i in range(n_points):
        # Get distances to k nearest neighbors
        neighbor_indices = indices[i]
        neighbor_sq_distances = squared_distances[i, neighbor_indices]

        # Binary search for optimal sigma
        sigma = _binary_search_sigma(neighbor_sq_distances, perplexity)

        # Compute conditional probabilities
        neg_sq_dist_scaled = -neighbor_sq_distances / (2.0 * sigma * sigma)
        neg_sq_dist_scaled = neg_sq_dist_scaled - neg_sq_dist_scaled.max()
        exp_vals = torch.exp(neg_sq_dist_scaled)
        probs = exp_vals / exp_vals.sum()

        # Store in conditional probability matrix
        P_conditional[i, neighbor_indices] = probs

    # Symmetrize: V_ij = (p_j|i + p_i|j) / (2*n)
    V = (P_conditional + P_conditional.t()) / (2.0 * n_points)

    # Ensure V sums to 1 (it should already, but normalize for numerical stability)
    V = V / V.sum()

    if verbose:
        print(f"Affinity matrix computed, sum={V.sum().item():.6f}")

    return V
