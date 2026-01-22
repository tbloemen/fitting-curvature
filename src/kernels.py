"""
Kernel functions for t-SNE embedding.

This module provides the Student-t distribution kernel used in the
low-dimensional space of t-SNE.

References:
- van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
"""

import torch
from torch import Tensor

from src.manifolds import Manifold


def t_distribution_kernel(distances: Tensor, dof: float = 1.0) -> Tensor:
    """
    Compute Student-t distribution kernel values.

    The kernel is: k(d) = (1 + d^2/dof)^(-(dof+1)/2)

    For t-SNE, dof=1 gives the standard Cauchy kernel:
    k(d) = (1 + d^2)^(-1) = 1 / (1 + d^2)

    Parameters
    ----------
    distances : Tensor
        Geodesic distances in the embedding space
    dof : float
        Degrees of freedom for the t-distribution. Default is 1 (Cauchy).

    Returns
    -------
    Tensor
        Kernel values, same shape as distances
    """
    return torch.pow(1.0 + distances**2 / dof, -(dof + 1) / 2)


def compute_q_matrix(manifold: Manifold, points: Tensor, dof: float = 1.0) -> Tensor:
    """
    Compute normalized Q matrix for t-SNE using geodesic distances.

    Q_ij represents the probability that points i and j are neighbors in
    the low-dimensional space, based on the t-distribution kernel applied
    to geodesic distances on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The manifold (Euclidean, Sphere, or Hyperboloid) for distance computation
    points : Tensor, shape (n_points, ambient_dim)
        Points in the embedding space
    dof : float
        Degrees of freedom for the t-distribution kernel

    Returns
    -------
    Tensor, shape (n_points, n_points)
        Normalized probability matrix Q where Q_ij = kernel(d_ij) / sum_k!=l kernel(d_kl)
        Diagonal is set to 0.
    """
    # Compute pairwise geodesic distances on the manifold
    distances = manifold.pairwise_distances(points)

    # Apply t-distribution kernel
    kernel_values = t_distribution_kernel(distances, dof)

    # Set diagonal to 0 (self-similarity not counted)
    kernel_values.fill_diagonal_(0.0)

    # Normalize to get probabilities
    total = kernel_values.sum()
    if total == 0:
        total = torch.tensor(1e-10, device=points.device)

    Q = kernel_values / total

    return Q
