import torch
from torch import Tensor


def similarity_metric(v1: Tensor, v2: Tensor) -> float:
    """
    Compute Euclidean distance for use in constructing target distance matrix.
    This is the distance in the original high-dimensional space.

    Parameters
    ----------
    v1, v2 : Tensor
        Input vectors in high-dimensional space

    Returns
    -------
    float
        Euclidean distance between v1 and v2
    """
    return torch.norm(v1 - v2).item()


def calculate_distance_matrix(X: Tensor) -> Tensor:
    N = len(X)
    A = torch.zeros((N, N))
    for i, v1 in enumerate(X):
        for j, v2 in enumerate(X):
            if i == j:
                continue
            similarity = similarity_metric(v1, v2)
            A[i, j] = similarity

    return A
