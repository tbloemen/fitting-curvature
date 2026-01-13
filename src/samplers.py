"""Pair sampling strategies for batched training on large datasets."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class PairSampler(ABC):
    """Abstract base class for pair sampling strategies."""

    def __init__(self, n_points: int, batch_size: int, device: torch.device):
        """
        Initialize pair sampler.

        Parameters
        ----------
        n_points : int
            Number of points in the dataset
        batch_size : int
            Number of pairs to sample per iteration
        device : torch.device
            Device to place tensors on
        """
        self.n_points = n_points
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def precompute(self, distance_matrix: Tensor) -> None:
        """
        Precompute any necessary data structures.

        Parameters
        ----------
        distance_matrix : Tensor, shape (n_points, n_points)
            Pairwise distance matrix for the dataset
        """
        pass

    @abstractmethod
    def sample_pairs(self) -> Tuple[Tensor, Tensor]:
        """
        Sample a batch of point pairs.

        Returns
        -------
        indices_i : Tensor, shape (batch_size,)
            First point indices
        indices_j : Tensor, shape (batch_size,)
            Second point indices
        """
        pass


class RandomSampler(PairSampler):
    """Uniformly random pair sampling.

    Samples random pairs (i, j) where i ≠ j. No precomputation required.
    Simple baseline that provides unbiased gradient estimates.
    """

    def precompute(self, distance_matrix: Tensor) -> None:
        """No precomputation needed for random sampling."""
        pass

    def sample_pairs(self) -> Tuple[Tensor, Tensor]:
        """
        Sample random pairs.

        Returns
        -------
        indices_i : Tensor, shape (batch_size,)
            First point indices
        indices_j : Tensor, shape (batch_size,)
            Second point indices (i ≠ j guaranteed)
        """
        # Sample first indices
        indices_i = torch.randint(
            0, self.n_points, (self.batch_size,), device=self.device
        )

        # Sample second indices
        indices_j = torch.randint(
            0, self.n_points, (self.batch_size,), device=self.device
        )

        # Ensure i != j by resampling collisions
        mask = indices_i == indices_j
        while mask.any():
            indices_j[mask] = torch.randint(
                0, self.n_points, (int(mask.sum()),), device=self.device
            )
            mask = indices_i == indices_j

        return indices_i, indices_j


class KNNSampler(PairSampler):
    """K-nearest neighbor pair sampling.

    Preserves local structure by sampling pairs from k-nearest neighbor graphs.
    Focuses optimization on local geometry which is important for embeddings.
    """

    def __init__(
        self, n_points: int, batch_size: int, device: torch.device, k: int = 15
    ):
        """
        Initialize KNN sampler.

        Parameters
        ----------
        n_points : int
            Number of points
        batch_size : int
            Batch size
        device : torch.device
            Device for tensors
        k : int
            Number of nearest neighbors to consider (default: 15)
        """
        super().__init__(n_points, batch_size, device)
        self.k = k
        self.knn_indices = None

    def precompute(self, distance_matrix: Tensor) -> None:
        """
        Precompute k-nearest neighbor graph.

        Parameters
        ----------
        distance_matrix : Tensor, shape (n_points, n_points)
            Pairwise distances
        """
        # Create a copy to avoid modifying original
        dist_no_diag = distance_matrix.clone()

        # Set diagonal to infinity to exclude self-pairs
        dist_no_diag.fill_diagonal_(float("inf"))

        # Get k nearest neighbors for each point
        # torch.topk with largest=False gives k smallest distances
        _, self.knn_indices = torch.topk(dist_no_diag, k=self.k, dim=1, largest=False)

    def sample_pairs(self) -> Tuple[Tensor, Tensor]:
        """
        Sample pairs from k-NN neighborhoods.

        Returns
        -------
        indices_i : Tensor, shape (batch_size,)
            First point indices (randomly chosen)
        indices_j : Tensor, shape (batch_size,)
            Second point indices (k-NN neighbors of indices_i)
        """
        if self.knn_indices is None:
            raise RuntimeError("Must call precompute() before sampling")

        # Sample random points as anchors
        indices_i = torch.randint(
            0, self.n_points, (self.batch_size,), device=self.device
        )

        # For each anchor, sample a random neighbor from its k-NN
        neighbor_positions = torch.randint(
            0, self.k, (self.batch_size,), device=self.device
        )
        indices_j = self.knn_indices[indices_i, neighbor_positions]

        return indices_i, indices_j


class StratifiedSampler(PairSampler):
    """Distance-stratified pair sampling.

    Groups pairs by distance percentiles and samples with bias toward closer pairs.
    Balances local and global structure preservation.
    """

    def __init__(
        self,
        n_points: int,
        batch_size: int,
        device: torch.device,
        n_bins: int = 10,
        close_weight: float = 3.0,
    ):
        """
        Initialize stratified sampler.

        Parameters
        ----------
        n_points : int
            Number of points
        batch_size : int
            Batch size
        device : torch.device
            Device for tensors
        n_bins : int
            Number of distance bins (default: 10)
        close_weight : float
            Relative weight for sampling from close-pair bins (default: 3.0)
            Higher values bias toward close pairs
        """
        super().__init__(n_points, batch_size, device)
        self.n_bins = n_bins
        self.close_weight = close_weight
        self.distance_bins = None
        self.bin_probs = None

    def precompute(self, distance_matrix: Tensor) -> None:
        """
        Precompute distance bins and bin probabilities.

        Parameters
        ----------
        distance_matrix : Tensor, shape (n_points, n_points)
            Pairwise distances
        """
        # Extract upper triangular distances (avoid duplicates and diagonal)
        triu_indices = torch.triu_indices(
            self.n_points, self.n_points, offset=1, device=self.device
        )
        distances = distance_matrix[triu_indices[0], triu_indices[1]]

        # Compute percentile-based bin edges
        percentiles = torch.linspace(0, 100, self.n_bins + 1, device=self.device)
        bin_edges = torch.quantile(distances, percentiles / 100.0)

        # Assign pairs to distance bins
        self.distance_bins = []
        for i in range(self.n_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            # Include upper edge in last bin
            if i == self.n_bins - 1:
                mask |= distances == bin_edges[i + 1]

            # Store (i_indices, j_indices) for this bin
            bin_i = triu_indices[0][mask]
            bin_j = triu_indices[1][mask]
            self.distance_bins.append((bin_i, bin_j))

        # Create bin sampling probabilities: higher weight for close pairs
        weights = torch.linspace(
            self.close_weight, 1.0, self.n_bins, device=self.device
        )
        self.bin_probs = weights / weights.sum()

    def sample_pairs(self) -> Tuple[Tensor, Tensor]:
        """
        Sample pairs with stratification by distance.

        Returns
        -------
        indices_i : Tensor, shape (batch_size,)
            First point indices
        indices_j : Tensor, shape (batch_size,)
            Second point indices
        """
        if self.distance_bins is None or self.bin_probs is None:
            raise RuntimeError("Must call precompute() before sampling")

        # Sample which bins to draw from
        bin_indices = torch.multinomial(
            self.bin_probs, self.batch_size, replacement=True
        )

        # Initialize output tensors
        indices_i = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        indices_j = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # Sample from each selected bin
        for bin_idx in range(self.n_bins):
            mask = bin_indices == bin_idx
            n_from_bin = mask.sum().item()

            if n_from_bin == 0:
                continue

            bin_i, bin_j = self.distance_bins[bin_idx]
            bin_size = len(bin_i)

            if bin_size == 0:
                continue

            # Sample random pair indices from this bin
            pair_indices = torch.randint(
                0, bin_size, (int(n_from_bin),), device=self.device
            )
            indices_i[mask] = bin_i[pair_indices]
            indices_j[mask] = bin_j[pair_indices]

        return indices_i, indices_j


class NegativeSampler(PairSampler):
    """Contrastive pair sampling with positive and negative pairs.

    Samples a mix of:
    - Positive pairs: from k-nearest neighbors (similar points)
    - Negative pairs: random pairs (dissimilar points)

    Inspired by contrastive learning methods.
    """

    def __init__(
        self,
        n_points: int,
        batch_size: int,
        device: torch.device,
        k: int = 15,
        positive_ratio: float = 0.7,
    ):
        """
        Initialize negative sampler.

        Parameters
        ----------
        n_points : int
            Number of points
        batch_size : int
            Batch size
        device : torch.device
            Device for tensors
        k : int
            Number of neighbors for positive pairs (default: 15)
        positive_ratio : float
            Fraction of batch from k-NN (positive) pairs (default: 0.7)
        """
        super().__init__(n_points, batch_size, device)
        self.k = k
        self.positive_ratio = positive_ratio
        self.knn_indices = None

    def precompute(self, distance_matrix: Tensor) -> None:
        """
        Precompute k-nearest neighbor graph.

        Parameters
        ----------
        distance_matrix : Tensor, shape (n_points, n_points)
            Pairwise distances
        """
        # Create a copy to avoid modifying original
        dist_no_diag = distance_matrix.clone()

        # Set diagonal to infinity to exclude self-pairs
        dist_no_diag.fill_diagonal_(float("inf"))

        # Get k nearest neighbors for each point
        _, self.knn_indices = torch.topk(dist_no_diag, k=self.k, dim=1, largest=False)

    def sample_pairs(self) -> Tuple[Tensor, Tensor]:
        """
        Sample pairs with positive (k-NN) and negative (random) components.

        Returns
        -------
        indices_i : Tensor, shape (batch_size,)
            First point indices
        indices_j : Tensor, shape (batch_size,)
            Second point indices (mix of k-NN and random)
        """
        if self.knn_indices is None:
            raise RuntimeError("Must call precompute() before sampling")

        # Determine number of positive and negative pairs
        n_positive = int(self.batch_size * self.positive_ratio)
        n_negative = self.batch_size - n_positive

        # Sample positive pairs (from k-NN)
        pos_i = torch.randint(0, self.n_points, (n_positive,), device=self.device)
        neighbor_positions = torch.randint(0, self.k, (n_positive,), device=self.device)
        pos_j = self.knn_indices[pos_i, neighbor_positions]

        # Sample negative pairs (random)
        neg_i = torch.randint(0, self.n_points, (n_negative,), device=self.device)
        neg_j = torch.randint(0, self.n_points, (n_negative,), device=self.device)

        # Ensure neg_i ≠ neg_j
        mask = neg_i == neg_j
        while mask.any():
            neg_j[mask] = torch.randint(
                0, self.n_points, (int(mask.sum()),), device=self.device
            )
            mask = neg_i == neg_j

        # Concatenate positive and negative pairs
        indices_i = torch.cat([pos_i, neg_i])
        indices_j = torch.cat([pos_j, neg_j])

        return indices_i, indices_j


def create_sampler(
    sampler_type: str,
    n_points: int,
    batch_size: int,
    device: torch.device,
    **kwargs,
) -> PairSampler:
    """
    Factory function to create pair samplers.

    Parameters
    ----------
    sampler_type : str
        Type of sampler: 'random', 'knn', 'stratified', 'negative'
    n_points : int
        Number of points in dataset
    batch_size : int
        Number of pairs to sample per iteration
    device : torch.device
        Device to create tensors on
    **kwargs
        Additional sampler-specific arguments:
        - k (int): number of neighbors for KNN/Negative samplers
        - n_bins (int): number of bins for Stratified sampler
        - close_weight (float): weight for close pairs in Stratified sampler
        - positive_ratio (float): ratio of positive pairs in Negative sampler

    Returns
    -------
    PairSampler
        Configured sampler instance

    Raises
    ------
    ValueError
        If sampler_type is not recognized
    """
    sampler_type = sampler_type.lower()

    if sampler_type == "random":
        return RandomSampler(n_points, batch_size, device)
    elif sampler_type == "knn":
        k = kwargs.get("k", 15)
        return KNNSampler(n_points, batch_size, device, k=k)
    elif sampler_type == "stratified":
        n_bins = kwargs.get("n_bins", 10)
        close_weight = kwargs.get("close_weight", 3.0)
        return StratifiedSampler(
            n_points, batch_size, device, n_bins=n_bins, close_weight=close_weight
        )
    elif sampler_type == "negative":
        k = kwargs.get("k", 15)
        positive_ratio = kwargs.get("positive_ratio", 0.7)
        return NegativeSampler(
            n_points, batch_size, device, k=k, positive_ratio=positive_ratio
        )
    else:
        raise ValueError(
            f"Unknown sampler_type: {sampler_type}. "
            "Use 'random', 'knn', 'stratified', or 'negative'."
        )
