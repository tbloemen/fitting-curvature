"""Pair sampling strategies for batched training on large datasets."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from src.matrices import compute_euclidean_distances_batched


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
    def precompute(self, data: Tensor) -> None:
        """
        Precompute any necessary data structures from raw data.

        Parameters
        ----------
        data : Tensor, shape (n_points, D)
            Raw data points (NOT distance matrix)
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

    def precompute(self, data: Tensor) -> None:
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

    def precompute(self, data: Tensor) -> None:
        """
        Precompute k-nearest neighbor graph from raw data.

        Parameters
        ----------
        data : Tensor, shape (n_points, D)
            Raw data points
        """
        # Use chunked k-NN computation to avoid full distance matrix
        self.knn_indices = _compute_knn_chunked(data, k=self.k)

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

    For memory efficiency with large datasets, this uses sampling-based estimation
    of distance percentiles rather than computing all pairwise distances.
    """

    def __init__(
        self,
        n_points: int,
        batch_size: int,
        device: torch.device,
        n_bins: int = 10,
        close_weight: float = 3.0,
        n_percentile_samples: int = 100000,
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
        n_percentile_samples : int
            Number of pairs to sample for estimating percentiles (default: 100000)
        """
        super().__init__(n_points, batch_size, device)
        self.n_bins = n_bins
        self.close_weight = close_weight
        self.n_percentile_samples = n_percentile_samples
        self.bin_edges = None
        self.bin_probs = None
        self._data = None  # Store reference to data for on-the-fly distance computation

    def precompute(self, data: Tensor) -> None:
        """
        Precompute bin edges from sampled distances.

        Parameters
        ----------
        data : Tensor, shape (n_points, D)
            Raw data points
        """
        from src.matrices import compute_euclidean_distances_batched

        self._data = data

        # Sample random pairs to estimate distance percentiles
        n_samples = min(
            self.n_percentile_samples, self.n_points * (self.n_points - 1) // 2
        )

        indices_i = torch.randint(0, self.n_points, (n_samples,), device=self.device)
        indices_j = torch.randint(0, self.n_points, (n_samples,), device=self.device)

        # Ensure i != j
        mask = indices_i == indices_j
        while mask.any():
            indices_j[mask] = torch.randint(
                0, self.n_points, (int(mask.sum()),), device=self.device
            )
            mask = indices_i == indices_j

        # Compute distances for sampled pairs
        distances = compute_euclidean_distances_batched(data, indices_i, indices_j)

        # Compute percentile-based bin edges
        percentiles = torch.linspace(0, 100, self.n_bins + 1, device=self.device)
        self.bin_edges = torch.quantile(distances, percentiles / 100.0)

        # Create bin sampling probabilities: higher weight for close pairs
        weights = torch.linspace(
            self.close_weight, 1.0, self.n_bins, device=self.device
        )
        self.bin_probs = weights / weights.sum()

    def sample_pairs(self) -> Tuple[Tensor, Tensor]:
        """
        Sample pairs with stratification by distance.

        Uses rejection sampling to sample pairs that fall into each distance bin.

        Returns
        -------
        indices_i : Tensor, shape (batch_size,)
            First point indices
        indices_j : Tensor, shape (batch_size,)
            Second point indices
        """

        if self.bin_edges is None or self.bin_probs is None:
            raise RuntimeError("Must call precompute() before sampling")

        # Sample which bins to draw from
        bin_indices = torch.multinomial(
            self.bin_probs, self.batch_size, replacement=True
        )

        # Initialize output tensors
        indices_i = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        indices_j = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # Oversample random pairs and assign to bins
        oversample_factor = 4  # Sample more pairs than needed
        n_oversample = self.batch_size * oversample_factor

        cand_i = torch.randint(0, self.n_points, (n_oversample,), device=self.device)
        cand_j = torch.randint(0, self.n_points, (n_oversample,), device=self.device)

        # Ensure i != j
        mask = cand_i == cand_j
        while mask.any():
            cand_j[mask] = torch.randint(
                0, self.n_points, (int(mask.sum()),), device=self.device
            )
            mask = cand_i == cand_j

        # Compute distances for candidates
        assert (
            self._data is not None
        ), "Data is not initialized in sampler, unable to calculate distances"
        cand_distances = compute_euclidean_distances_batched(self._data, cand_i, cand_j)

        # Assign candidates to bins
        cand_bins = torch.zeros(n_oversample, dtype=torch.long, device=self.device)
        for bin_idx in range(self.n_bins):
            if bin_idx == self.n_bins - 1:
                # Last bin includes upper edge
                bin_mask = (cand_distances >= self.bin_edges[bin_idx]) & (
                    cand_distances <= self.bin_edges[bin_idx + 1]
                )
            else:
                bin_mask = (cand_distances >= self.bin_edges[bin_idx]) & (
                    cand_distances < self.bin_edges[bin_idx + 1]
                )
            cand_bins[bin_mask] = bin_idx

        # Fill output from each bin
        filled = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        for bin_idx in range(self.n_bins):
            # Positions that need pairs from this bin
            need_mask = (bin_indices == bin_idx) & ~filled
            n_need = need_mask.sum().item()

            if n_need == 0:
                continue

            # Candidates in this bin
            have_mask = cand_bins == bin_idx
            have_indices = torch.where(have_mask)[0]
            n_have = len(have_indices)

            if n_have == 0:
                # Fall back to random sampling if no candidates in bin
                indices_i[need_mask] = torch.randint(
                    0, self.n_points, (int(n_need),), device=self.device
                )
                indices_j[need_mask] = torch.randint(
                    0, self.n_points, (int(n_need),), device=self.device
                )
                continue

            # Take min(n_need, n_have) from candidates
            n_take = min(int(n_need), n_have)
            take_idx = have_indices[:n_take]

            # Find positions to fill
            need_positions = torch.where(need_mask)[0][:n_take]

            indices_i[need_positions] = cand_i[take_idx]
            indices_j[need_positions] = cand_j[take_idx]
            filled[need_positions] = True

        # Fill any remaining with random pairs
        unfilled = ~filled
        if unfilled.any():
            n_unfilled = unfilled.sum().item()
            indices_i[unfilled] = torch.randint(
                0, self.n_points, (int(n_unfilled),), device=self.device
            )
            indices_j[unfilled] = torch.randint(
                0, self.n_points, (int(n_unfilled),), device=self.device
            )

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

    def precompute(self, data: Tensor) -> None:
        """
        Precompute k-nearest neighbor graph from raw data.

        Parameters
        ----------
        data : Tensor, shape (n_points, D)
            Raw data points
        """
        # Use chunked k-NN computation to avoid full distance matrix
        self.knn_indices = _compute_knn_chunked(data, k=self.k)

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


def _compute_knn_chunked(X: Tensor, k: int, chunk_size: int = 1000) -> Tensor:
    """
    Compute k-nearest neighbor indices using chunked computation.

    This avoids storing the full N×N distance matrix by processing
    the data in chunks.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points
    k : int
        Number of nearest neighbors
    chunk_size : int
        Number of query points to process at once

    Returns
    -------
    Tensor, shape (N, k)
        Indices of k-nearest neighbors for each point
    """
    n_points = X.shape[0]
    device = X.device

    # Precompute squared norms for all points
    squared_norms = (X**2).sum(dim=1)  # (N,)

    # Initialize output tensor
    knn_indices = torch.zeros((n_points, k), dtype=torch.long, device=device)

    # Process in chunks
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        chunk = X[start:end]  # (chunk_size, D)
        chunk_norms = squared_norms[start:end]  # (chunk_size,)

        # Compute squared distances from chunk to all points
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        dot_products = torch.mm(chunk, X.t())  # (chunk_size, N)
        squared_distances = (
            chunk_norms.unsqueeze(1) + squared_norms.unsqueeze(0) - 2 * dot_products
        )

        # Set self-distances to infinity
        for i, idx in enumerate(range(start, end)):
            squared_distances[i, idx] = float("inf")

        # Get k smallest (nearest neighbors)
        _, indices = torch.topk(squared_distances, k=k, dim=1, largest=False)
        knn_indices[start:end] = indices

    return knn_indices
