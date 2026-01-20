"""Tests for pair sampling strategies in batched training.

Note: Samplers now take raw data instead of distance matrices.
This allows for on-the-fly distance computation in large datasets.
"""

import pytest
import torch
from conftest import calculate_distance_matrix

from src.samplers import (KNNSampler, NegativeSampler, RandomSampler,
                          SamplerType, StratifiedSampler, create_sampler)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    torch.manual_seed(42)
    n_points = 100
    X = torch.randn(n_points, 10)
    # Keep distance_matrix for tests that verify KNN correctness
    distance_matrix = calculate_distance_matrix(X)
    return n_points, X, distance_matrix


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_output_shape(self, sample_data):
        """Test RandomSampler returns correct shape."""
        n_points, _, _ = sample_data
        batch_size = 32
        sampler = RandomSampler(n_points, batch_size, torch.device("cpu"))
        indices_i, indices_j = sampler.sample_pairs()

        assert indices_i.shape == (batch_size,)
        assert indices_j.shape == (batch_size,)

    def test_no_self_pairs(self, sample_data):
        """Test RandomSampler doesn't sample (i, i) pairs."""
        n_points, _, _ = sample_data
        batch_size = 1000
        sampler = RandomSampler(n_points, batch_size, torch.device("cpu"))
        indices_i, indices_j = sampler.sample_pairs()

        assert (indices_i != indices_j).all()

    def test_valid_indices(self, sample_data):
        """Test RandomSampler returns valid indices."""
        n_points, _, _ = sample_data
        batch_size = 100
        sampler = RandomSampler(n_points, batch_size, torch.device("cpu"))
        indices_i, indices_j = sampler.sample_pairs()

        assert (indices_i >= 0).all() and (indices_i < n_points).all()
        assert (indices_j >= 0).all() and (indices_j < n_points).all()

    def test_no_precompute_needed(self, sample_data):
        """Test RandomSampler doesn't need precompute."""
        n_points, _, _ = sample_data
        batch_size = 32
        sampler = RandomSampler(n_points, batch_size, torch.device("cpu"))
        # Should work without calling precompute
        indices_i, _ = sampler.sample_pairs()
        assert indices_i.shape == (batch_size,)


class TestKNNSampler:
    """Tests for KNNSampler."""

    def test_output_shape(self, sample_data):
        """Test KNNSampler returns correct shape."""
        n_points, X, _ = sample_data
        batch_size = 32
        sampler = KNNSampler(n_points, batch_size, torch.device("cpu"), k=5)
        sampler.precompute(X)  # Now takes raw data
        indices_i, indices_j = sampler.sample_pairs()

        assert indices_i.shape == (batch_size,)
        assert indices_j.shape == (batch_size,)

    def test_knn_graph_shape(self, sample_data):
        """Test KNNSampler precomputes correct shape."""
        n_points, X, _ = sample_data
        k = 10
        sampler = KNNSampler(n_points, 32, torch.device("cpu"), k=k)
        sampler.precompute(X)

        assert sampler.knn_indices is not None
        assert sampler.knn_indices.shape == (n_points, k)

    def test_knn_graph_correctness(self, sample_data):
        """Test KNNSampler precomputes correct k-NN graph."""
        n_points, X, distance_matrix = sample_data
        k = 5
        sampler = KNNSampler(n_points, 32, torch.device("cpu"), k=k)
        sampler.precompute(X)
        assert sampler.knn_indices is not None

        # Verify first point's k-NN against full distance matrix
        dist_no_diag = distance_matrix.clone()
        dist_no_diag.fill_diagonal_(float("inf"))
        true_knn = torch.argsort(dist_no_diag[0])[:k]

        computed_knn = sampler.knn_indices[0].sort()[0]
        true_knn_sorted = true_knn.sort()[0]

        assert torch.allclose(computed_knn.float(), true_knn_sorted.float(), atol=1e-5)

    def test_samples_from_knn(self, sample_data):
        """Test KNNSampler samples only from k-NN."""
        n_points, X, _ = sample_data
        k = 5
        batch_size = 100
        sampler = KNNSampler(n_points, batch_size, torch.device("cpu"), k=k)
        sampler.precompute(X)
        assert sampler.knn_indices is not None

        indices_i, indices_j = sampler.sample_pairs()

        # For each sample, verify j is in k-NN of i
        for i_idx, j_idx in zip(indices_i, indices_j):
            assert j_idx in sampler.knn_indices[i_idx]

    def test_requires_precompute(self, sample_data):
        """Test KNNSampler requires precompute."""
        n_points, _, _ = sample_data
        sampler = KNNSampler(n_points, 32, torch.device("cpu"), k=5)

        with pytest.raises(RuntimeError):
            sampler.sample_pairs()


class TestStratifiedSampler:
    """Tests for StratifiedSampler."""

    def test_output_shape(self, sample_data):
        """Test StratifiedSampler returns correct shape."""
        n_points, X, _ = sample_data
        batch_size = 32
        sampler = StratifiedSampler(n_points, batch_size, torch.device("cpu"), n_bins=5)
        sampler.precompute(X)  # Now takes raw data
        indices_i, indices_j = sampler.sample_pairs()

        assert indices_i.shape == (batch_size,)
        assert indices_j.shape == (batch_size,)

    def test_bins_created(self, sample_data):
        """Test StratifiedSampler creates bin edges and probabilities."""
        n_points, X, _ = sample_data
        n_bins = 5
        sampler = StratifiedSampler(n_points, 32, torch.device("cpu"), n_bins=n_bins)
        sampler.precompute(X)
        assert sampler.bin_edges is not None
        assert sampler.bin_probs is not None

        # Check bin_edges has n_bins + 1 elements (fence posts)
        assert sampler.bin_edges.shape == (n_bins + 1,)
        assert sampler.bin_probs.shape == (n_bins,)
        assert torch.allclose(sampler.bin_probs.sum(), torch.tensor(1.0))

    def test_bin_probabilities(self, sample_data):
        """Test bin probabilities are properly weighted."""
        n_points, X, _ = sample_data
        n_bins = 5
        close_weight = 3.0
        sampler = StratifiedSampler(
            n_points, 32, torch.device("cpu"), n_bins=n_bins, close_weight=close_weight
        )
        sampler.precompute(X)
        assert sampler.bin_probs is not None

        # Verify probabilities are higher for close pairs (first bin)
        assert sampler.bin_probs[0] > sampler.bin_probs[-1]

    def test_requires_precompute(self, sample_data):
        """Test StratifiedSampler requires precompute."""
        n_points, _, _ = sample_data
        sampler = StratifiedSampler(n_points, 32, torch.device("cpu"), n_bins=5)

        with pytest.raises(RuntimeError):
            sampler.sample_pairs()


class TestNegativeSampler:
    """Tests for NegativeSampler."""

    def test_output_shape(self, sample_data):
        """Test NegativeSampler returns correct shape."""
        n_points, X, _ = sample_data
        batch_size = 100
        sampler = NegativeSampler(
            n_points, batch_size, torch.device("cpu"), k=5, positive_ratio=0.7
        )
        sampler.precompute(X)  # Now takes raw data
        indices_i, indices_j = sampler.sample_pairs()

        assert indices_i.shape == (batch_size,)
        assert indices_j.shape == (batch_size,)

    def test_positive_ratio(self, sample_data):
        """Test NegativeSampler respects positive/negative ratio."""
        n_points, X, _ = sample_data
        k = 5
        batch_size = 1000
        positive_ratio = 0.7
        sampler = NegativeSampler(
            n_points,
            batch_size,
            torch.device("cpu"),
            k=k,
            positive_ratio=positive_ratio,
        )
        sampler.precompute(X)
        assert sampler.knn_indices is not None

        indices_i, indices_j = sampler.sample_pairs()

        # Count positive pairs (those in k-NN)
        n_positive = int(batch_size * positive_ratio)
        positive_pairs = 0
        for i in range(n_positive):
            if indices_j[i] in sampler.knn_indices[indices_i[i]]:
                positive_pairs += 1

        # Should have close to expected number (allow some variance)
        assert positive_pairs >= n_positive * 0.9

    def test_no_self_pairs_negative(self, sample_data):
        """Test NegativeSampler doesn't sample self-pairs in negative samples."""
        n_points, X, _ = sample_data
        batch_size = 1000
        sampler = NegativeSampler(
            n_points,
            batch_size,
            torch.device("cpu"),
            k=5,
            positive_ratio=0.3,  # More negative pairs
        )
        sampler.precompute(X)

        indices_i, indices_j = sampler.sample_pairs()
        assert (indices_i != indices_j).all()

    def test_requires_precompute(self, sample_data):
        """Test NegativeSampler requires precompute."""
        n_points, _, _ = sample_data
        sampler = NegativeSampler(n_points, 32, torch.device("cpu"), k=5)

        with pytest.raises(RuntimeError):
            sampler.sample_pairs()


class TestCreateSampler:
    """Tests for sampler factory function."""

    def test_create_random(self, sample_data):
        """Test create_sampler creates RandomSampler."""
        n_points, _, _ = sample_data
        sampler = create_sampler(SamplerType.RANDOM, n_points, 32, torch.device("cpu"))
        assert isinstance(sampler, RandomSampler)

    def test_create_knn(self, sample_data):
        """Test create_sampler creates KNNSampler."""
        n_points, _, _ = sample_data
        sampler = create_sampler(SamplerType.KNN, n_points, 32, torch.device("cpu"))
        assert isinstance(sampler, KNNSampler)

    def test_create_stratified(self, sample_data):
        """Test create_sampler creates StratifiedSampler."""
        n_points, _, _ = sample_data
        sampler = create_sampler(
            SamplerType.STRATIFIED, n_points, 32, torch.device("cpu")
        )
        assert isinstance(sampler, StratifiedSampler)

    def test_create_negative(self, sample_data):
        """Test create_sampler creates NegativeSampler."""
        n_points, _, _ = sample_data
        sampler = create_sampler(
            SamplerType.NEGATIVE, n_points, 32, torch.device("cpu")
        )
        assert isinstance(sampler, NegativeSampler)

    def test_create_with_kwargs(self, sample_data):
        """Test create_sampler passes kwargs correctly."""
        n_points, _, _ = sample_data
        k = 20
        sampler = create_sampler(
            SamplerType.KNN, n_points, 32, torch.device("cpu"), k=k
        )
        assert isinstance(sampler, KNNSampler)
        assert sampler.k == k


class TestSamplerConsistency:
    """Tests for sampler consistency and correctness."""

    def test_all_samplers_convergence(self):
        """Test all sampler types can be used for training."""
        torch.manual_seed(42)
        n_points = 50
        X = torch.randn(n_points, 10)

        sampler_types = [
            SamplerType.RANDOM,
            SamplerType.KNN,
            SamplerType.STRATIFIED,
            SamplerType.NEGATIVE,
        ]
        device = torch.device("cpu")

        for sampler_type in sampler_types:
            sampler = create_sampler(sampler_type, n_points, 128, device)
            sampler.precompute(X)  # Now takes raw data

            # Sample a few batches without error
            for _ in range(10):
                indices_i, indices_j = sampler.sample_pairs()
                assert indices_i.shape == (128,)
                assert indices_j.shape == (128,)

    def test_sampler_on_gpu(self):
        """Test samplers work on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        n_points = 50

        sampler = RandomSampler(n_points, 32, torch.device("cuda"))
        indices_i, indices_j = sampler.sample_pairs()

        assert indices_i.device.type == "cuda"
        assert indices_j.device.type == "cuda"
