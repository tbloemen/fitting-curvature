"""
Integration test for manifold-based architecture.

Tests the full embedding workflow with the new manifold-based architecture,
similar to main.py but without verbose tqdm output.
"""

import torch

from src.embedding import fit_embedding
from src.matrices import calculate_distance_matrix


def _create_test_data(n_samples: int = 100) -> torch.Tensor:
    """Create synthetic test data."""
    return torch.randn(n_samples, 10)


def test_integration_hyperbolic():
    """Test full embedding workflow in hyperbolic space."""
    # Create synthetic test data
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    # Fit embedding in hyperbolic space
    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=-1.0,
        init_scale=0.1,
        n_iterations=50,
        lr=0.0001,
        verbose=False,  # No progress bar
        loss_type="gu2019",
    )

    # Verify output
    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert embeddings.shape[1] == 3  # Ambient dimension for hyperbolic
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()

    # Verify distances
    distances = model()
    assert distances.shape == (100, 100)
    assert not torch.isnan(distances).any()
    assert not torch.isinf(distances).any()


def test_integration_euclidean():
    """Test full embedding workflow in Euclidean space."""
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=0.0,
        init_scale=0.1,
        n_iterations=50,
        lr=0.001,
        verbose=False,
        loss_type="gu2019",
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert embeddings.shape[1] == 2  # Ambient dimension for Euclidean
    assert not torch.isnan(embeddings).any()

    distances = model()
    assert distances.shape == (100, 100)
    assert not torch.isnan(distances).any()


def test_integration_spherical():
    """Test full embedding workflow in spherical space."""
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=1.0,
        init_scale=0.1,
        n_iterations=50,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert embeddings.shape[1] == 3  # Ambient dimension for spherical
    assert not torch.isnan(embeddings).any()

    distances = model()
    assert distances.shape == (100, 100)
    assert not torch.isnan(distances).any()


def test_integration_mse_loss():
    """Test embedding with MSE loss function."""
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=-1.0,
        init_scale=0.1,
        n_iterations=50,
        lr=0.0001,
        verbose=False,
        loss_type="mse",
    )

    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any()

    distances = model()
    assert not torch.isnan(distances).any()


def test_integration_multiple_curvatures():
    """Test embedding workflow across different curvatures."""
    X = _create_test_data(50)
    distance_matrix = calculate_distance_matrix(X)

    curvatures = [-1.0, 0.0, 1.0]
    for k in curvatures:
        model = fit_embedding(
            distance_matrix=distance_matrix,
            embed_dim=2,
            curvature=k,
            init_scale=0.1,
            n_iterations=30,
            lr=0.0001,
            verbose=False,
            loss_type="gu2019",
        )

        embeddings = model.get_embeddings()
        distances = model()

        assert not torch.isnan(embeddings).any(), f"NaN in embeddings for k={k}"
        assert not torch.isnan(distances).any(), f"NaN in distances for k={k}"


# Tests for batched training with different samplers


def test_batched_training_random_sampler():
    """Test batched training with random sampler."""
    X = _create_test_data(200)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=-1.0,
        init_scale=0.1,
        n_iterations=50,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
        sampler_type="random",
        batch_size=512,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 200
    assert not torch.isnan(embeddings).any()


def test_batched_training_knn_sampler():
    """Test batched training with KNN sampler."""
    X = _create_test_data(200)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=0.0,
        init_scale=0.1,
        n_iterations=50,
        lr=0.001,
        verbose=False,
        loss_type="gu2019",
        sampler_type="knn",
        batch_size=256,
        sampler_kwargs={"k": 10},
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 200
    assert not torch.isnan(embeddings).any()


def test_batched_training_stratified_sampler():
    """Test batched training with stratified sampler."""
    X = _create_test_data(150)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=1.0,
        init_scale=0.1,
        n_iterations=40,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
        sampler_type="stratified",
        batch_size=256,
        sampler_kwargs={"n_bins": 5, "close_weight": 2.0},
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 150
    assert not torch.isnan(embeddings).any()


def test_batched_training_negative_sampler():
    """Test batched training with negative sampler."""
    X = _create_test_data(150)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=-1.0,
        init_scale=0.1,
        n_iterations=40,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
        sampler_type="negative",
        batch_size=256,
        sampler_kwargs={"k": 10, "positive_ratio": 0.7},
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 150
    assert not torch.isnan(embeddings).any()


def test_batched_training_all_samplers():
    """Test all sampler types converge without errors."""
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    sampler_types = ["random", "knn", "stratified", "negative"]

    for sampler_type in sampler_types:
        model = fit_embedding(
            distance_matrix=distance_matrix,
            embed_dim=2,
            curvature=0.0,
            init_scale=0.1,
            n_iterations=30,
            lr=0.001,
            verbose=False,
            loss_type="gu2019",
            sampler_type=sampler_type,
            batch_size=128,
        )

        embeddings = model.get_embeddings()
        assert not torch.isnan(embeddings).any(), f"NaN for sampler {sampler_type}"


def test_batched_training_large_dataset():
    """Test batched training on a larger dataset."""
    X = _create_test_data(500)
    distance_matrix = calculate_distance_matrix(X)

    # Use smaller number of iterations for faster test
    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=-1.0,
        init_scale=0.1,
        n_iterations=20,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
        sampler_type="knn",
        batch_size=1024,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 500
    assert not torch.isnan(embeddings).any()


def test_batched_vs_default_sampler_type():
    """Test batched training works with default sampler parameters."""
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    # Use default sampler_type (should be "random")
    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=0.0,
        init_scale=0.1,
        n_iterations=30,
        lr=0.001,
        verbose=False,
        loss_type="gu2019",
        batch_size=256,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert not torch.isnan(embeddings).any()


def test_batched_training_mse_loss():
    """Test batched training with MSE loss."""
    X = _create_test_data(100)
    distance_matrix = calculate_distance_matrix(X)

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=-1.0,
        init_scale=0.1,
        n_iterations=30,
        lr=0.0001,
        verbose=False,
        loss_type="mse",
        sampler_type="knn",
        batch_size=256,
    )

    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any()
