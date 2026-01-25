"""Tests for initialization methods (random and PCA)."""

import pytest
import torch

from src.embedding import ConstantCurvatureEmbedding
from src.types import InitMethod


def test_random_initialization_euclidean():
    """Test random initialization in Euclidean space."""
    n_points = 100
    embed_dim = 2
    curvature = 0.0
    init_scale = 0.01
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Check shape
    assert model.points.shape == (n_points, embed_dim)


def test_random_initialization_spherical():
    """Test random initialization on sphere."""
    n_points = 100
    embed_dim = 2
    curvature = 1.0
    init_scale = 0.01
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Check shape (ambient dim = embed_dim + 1)
    assert model.points.shape == (n_points, embed_dim + 1)


def test_random_initialization_hyperbolic():
    """Test random initialization on hyperboloid."""
    n_points = 100
    embed_dim = 2
    curvature = -1.0
    init_scale = 0.01
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Check shape (ambient dim = embed_dim + 1)
    assert model.points.shape == (n_points, embed_dim + 1)


def test_pca_initialization_euclidean():
    """Test PCA initialization in Euclidean space."""
    n_points = 100
    embed_dim = 2
    data_dim = 10
    curvature = 0.0
    init_scale = 0.01
    device = torch.device("cpu")

    # Create synthetic data
    data = torch.randn(n_points, data_dim, device=device)

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.PCA,
        data=data,
    )

    # Check shape
    assert model.points.shape == (n_points, embed_dim)


def test_pca_initialization_spherical():
    """Test PCA initialization on sphere."""
    n_points = 100
    embed_dim = 2
    data_dim = 10
    curvature = 1.0
    init_scale = 0.01
    device = torch.device("cpu")

    # Create synthetic data
    data = torch.randn(n_points, data_dim, device=device)

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.PCA,
        data=data,
    )

    # Check shape (ambient dim = embed_dim + 1)
    assert model.points.shape == (n_points, embed_dim + 1)


def test_pca_initialization_hyperbolic():
    """Test PCA initialization on hyperboloid."""
    n_points = 100
    embed_dim = 2
    data_dim = 10
    curvature = -1.0
    init_scale = 0.01
    device = torch.device("cpu")

    # Create synthetic data
    data = torch.randn(n_points, data_dim, device=device)

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.PCA,
        data=data,
    )

    # Check shape (ambient dim = embed_dim + 1)
    assert model.points.shape == (n_points, embed_dim + 1)


def test_pca_requires_data():
    """Test that PCA initialization raises error without data."""
    n_points = 100
    embed_dim = 2
    curvature = 0.0
    init_scale = 0.01
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Data needs to be supplied for PCA"):
        ConstantCurvatureEmbedding(
            n_points=n_points,
            embed_dim=embed_dim,
            curvature=curvature,
            init_scale=init_scale,
            device=device,
            init_method=InitMethod.PCA,
            data=None,
        )


def test_pca_vs_random_different():
    """Test that PCA and random initialization produce different results."""
    n_points = 50
    embed_dim = 2
    data_dim = 10
    curvature = 0.0
    init_scale = 0.01
    device = torch.device("cpu")

    # Create synthetic data
    torch.manual_seed(42)
    data = torch.randn(n_points, data_dim, device=device)

    # PCA initialization
    torch.manual_seed(42)
    model_pca = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.PCA,
        data=data,
    )

    # Random initialization
    torch.manual_seed(42)
    model_random = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Points should be different
    assert not torch.allclose(model_pca.points, model_random.points)


def test_pca_initialization_scaling():
    """Test that PCA initialization respects init_scale parameter."""
    n_points = 100
    embed_dim = 2
    data_dim = 10
    curvature = 0.0
    init_scale = 0.01
    device = torch.device("cpu")

    # Create synthetic data with large variance
    data = torch.randn(n_points, data_dim, device=device) * 100

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.PCA,
        data=data,
    )

    # Points should be rescaled to have std ≈ init_scale
    # Allow some tolerance due to PCA projection
    points_std = model.points.std().item()
    assert abs(points_std - init_scale) < 0.02, f"Expected std ≈ {init_scale}, got {points_std}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pca_initialization_gpu():
    """Test PCA initialization on GPU."""
    n_points = 100
    embed_dim = 2
    data_dim = 10
    curvature = 0.0
    init_scale = 0.01
    device = torch.device("cuda")

    # Create synthetic data on GPU
    data = torch.randn(n_points, data_dim, device=device)

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=InitMethod.PCA,
        data=data,
    )

    # Check shape and device
    assert model.points.shape == (n_points, embed_dim)
    assert model.points.device.type == "cuda"
