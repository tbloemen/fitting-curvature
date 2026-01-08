"""Test gradient flow for constant curvature embeddings."""

import torch
import pytest
from src.embedding import ConstantCurvatureEmbedding


@pytest.fixture
def hyperbolic_model():
    """Create a small hyperbolic embedding model for testing."""
    n_points = 10
    embed_dim = 2
    curvature = -1.0
    return ConstantCurvatureEmbedding(n_points, embed_dim, curvature, init_scale=0.1)


def test_hyperbolic_model_initialization(hyperbolic_model):
    """Test that hyperbolic model initializes correctly."""
    assert hyperbolic_model.points.shape == (10, 3)
    assert hyperbolic_model.points.requires_grad is True
    assert hyperbolic_model.curvature == -1.0
    assert hyperbolic_model.embed_dim == 2


def test_project_to_manifold_shape(hyperbolic_model):
    """Test that get_embeddings produces correct shape."""
    points = hyperbolic_model.get_embeddings()
    # Hyperbolic uses hyperboloid model in R^(d+1)
    assert points.shape == (10, 3)


def test_pairwise_distances_shape(hyperbolic_model):
    """Test that pairwise distances have correct shape."""
    distances = hyperbolic_model()
    assert distances.shape == (10, 10)


def test_gradient_flow_hyperbolic(hyperbolic_model):
    """Test that gradients flow correctly through hyperbolic embedding."""
    # Forward pass
    distances = hyperbolic_model()

    # Compute loss
    loss = distances.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist and are non-zero
    assert hyperbolic_model.points.grad is not None
    grad_norm = hyperbolic_model.points.grad.norm().item()
    assert grad_norm > 0, "Gradients should be non-zero"
    assert not torch.isnan(hyperbolic_model.points.grad).any(), (
        "Gradients should not contain NaN"
    )


def test_distances_non_negative(hyperbolic_model):
    """Test that all distances are non-negative."""
    distances = hyperbolic_model()
    assert (distances >= 0).all(), "All distances should be non-negative"


def test_distances_symmetric(hyperbolic_model):
    """Test that distance matrix is symmetric."""
    distances = hyperbolic_model()
    assert torch.allclose(distances, distances.t(), atol=1e-5), (
        "Distance matrix should be symmetric"
    )


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
def test_different_curvatures(curvature):
    """Test gradient flow works for all curvature types."""
    n_points = 5
    embed_dim = 2
    model = ConstantCurvatureEmbedding(n_points, embed_dim, curvature, init_scale=0.1)

    # Forward pass
    distances = model()
    loss = distances.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert model.points.grad is not None
    assert model.points.grad.norm().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_flow_on_gpu():
    """Test that gradients flow correctly on GPU."""
    n_points = 10
    embed_dim = 2
    curvature = -1.0
    model = ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, init_scale=0.1, device="cuda"
    )

    assert model.points.device.type == "cuda"

    # Forward pass
    distances = model()
    loss = distances.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist and are on GPU
    assert model.points.grad is not None
    assert model.points.grad.device.type == "cuda"
    assert model.points.grad.norm().item() > 0
