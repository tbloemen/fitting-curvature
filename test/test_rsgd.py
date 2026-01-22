"""
Tests for Riemannian SGD implementation with t-SNE.

Tests verify that the optimizer produces valid embeddings that maintain
manifold constraints for hyperbolic, Euclidean, and spherical geometries.
"""

import pytest
import torch

from src.embedding import fit_embedding
from src.types import InitMethod


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    torch.manual_seed(42)
    n_samples = 50
    dim = 5
    X = torch.randn(n_samples, dim)
    return X


def test_rsgd_basic_convergence(synthetic_dataset):
    """Test that RSGD optimizer converges without producing NaN values."""
    X = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        perplexity=10.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    # Check that embeddings are valid (not NaN)
    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any(), "RSGD produced NaN embeddings"


def test_rsgd_convergence_quality(synthetic_dataset):
    """Test RSGD convergence quality on different geometries."""
    X = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test hyperbolic
    model_hyp = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        perplexity=10.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    # Test Euclidean
    model_euc = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=0.0,
        perplexity=10.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=100.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.1,
        verbose=False,
    )

    # Test spherical
    model_sph = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=1.0,
        perplexity=10.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    # All should produce valid embeddings
    assert not torch.isnan(model_hyp.get_embeddings()).any()
    assert not torch.isnan(model_euc.get_embeddings()).any()
    assert not torch.isnan(model_sph.get_embeddings()).any()


def test_rsgd_hyperboloid_constraint(synthetic_dataset):
    """Test that RSGD maintains the hyperboloid constraint."""
    X = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        perplexity=10.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    # Get full embeddings (including time coordinate)
    embeddings = model.get_embeddings()

    # Check hyperboloid constraint: -x0^2 + ||x_spatial||^2 = -1
    x0 = embeddings[:, 0]
    spatial = embeddings[:, 1:]

    constraint = -(x0**2) + (spatial**2).sum(dim=1)
    expected = torch.ones_like(constraint) * (-1.0)

    # Allow small numerical error
    assert torch.allclose(
        constraint, expected, atol=1e-3
    ), "Hyperboloid constraint violated"


def test_rsgd_spherical_constraint():
    """Test that RSGD maintains the spherical constraint."""
    torch.manual_seed(42)
    X = torch.randn(30, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_spherical = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=1.0,
        perplexity=10.0,
        n_iterations=30,
        early_exaggeration_iterations=10,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    # Get embeddings and check they lie on the sphere
    embeddings = model_spherical.get_embeddings()
    assert not torch.isnan(embeddings).any()

    # Check sphere constraint: ||x||^2 = 1 (radius = 1 for curvature k=1)
    norms = (embeddings**2).sum(dim=1)
    expected = torch.ones_like(norms)

    # Allow small numerical error
    assert torch.allclose(norms, expected, atol=1e-4), "Spherical constraint violated"


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
def test_all_curvatures(curvature, synthetic_dataset):
    """Test embedding works for all curvature types."""
    X = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        device=device,
        embed_dim=embed_dim,
        curvature=curvature,
        perplexity=10.0,
        n_iterations=30,
        early_exaggeration_iterations=10,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    # Check that embeddings are valid
    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any()

    # Check manifold constraints are maintained
    if curvature > 0:
        # Sphere: ||x||^2 = 1/k (radius^2)
        expected_norm_sq = 1.0 / curvature
        actual_norm_sq = (embeddings**2).sum(dim=1)
        assert torch.allclose(
            actual_norm_sq, torch.tensor(expected_norm_sq), atol=1e-4
        )
    elif curvature < 0:
        # Hyperboloid: -x0^2 + ||x_spatial||^2 = -radius^2 = -1/|k|
        x0 = embeddings[:, 0]
        spatial = embeddings[:, 1:]
        constraint = -(x0**2) + (spatial**2).sum(dim=1)
        expected = torch.ones_like(constraint) * (-1.0 / abs(curvature))
        assert torch.allclose(constraint, expected, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
