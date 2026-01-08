"""
Tests for Riemannian SGD implementation.

Following:
- Gu et al. (2019): "Learning Mixed-Curvature Representations in Products of Model Spaces"
- Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"
"""

import pytest
import torch
from src.embedding import fit_embedding
from src.matrices import calculate_distance_matrix, get_init_scale


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    torch.manual_seed(42)
    n_samples = 50
    dim = 5
    X = torch.randn(n_samples, dim)
    distance_matrix = calculate_distance_matrix(X)
    return distance_matrix


def test_rsgd_basic_convergence(synthetic_dataset):
    """Test that RSGD optimizer converges without producing NaN values."""
    distance_matrix = synthetic_dataset
    embed_dim = 2

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=embed_dim,
        curvature=-1.0,
        init_scale=0.001,  # Paper uses very small initialization
        n_iterations=50,
        lr=0.0001,  # Small learning rate for RSGD in curved spaces
        verbose=False,
    )

    # Check that embeddings are valid (not NaN)
    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any(), "RSGD produced NaN embeddings"

    # Check that loss is finite
    with torch.no_grad():
        distances = model().cpu()
        loss = torch.sum((distances - distance_matrix.cpu()) ** 2).item()

    assert torch.isfinite(torch.tensor(loss)), "RSGD produced infinite loss"


def test_rsgd_convergence_quality(synthetic_dataset):
    """Test RSGD convergence quality on different geometries."""
    distance_matrix = synthetic_dataset
    embed_dim = 2

    # Test hyperbolic - use smaller lr for curved space
    model_hyp = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=embed_dim,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,  # Smaller lr for hyperbolic to avoid exponential explosion
        verbose=False,
    )

    # Test Euclidean - can use larger lr
    model_euc = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=embed_dim,
        curvature=0.0,
        init_scale=0.1,
        n_iterations=100,
        lr=0.01,
        verbose=False,
    )

    # Test spherical - use smaller lr for curved space
    model_sph = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=embed_dim,
        curvature=1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,  # Smaller lr for spherical to avoid constraint violations
        verbose=False,
    )

    # Check losses are reasonable for all geometries
    with torch.no_grad():
        distance_matrix_cpu = distance_matrix.cpu()

        hyp_loss = torch.sum((model_hyp().cpu() - distance_matrix_cpu) ** 2).item()
        euc_loss = torch.sum((model_euc().cpu() - distance_matrix_cpu) ** 2).item()
        sph_loss = torch.sum((model_sph().cpu() - distance_matrix_cpu) ** 2).item()

    # All should produce finite, reasonable losses
    assert hyp_loss < 10000, f"Hyperbolic loss too high: {hyp_loss}"
    assert euc_loss < 10000, f"Euclidean loss too high: {euc_loss}"
    assert sph_loss < 10000, f"Spherical loss too high: {sph_loss}"

    print(f"\nRSGD performance across geometries:")
    print(f"  Hyperbolic loss: {hyp_loss:.2f}")
    print(f"  Euclidean loss: {euc_loss:.2f}")
    print(f"  Spherical loss: {sph_loss:.2f}")


def test_rsgd_hyperboloid_constraint(synthetic_dataset):
    """Test that RSGD maintains the hyperboloid constraint."""
    distance_matrix = synthetic_dataset
    embed_dim = 2

    model = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=embed_dim,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=50,
        lr=0.0001,  # Small lr to avoid exponential explosion in hyperbolic space
        verbose=False,
    )

    # Get full embeddings (including time coordinate)
    embeddings = model.get_embeddings()

    # Check hyperboloid constraint: -x0^2 + ||x_spatial||^2 = -1
    # (assuming radius = 1)
    x0 = embeddings[:, 0]
    spatial = embeddings[:, 1:]

    constraint = -(x0**2) + (spatial**2).sum(dim=1)
    expected = torch.ones_like(constraint) * (-1.0)

    # Allow small numerical error
    assert torch.allclose(constraint, expected, atol=1e-5), (
        "Hyperboloid constraint violated"
    )


def test_rsgd_spherical_constraint():
    """Test that RSGD maintains the spherical constraint."""
    torch.manual_seed(42)
    X = torch.randn(20, 3)
    distance_matrix = calculate_distance_matrix(X)

    # RSGD should work for spherical geometry
    model_spherical = fit_embedding(
        distance_matrix=distance_matrix,
        embed_dim=2,
        curvature=1.0,
        init_scale=0.001,
        n_iterations=10,
        lr=0.0001,  # Small lr for spherical space to maintain constraint
        verbose=False,
    )

    # Get embeddings and check they lie on the sphere
    embeddings = model_spherical.get_embeddings()
    assert not torch.isnan(embeddings).any()

    # Check sphere constraint: ||x||^2 = 1 (radius = 1 for curvature k=1)
    norms = (embeddings**2).sum(dim=1)
    expected = torch.ones_like(norms)

    # Allow small numerical error
    assert torch.allclose(norms, expected, atol=1e-5), "Spherical constraint violated"


if __name__ == "__main__":
    # Allow running as a script for debugging
    pytest.main([__file__, "-v", "-s"])
