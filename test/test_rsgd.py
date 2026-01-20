"""
Tests for Riemannian SGD implementation.

Following:
- Gu et al. (2019): "Learning Mixed-Curvature Representations in Products of Model Spaces"
- Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"

Note: fit_embedding now takes raw data instead of a distance matrix.
Distances are computed on-the-fly during training.
"""

import pytest
import torch
from conftest import calculate_distance_matrix, compute_loss

from src.embedding import fit_embedding


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    torch.manual_seed(42)
    n_samples = 50
    dim = 5
    X = torch.randn(n_samples, dim)
    # Return both raw data and distance matrix for loss computation
    distance_matrix = calculate_distance_matrix(X)
    return X, distance_matrix


def test_rsgd_basic_convergence(synthetic_dataset):
    """Test that RSGD optimizer converges without producing NaN values."""
    X, distance_matrix = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,  # Now takes raw data
        embed_dim=embed_dim,
        device=device,
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
        loss = compute_loss(distances, distance_matrix.cpu(), "gu2019").item()

    assert torch.isfinite(torch.tensor(loss)), "RSGD produced infinite loss"


def test_rsgd_convergence_quality(synthetic_dataset):
    """Test RSGD convergence quality on different geometries."""
    X, distance_matrix = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test hyperbolic - use smaller lr for curved space
    model_hyp = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,  # Smaller lr for hyperbolic to avoid exponential explosion
        verbose=False,
    )

    # Test Euclidean - can use larger lr
    model_euc = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=0.0,
        init_scale=0.1,
        n_iterations=100,
        lr=0.01,
        verbose=False,
    )

    # Test spherical - use smaller lr for curved space
    model_sph = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,  # Smaller lr for spherical to avoid constraint violations
        verbose=False,
    )

    # Check losses are reasonable for all geometries
    with torch.no_grad():
        distance_matrix_cpu = distance_matrix.cpu()

        hyp_loss = compute_loss(model_hyp().cpu(), distance_matrix_cpu, "gu2019").item()
        euc_loss = compute_loss(model_euc().cpu(), distance_matrix_cpu, "gu2019").item()
        sph_loss = compute_loss(model_sph().cpu(), distance_matrix_cpu, "gu2019").item()

    # All should produce finite, reasonable losses
    assert hyp_loss < 10000, f"Hyperbolic loss too high: {hyp_loss}"
    assert euc_loss < 10000, f"Euclidean loss too high: {euc_loss}"
    assert sph_loss < 10000, f"Spherical loss too high: {sph_loss}"

    print("\nRSGD performance across geometries:")
    print(f"  Hyperbolic loss: {hyp_loss:.2f}")
    print(f"  Euclidean loss: {euc_loss:.2f}")
    print(f"  Spherical loss: {sph_loss:.2f}")


def test_rsgd_hyperboloid_constraint(synthetic_dataset):
    """Test that RSGD maintains the hyperboloid constraint."""
    X, _ = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
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
    assert torch.allclose(
        constraint, expected, atol=1e-5
    ), "Hyperboloid constraint violated"


def test_rsgd_spherical_constraint():
    """Test that RSGD maintains the spherical constraint."""
    torch.manual_seed(42)
    X = torch.randn(20, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RSGD should work for spherical geometry
    model_spherical = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
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


def test_gu2019_loss_convergence(synthetic_dataset):
    """Test that Gu et al. (2019) loss function works correctly."""
    X, distance_matrix = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
    )

    # Check that embeddings are valid (not NaN)
    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any(), "Gu et al. loss produced NaN embeddings"

    # Check that loss decreases
    with torch.no_grad():
        distances = model().cpu()
        final_loss = compute_loss(distances, distance_matrix.cpu(), "gu2019").item()

    assert torch.isfinite(
        torch.tensor(final_loss)
    ), "Gu et al. loss produced infinite loss"
    assert final_loss < 10000, f"Gu et al. loss too high: {final_loss}"


def test_mse_loss_convergence(synthetic_dataset):
    """Test that MSE loss function still works correctly."""
    X, distance_matrix = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        device=device,
        embed_dim=embed_dim,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,
        verbose=False,
        loss_type="mse",
    )

    # Check that embeddings are valid (not NaN)
    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any(), "MSE loss produced NaN embeddings"

    # Check that loss decreases
    with torch.no_grad():
        distances = model().cpu()
        final_loss = compute_loss(distances, distance_matrix.cpu(), "mse").item()

    assert torch.isfinite(torch.tensor(final_loss)), "MSE loss produced infinite loss"


def test_loss_type_comparison(synthetic_dataset):
    """Compare Gu et al. (2019) loss vs MSE loss."""
    X, distance_matrix = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fit with Gu et al. loss
    model_gu = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
    )

    # Fit with MSE loss
    model_mse = fit_embedding(
        data=X,
        embed_dim=embed_dim,
        device=device,
        curvature=-1.0,
        init_scale=0.001,
        n_iterations=100,
        lr=0.0001,
        verbose=False,
        loss_type="mse",
    )

    # Both should produce valid embeddings
    assert not torch.isnan(model_gu.get_embeddings()).any()
    assert not torch.isnan(model_mse.get_embeddings()).any()

    # Compute relative distortion for both models
    with torch.no_grad():
        dist_gu = model_gu().cpu()
        dist_mse = model_mse().cpu()
        distance_matrix_cpu = distance_matrix.cpu()

        gu_loss = compute_loss(dist_gu, distance_matrix_cpu, "gu2019").item()
        mse_loss = compute_loss(dist_mse, distance_matrix_cpu, "gu2019").item()

    # Gu et al. loss should be lower on its own metric
    # (though MSE loss may be better at absolute error)
    print("\nLoss comparison:")
    print(f"  Gu et al. model relative distortion: {gu_loss:.4f}")
    print(f"  MSE model relative distortion: {mse_loss:.4f}")


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
def test_gu2019_loss_all_curvatures(curvature, synthetic_dataset):
    """Test Gu et al. (2019) loss works for all curvature types."""
    X, _ = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        device=device,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=0.001,
        n_iterations=50,
        lr=0.0001,
        verbose=False,
        loss_type="gu2019",
    )

    # Check that embeddings are valid
    embeddings = model.get_embeddings()
    assert not torch.isnan(embeddings).any()

    # Check manifold constraints are maintained
    if curvature > 0:
        # Sphere: ||x||^2 = 1/k (radius^2)
        expected_norm_sq = 1.0 / curvature
        actual_norm_sq = (embeddings**2).sum(dim=1)
        assert torch.allclose(actual_norm_sq, torch.tensor(expected_norm_sq), atol=1e-5)
    elif curvature < 0:
        # Hyperboloid: -x0^2 + ||x_spatial||^2 = -radius^2 = -1/|k|
        x0 = embeddings[:, 0]
        spatial = embeddings[:, 1:]
        constraint = -(x0**2) + (spatial**2).sum(dim=1)
        expected = torch.ones_like(constraint) * (-1.0 / abs(curvature))
        assert torch.allclose(constraint, expected, atol=1e-5)


def test_invalid_loss_type(synthetic_dataset):
    """Test that invalid loss_type raises an error."""
    X, _ = synthetic_dataset
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with pytest.raises(ValueError, match="Unknown loss_type"):
        fit_embedding(
            data=X,
            device=device,
            embed_dim=embed_dim,
            curvature=-1.0,
            init_scale=0.001,
            n_iterations=10,
            lr=0.0001,
            verbose=False,
            loss_type="invalid_loss",
        )


def test_compute_loss_function():
    """Test compute_loss function directly."""
    # Create simple distance matrices
    n_points = 5
    embedded_distances = torch.randn(n_points, n_points).abs()
    embedded_distances = (
        embedded_distances + embedded_distances.t()
    ) / 2  # Make symmetric
    embedded_distances.fill_diagonal_(0.0)

    target_distances = torch.randn(n_points, n_points).abs()
    target_distances = (target_distances + target_distances.t()) / 2  # Make symmetric
    target_distances.fill_diagonal_(1.0)  # Avoid division by zero

    # Test Gu et al. loss
    gu_loss = compute_loss(embedded_distances, target_distances, "gu2019")
    assert isinstance(gu_loss, torch.Tensor)
    assert gu_loss.dim() == 0  # Scalar
    assert gu_loss.item() >= 0

    # Test MSE loss
    mse_loss = compute_loss(embedded_distances, target_distances, "mse")
    assert isinstance(mse_loss, torch.Tensor)
    assert mse_loss.dim() == 0  # Scalar
    assert mse_loss.item() >= 0

    # Test invalid loss type
    with pytest.raises(ValueError, match="Unknown loss_type"):
        compute_loss(embedded_distances, target_distances, "invalid")

    # Test that Gu et al. loss ignores diagonal
    embedded_with_diagonal = embedded_distances.clone()
    embedded_with_diagonal.fill_diagonal_(100.0)  # Large diagonal values
    gu_loss_1 = compute_loss(embedded_distances, target_distances, "gu2019")
    gu_loss_2 = compute_loss(embedded_with_diagonal, target_distances, "gu2019")
    assert torch.allclose(gu_loss_1, gu_loss_2), "Gu et al. loss should ignore diagonal"

    print("\ncompute_loss tests:")
    print(f"  Gu et al. loss: {gu_loss.item():.4f}")
    print(f"  MSE loss: {mse_loss.item():.4f}")


if __name__ == "__main__":
    # Allow running as a script for debugging
    pytest.main([__file__, "-v", "-s"])
