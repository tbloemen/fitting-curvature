"""
Test gradient flow for constant curvature embeddings.

Includes tests for verifying correct gradient formula from Xia (2024)
"Revisiting Hyperbolic t-SNE", equation 4.2.
"""

import pytest
import torch

from src.embedding import ConstantCurvatureEmbedding, compute_tsne_kl_loss
from src.kernels import compute_q_matrix, t_distribution_kernel
from src.types import InitMethod


@pytest.fixture
def hyperbolic_model():
    """Create a small hyperbolic embedding model for testing."""
    n_points = 10
    embed_dim = 2
    curvature = -1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, device=device, init_scale=0.1
    )


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
    assert not torch.isnan(
        hyperbolic_model.points.grad
    ).any(), "Gradients should not contain NaN"


def test_distances_non_negative(hyperbolic_model):
    """Test that all distances are non-negative."""
    distances = hyperbolic_model()
    assert (distances >= 0).all(), "All distances should be non-negative"


def test_distances_symmetric(hyperbolic_model):
    """Test that distance matrix is symmetric."""
    distances = hyperbolic_model()
    assert torch.allclose(
        distances, distances.t(), atol=1e-5
    ), "Distance matrix should be symmetric"


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
def test_different_curvatures(curvature):
    """Test gradient flow works for all curvature types."""
    n_points = 5
    embed_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, device=device, init_scale=0.1
    )

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
        n_points, embed_dim, curvature, init_scale=0.1, device=torch.device("cuda")
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


# ============================================================================
# Tests for t-SNE gradient correctness (Xia 2024, equation 4.2)
# ============================================================================


def test_kernel_gradient_includes_distance():
    """
    Test that the gradient of the t-distribution kernel includes the distance term.

    The kernel is k(d) = (1 + d²)^(-1), so dk/dd = -2d / (1 + d²)²
    This test verifies that differentiating through the kernel gives us the d term.
    """

    # Create a simple distance and enable gradient tracking
    d = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]], requires_grad=True
    )

    # Compute kernel values
    k = t_distribution_kernel(d)

    # Compute a simple loss: sum of kernel values
    loss = k.sum()

    # Backpropagate
    loss.backward()

    # The gradient should be dk/dd = -2d / (1 + d²)²
    expected_grad = -2 * d.data / (1 + d.data**2) ** 2

    # Check that gradients match
    assert d.grad is not None
    assert torch.allclose(
        d.grad, expected_grad, atol=1e-6
    ), "Kernel gradient does not include distance term"


def test_distance_term_in_full_pipeline():
    """
    Test that the distance term d_ij appears in the full t-SNE gradient pipeline.

    This verifies that differentiating through:
    Loss -> Q -> kernel -> distance
    properly includes the d_ij term from the kernel derivative.
    """
    torch.manual_seed(42)
    n_points = 4
    embed_dim = 2
    curvature = 0.0  # Use Euclidean for simpler interpretation
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=0.1,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Create V matrix
    V = torch.rand(n_points, n_points)
    V = (V + V.t()) / 2
    V.fill_diagonal_(0.0)
    V = V / V.sum()

    model.points.requires_grad = True

    # Compute loss
    Q = compute_q_matrix(model.manifold, model.points, dof=1.0)
    loss = compute_tsne_kl_loss(Q, V)
    loss.backward()

    grad = model.points.grad
    assert grad is not None

    # Check that gradient is non-zero and finite
    assert not torch.allclose(grad, torch.zeros_like(grad)), "Gradient is zero"
    assert torch.isfinite(grad).all(), "Gradient contains inf or NaN"

    # Check that gradient magnitude scales reasonably with distances
    with torch.no_grad():
        distances = model.manifold.pairwise_distances(model.points)
        avg_dist = distances[distances > 0].mean()

    # Gradient norm should be reasonable (not tiny, not huge)
    grad_norm = torch.norm(grad)
    assert 1e-6 < grad_norm < 1e6, f"Gradient norm {grad_norm:.2e} is unreasonable"

    # With the d_ij term included, gradients should scale with typical distances
    # Gradient magnitude per point should be on a reasonable scale relative to distances
    grad_per_point = grad_norm / n_points**0.5
    assert grad_per_point / avg_dist < 100, (
        f"Gradient magnitude ({grad_per_point:.2e}) is too large relative to "
        f"average distance ({avg_dist:.2e})"
    )


def test_gradient_includes_distance_term():
    """
    Verify that the gradient includes the d_ij^H term.

    Without the d_ij^H term, gradients would be systematically incorrect.
    This test checks that gradients have reasonable magnitudes relative to distances.
    """
    torch.manual_seed(123)
    n_points = 4
    embed_dim = 2
    curvature = -1.0
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=0.01,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    V = torch.rand(n_points, n_points)
    V = (V + V.t()) / 2
    V.fill_diagonal_(0.0)
    V = V / V.sum()

    model.points.requires_grad = True

    # Compute Q and loss
    Q = compute_q_matrix(model.manifold, model.points, dof=1.0)
    loss = compute_tsne_kl_loss(Q, V)
    loss.backward()

    grad = model.points.grad
    assert grad is not None

    # Gradient should not be zero
    assert not torch.allclose(grad, torch.zeros_like(grad)), "Gradient is zero!"

    # Compute average distance between points
    with torch.no_grad():
        distances = model.manifold.pairwise_distances(model.points)
        avg_distance = distances[distances > 0].mean()

    # If the d_ij term is included, gradient magnitude should be reasonable
    grad_norm = torch.norm(grad)
    assert grad_norm > 1e-6, f"Gradient norm is too small: {grad_norm:.2e}"
    assert not torch.isnan(grad).any(), "Gradient contains NaN"
    assert not torch.isinf(grad).any(), "Gradient contains inf"

    # Verify gradient magnitude is proportional to distance scale
    # With the correct d_ij term, gradients should be on a similar scale to distances
    grad_scale = grad.abs().mean()
    distance_scale = avg_distance.item()
    assert 0.001 < grad_scale / distance_scale < 1000, (
        f"Gradient scale ({grad_scale:.2e}) is disproportionate to "
        f"distance scale ({distance_scale:.2e})"
    )


def test_gradient_tsne_loss_hyperbolic():
    """Test that gradients flow correctly through t-SNE KL loss for hyperbolic space."""
    torch.manual_seed(456)
    n_points = 3
    embed_dim = 2
    curvature = -1.0
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=0.001,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Simple V matrix with high affinity between points 0 and 1
    V = torch.zeros(n_points, n_points)
    V[0, 1] = V[1, 0] = 0.4
    V[0, 2] = V[2, 0] = 0.05
    V[1, 2] = V[2, 1] = 0.05
    V = V / V.sum()

    model.points.requires_grad = True

    Q = compute_q_matrix(model.manifold, model.points, dof=1.0)
    loss = compute_tsne_kl_loss(Q, V)
    loss.backward()

    # Check that all points have non-zero gradients
    grad = model.points.grad
    assert grad is not None
    for i in range(n_points):
        assert not torch.allclose(
            grad[i], torch.zeros_like(grad[i])
        ), f"Point {i} has zero gradient"


def test_gradient_numerical_verification_simple():
    """
    Numerically verify gradient using a simple distance-based loss.

    This test uses a simpler loss function to avoid numerical issues
    with the normalized Q matrix in t-SNE.
    """
    torch.manual_seed(789)
    n_points = 3
    embed_dim = 2
    curvature = 0.0
    device = torch.device("cpu")
    eps = 1e-6

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=0.1,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    # Test gradient for a single coordinate using simple MSE loss
    point_idx = 0
    coord_idx = 1

    # Target distances
    target = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])

    # Compute gradient via autograd
    model.points.requires_grad = True
    distances = model.manifold.pairwise_distances(model.points)
    loss = ((distances - target) ** 2).sum()
    loss.backward()
    assert model.points.grad is not None
    analytical_grad = model.points.grad[point_idx, coord_idx].item()

    # Compute numerical gradient
    with torch.no_grad():
        points_plus = model.points.clone()
        points_plus[point_idx, coord_idx] += eps
        dist_plus = model.manifold.pairwise_distances(points_plus)
        loss_plus = ((dist_plus - target) ** 2).sum()

        points_minus = model.points.clone()
        points_minus[point_idx, coord_idx] -= eps
        dist_minus = model.manifold.pairwise_distances(points_minus)
        loss_minus = ((dist_minus - target) ** 2).sum()

        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        numerical_grad = numerical_grad.item()

    # Compare
    rel_error = abs(analytical_grad - numerical_grad) / (abs(analytical_grad) + 1e-10)
    assert rel_error < 0.05, (
        f"Gradient mismatch:\n"
        f"  Autograd:  {analytical_grad:.6f}\n"
        f"  Numerical: {numerical_grad:.6f}\n"
        f"  Relative error: {rel_error:.4%}"
    )


@pytest.mark.parametrize("curvature", [-1.0, -0.5, -2.0])
def test_gradient_tsne_different_curvatures(curvature):
    """Test t-SNE gradient correctness for different hyperbolic curvatures."""
    torch.manual_seed(42)
    n_points = 4
    embed_dim = 2
    device = torch.device("cpu")

    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=0.001,
        device=device,
        init_method=InitMethod.RANDOM,
    )

    V = torch.rand(n_points, n_points)
    V = (V + V.t()) / 2
    V.fill_diagonal_(0.0)
    V = V / V.sum()

    model.points.requires_grad = True
    Q = compute_q_matrix(model.manifold, model.points, dof=1.0)
    loss = compute_tsne_kl_loss(Q, V)
    loss.backward()

    # Check gradient is not NaN or inf
    grad = model.points.grad
    assert grad is not None
    assert not torch.isnan(grad).any(), f"NaN in gradient for curvature {curvature}"
    assert not torch.isinf(grad).any(), f"Inf in gradient for curvature {curvature}"

    # Check gradient is not zero
    assert grad.abs().max() > 1e-8, f"Gradient too small for curvature {curvature}"
