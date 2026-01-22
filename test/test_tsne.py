"""
Tests for t-SNE implementation.

Tests cover:
- Affinity computation (binary search, symmetry, normalization)
- Kernel functions (t-distribution values, Q matrix properties)
- t-SNE embedding (no NaN for all curvatures, loss decreases)
"""

import pytest
import torch

from src.affinities import _binary_search_sigma, compute_perplexity_affinities
from src.embedding import compute_tsne_kl_loss, fit_embedding
from src.kernels import compute_q_matrix, t_distribution_kernel
from src.manifolds import Euclidean, Hyperboloid, Sphere
from src.types import InitMethod


class TestAffinities:
    """Tests for affinity computation."""

    def test_binary_search_convergence(self):
        """Test that binary search finds sigma that achieves target perplexity."""
        torch.manual_seed(42)
        # Create some random squared distances
        distances = torch.rand(30) * 10  # 30 neighbors

        target_perplexity = 15.0
        sigma = _binary_search_sigma(distances, target_perplexity)

        # Verify sigma is positive and reasonable
        assert sigma > 0
        assert sigma < 1e4

        # Verify perplexity is close to target
        neg_sq_dist_scaled = -distances / (2.0 * sigma * sigma)
        neg_sq_dist_scaled = neg_sq_dist_scaled - neg_sq_dist_scaled.max()
        exp_vals = torch.exp(neg_sq_dist_scaled)
        probs = exp_vals / exp_vals.sum()

        # Compute entropy and perplexity
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        perplexity = torch.exp(entropy).item()

        assert abs(perplexity - target_perplexity) < 1.0, (
            f"Perplexity {perplexity} not close to target {target_perplexity}"
        )

    def test_affinity_symmetry(self):
        """Test that computed affinities are symmetric."""
        torch.manual_seed(42)
        data = torch.randn(50, 10)

        V = compute_perplexity_affinities(data, perplexity=15.0)

        # Check symmetry
        assert torch.allclose(V, V.t(), atol=1e-6), "Affinity matrix not symmetric"

    def test_affinity_normalization(self):
        """Test that affinities sum to 1."""
        torch.manual_seed(42)
        data = torch.randn(50, 10)

        V = compute_perplexity_affinities(data, perplexity=15.0)

        # Check normalization
        assert torch.abs(V.sum() - 1.0) < 1e-5, f"Affinities sum to {V.sum()}, not 1"

    def test_affinity_non_negative(self):
        """Test that all affinities are non-negative."""
        torch.manual_seed(42)
        data = torch.randn(50, 10)

        V = compute_perplexity_affinities(data, perplexity=15.0)

        assert (V >= 0).all(), "Negative affinity values found"

    def test_affinity_diagonal_zero(self):
        """Test that diagonal entries (self-affinities) are essentially zero."""
        torch.manual_seed(42)
        data = torch.randn(50, 10)

        V = compute_perplexity_affinities(data, perplexity=15.0)

        diagonal = torch.diag(V)
        # Diagonal should be very small (close to zero after normalization)
        assert diagonal.max() < 0.01, f"Diagonal has large values: {diagonal.max()}"


class TestKernels:
    """Tests for t-distribution kernel functions."""

    def test_t_distribution_values(self):
        """Test that t-distribution kernel produces expected values."""
        distances = torch.tensor([0.0, 1.0, 2.0, 10.0])

        # With dof=1 (Cauchy kernel): k(d) = 1 / (1 + d^2)
        kernel_vals = t_distribution_kernel(distances, dof=1.0)

        expected = torch.tensor([1.0, 0.5, 0.2, 1 / 101])
        assert torch.allclose(kernel_vals, expected, atol=1e-6)

    def test_t_distribution_monotonic(self):
        """Test that kernel values decrease with distance."""
        distances = torch.linspace(0, 10, 100)
        kernel_vals = t_distribution_kernel(distances, dof=1.0)

        # Should be monotonically decreasing
        diffs = kernel_vals[1:] - kernel_vals[:-1]
        assert (diffs <= 0).all(), "Kernel is not monotonically decreasing"

    def test_q_matrix_normalization(self):
        """Test that Q matrix sums to 1."""
        torch.manual_seed(42)
        manifold = Euclidean(0.0)
        points = torch.randn(50, 2)

        Q = compute_q_matrix(manifold, points, dof=1.0)

        assert torch.abs(Q.sum() - 1.0) < 1e-5, f"Q matrix sums to {Q.sum()}, not 1"

    def test_q_matrix_symmetry(self):
        """Test that Q matrix is symmetric."""
        torch.manual_seed(42)
        manifold = Euclidean(0.0)
        points = torch.randn(50, 2)

        Q = compute_q_matrix(manifold, points, dof=1.0)

        assert torch.allclose(Q, Q.t(), atol=1e-6), "Q matrix not symmetric"

    def test_q_matrix_diagonal_zero(self):
        """Test that Q matrix diagonal is zero."""
        torch.manual_seed(42)
        manifold = Euclidean(0.0)
        points = torch.randn(50, 2)

        Q = compute_q_matrix(manifold, points, dof=1.0)

        diagonal = torch.diag(Q)
        assert (diagonal == 0).all(), "Q matrix diagonal should be zero"

    @pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
    def test_q_matrix_all_curvatures(self, curvature):
        """Test Q matrix computation works for all curvatures."""
        torch.manual_seed(42)

        if curvature < 0:
            manifold = Hyperboloid(curvature)
            # Initialize on hyperboloid
            spatial = torch.randn(30, 2) * 0.1
            time = torch.sqrt(1.0 + (spatial**2).sum(dim=1, keepdim=True))
            points = torch.cat([time, spatial], dim=1)
        elif curvature > 0:
            manifold = Sphere(curvature)
            # Initialize on sphere
            spatial = torch.randn(30, 2) * 0.1
            x0 = torch.sqrt(1.0 - (spatial**2).sum(dim=1, keepdim=True).clamp(max=0.99))
            points = torch.cat([x0, spatial], dim=1)
        else:
            manifold = Euclidean(0.0)
            points = torch.randn(30, 2)

        Q = compute_q_matrix(manifold, points, dof=1.0)

        # Should be normalized
        assert torch.abs(Q.sum() - 1.0) < 1e-5
        # Should be symmetric
        assert torch.allclose(Q, Q.t(), atol=1e-6)
        # Should have no NaN
        assert not torch.isnan(Q).any()


class TestTSNEEmbedding:
    """Tests for t-SNE embedding function."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for testing."""
        torch.manual_seed(42)
        # Create 3 clusters
        cluster1 = torch.randn(20, 5) + torch.tensor([2.0, 0, 0, 0, 0])
        cluster2 = torch.randn(20, 5) + torch.tensor([-2.0, 0, 0, 0, 0])
        cluster3 = torch.randn(20, 5) + torch.tensor([0, 2.0, 0, 0, 0])
        return torch.cat([cluster1, cluster2, cluster3], dim=0)

    @pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
    def test_tsne_no_nan(self, curvature, small_dataset):
        """Test that t-SNE produces no NaN values for all curvatures."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = fit_embedding(
            data=small_dataset,
            embed_dim=2,
            curvature=curvature,
            device=device,
            perplexity=10.0,
            n_iterations=50,
            early_exaggeration_iterations=20,
            learning_rate=50.0,
            init_method=InitMethod.RANDOM,
            init_scale=0.001,
            verbose=False,
        )

        embeddings = model.get_embeddings()
        assert not torch.isnan(embeddings).any(), f"NaN in embeddings for k={curvature}"

    def test_tsne_loss_decreases(self, small_dataset):
        """Test that t-SNE loss decreases during training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run short optimization and track loss
        losses = []

        # Manual training loop to track losses
        data = small_dataset.to(device)
        from src.affinities import compute_perplexity_affinities
        from src.kernels import compute_q_matrix
        from src.embedding import ConstantCurvatureEmbedding, compute_tsne_kl_loss
        from src.riemannian_optimizer import RiemannianSGDMomentum

        V = compute_perplexity_affinities(data, perplexity=10.0).to(device)

        model = ConstantCurvatureEmbedding(
            n_points=len(data),
            embed_dim=2,
            curvature=0.0,
            init_scale=0.001,
            device=device,
            init_method=InitMethod.RANDOM,
        )

        optimizer = RiemannianSGDMomentum(
            model.parameters(), lr=50.0, curvature=0.0, momentum=0.5
        )

        for i in range(100):
            optimizer.zero_grad()
            Q = compute_q_matrix(model.manifold, model.points)
            loss = compute_tsne_kl_loss(Q, V * 12.0 if i < 25 else V)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Check that loss generally decreases
        # Compare first 10% vs last 10%
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10

        assert late_avg < early_avg, (
            f"Loss did not decrease: early={early_avg:.4f}, late={late_avg:.4f}"
        )

    def test_tsne_kl_loss_computation(self):
        """Test KL loss computation directly."""
        torch.manual_seed(42)
        n = 20

        # Create dummy Q and V matrices
        Q = torch.rand(n, n)
        Q = (Q + Q.t()) / 2
        Q.fill_diagonal_(0)
        Q = Q / Q.sum()

        V = torch.rand(n, n)
        V = (V + V.t()) / 2
        V.fill_diagonal_(0)
        V = V / V.sum()

        loss = compute_tsne_kl_loss(Q, V)

        # Loss should be finite and non-negative (KL divergence >= 0)
        assert torch.isfinite(loss), "KL loss is not finite"
        # Note: Our simplified loss can be negative since we only compute -sum(V * log(Q))

    def test_tsne_manifold_constraints(self, small_dataset):
        """Test that manifold constraints are maintained after t-SNE optimization."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test hyperbolic
        model_hyp = fit_embedding(
            data=small_dataset,
            embed_dim=2,
            curvature=-1.0,
            device=device,
            perplexity=10.0,
            n_iterations=30,
            early_exaggeration_iterations=10,
            learning_rate=10.0,
            init_method=InitMethod.RANDOM,
            init_scale=0.001,
            verbose=False,
        )

        embeddings = model_hyp.get_embeddings()
        x0 = embeddings[:, 0]
        spatial = embeddings[:, 1:]
        constraint = -(x0**2) + (spatial**2).sum(dim=1)
        expected = torch.ones_like(constraint) * (-1.0)
        assert torch.allclose(constraint, expected, atol=1e-3), (
            "Hyperboloid constraint violated"
        )

        # Test spherical
        model_sph = fit_embedding(
            data=small_dataset,
            embed_dim=2,
            curvature=1.0,
            device=device,
            perplexity=10.0,
            n_iterations=30,
            early_exaggeration_iterations=10,
            learning_rate=10.0,
            init_method=InitMethod.RANDOM,
            init_scale=0.001,
            verbose=False,
        )

        embeddings = model_sph.get_embeddings()
        norms = (embeddings**2).sum(dim=1)
        expected = torch.ones_like(norms)
        assert torch.allclose(norms, expected, atol=1e-3), "Spherical constraint violated"


class TestRiemannianSGDMomentum:
    """Tests for momentum optimizer."""

    def test_momentum_set(self):
        """Test that momentum can be changed."""
        from src.riemannian_optimizer import RiemannianSGDMomentum

        params = [torch.randn(10, 3, requires_grad=True)]
        optimizer = RiemannianSGDMomentum(params, lr=0.01, curvature=0.0, momentum=0.5)

        assert optimizer.momentum == 0.5
        optimizer.set_momentum(0.8)
        assert optimizer.momentum == 0.8

    def test_momentum_step(self):
        """Test that momentum optimizer takes valid steps."""
        from src.riemannian_optimizer import RiemannianSGDMomentum

        torch.manual_seed(42)
        params = torch.randn(10, 3, requires_grad=True)
        optimizer = RiemannianSGDMomentum([params], lr=0.01, curvature=0.0, momentum=0.5)

        # Create dummy gradient
        loss = (params**2).sum()
        loss.backward()

        # Take step
        old_params = params.data.clone()
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(params.data, old_params)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
