"""Tests for synthetic dataset generators."""

import pytest
import torch

from src.load_data import load_raw_data
from src.synthetic_data import (
    SYNTHETIC_DATASETS,
    generate_antipodal_clusters,
    generate_concentric_circles,
    generate_gaussian_blob,
    generate_hyperbolic_shells,
    generate_tree_structured,
    generate_uniform_grid,
    generate_uniform_hyperbolic,
    generate_uniform_sphere,
    generate_von_mises_fisher,
    load_synthetic,
)

EUCLIDEAN_GENERATORS = [
    generate_uniform_grid,
    generate_gaussian_blob,
    generate_concentric_circles,
]
SPHERICAL_GENERATORS = [
    generate_uniform_sphere,
    generate_von_mises_fisher,
    generate_antipodal_clusters,
]
HYPERBOLIC_GENERATORS = [
    generate_uniform_hyperbolic,
    generate_tree_structured,
    generate_hyperbolic_shells,
]
ALL_GENERATORS = EUCLIDEAN_GENERATORS + SPHERICAL_GENERATORS + HYPERBOLIC_GENERATORS


class TestEuclideanGenerators:
    """Euclidean generators return D=None and correct shapes."""

    @pytest.mark.parametrize("gen", EUCLIDEAN_GENERATORS)
    def test_returns_none_distances(self, gen):
        _, _, D = gen(100)
        assert D is None

    @pytest.mark.parametrize("gen", EUCLIDEAN_GENERATORS)
    def test_shape(self, gen):
        X, y, _ = gen(100)
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2
        assert X.shape[1] == 2

    def test_uniform_grid_labels(self):
        _, y, _ = generate_uniform_grid(100)
        assert set(y.tolist()).issubset({0, 1, 2, 3})

    def test_concentric_circles_labels(self):
        _, y, _ = generate_concentric_circles(100)
        assert set(y.tolist()) == {0, 1}


class TestSphericalGenerators:
    """Spherical generators return correct distances and shapes."""

    @pytest.mark.parametrize("gen", SPHERICAL_GENERATORS)
    def test_returns_distance_matrix(self, gen):
        X, _, D = gen(50)
        assert D is not None
        assert D.shape == (X.shape[0], X.shape[0])

    @pytest.mark.parametrize("gen", SPHERICAL_GENERATORS)
    def test_points_on_unit_sphere(self, gen):
        X, _, _ = gen(50)
        norms = torch.norm(X, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize("gen", SPHERICAL_GENERATORS)
    def test_ambient_dim(self, gen):
        X, _, _ = gen(50)
        assert X.shape[1] == 3

    @pytest.mark.parametrize("gen", SPHERICAL_GENERATORS)
    def test_distances_match_arccos(self, gen):
        """Verify D matches manual arccos computation."""
        X, _, D = gen(30)
        dots = X @ X.t()
        dots = torch.clamp(dots, -1.0, 1.0)
        D_manual = torch.acos(dots)
        D_manual.fill_diagonal_(0.0)
        assert torch.allclose(D, D_manual, atol=1e-5)

    @pytest.mark.parametrize("gen", SPHERICAL_GENERATORS)
    def test_diagonal_zero(self, gen):
        _, _, D = gen(30)
        assert torch.allclose(D.diag(), torch.zeros(D.shape[0]), atol=1e-5)

    @pytest.mark.parametrize("gen", SPHERICAL_GENERATORS)
    def test_symmetric(self, gen):
        _, _, D = gen(30)
        assert torch.allclose(D, D.t(), atol=1e-5)


class TestHyperbolicGenerators:
    """Hyperbolic generators return correct distances and shapes."""

    @pytest.mark.parametrize("gen", HYPERBOLIC_GENERATORS)
    def test_returns_distance_matrix(self, gen):
        X, _, D = gen(50)
        assert D is not None
        assert D.shape == (X.shape[0], X.shape[0])

    @pytest.mark.parametrize("gen", HYPERBOLIC_GENERATORS)
    def test_ambient_dim(self, gen):
        X, _, _ = gen(50)
        assert X.shape[1] == 3

    @pytest.mark.parametrize("gen", HYPERBOLIC_GENERATORS)
    def test_points_on_hyperboloid(self, gen):
        """Check -x0^2 + x1^2 + x2^2 = -1 and x0 > 0."""
        X, _, _ = gen(50)
        lorentz_norm = -X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2
        assert torch.allclose(lorentz_norm, -torch.ones_like(lorentz_norm), atol=1e-4)
        assert (X[:, 0] > 0).all()

    @pytest.mark.parametrize("gen", HYPERBOLIC_GENERATORS)
    def test_distances_match_acosh(self, gen):
        """Verify D matches manual acosh computation."""
        X, _, D = gen(30)
        # Lorentzian inner product
        L = X.clone()
        L[:, 0] = -L[:, 0]
        inner = X @ L.t()
        minus_inner = torch.clamp(-inner, min=1.0)
        D_manual = torch.acosh(minus_inner)
        D_manual.fill_diagonal_(0.0)
        assert torch.allclose(D, D_manual, atol=1e-4)

    @pytest.mark.parametrize("gen", HYPERBOLIC_GENERATORS)
    def test_diagonal_zero(self, gen):
        _, _, D = gen(30)
        assert torch.allclose(D.diag(), torch.zeros(D.shape[0]), atol=1e-5)

    @pytest.mark.parametrize("gen", HYPERBOLIC_GENERATORS)
    def test_symmetric(self, gen):
        _, _, D = gen(30)
        assert torch.allclose(D, D.t(), atol=1e-5)


class TestNSamples:
    """n_samples parameter works correctly."""

    @pytest.mark.parametrize("gen", ALL_GENERATORS)
    def test_sample_count(self, gen):
        X, y, D = gen(200)
        assert X.shape[0] == y.shape[0]
        # For most generators, should get exactly n_samples
        # uniform_grid rounds to perfect square
        if gen != generate_uniform_grid:
            assert X.shape[0] == 200
        if D is not None:
            assert D.shape == (X.shape[0], X.shape[0])


class TestLabels:
    """Labels have expected ranges."""

    @pytest.mark.parametrize("gen", ALL_GENERATORS)
    def test_labels_are_integers(self, gen):
        _, y, _ = gen(100)
        assert y.dtype == torch.long

    @pytest.mark.parametrize("gen", ALL_GENERATORS)
    def test_labels_nonnegative(self, gen):
        _, y, _ = gen(100)
        assert (y >= 0).all()


class TestDispatcher:
    """load_synthetic and load_raw_data dispatch correctly."""

    def test_load_synthetic_all_datasets(self):
        for name in SYNTHETIC_DATASETS:
            X, y, _ = load_synthetic(name, n_samples=50)
            assert X.shape[0] == y.shape[0] or name == "uniform_grid"

    def test_load_synthetic_unknown(self):
        with pytest.raises(ValueError, match="Unknown synthetic dataset"):
            load_synthetic("nonexistent")

    def test_load_raw_data_dispatches_synthetic(self):
        X, _, D = load_raw_data("gaussian_blob", n_samples=50)
        assert D is None
        assert X.shape[0] == 50

    def test_load_raw_data_dispatches_spherical(self):
        _, _, D = load_raw_data("uniform_sphere", n_samples=50)
        assert D is not None
        assert D.shape == (50, 50)

    def test_load_raw_data_unknown(self):
        with pytest.raises(ValueError):
            load_raw_data("nonexistent")
