"""Tests for embedding quality metrics."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.metrics import (
    area_utilisation,
    cluster_interpretability,
    compute_all_metrics,
    continuity,
    false_structure,
    geodesic_distortion,
    knn_overlap,
    over_smoothing,
    radial_distribution,
    spectral_distortion,
    volume_distortion,
)


@pytest.fixture
def simple_data():
    """Create simple test data: two well-separated clusters."""
    rng = np.random.default_rng(42)
    cluster1 = rng.normal(loc=[0, 0, 0], scale=0.5, size=(30, 3))
    cluster2 = rng.normal(loc=[5, 5, 5], scale=0.5, size=(30, 3))
    high_dim = np.vstack([cluster1, cluster2])
    labels = np.array([0] * 30 + [1] * 30)
    return high_dim, labels


@pytest.fixture
def identity_embedding(simple_data):
    """Embedding that preserves distances perfectly (identity map to 2D via PCA-like)."""
    high_dim, labels = simple_data
    # Use first 2 dimensions as "embedding" - preserves structure well
    embedded = high_dim[:, :2]
    high_dist = squareform(pdist(high_dim))
    embed_dist = squareform(pdist(embedded))
    return high_dim, embedded, high_dist, embed_dist, labels


@pytest.fixture
def random_embedding(simple_data):
    """Random embedding that does NOT preserve structure."""
    high_dim, labels = simple_data
    rng = np.random.default_rng(99)
    embedded = rng.normal(size=(60, 2))
    high_dist = squareform(pdist(high_dim))
    embed_dist = squareform(pdist(embedded))
    return high_dim, embedded, high_dist, embed_dist, labels


# ---------------------------------------------------------------------------
# A. Local structure preservation
# ---------------------------------------------------------------------------


class TestLocalStructure:
    def test_continuity_range(self, identity_embedding):
        high_dim, embedded, _, _, _ = identity_embedding
        score = continuity(high_dim, embedded, n_neighbors=5)
        assert 0.0 <= score <= 1.0

    def test_knn_overlap_identity(self, identity_embedding):
        _, _, high_dist, embed_dist, _ = identity_embedding
        score = knn_overlap(high_dist, embed_dist, k=5)
        assert 0.0 <= score <= 1.0
        # Identity-like embedding should preserve most neighbors
        assert score > 0.5

    def test_knn_overlap_random(self, random_embedding):
        _, _, high_dist, embed_dist, _ = random_embedding
        score = knn_overlap(high_dist, embed_dist, k=5)
        assert 0.0 <= score <= 1.0

    def test_knn_overlap_perfect(self):
        """Same distance matrix should give perfect overlap."""
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(20, 3))
        dist = squareform(pdist(pts))
        assert knn_overlap(dist, dist, k=5) == pytest.approx(1.0)

    def test_knn_overlap_k_clipped(self):
        """k larger than n-1 should not crash."""
        dist = squareform(pdist(np.eye(5)))
        score = knn_overlap(dist, dist, k=100)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# B. Global geometry preservation
# ---------------------------------------------------------------------------


class TestGlobalGeometry:
    def test_geodesic_distortion_perfect(self):
        """Identical distances should give correlation ~1."""
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(30, 3))
        dist = squareform(pdist(pts))
        score = geodesic_distortion(dist, dist)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_geodesic_distortion_range(self, identity_embedding):
        _, _, high_dist, embed_dist, _ = identity_embedding
        score = geodesic_distortion(high_dist, embed_dist)
        assert -1.0 <= score <= 1.0

    def test_geodesic_distortion_identity_is_good(self, identity_embedding):
        _, _, high_dist, embed_dist, _ = identity_embedding
        score = geodesic_distortion(high_dist, embed_dist)
        assert score > 0.5

    def test_volume_distortion_perfect(self):
        """Same distances should give zero distortion."""
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(30, 3))
        dist = squareform(pdist(pts))
        score = volume_distortion(dist, dist, k=5)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_volume_distortion_nonnegative(self, identity_embedding):
        _, _, high_dist, embed_dist, _ = identity_embedding
        score = volume_distortion(high_dist, embed_dist, k=5)
        assert score >= 0.0

    def test_spectral_distortion_perfect(self):
        """Same distances should give correlation ~1."""
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(30, 3))
        dist = squareform(pdist(pts))
        score = spectral_distortion(dist, dist)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_spectral_distortion_range(self, identity_embedding):
        _, _, high_dist, embed_dist, _ = identity_embedding
        score = spectral_distortion(high_dist, embed_dist)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# C. Space efficiency
# ---------------------------------------------------------------------------


class TestSpaceEfficiency:
    def test_area_utilisation_range(self):
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(50, 2))
        score = area_utilisation(pts)
        assert 0.0 < score <= 1.0

    def test_area_utilisation_square(self):
        """Points on a square should use all the bounding box."""
        pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
        score = area_utilisation(pts)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_area_utilisation_too_few_points(self):
        pts = np.array([[0, 0], [1, 1]])
        score = area_utilisation(pts)
        assert score == 0.0

    def test_area_utilisation_higher_dim(self):
        """For 3D+ embeddings (curved spaces), uses columns 1:3."""
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(50, 3))
        score = area_utilisation(pts)
        assert 0.0 < score <= 1.0

    def test_radial_distribution_uniform(self):
        """Points on a circle have zero radial variation (CV = 0)."""
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        score = radial_distribution(pts)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_radial_distribution_positive(self):
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(50, 2))
        score = radial_distribution(pts)
        assert score >= 0.0

    def test_radial_distribution_collapsed(self):
        """All points at origin should return 0."""
        pts = np.zeros((10, 2))
        score = radial_distribution(pts)
        assert score == 0.0


# ---------------------------------------------------------------------------
# D. Perceptual evaluation
# ---------------------------------------------------------------------------


class TestPerceptualQuality:
    def test_cluster_interpretability_separated(self, identity_embedding):
        _, _, _, embed_dist, labels = identity_embedding
        score = cluster_interpretability(embed_dist, labels)
        # Well-separated clusters should have high silhouette
        assert score > 0.3

    def test_cluster_interpretability_single_label(self):
        """Single label should return 0."""
        dist = squareform(pdist(np.eye(5)))
        np.fill_diagonal(dist, 0.0)
        labels = np.zeros(5, dtype=int)
        score = cluster_interpretability(dist, labels)
        assert score == 0.0

    def test_over_smoothing_separated(self, identity_embedding):
        _, _, _, embed_dist, labels = identity_embedding
        score = over_smoothing(embed_dist, labels)
        # Well-separated clusters: inter > intra, so ratio > 1
        assert score > 1.0

    def test_over_smoothing_single_label(self):
        dist = squareform(pdist(np.eye(5)))
        labels = np.zeros(5, dtype=int)
        score = over_smoothing(dist, labels)
        assert score == 0.0

    def test_false_structure_identical(self):
        """Same distance matrix should give 0 false structure."""
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(30, 3))
        dist = squareform(pdist(pts))
        score = false_structure(dist, dist)
        assert score == 0.0

    def test_false_structure_nonnegative(self, identity_embedding):
        _, _, high_dist, embed_dist, _ = identity_embedding
        score = false_structure(high_dist, embed_dist)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_all_keys_present(self, identity_embedding):
        high_dim, embedded, high_dist, embed_dist, labels = identity_embedding
        result = compute_all_metrics(
            embedded_distances=embed_dist,
            embeddings=embedded,
            high_dim_data=high_dim,
            high_dim_distances=high_dist,
            labels=labels,
        )
        expected_keys = {
            "trustworthiness",
            "continuity",
            "knn_overlap",
            "geodesic_distortion",
            "volume_distortion",
            "spectral_distortion",
            "area_utilisation",
            "radial_distribution",
            "cluster_interpretability",
            "over_smoothing",
            "false_structure",
        }
        assert set(result.keys()) == expected_keys

    def test_no_none_with_full_inputs(self, identity_embedding):
        high_dim, embedded, high_dist, embed_dist, labels = identity_embedding
        result = compute_all_metrics(
            embedded_distances=embed_dist,
            embeddings=embedded,
            high_dim_data=high_dim,
            high_dim_distances=high_dist,
            labels=labels,
        )
        for key, val in result.items():
            assert val is not None, f"{key} is None with full inputs"

    def test_without_labels(self, identity_embedding):
        high_dim, embedded, high_dist, embed_dist, _ = identity_embedding
        result = compute_all_metrics(
            embedded_distances=embed_dist,
            embeddings=embedded,
            high_dim_data=high_dim,
            high_dim_distances=high_dist,
            labels=None,
        )
        assert result["cluster_interpretability"] is None
        assert result["over_smoothing"] is None
        # Non-label metrics should still work
        assert result["trustworthiness"] is not None
        assert result["knn_overlap"] is not None

    def test_without_high_dim_data(self, identity_embedding):
        _, embedded, high_dist, embed_dist, labels = identity_embedding
        result = compute_all_metrics(
            embedded_distances=embed_dist,
            embeddings=embedded,
            high_dim_data=None,
            high_dim_distances=high_dist,
            labels=labels,
        )
        assert result["trustworthiness"] is None
        assert result["continuity"] is None
        # Distance-based metrics should still work
        assert result["knn_overlap"] is not None
        assert result["geodesic_distortion"] is not None

    def test_without_any_high_dim(self, identity_embedding):
        _, embedded, _, embed_dist, labels = identity_embedding
        result = compute_all_metrics(
            embedded_distances=embed_dist,
            embeddings=embedded,
            high_dim_data=None,
            high_dim_distances=None,
            labels=labels,
        )
        # All metrics needing high-dim should be None
        assert result["knn_overlap"] is None
        assert result["geodesic_distortion"] is None
        assert result["false_structure"] is None
        # Space efficiency metrics should still work
        assert result["area_utilisation"] is not None
        assert result["radial_distribution"] is not None

    def test_high_dim_distances_computed_from_data(self, identity_embedding):
        """When only high_dim_data is given, distances should be computed automatically."""
        high_dim, embedded, _, embed_dist, labels = identity_embedding
        result = compute_all_metrics(
            embedded_distances=embed_dist,
            embeddings=embedded,
            high_dim_data=high_dim,
            high_dim_distances=None,
            labels=labels,
        )
        # Should compute distances from data and produce results
        assert result["knn_overlap"] is not None
        assert result["geodesic_distortion"] is not None
