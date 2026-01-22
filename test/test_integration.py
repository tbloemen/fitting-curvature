"""
Integration tests for t-SNE embedding workflow.

Tests the full embedding workflow with t-SNE across different geometries.
"""

import torch

from src.embedding import fit_embedding
from src.matrices import get_default_init_scale, normalize_data
from src.types import InitMethod


def _create_test_data(n_samples: int = 100, n_features: int = 10) -> torch.Tensor:
    """Create synthetic test data."""
    torch.manual_seed(42)
    return torch.randn(n_samples, n_features)


def test_integration_hyperbolic():
    """Test full embedding workflow in hyperbolic space."""
    X = _create_test_data(100)
    X = normalize_data(X, verbose=False)
    init_scale = get_default_init_scale(embed_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=-1.0,
        perplexity=15.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=init_scale,
        verbose=False,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert embeddings.shape[1] == 3  # Ambient dimension for hyperbolic
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()

    distances = model()
    assert distances.shape == (100, 100)
    assert not torch.isnan(distances).any()
    assert not torch.isinf(distances).any()


def test_integration_euclidean():
    """Test full embedding workflow in Euclidean space."""
    X = _create_test_data(100)
    X = normalize_data(X, verbose=False)
    init_scale = get_default_init_scale(embed_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=2,
        curvature=0.0,
        device=device,
        perplexity=15.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=100.0,
        init_method=InitMethod.RANDOM,
        init_scale=init_scale,
        verbose=False,
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
    X = normalize_data(X, verbose=False)
    init_scale = get_default_init_scale(embed_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=1.0,
        perplexity=15.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=init_scale,
        verbose=False,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert embeddings.shape[1] == 3  # Ambient dimension for spherical
    assert not torch.isnan(embeddings).any()

    distances = model()
    assert distances.shape == (100, 100)
    assert not torch.isnan(distances).any()


def test_integration_multiple_curvatures():
    """Test embedding workflow across different curvatures."""
    X = _create_test_data(50)
    X = normalize_data(X, verbose=False)
    init_scale = get_default_init_scale(embed_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    curvatures = [-1.0, 0.0, 1.0]
    for k in curvatures:
        model = fit_embedding(
            data=X,
            device=device,
            embed_dim=2,
            curvature=k,
            perplexity=10.0,
            n_iterations=30,
            early_exaggeration_iterations=10,
            learning_rate=50.0,
            init_method=InitMethod.RANDOM,
            init_scale=init_scale,
            verbose=False,
        )

        embeddings = model.get_embeddings()
        distances = model()

        assert not torch.isnan(embeddings).any(), f"NaN in embeddings for k={k}"
        assert not torch.isnan(distances).any(), f"NaN in distances for k={k}"


def test_integration_pca_initialization():
    """Test embedding with PCA initialization."""
    X = _create_test_data(100)
    X = normalize_data(X, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=0.0,
        perplexity=15.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=100.0,
        init_method=InitMethod.PCA,
        verbose=False,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 100
    assert not torch.isnan(embeddings).any()


def test_integration_larger_dataset():
    """Test embedding on a larger dataset (500 samples)."""
    X = _create_test_data(500)
    X = normalize_data(X, verbose=False)
    init_scale = get_default_init_scale(embed_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=-1.0,
        perplexity=30.0,
        n_iterations=30,
        early_exaggeration_iterations=10,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=init_scale,
        verbose=False,
    )

    embeddings = model.get_embeddings()
    assert embeddings.shape[0] == 500
    assert not torch.isnan(embeddings).any()


def test_integration_different_perplexities():
    """Test embedding with different perplexity values."""
    X = _create_test_data(100)
    X = normalize_data(X, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perplexities = [5.0, 15.0, 30.0]
    for perp in perplexities:
        model = fit_embedding(
            data=X,
            embed_dim=2,
            device=device,
            curvature=0.0,
            perplexity=perp,
            n_iterations=30,
            early_exaggeration_iterations=10,
            learning_rate=100.0,
            init_method=InitMethod.RANDOM,
            init_scale=0.001,
            verbose=False,
        )

        embeddings = model.get_embeddings()
        assert not torch.isnan(embeddings).any(), f"NaN for perplexity={perp}"


def test_integration_manifold_constraints_preserved():
    """Test that manifold constraints are preserved after optimization."""
    X = _create_test_data(80)
    X = normalize_data(X, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test hyperbolic constraint
    model_hyp = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=-1.0,
        perplexity=15.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    embeddings_hyp = model_hyp.get_embeddings()
    x0 = embeddings_hyp[:, 0]
    spatial = embeddings_hyp[:, 1:]
    constraint = -(x0**2) + (spatial**2).sum(dim=1)
    expected = torch.ones_like(constraint) * (-1.0)
    assert torch.allclose(constraint, expected, atol=1e-3), "Hyperboloid constraint violated"

    # Test spherical constraint
    model_sph = fit_embedding(
        data=X,
        embed_dim=2,
        device=device,
        curvature=1.0,
        perplexity=15.0,
        n_iterations=50,
        early_exaggeration_iterations=20,
        learning_rate=50.0,
        init_method=InitMethod.RANDOM,
        init_scale=0.001,
        verbose=False,
    )

    embeddings_sph = model_sph.get_embeddings()
    norms = (embeddings_sph**2).sum(dim=1)
    expected_norms = torch.ones_like(norms)
    assert torch.allclose(norms, expected_norms, atol=1e-4), "Spherical constraint violated"
