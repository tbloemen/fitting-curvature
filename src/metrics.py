from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

# Maximum number of samples before switching to sampling-based evaluation
# For N samples, sklearn computes an N×N distance matrix
# At 10k samples, this is 10k × 10k × 4 bytes = 400MB (manageable)
MAX_SAMPLES_FOR_FULL_EVALUATION = 10000


class LossType(Enum):
    GU2019 = "gu2019"
    MSE = "mse"


# ---------------------------------------------------------------------------
# A. Local structure preservation
# ---------------------------------------------------------------------------


def continuity(
    high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int = 5
) -> float:
    """
    Compute continuity metric between high-dimensional and low-dimensional embeddings.

    Continuity measures how well the local neighborhood structure is preserved when
    going from high-dimensional to low-dimensional space. It's the complement of
    trustworthiness in the reverse direction.

    Parameters
    ----------
    high_dim : np.ndarray, shape (n_samples, n_features_high)
        High-dimensional data
    low_dim : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional embedding
    n_neighbors : int, default=5
        Number of neighbors to consider

    Returns
    -------
    continuity : float
        Continuity score between 0 and 1, higher is better
    """
    # Compute trustworthiness in the reverse direction (low -> high)
    # This gives us continuity (high -> low)
    return trustworthiness(low_dim, high_dim, n_neighbors=n_neighbors)


def knn_overlap(
    high_dim_distances: np.ndarray,
    embedded_distances: np.ndarray,
    k: int = 10,
) -> float:
    """
    Fraction of k-nearest neighbors preserved between high-dim and embedded spaces.

    Parameters
    ----------
    high_dim_distances : np.ndarray, shape (n, n)
        Pairwise distance matrix in high-dimensional space
    embedded_distances : np.ndarray, shape (n, n)
        Pairwise geodesic distance matrix in embedded space
    k : int
        Number of neighbors

    Returns
    -------
    float
        Mean fraction of preserved neighbors (0 to 1, higher is better)
    """
    n = high_dim_distances.shape[0]
    k = min(k, n - 1)

    # Get k-NN indices from each distance matrix (exclude self)
    high_knn = np.argsort(high_dim_distances, axis=1)[:, 1 : k + 1]
    embed_knn = np.argsort(embedded_distances, axis=1)[:, 1 : k + 1]

    overlap = 0.0
    for i in range(n):
        overlap += len(np.intersect1d(high_knn[i], embed_knn[i])) / k
    return overlap / n


# ---------------------------------------------------------------------------
# B. Global geometry preservation
# ---------------------------------------------------------------------------


def geodesic_distortion(
    high_dim_distances: np.ndarray, embedded_distances: np.ndarray, method: LossType
) -> float:
    """
    Distortion between high-dim and embedded pairwise distances as presented by Gu et al (2019).

    Parameters
    ----------
    high_dim_distances : np.ndarray, shape (n, n)
        High-dimensional pairwise distance matrix
    embedded_distances : np.ndarray, shape (n, n)
        Embedded geodesic pairwise distance matrix

    Returns
    -------
    float
        Relative distortion (0 to infinity, lower is better)
    """
    # Extract upper triangle (exclude diagonal)
    idx = np.triu_indices(high_dim_distances.shape[0], k=1)
    high_flat = high_dim_distances[idx]
    embed_flat = embedded_distances[idx]
    if method == LossType.GU2019:
        distortion = np.mean(np.abs((embed_flat / high_flat) ** 2 - 1))
        return float(distortion)
    elif method == LossType.MSE:
        distortion = np.mean((embed_flat - high_flat) ** 2)
        return float(distortion)


def volume_distortion(
    high_dim_distances: np.ndarray,
    embedded_distances: np.ndarray,
    k: int = 10,
) -> float:
    """
    Ratio of local neighborhood volumes (high-dim vs embedded), averaged.

    Uses the k-th neighbor distance as a proxy for local volume.
    Returns the mean log-ratio (0 = no distortion).

    Parameters
    ----------
    high_dim_distances : np.ndarray, shape (n, n)
    embedded_distances : np.ndarray, shape (n, n)
    k : int
        Neighborhood size

    Returns
    -------
    float
        Mean absolute log-ratio of k-th neighbor distances.
        0 = perfect, higher = more distortion.
    """
    n = high_dim_distances.shape[0]
    k = min(k, n - 1)

    # k-th neighbor distance for each point
    high_sorted = np.sort(high_dim_distances, axis=1)[:, k]
    embed_sorted = np.sort(embedded_distances, axis=1)[:, k]

    # Avoid log(0)
    eps = 1e-10
    high_sorted = np.maximum(high_sorted, eps)
    embed_sorted = np.maximum(embed_sorted, eps)

    log_ratio = np.abs(np.log(embed_sorted / high_sorted))
    return float(np.mean(log_ratio))


def spectral_distortion(
    high_dim_distances: np.ndarray,
    embedded_distances: np.ndarray,
    n_eigenvalues: int = 20,
) -> float:
    """
    Compare top eigenvalues of distance matrices via Pearson correlation.

    Parameters
    ----------
    high_dim_distances : np.ndarray, shape (n, n)
    embedded_distances : np.ndarray, shape (n, n)
    n_eigenvalues : int
        Number of top eigenvalues to compare

    Returns
    -------
    float
        Pearson correlation of top eigenvalues (0 to 1, higher is better)
    """
    n = high_dim_distances.shape[0]
    n_eig = min(n_eigenvalues, n - 1)

    # Double-center distance matrices (classical MDS kernel)
    def double_center(D):
        D_sq = D**2
        row_mean = D_sq.mean(axis=1, keepdims=True)
        col_mean = D_sq.mean(axis=0, keepdims=True)
        total_mean = D_sq.mean()
        return -0.5 * (D_sq - row_mean - col_mean + total_mean)

    B_high = double_center(high_dim_distances)
    B_embed = double_center(embedded_distances)

    # Top eigenvalues (largest)
    eig_high = np.linalg.eigvalsh(B_high)[-n_eig:][::-1]
    eig_embed = np.linalg.eigvalsh(B_embed)[-n_eig:][::-1]

    corr = np.corrcoef(eig_high, eig_embed)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# C. Space efficiency
# ---------------------------------------------------------------------------


def area_utilisation(embeddings: np.ndarray) -> float:
    """
    Fraction of bounding area used by the convex hull of the embedding.

    Uses the 2D projection (first two spatial coordinates).

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
        Embedded points (ambient coordinates)

    Returns
    -------
    float
        Ratio of convex hull area to bounding box area (0 to 1)
    """
    # Use first two spatial dimensions
    if embeddings.shape[1] <= 2:
        pts = embeddings[:, :2]
    else:
        # For curved spaces the first column is x0 (time/radial); skip it
        pts = embeddings[:, 1:3]

    if pts.shape[0] < 3:
        return 0.0

    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, ConvexHull.volume is the area
    except Exception:
        return 0.0

    # Bounding box area
    ranges = pts.max(axis=0) - pts.min(axis=0)
    bbox_area = float(np.prod(ranges))
    if bbox_area < 1e-12:
        return 0.0

    return float(hull_area / bbox_area)


def radial_distribution(embeddings: np.ndarray) -> float:
    """
    Normalized standard deviation of radial distances from centroid.

    Higher values indicate more uniform spread across the space.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)

    Returns
    -------
    float
        Coefficient of variation of radial distances (std / mean).
        Lower = more uniform spread.
    """
    centroid = embeddings.mean(axis=0)
    radii = np.linalg.norm(embeddings - centroid, axis=1)
    mean_r = radii.mean()
    if mean_r < 1e-12:
        return 0.0
    return float(radii.std() / mean_r)


# ---------------------------------------------------------------------------
# D. Perceptual evaluation
# ---------------------------------------------------------------------------


def cluster_interpretability(
    embedded_distances: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Silhouette score using labels on embedded distances.

    Parameters
    ----------
    embedded_distances : np.ndarray, shape (n, n)
        Pairwise geodesic distance matrix in embedded space
    labels : np.ndarray, shape (n,)

    Returns
    -------
    float
        Silhouette score (−1 to 1, higher = better separated clusters)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    return float(silhouette_score(embedded_distances, labels, metric="precomputed"))


# ---------------------------------------------------------------------------
# Aggregated metric computation
# ---------------------------------------------------------------------------


def evaluate_embedding(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    n_neighbors: int = 5,
    max_samples: int = MAX_SAMPLES_FOR_FULL_EVALUATION,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Evaluate embedding quality using trustworthiness and continuity metrics.

    For large datasets (> max_samples), uses random sampling to estimate metrics
    without computing the full N×N distance matrix.

    Parameters
    ----------
    high_dim : np.ndarray, shape (n_samples, n_features_high)
        Original high-dimensional data
    low_dim : np.ndarray, shape (n_samples, n_features_low)
        Low-dimensional embedding
    n_neighbors : int, default=5
        Number of neighbors to consider for metrics
    max_samples : int, default=10000
        Maximum samples before using sampling-based evaluation
    random_state : int, default=42
        Random seed for reproducible sampling

    Returns
    -------
    trustworthiness_score : float
        Trustworthiness metric (0-1, higher is better)
    continuity_score : float
        Continuity metric (0-1, higher is better)
    """
    n_samples = high_dim.shape[0]

    if n_samples > max_samples:
        # Use sampling-based evaluation for large datasets
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, size=max_samples, replace=False)
        high_dim = high_dim[indices]
        low_dim = low_dim[indices]
        print(f"  (Using {max_samples} sampled points for metric evaluation)")

    trust_score = trustworthiness(high_dim, low_dim, n_neighbors=n_neighbors)
    cont_score = continuity(high_dim, low_dim, n_neighbors=n_neighbors)

    return trust_score, cont_score


def compute_all_metrics(
    embedded_distances: np.ndarray,
    embeddings: np.ndarray,
    high_dim_data: Optional[np.ndarray] = None,
    high_dim_distances: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    n_neighbors: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Compute all embedding quality metrics.

    Parameters
    ----------
    embedded_distances : np.ndarray, shape (n, n)
        Pairwise geodesic distances in embedded space
    embeddings : np.ndarray, shape (n, d)
        Embedded points in ambient coordinates
    high_dim_data : np.ndarray, optional, shape (n, D)
        Original high-dimensional data (needed for trustworthiness/continuity)
    high_dim_distances : np.ndarray, optional, shape (n, n)
        High-dimensional pairwise distance matrix
    labels : np.ndarray, optional, shape (n,)
        Class labels for cluster-based metrics
    n_neighbors : int
        Number of neighbors for local metrics

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary of metric name -> value. None if metric could not be computed.
    """
    results: Dict[str, Optional[float]] = {}

    # Compute high-dim distances from data if not provided
    if high_dim_distances is None and high_dim_data is not None:
        high_dim_distances = squareform(pdist(high_dim_data))

    has_high_dist = high_dim_distances is not None
    has_labels = labels is not None and len(np.unique(labels)) >= 2

    # --- A. Local structure preservation ---
    if high_dim_data is not None:
        results["trustworthiness"] = float(
            trustworthiness(high_dim_data, embeddings, n_neighbors=n_neighbors)
        )
        results["continuity"] = continuity(
            high_dim_data, embeddings, n_neighbors=n_neighbors
        )
    else:
        results["trustworthiness"] = None
        results["continuity"] = None

    if has_high_dist:
        results["knn_overlap"] = knn_overlap(
            high_dim_distances, embedded_distances, k=n_neighbors
        )
    else:
        results["knn_overlap"] = None

    # --- B. Global geometry preservation ---
    if has_high_dist:
        results["geodesic_distortion_gu"] = geodesic_distortion(
            high_dim_distances, embedded_distances, LossType.GU2019
        )
        results["geodesic_distortion_mse"] = geodesic_distortion(
            high_dim_distances, embedded_distances, LossType.MSE
        )
    else:
        results["geodesic_distortion_gu"] = None
        results["geodesic_distortion_mse"] = None

    # --- C. Space efficiency ---
    results["area_utilisation"] = area_utilisation(embeddings)
    results["radial_distribution"] = radial_distribution(embeddings)

    # --- D. Perceptual evaluation ---
    if has_labels:
        assert labels is not None
        results["cluster_interpretability"] = cluster_interpretability(
            embedded_distances, labels
        )
    else:
        results["cluster_interpretability"] = None

    print(results)
    return results
