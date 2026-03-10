"""Type definitions and enums for the embedding module."""

from enum import Enum


class InitMethod(Enum):
    """Initialization methods for embedding."""

    RANDOM = "random"  # Random initialization
    PCA = "pca"  # PCA-based initialization


class ScalingLossType(Enum):
    """Scaling loss strategies for hyperbolic embeddings."""

    RMS = "rms"  # (RMS_geodesic_dist - 1)^2 — original, tends to form rings
    HARD_BARRIER = "hard_barrier"  # mean(relu(d - d_max)^2) — hard fence at d_max
    SOFTPLUS_BARRIER = "softplus_barrier"  # mean(softplus(d - d_max)) — smooth barrier
    MEAN_DISTANCE = "mean_distance"  # mean(d) — simple linear penalty on spread
