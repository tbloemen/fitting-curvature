"""Type definitions and enums for the embedding module."""

from enum import Enum


class InitMethod(Enum):
    """Initialization methods for embedding."""

    RANDOM = "random"  # Random initialization
    PCA = "pca"  # PCA-based initialization
