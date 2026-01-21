"""Type definitions and enums for the embedding module."""

from enum import Enum


class LossType(Enum):
    """Loss function types for embedding optimization."""

    GU2019 = "gu2019"  # Relative distortion loss from Gu et al. (2019)
    MSE = "mse"  # Mean squared error (stress function)


class InitMethod(Enum):
    """Initialization methods for embedding."""

    RANDOM = "random"  # Random initialization
    PCA = "pca"  # PCA-based initialization
