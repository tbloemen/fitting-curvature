import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.manifolds import Euclidean, Hyperboloid, Manifold, Sphere
from src.matrices import compute_euclidean_distances_batched
from src.riemannian_optimizer import RiemannianSGD
from src.samplers import create_sampler


def compute_loss_batched(
    embedded_distances: Tensor,
    target_distances: Tensor,
    loss_type: str = "gu2019",
) -> Tensor:
    """
    Compute loss function for batched pairs.

    Parameters
    ----------
    embedded_distances : Tensor, shape (batch_size,)
        Pairwise distances in the embedded space for sampled pairs
    target_distances : Tensor, shape (batch_size,)
        Target pairwise distances for sampled pairs
    loss_type : str
        Type of loss function: 'gu2019' for relative distortion or 'mse' for mean squared error

    Returns
    -------
    Tensor
        Scalar loss value
    """
    if loss_type == "gu2019":
        # Gu et al. (2019) relative distortion loss
        # L = sum((d_embedded / d_target - 1)^2)
        # Add small epsilon to avoid division by zero
        loss = torch.sum((embedded_distances / (target_distances + 1e-8) - 1) ** 2)
    elif loss_type == "mse":
        # Mean squared error (stress function)
        loss = torch.sum((embedded_distances - target_distances) ** 2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'gu2019' or 'mse'")

    return loss


class ConstantCurvatureEmbedding(nn.Module):
    """
    Learn embeddings in constant curvature spaces using distance preservation.

    Supports three geometries:
    - k > 0: Spherical (embedded in R^(d+1))
    - k = 0: Euclidean (embedded in R^d)
    - k < 0: Hyperbolic (hyperboloid model in R^(d+1))
    """

    def __init__(
        self,
        n_points: int,
        embed_dim: int,
        curvature: float,
        init_scale: float = 0.1,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.n_points = n_points
        self.embed_dim = embed_dim
        self.curvature = curvature

        # Set device (defaults to CUDA if available, else CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create appropriate manifold
        self.manifold = self._create_manifold(curvature)

        # Initialize points on manifold
        points_init = self.manifold.init_points(
            n_points, embed_dim, init_scale, self.device
        )
        self.points = nn.Parameter(points_init)

        # Set ambient_dim for backward compatibility
        self.ambient_dim = self.manifold.ambient_dim_for_embed_dim(embed_dim)

    def _create_manifold(self, curvature: float) -> Manifold:
        """Factory method to create appropriate manifold."""
        if curvature > 0:
            return Sphere(curvature)
        elif curvature == 0:
            return Euclidean(curvature)
        else:
            return Hyperboloid(curvature)

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """
        Compute pairwise distances in the constant curvature space.

        Returns
        -------
        Tensor of shape (n_points, n_points)
        """
        return self.manifold.pairwise_distances(points)

    def pairwise_distances_batched(
        self, indices_i: Tensor, indices_j: Tensor
    ) -> Tensor:
        """
        Compute distances for batched pairs.

        Parameters
        ----------
        indices_i : Tensor, shape (batch_size,)
            First point indices
        indices_j : Tensor, shape (batch_size,)
            Second point indices

        Returns
        -------
        Tensor, shape (batch_size,)
            Distances for sampled pairs
        """
        return self.manifold.pairwise_distances_batched(
            self.points, indices_i, indices_j
        )

    def forward(self) -> Tensor:
        """Return current embedding distances."""
        return self.pairwise_distances(self.points)

    def get_embeddings(self) -> Tensor:
        """Return the current embedded points."""
        return self.points


def fit_embedding(
    data: Tensor,
    embed_dim: int,
    curvature: float,
    init_scale: float,
    n_iterations: int = 1000,
    lr: float = 0.01,
    verbose: bool = True,
    device: torch.device | str | None = None,
    loss_type: str = "gu2019",
    sampler_type: str = "random",
    batch_size: int = 4096,
    sampler_kwargs: dict | None = None,
) -> ConstantCurvatureEmbedding:
    """
    Fit a constant curvature embedding to preserve distances from raw data.

    This function computes Euclidean distances on-the-fly during training,
    avoiding the need to store an N×N distance matrix. This enables training
    on very large datasets with O(N×D) memory instead of O(N²).

    Parameters
    ----------
    data : Tensor, shape (N, D)
        Input data points in original space
    embed_dim : int
        Embedding dimensionality
    curvature : float
        Curvature of the space (k > 0: sphere, k = 0: Euclidean, k < 0: hyperbolic)
    init_scale : float
        Initialization scale for embeddings
    n_iterations : int
        Number of optimization iterations (default: 1000)
    lr : float
        Learning rate (for RSGD, typically use smaller values than Adam)
    verbose : bool
        Print progress (default: True)
    device : torch.device, str, or None
        Device to use for computation (defaults to CUDA if available, else CPU)
    loss_type : str
        Type of loss function: 'gu2019' for relative distortion (default) or 'mse' for mean squared error
    sampler_type : str
        Type of pair sampler: 'random', 'knn', 'stratified', 'negative' (default: 'random')
    batch_size : int
        Number of pairs to sample per iteration (default: 4096)
    sampler_kwargs : dict, optional
        Additional arguments for sampler (e.g., {'k': 15} for KNN)

    Returns
    -------
    ConstantCurvatureEmbedding
        Fitted embedding model

    Notes
    -----
    Always uses Riemannian SGD optimizer following Gu et al. (2019).
    Uses batched training with configurable pair sampling strategy.
    Distances are computed on-the-fly from raw data to minimize memory usage.

    Loss functions:
    - 'gu2019': Relative distortion loss from Gu et al. (2019) Eq 2:
                L = sum((d_P(xi,xj)/d_G(Xi,Xj) - 1)^2) for sampled pairs
    - 'mse': Mean squared error: L = sum((d_P(xi,xj) - d_G(Xi,Xj))^2) for sampled pairs
    """
    # Set device (defaults to CUDA if available, else CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if verbose:
        print(f"Using device: {device}")

    # Move data to device
    data = data.to(device)

    n_points = data.shape[0]

    # Initialize model with data-driven scale on the specified device
    model = ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, init_scale=init_scale, device=device
    )

    # Create sampler
    if sampler_kwargs is None:
        sampler_kwargs = {}

    sampler = create_sampler(
        sampler_type=sampler_type,
        n_points=n_points,
        batch_size=batch_size,
        device=device,
        **sampler_kwargs,
    )

    # Precompute sampler data structures (e.g., k-NN graph) from raw data
    if verbose:
        print(f"Initializing {sampler_type} sampler with batch_size={batch_size}...")
    sampler.precompute(data)

    optimizer = RiemannianSGD(model.parameters(), lr=lr, curvature=curvature)
    if verbose:
        manifold_type = "Euclidean"
        if curvature > 0:
            manifold_type = "spherical"
        elif curvature < 0:
            manifold_type = "hyperbolic"
        print(f"Using Riemannian SGD optimizer for {manifold_type} space")

    # Training loop
    pbar = tqdm(range(n_iterations), disable=not verbose, desc="Training")
    for _ in pbar:
        optimizer.zero_grad()

        # Sample pairs
        indices_i, indices_j = sampler.sample_pairs()

        # Compute target distances on-the-fly from raw data
        target_distances = compute_euclidean_distances_batched(
            data, indices_i, indices_j
        )

        # Get embedded distances for sampled pairs
        embedded_distances = model.pairwise_distances_batched(indices_i, indices_j)

        # Compute loss
        loss = compute_loss_batched(embedded_distances, target_distances, loss_type)

        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return model
