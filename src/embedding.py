import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.manifolds import Euclidean, Hyperboloid, Manifold, Sphere
from src.riemannian_optimizer import RiemannianSGD


def compute_loss(
    embedded_distances: Tensor,
    distance_matrix: Tensor,
    loss_type: str = "gu2019",
) -> Tensor:
    """
    Compute loss function for embedding optimization.

    Parameters
    ----------
    embedded_distances : Tensor, shape (N, N)
        Pairwise distances in the embedded space
    distance_matrix : Tensor, shape (N, N)
        Target pairwise distances to preserve
    loss_type : str
        Type of loss function: 'gu2019' for relative distortion or 'mse' for mean squared error

    Returns
    -------
    Tensor
        Scalar loss value
    """
    if loss_type == "gu2019":
        # Gu et al. (2019) relative distortion loss (Eq 2)
        # L = sum((d_P(xi,xj)/d_G(Xi,Xj) - 1)^2) for i<j
        n_points = distance_matrix.shape[0]
        mask = torch.triu(torch.ones(n_points, n_points), diagonal=1).bool()
        loss = torch.sum((embedded_distances[mask] / distance_matrix[mask] - 1) ** 2)
    elif loss_type == "mse":
        # Mean squared error (stress function)
        loss = torch.sum((embedded_distances - distance_matrix) ** 2)
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

    def forward(self) -> Tensor:
        """Return current embedding distances."""
        return self.pairwise_distances(self.points)

    def get_embeddings(self) -> Tensor:
        """Return the current embedded points."""
        return self.points


def fit_embedding(
    distance_matrix: Tensor,
    embed_dim: int,
    curvature: float,
    init_scale: float,
    n_iterations: int = 1000,
    lr: float = 0.01,
    verbose: bool = True,
    device: torch.device | str | None = None,
    loss_type: str = "gu2019",
) -> ConstantCurvatureEmbedding:
    """
    Fit a constant curvature embedding to preserve given distances.

    Parameters
    ----------
    distance_matrix : Tensor, shape (N, N)
        Target pairwise distances to preserve
    embed_dim : int
        Embedding dimensionality
    curvature : float
        Curvature of the space (k > 0: sphere, k = 0: Euclidean, k < 0: hyperbolic)
    n_iterations : int
        Number of optimization iterations
    lr : float
        Learning rate (for RSGD, typically use smaller values than Adam)
    verbose : bool
        Print progress
    device : torch.device, str, or None
        Device to use for computation (defaults to CUDA if available, else CPU)
    loss_type : str
        Type of loss function: 'gu2019' for relative distortion (default) or 'mse' for mean squared error

    Returns
    -------
    ConstantCurvatureEmbedding
        Fitted embedding model

    Notes
    -----
    Always uses Riemannian SGD optimizer following Gu et al. (2019).

    Loss functions:
    - 'gu2019': Relative distortion loss from Gu et al. (2019) Eq 2:
                L = sum((d_P(xi,xj)/d_G(Xi,Xj) - 1)^2) for i<j
    - 'mse': Mean squared error: L = sum((d_P(xi,xj) - d_G(Xi,Xj))^2)
    """
    # Set device (defaults to CUDA if available, else CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if verbose:
        print(f"Using device: {device}")

    # Move distance matrix to device
    distance_matrix = distance_matrix.to(device)

    n_points = distance_matrix.shape[0]

    # Initialize model with data-driven scale on the specified device
    model = ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, init_scale=init_scale, device=device
    )

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

        # Get embedded distances and compute loss
        embedded_distances = model()
        loss = compute_loss(embedded_distances, distance_matrix, loss_type)

        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return model
