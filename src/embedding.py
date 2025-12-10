import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm


class ConstantCurvatureEmbedding(nn.Module):
    """
    Learn embeddings in constant curvature spaces using distance preservation.

    Supports three geometries:
    - k > 0: Spherical (embedded in R^(d+1))
    - k = 0: Euclidean (embedded in R^d)
    - k < 0: Hyperbolic (hyperboloid model in R^(d+1))
    """

    def __init__(self, n_points: int, embed_dim: int, curvature: float):
        super().__init__()
        self.n_points = n_points
        self.embed_dim = embed_dim
        self.curvature = curvature

        # Initialize embeddings based on curvature
        if curvature > 0:
            # Spherical: points on sphere in R^(d+1)
            self.ambient_dim = embed_dim + 1
            points = torch.randn(n_points, self.ambient_dim)
            # Project to sphere
            radius = 1.0 / torch.sqrt(torch.tensor(curvature))
            points = radius * points / points.norm(dim=1, keepdim=True)
            self.points = nn.Parameter(points)

        elif curvature == 0:
            # Euclidean: points in R^d
            self.ambient_dim = embed_dim
            self.points = nn.Parameter(torch.randn(n_points, embed_dim) * 0.01)

        else:  # curvature < 0
            # Hyperbolic: hyperboloid model in R^(d+1)
            self.ambient_dim = embed_dim + 1
            radius = 1.0 / torch.sqrt(torch.tensor(-curvature))

            # Initialize on hyperboloid: x0^2 - x1^2 - ... - xd^2 = r^2
            spatial = torch.randn(n_points, embed_dim) * 0.1
            # Compute time coordinate to satisfy constraint
            time = torch.sqrt(radius**2 + (spatial**2).sum(dim=1, keepdim=True))
            points = torch.cat([time, spatial], dim=1)
            self.points = nn.Parameter(points)

    def project_to_manifold(self) -> Tensor:
        """Project points back to the manifold (sphere or hyperboloid)."""
        if self.curvature > 0:
            # Project to sphere
            radius = 1.0 / torch.sqrt(torch.tensor(self.curvature))
            return radius * self.points / self.points.norm(dim=1, keepdim=True)

        elif self.curvature == 0:
            # No constraint for Euclidean
            return self.points

        else:  # curvature < 0
            # Project to hyperboloid: x0^2 - ||x_space||^2 = r^2
            radius = 1.0 / torch.sqrt(torch.tensor(-self.curvature))
            spatial = self.points[:, 1:]

            # Ensure time coordinate satisfies constraint
            time_corrected = torch.sqrt(
                radius**2 + (spatial**2).sum(dim=1, keepdim=True)
            )
            # Make sure time is positive
            time_corrected = torch.abs(time_corrected)

            return torch.cat([time_corrected, spatial], dim=1)

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """
        Compute pairwise distances in the constant curvature space.

        Returns
        -------
        Tensor of shape (n_points, n_points)
        """
        if self.curvature > 0:
            # Spherical distance: d(x,y) = r * arccos(<x,y> / r^2)
            radius = 1.0 / torch.sqrt(torch.tensor(self.curvature))
            # Compute dot products
            dots = torch.mm(points, points.t()) / (radius**2)
            # Clamp to avoid numerical issues with arccos
            dots = torch.clamp(dots, -1.0 + 1e-7, 1.0 - 1e-7)
            distances = radius * torch.acos(dots)

        elif self.curvature == 0:
            # Euclidean distance
            diff = points.unsqueeze(0) - points.unsqueeze(1)
            distances = torch.norm(diff, dim=2)

        else:  # curvature < 0
            # Hyperbolic distance in hyperboloid model
            # d(x,y) = r * arccosh(-<x,y>_L / r^2)
            # where <x,y>_L = x0*y0 - x1*y1 - ... - xd*yd (Lorentzian inner product)
            radius = 1.0 / torch.sqrt(torch.tensor(-self.curvature))

            # Lorentzian inner product
            time = points[:, 0:1]
            spatial = points[:, 1:]

            lorentz_prod = torch.mm(time, time.t()) - torch.mm(spatial, spatial.t())
            lorentz_prod = -lorentz_prod / (radius**2)

            # Clamp to avoid numerical issues: arccosh needs input >= 1
            lorentz_prod = torch.clamp(lorentz_prod, 1.0 + 1e-7, None)
            distances = radius * torch.acosh(lorentz_prod)

        return distances

    def forward(self) -> Tensor:
        """Return current embedding distances."""
        points = self.project_to_manifold()
        return self.pairwise_distances(points)

    def get_embeddings(self) -> Tensor:
        """Return the current embedded points."""
        return self.project_to_manifold()


def fit_embedding(
    distance_matrix: Tensor,
    embed_dim: int,
    curvature: float,
    n_iterations: int = 1000,
    lr: float = 0.01,
    verbose: bool = True,
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
        Learning rate
    verbose : bool
        Print progress

    Returns
    -------
    ConstantCurvatureEmbedding
        Fitted embedding model
    """
    n_points = distance_matrix.shape[0]

    # Initialize model
    model = ConstantCurvatureEmbedding(n_points, embed_dim, curvature)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    pbar = tqdm(range(n_iterations), disable=not verbose, desc="Training")
    for _ in pbar:
        optimizer.zero_grad()

        # Get embedded distances
        embedded_distances = model()

        # Loss: preserve pairwise distances (stress function)
        loss = torch.sum((embedded_distances - distance_matrix) ** 2)

        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return model
