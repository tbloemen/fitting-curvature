import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

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

        # Compute radius for curved spaces (store as buffer, not parameter)
        # We precompute radius_squared since it's used frequently in distance calculations
        if curvature != 0:
            radius_val = 1.0 / torch.sqrt(
                torch.tensor(abs(curvature), device=self.device)
            )
            self.register_buffer("radius", radius_val)
            self.register_buffer("radius_squared", radius_val * radius_val)
        else:
            # Dummy values for Euclidean (not used but stored for consistency)
            self.register_buffer("radius", torch.tensor(1.0, device=self.device))
            self.register_buffer(
                "radius_squared", torch.tensor(1.0, device=self.device)
            )

        # Initialize embeddings based on curvature
        # RSGD works with full ambient coordinates and maintains manifold constraints.

        if curvature > 0:
            # Spherical: points on sphere in R^(d+1)
            # Initialize with small random points and project to sphere
            self.ambient_dim = embed_dim + 1

            # Initialize random spatial coordinates
            spatial_init = (
                torch.randn(n_points, embed_dim, device=self.device) * init_scale
            )

            # Project to sphere by computing x0 from constraint
            radius_squared: Tensor = self.radius_squared  # type: ignore
            squared_norm = (spatial_init * spatial_init).sum(dim=1, keepdim=True)
            squared_norm = torch.clamp(squared_norm, max=radius_squared - 1e-7)
            x0 = torch.sqrt(radius_squared - squared_norm)
            points_init = torch.cat([x0, spatial_init], dim=1)

            self.points = nn.Parameter(points_init)

        elif curvature == 0:
            # Euclidean: points in R^d (no constraint)
            self.ambient_dim = embed_dim
            self.points = nn.Parameter(
                torch.randn(n_points, embed_dim, device=self.device) * init_scale
            )

        else:  # curvature < 0
            # Hyperbolic: hyperboloid model in R^(d+1)
            # Initialize spatial coordinates and project to hyperboloid
            self.ambient_dim = embed_dim + 1

            # Initialize random spatial coordinates
            spatial_init = (
                torch.randn(n_points, embed_dim, device=self.device) * init_scale
            )

            # Project to hyperboloid by computing x0 from constraint
            radius_squared: Tensor = self.radius_squared  # type: ignore
            squared_spatial_norm = (spatial_init * spatial_init).sum(
                dim=1, keepdim=True
            )
            time = torch.sqrt(radius_squared + squared_spatial_norm)
            points_init = torch.cat([time, spatial_init], dim=1)

            self.points = nn.Parameter(points_init)

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """
        Compute pairwise distances in the constant curvature space.

        Returns
        -------
        Tensor of shape (n_points, n_points)
        """
        if self.curvature > 0:
            # Spherical distance: d(x,y) = r * arccos(<x,y> / r^2)
            radius: Tensor = self.radius  # type: ignore
            radius_squared: Tensor = self.radius_squared  # type: ignore

            # Compute dot products
            dots = torch.mm(points, points.t()) / radius_squared
            # Clamp to avoid numerical issues with arccos
            dots = torch.clamp(dots, -1.0 + 1e-7, 1.0 - 1e-7)
            distances = radius * torch.acos(dots)

        elif self.curvature == 0:
            # Euclidean distance
            diff = points.unsqueeze(0) - points.unsqueeze(1)
            distances = torch.norm(diff, dim=2)

        else:  # curvature < 0
            # Hyperbolic distance in hyperboloid model (Nickel & Kiela 2017)
            # Upper sheet hyperboloid: ⟨x,x⟩L = -x0^2 + x1^2 + ... + xd^2 = -r^2, x0 > 0
            # Lorentzian inner product: ⟨x,y⟩L = -x0*y0 + x1*y1 + ... + xd*yd
            # Distance: d(x,y) = r * arcosh(-⟨x,y⟩L / r^2)
            # Note: For points on hyperboloid, -⟨x,y⟩L / r^2 >= 1
            radius: Tensor = self.radius  # type: ignore
            radius_squared: Tensor = self.radius_squared  # type: ignore

            # Lorentzian inner product with signature (-, +, +, ...)
            # Following Nickel & Kiela: ⟨x, y⟩L = -x0*y0 + x1*y1 + ... + xd*yd
            time = points[:, 0:1]
            spatial = points[:, 1:]

            lorentz_prod = -torch.mm(time, time.t()) + torch.mm(spatial, spatial.t())
            lorentz_prod_normalized = lorentz_prod / radius_squared

            # Following Nickel & Kiela: d(x,y) = arcosh(-⟨x,y⟩L / r^2)
            # Note: ⟨x,y⟩L is negative for points on hyperboloid, so -⟨x,y⟩L >= 1
            # Add small epsilon to avoid infinite gradient at x=1 (arcosh'(1) = ∞)
            # Use clamp to ensure all values are >= 1 + eps despite floating point errors
            eps = 1e-7
            input_to_acosh = torch.clamp(-lorentz_prod_normalized, min=1.0 + eps)
            distances = radius * torch.acosh(input_to_acosh)

        return distances

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
