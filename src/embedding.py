import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.riemannian_optimizer import RiemannianSGD


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
        # For RSGD, we work with full ambient coordinates.
        # The optimizer's exponential map will maintain the manifold constraints.

        if curvature > 0:
            # Spherical: points on sphere in R^(d+1)
            # Initialize with small random points and project to sphere
            self.ambient_dim = embed_dim + 1
            self.param_dim = embed_dim  # Spatial coordinates for RSGD

            # Initialize spatial coordinates
            spatial_init = (
                torch.randn(n_points, embed_dim, device=self.device) * init_scale
            )

            # RSGD will work with spatial coordinates only
            self.points = nn.Parameter(spatial_init)

        elif curvature == 0:
            # Euclidean: points in R^d (no constraint)
            self.ambient_dim = embed_dim
            self.param_dim = embed_dim
            self.points = nn.Parameter(
                torch.randn(n_points, embed_dim, device=self.device) * init_scale
            )

        else:  # curvature < 0
            # Hyperbolic: hyperboloid model in R^(d+1)
            # Initialize spatial coordinates, RSGD will maintain hyperboloid constraint
            self.ambient_dim = embed_dim + 1
            self.param_dim = embed_dim  # Spatial coordinates for RSGD

            # Initialize spatial coordinates
            spatial_init = (
                torch.randn(n_points, embed_dim, device=self.device) * init_scale
            )

            # RSGD will work with spatial coordinates only
            self.points = nn.Parameter(spatial_init)

    def project_to_manifold(self) -> Tensor:
        """
        Map free parameters to manifold coordinates.

        For spherical and hyperbolic, we only optimize free coordinates and
        compute the constrained coordinate from the manifold equation.
        This ensures the constraint is satisfied during optimization.
        """
        if self.curvature > 0:
            # Spherical: self.points contains (x1, ..., xd)
            # Compute x0 = sqrt(r^2 - x1^2 - ... - xd^2)
            # Then return (x0, x1, ..., xd) normalized to sphere

            free_coords = self.points  # shape: (n, d)
            squared_norm = (free_coords * free_coords).sum(dim=1, keepdim=True)

            # Type assertion for buffer
            radius_squared: Tensor = self.radius_squared  # type: ignore

            # Ensure we're inside the sphere by clamping
            squared_norm = torch.clamp(squared_norm, max=radius_squared - 1e-7)

            # Compute constrained coordinate
            x0 = torch.sqrt(radius_squared - squared_norm)

            # Full ambient coordinates
            full_coords = torch.cat([x0, free_coords], dim=1)

            return full_coords

        elif self.curvature == 0:
            # Euclidean: no constraint
            return self.points

        else:  # curvature < 0
            # Hyperbolic: self.points contains spatial coords (x1, ..., xd)
            # Compute time x0 = sqrt(r^2 + x1^2 + ... + xd^2)
            # Return (x0, x1, ..., xd) which satisfies ⟨x, x⟩L = -x0^2 + ||spatial||^2 = -r^2

            spatial = self.points  # shape: (n, d)
            squared_spatial_norm = (spatial * spatial).sum(dim=1, keepdim=True)

            # Type assertion for buffer
            radius_squared: Tensor = self.radius_squared  # type: ignore

            # Compute time coordinate from constraint
            time = torch.sqrt(radius_squared + squared_spatial_norm)

            # Full ambient coordinates
            full_coords = torch.cat([time, spatial], dim=1)

            return full_coords

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
        points = self.project_to_manifold()
        return self.pairwise_distances(points)

    def get_embeddings(self) -> Tensor:
        """Return the current embedded points."""
        return self.project_to_manifold()


def fit_embedding(
    distance_matrix: Tensor,
    embed_dim: int,
    curvature: float,
    init_scale: float,
    n_iterations: int = 1000,
    lr: float = 0.01,
    verbose: bool = True,
    device: torch.device | str | None = None,
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

    Returns
    -------
    ConstantCurvatureEmbedding
        Fitted embedding model

    Notes
    -----
    Always uses Riemannian SGD optimizer following Gu et al. (2019).
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

        # Get embedded distances
        embedded_distances = model()

        # Loss: preserve pairwise distances (stress function)
        loss = torch.sum((embedded_distances - distance_matrix) ** 2)

        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return model
