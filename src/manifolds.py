"""
Constant curvature manifolds for Riemannian optimization.

This module provides a base Manifold class and concrete implementations for
three constant curvature geometries:
- Euclidean (k=0)
- Spherical (k>0)
- Hyperbolic (k<0, hyperboloid model)

Design inspired by geoopt library.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.types import InitMethod, ScalingLossType


def _pca_init(
    data: Tensor, n_components: int, init_scale: float, device: torch.device
) -> Tensor:
    """
    Initialize points using PCA with specified scale.

    Performs PCA on high-dimensional data and scales the coordinates to have
    standard deviation equal to init_scale, matching the spread of random initialization.

    Parameters
    ----------
    data : Tensor
        High-dimensional data, shape (n_points, n_features)
    n_components : int
        Number of principal components to keep (embedding dimension)
    init_scale : float
        Target standard deviation for the initialized coordinates
    device : torch.device
        Target device for the output tensor

    Returns
    -------
    Tensor
        PCA-initialized points, shape (n_points, n_components)
        Scaled to have std ≈ init_scale
    """

    # Center the data
    data_centered = data - data.mean(dim=0, keepdim=True)

    # Compute SVD
    U, S, _ = torch.linalg.svd(data_centered, full_matrices=False)

    # Project data onto first n_components principal components
    X_pca = U[:, :n_components] @ torch.diag(S[:n_components])

    # Scale to match init_scale (similar to random initialization)
    current_std = X_pca.std()
    if current_std > 1e-10:
        X_pca = X_pca / current_std * init_scale
    else:
        # If std is near zero, just use init_scale directly
        X_pca = X_pca * init_scale

    return X_pca.to(device)


class Manifold(ABC):
    """
    Base class for constant curvature manifolds.

    This class defines the interface for manifold operations needed for
    Riemannian optimization in constant curvature spaces.
    """

    def __init__(self, curvature: float):
        """
        Initialize manifold with given curvature.

        Parameters
        ----------
        curvature : float
            Curvature of the manifold (k > 0: sphere, k = 0: Euclidean, k < 0: hyperbolic)
        """
        self.curvature = curvature
        if curvature != 0:
            self.radius = 1.0 / (abs(curvature) ** 0.5)
            self.radius_squared = 1.0 / abs(curvature)
        else:
            self.radius = 1.0
            self.radius_squared = 1.0

    @abstractmethod
    def init_points(
        self,
        n_points: int,
        embed_dim: int,
        init_scale: float,
        device: torch.device,
        init_method: InitMethod = InitMethod.RANDOM,
        data: Tensor | None = None,
    ) -> Tensor:
        """
        Initialize points on the manifold.

        Parameters
        ----------
        n_points : int
            Number of points to initialize
        embed_dim : int
            Intrinsic dimension of the manifold
        init_scale : float
            Scale for random initialization
        device : torch.device
            Device to create tensors on
        init_method : InitMethod
            Initialization method: InitMethod.RANDOM or InitMethod.PCA
        data : Tensor, optional
            High-dimensional data for PCA initialization (required if init_method=InitMethod.PCA)

        Returns
        -------
        Tensor
            Points in ambient space, shape (n_points, ambient_dim)
        """
        pass

    @abstractmethod
    def pairwise_distances(self, points: Tensor) -> Tensor:
        """
        Compute pairwise distances between points on the manifold.

        Parameters
        ----------
        points : Tensor
            Points in ambient space, shape (n_points, ambient_dim)

        Returns
        -------
        Tensor
            Distance matrix, shape (n_points, n_points)
        """
        pass

    @abstractmethod
    def pairwise_distances_batched(
        self, points: Tensor, indices_i: Tensor, indices_j: Tensor
    ) -> Tensor:
        """
        Compute distances for batched pairs on the manifold.

        Parameters
        ----------
        points : Tensor
            Points in ambient space, shape (n_points, ambient_dim)
        indices_i : Tensor
            First point indices, shape (batch_size,)
        indices_j : Tensor
            Second point indices, shape (batch_size,)

        Returns
        -------
        Tensor
            Distances for batched pairs, shape (batch_size,)
        """
        pass

    @abstractmethod
    def project_to_tangent(self, points: Tensor, grad: Tensor) -> Tensor:
        """
        Project gradient to tangent space at points.

        Converts Euclidean gradient to Riemannian gradient in tangent space.

        Parameters
        ----------
        points : Tensor
            Points on manifold, shape (..., ambient_dim)
        grad : Tensor
            Euclidean gradient, shape (..., ambient_dim)

        Returns
        -------
        Tensor
            Tangent vector in tangent space, shape (..., ambient_dim)
        """
        pass

    @abstractmethod
    def exp_map(self, points: Tensor, tangent_vec: Tensor) -> Tensor:
        """
        Exponential map: move from point along tangent vector.

        Maps tangent vectors to points on the manifold.

        Parameters
        ----------
        points : Tensor
            Base points on manifold, shape (..., ambient_dim)
        tangent_vec : Tensor
            Tangent vectors, shape (..., ambient_dim)

        Returns
        -------
        Tensor
            New points on manifold, shape (..., ambient_dim)
        """
        pass

    @abstractmethod
    def center_and_scale(self, points: Tensor) -> Tensor:
        """
        Center points so the mean lies at the manifold origin and scale
        so the RMS of geodesic distances from the origin equals 1.
        """
        pass

    @abstractmethod
    def center(self, points: Tensor) -> Tensor:
        """
        Center points so the Fréchet mean lies at the manifold origin,
        without rescaling.
        """
        pass

    @abstractmethod
    def scaling_loss(self, points: Tensor) -> Tensor:
        """
        Differentiable loss penalizing RMS geodesic distance from origin
        deviating from 1.

        Returns a scalar loss: (RMS_geodesic_dist - 1)^2
        """
        pass

    @abstractmethod
    def ambient_dim_for_embed_dim(self, embed_dim: int) -> int:
        """
        Return ambient dimension for given embedding dimension.

        Parameters
        ----------
        embed_dim : int
            Intrinsic dimension of the manifold

        Returns
        -------
        int
            Dimension of ambient space
        """
        pass


class Euclidean(Manifold):
    """Euclidean space (k=0)."""

    def __init__(self, curvature: float = 0.0):
        """Initialize Euclidean manifold. Curvature is always 0."""
        if abs(curvature) > 1e-10:
            raise ValueError("Euclidean manifold requires curvature = 0")
        super().__init__(0.0)

    def center_and_scale(self, points: Tensor) -> Tensor:
        """Center points at origin and scale so RMS distance from origin = 1."""
        # Center: subtract mean
        mean = points.mean(dim=0, keepdim=True)
        points = points - mean
        # Scale: divide by RMS of distances from origin
        norms_sq = (points**2).sum(dim=1)  # ||x_i||^2
        rms = torch.sqrt(norms_sq.mean())
        if rms > 1e-10:
            points = points / rms
        return points

    def center(self, points: Tensor) -> Tensor:
        """Center points at origin."""
        mean = points.mean(dim=0, keepdim=True)
        return points - mean

    def scaling_loss(self, points: Tensor) -> Tensor:
        """Scaling loss for Euclidean space: (RMS_dist - 1)^2."""
        rms = torch.sqrt((points**2).sum(dim=1).mean())
        return (rms - 1.0) ** 2

    def init_points(
        self,
        n_points: int,
        embed_dim: int,
        init_scale: float,
        device: torch.device,
        init_method: InitMethod = InitMethod.RANDOM,
        data: Tensor | None = None,
    ) -> Tensor:
        """
        Initialize points in R^d.

        Parameters
        ----------
        init_method : InitMethod
            InitMethod.RANDOM for random initialization, InitMethod.PCA for PCA-based initialization
        data : Tensor, optional
            High-dimensional data for PCA initialization
        """
        if init_method == InitMethod.RANDOM:
            return torch.randn(n_points, embed_dim, device=device) * init_scale
        elif init_method == InitMethod.PCA:
            if data is None:
                raise ValueError("Data needs to be supplied for PCA")
            return _pca_init(data, embed_dim, init_scale, device)

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """Compute Euclidean distances."""
        diff = points.unsqueeze(0) - points.unsqueeze(1)
        return torch.norm(diff, dim=2)

    def pairwise_distances_batched(
        self, points: Tensor, indices_i: Tensor, indices_j: Tensor
    ) -> Tensor:
        """
        Compute Euclidean distances for batched pairs.

        Parameters
        ----------
        points : Tensor, shape (n_points, embed_dim)
        indices_i : Tensor, shape (batch_size,)
        indices_j : Tensor, shape (batch_size,)

        Returns
        -------
        Tensor, shape (batch_size,)
        """
        diff = points[indices_i] - points[indices_j]
        return torch.norm(diff, dim=1)

    def project_to_tangent(self, points: Tensor, grad: Tensor) -> Tensor:
        """Tangent space is entire space, so projection is identity."""
        return grad

    def exp_map(self, points: Tensor, tangent_vec: Tensor) -> Tensor:
        """Exponential map is just addition."""
        return points + tangent_vec

    def ambient_dim_for_embed_dim(self, embed_dim: int) -> int:
        """Ambient dimension equals intrinsic dimension."""
        return embed_dim


class Sphere(Manifold):
    """Spherical manifold (k>0), embedded in R^(d+1)."""

    def __init__(self, curvature: float):
        """
        Initialize spherical manifold.

        Parameters
        ----------
        curvature : float
            Curvature k > 0. Points lie on sphere of radius r = 1/sqrt(k)
        """
        if curvature <= 0:
            raise ValueError("Spherical manifold requires curvature > 0")
        super().__init__(curvature)

    def center_and_scale(self, points: Tensor) -> Tensor:
        """Center points so Fréchet mean is at north pole, scale so RMS geodesic distance = 1."""
        r = self.radius

        # 1. Center via Householder reflection mapping mean direction to north pole
        mean = points.mean(dim=0)
        mean_norm = torch.norm(mean)
        if mean_norm > 1e-10:
            mean_dir = mean / mean_norm
            north = torch.zeros_like(mean_dir)
            north[0] = 1.0
            v = mean_dir - north
            v_norm_sq = (v * v).sum()
            if v_norm_sq > 1e-10:
                # Householder reflection: H(x) = x - 2*(v·x)/(v·v) * v
                proj = points @ v  # (n,)
                points = points - (2.0 * proj / v_norm_sq).unsqueeze(1) * v.unsqueeze(0)

        # 2. Scale geodesic distances from north pole so RMS = 1
        cos_theta = torch.clamp(points[:, 0] / r, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)  # angular distance from pole
        geodesic_dist = r * theta
        rms = torch.sqrt(torch.mean(geodesic_dist**2))
        if rms < 1e-10:
            return points

        new_dist = geodesic_dist / rms
        new_theta = new_dist / r

        # Reconstruct points with scaled angular distances
        spatial = points[:, 1:]
        spatial_norms = torch.norm(spatial, dim=1, keepdim=True)
        spatial_norms = torch.clamp(spatial_norms, min=1e-10)
        spatial_dir = spatial / spatial_norms

        new_x0 = r * torch.cos(new_theta)
        new_spatial = r * torch.sin(new_theta).unsqueeze(1) * spatial_dir

        return torch.cat([new_x0.unsqueeze(1), new_spatial], dim=1)

    def center(self, points: Tensor) -> Tensor:
        """No-op: centering on the sphere is a rotation (isometry), doesn't affect KL loss."""
        return points

    def scaling_loss(self, points: Tensor) -> Tensor:
        """No-op: scaling is constrained by the compact geometry of the sphere."""
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)

    def init_points(
        self,
        n_points: int,
        embed_dim: int,
        init_scale: float,
        device: torch.device,
        init_method: InitMethod = InitMethod.RANDOM,
        data: Tensor | None = None,
    ) -> Tensor:
        """
        Initialize points on sphere in R^(d+1).

        Uses constraint: ||x||^2 = r^2 where r = 1/sqrt(k)

        Parameters
        ----------
        init_method : InitMethod
            InitMethod.RANDOM for random initialization, InitMethod.PCA for PCA-based initialization
        data : Tensor, optional
            High-dimensional data for PCA initialization
        """
        if init_method == InitMethod.RANDOM:
            # Initialize random spatial coordinates
            spatial_init = torch.randn(n_points, embed_dim, device=device) * init_scale
        elif init_method == InitMethod.PCA:
            # Use PCA to get spatial coordinates
            if data is None:
                raise ValueError("Data needs to be supplied for PCA")
            spatial_init = _pca_init(data, embed_dim, init_scale, device)

        # Project to sphere by computing x0 from constraint
        radius_squared = self.radius_squared
        squared_norm = (spatial_init * spatial_init).sum(dim=1, keepdim=True)
        squared_norm = torch.clamp(squared_norm, max=radius_squared - 1e-7)
        x0 = torch.sqrt(radius_squared - squared_norm)

        points = torch.cat([x0, spatial_init], dim=1)
        return points

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """
        Compute spherical distances.

        Distance: d(x,y) = r * arccos(<x,y> / r^2)
        """
        radius = self.radius
        radius_squared = self.radius_squared

        # Compute dot products
        dots = torch.mm(points, points.t()) / radius_squared
        # Clamp to avoid numerical issues with arccos
        dots = torch.clamp(dots, -1.0 + 1e-7, 1.0 - 1e-7)
        distances = radius * torch.acos(dots)

        return distances

    def pairwise_distances_batched(
        self, points: Tensor, indices_i: Tensor, indices_j: Tensor
    ) -> Tensor:
        """
        Compute spherical distances for batched pairs.

        Parameters
        ----------
        points : Tensor, shape (n_points, embed_dim+1)
        indices_i : Tensor, shape (batch_size,)
        indices_j : Tensor, shape (batch_size,)

        Returns
        -------
        Tensor, shape (batch_size,)
        """
        radius = self.radius
        radius_squared = self.radius_squared

        # Compute dot products for selected pairs
        dots = (points[indices_i] * points[indices_j]).sum(dim=1) / radius_squared
        # Clamp to avoid numerical issues with arccos
        dots = torch.clamp(dots, -1.0 + 1e-7, 1.0 - 1e-7)
        distances = radius * torch.acos(dots)

        return distances

    def project_to_tangent(self, points: Tensor, grad: Tensor) -> Tensor:
        """
        Project gradient to tangent space of sphere.

        Tangent space at x: {v : <v, x> = 0}
        Projection: proj_x(h) = h - <h, x> * x
        """
        # Project to tangent space: proj_x(h) = h - ⟨h, x⟩x
        inner_prod = (grad * points).sum(dim=-1, keepdim=True)
        v = grad - inner_prod * points
        return v

    def exp_map(self, points: Tensor, tangent_vec: Tensor) -> Tensor:
        """
        Exponential map on sphere.

        exp_x(v) = cos(||v||/r) * x + sin(||v||/r) * (r * v/||v||)
        """
        radius = self.radius
        radius_squared = self.radius_squared

        # Compute norm of tangent vector
        v_norm = torch.norm(tangent_vec, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=1e-10)

        # Exponential map: exp_x(v) = cos(||v||/r) x + sin(||v||/r) (r * v/||v||)
        x_new = (
            torch.cos(v_norm / radius) * points
            + torch.sin(v_norm / radius) * tangent_vec / v_norm * radius
        )

        # Normalize to maintain constraint exactly
        x_new_norm = torch.norm(x_new, dim=-1, keepdim=True)
        x_new = x_new / x_new_norm * (radius_squared**0.5)

        return x_new

    def ambient_dim_for_embed_dim(self, embed_dim: int) -> int:
        """Ambient dimension is embed_dim + 1."""
        return embed_dim + 1


class Hyperboloid(Manifold):
    """Hyperbolic manifold (k<0), hyperboloid model in R^(d+1)."""

    def __init__(
        self,
        curvature: float,
        scaling_loss_type: ScalingLossType = ScalingLossType.HARD_BARRIER,
    ):
        """
        Initialize hyperbolic manifold.

        Parameters
        ----------
        curvature : float
            Curvature k < 0. Points lie on hyperboloid in R^(d+1)
        scaling_loss_type : ScalingLossType
            Strategy for the scaling regularization loss.

        Notes
        -----
        Uses the hyperboloid model with Lorentzian metric (-, +, +, ...).
        Points satisfy: -x0^2 + x1^2 + ... + xd^2 = -r^2, x0 > 0
        """
        if curvature >= 0:
            raise ValueError("Hyperbolic manifold requires curvature < 0")
        super().__init__(curvature)
        self.scaling_loss_type = scaling_loss_type

    def center_and_scale(self, points: Tensor) -> Tensor:
        """Center points using Fréchet mean (Lorentz boost) and scale so RMS geodesic distance = 1."""
        r = self.radius
        r_sq = self.radius_squared

        # 1. Compute Fréchet mean via extrinsic approach
        # Average ambient coordinates and project to hyperboloid
        mean = points.mean(dim=0)
        lorentz_sq = -(mean[0] ** 2) + (mean[1:] ** 2).sum()
        if lorentz_sq >= 0:
            raise ValueError(
                f"The lorentz square was positive, while it should have been equal to -{r_sq}"
            )
        # Project mean to hyperboloid (Lorentzian normalization)
        scale = r / torch.sqrt(-lorentz_sq)
        frechet_mean = mean * scale

        # 2. Apply inverse Lorentz boost to move Fréchet mean to origin (r, 0, ..., 0)
        mu_spatial = frechet_mean[1:]
        mu_spatial_norm = torch.norm(mu_spatial)

        if mu_spatial_norm > 1e-10:
            n_dir = mu_spatial / mu_spatial_norm
            cosh_alpha = frechet_mean[0] / r
            sinh_alpha = mu_spatial_norm / r

            x0 = points[:, 0]  # (n_pts,)
            x_par = points[:, 1:] @ n_dir  # (n_pts,) projection along boost direction
            x_perp = points[:, 1:] - x_par.unsqueeze(1) * n_dir.unsqueeze(0)

            # Inverse boost
            new_x0 = cosh_alpha * x0 - sinh_alpha * x_par
            new_x_par = -sinh_alpha * x0 + cosh_alpha * x_par
            new_spatial = x_perp + new_x_par.unsqueeze(1) * n_dir.unsqueeze(0)

            points = torch.cat([new_x0.unsqueeze(1), new_spatial], dim=1)

        # 3. Scale geodesic distances from origin so RMS = 1
        # Distance from origin (r, 0, ...): d = r * arcosh(x0 / r)
        x0 = points[:, 0]
        alpha = torch.acosh(torch.clamp(x0 / r, min=1.0 + 1e-7))
        geodesic_dist = r * alpha
        rms = torch.sqrt(torch.mean(geodesic_dist**2))
        if rms < 1e-10:
            return points

        new_dist = geodesic_dist / rms
        new_alpha = new_dist / r

        # Reconstruct points with scaled hyperbolic angles
        spatial = points[:, 1:]
        spatial_norms = torch.norm(spatial, dim=1, keepdim=True)
        spatial_norms = torch.clamp(spatial_norms, min=1e-10)
        spatial_dir = spatial / spatial_norms

        new_x0 = r * torch.cosh(new_alpha)
        new_spatial = r * torch.sinh(new_alpha).unsqueeze(1) * spatial_dir

        return torch.cat([new_x0.unsqueeze(1), new_spatial], dim=1)

    def center(self, points: Tensor) -> Tensor:
        """Center points so Fréchet mean is at origin (r, 0, ..., 0)."""
        r = self.radius
        r_sq = self.radius_squared

        mean = points.mean(dim=0)
        lorentz_sq = -(mean[0] ** 2) + (mean[1:] ** 2).sum()
        if lorentz_sq >= 0:
            raise ValueError(
                f"The lorentz square was positive, while it should have been equal to -{r_sq}"
            )
        scale = r / torch.sqrt(-lorentz_sq)
        frechet_mean = mean * scale

        mu_spatial = frechet_mean[1:]
        mu_spatial_norm = torch.norm(mu_spatial)

        if mu_spatial_norm > 1e-10:
            n_dir = mu_spatial / mu_spatial_norm
            cosh_alpha = frechet_mean[0] / r
            sinh_alpha = mu_spatial_norm / r

            x0 = points[:, 0]
            x_par = points[:, 1:] @ n_dir
            x_perp = points[:, 1:] - x_par.unsqueeze(1) * n_dir.unsqueeze(0)

            new_x0 = cosh_alpha * x0 - sinh_alpha * x_par
            new_x_par = -sinh_alpha * x0 + cosh_alpha * x_par
            new_spatial = x_perp + new_x_par.unsqueeze(1) * n_dir.unsqueeze(0)

            points = torch.cat([new_x0.unsqueeze(1), new_spatial], dim=1)

        return points

    def scaling_loss(self, points: Tensor) -> Tensor:
        """Scaling regularization loss, strategy depends on scaling_loss_type."""
        r = self.radius
        x0 = points[:, 0]
        alpha = torch.acosh(torch.clamp(x0 / r, min=1.0 + 1e-7))
        geodesic_dist = r * alpha

        if self.scaling_loss_type == ScalingLossType.RMS:
            rms = torch.sqrt(torch.mean(geodesic_dist**2))
            return (rms - 1.0) ** 2
        elif self.scaling_loss_type == ScalingLossType.HARD_BARRIER:
            d_max = 3.0 * r
            excess = torch.relu(geodesic_dist - d_max)
            return torch.mean(excess**2)
        elif self.scaling_loss_type == ScalingLossType.SOFTPLUS_BARRIER:
            d_max = 3.0 * r
            return torch.mean(torch.nn.functional.softplus(geodesic_dist - d_max))
        elif self.scaling_loss_type == ScalingLossType.MEAN_DISTANCE:
            return torch.mean(geodesic_dist)
        else:
            return torch.tensor(0.0, device=points.device, dtype=points.dtype)

    def init_points(
        self,
        n_points: int,
        embed_dim: int,
        init_scale: float,
        device: torch.device,
        init_method: InitMethod = InitMethod.RANDOM,
        data: Tensor | None = None,
    ) -> Tensor:
        """
        Initialize points on hyperboloid in R^(d+1).

        Uses constraint: -x0^2 + ||spatial||^2 = -r^2, x0 > 0

        Parameters
        ----------
        init_method : InitMethod
            InitMethod.RANDOM for random initialization, InitMethod.PCA for PCA-based initialization
        data : Tensor, optional
            High-dimensional data for PCA initialization
        """
        if init_method == InitMethod.RANDOM:
            # Initialize random spatial coordinates
            spatial_init = torch.randn(n_points, embed_dim, device=device) * init_scale
        elif init_method == InitMethod.PCA:
            # Use PCA to get spatial coordinates
            if data is None:
                raise ValueError("Data needs to be supplied for PCA")
            spatial_init = _pca_init(data, embed_dim, init_scale, device)

        # Project to hyperboloid by computing x0 from constraint
        radius_squared = self.radius_squared
        squared_spatial_norm = (spatial_init * spatial_init).sum(dim=1, keepdim=True)
        time = torch.sqrt(radius_squared + squared_spatial_norm)

        points = torch.cat([time, spatial_init], dim=1)
        return points

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """
        Compute hyperbolic distances using hyperboloid model.

        Distance: d(x,y) = r * arcosh(-⟨x,y⟩_L / r^2)
        where ⟨x,y⟩_L = -x0*y0 + x1*y1 + ... + xd*yd (Lorentzian inner product)
        """
        radius = self.radius
        radius_squared = self.radius_squared

        # Lorentzian inner product with signature (-, +, +, ...)
        time = points[:, 0:1]
        spatial = points[:, 1:]

        lorentz_prod = -torch.mm(time, time.t()) + torch.mm(spatial, spatial.t())
        lorentz_prod_normalized = lorentz_prod / radius_squared

        # Following Nickel & Kiela: d(x,y) = arcosh(-⟨x,y⟩_L / r^2)
        # Note: ⟨x,y⟩_L is negative for points on hyperboloid, so -⟨x,y⟩_L >= 1
        # Add small epsilon to avoid infinite gradient at x=1 (arcosh'(1) = ∞)
        eps = 1e-7
        input_to_acosh = torch.clamp(-lorentz_prod_normalized, min=1.0 + eps)
        distances = radius * torch.acosh(input_to_acosh)

        return distances

    def pairwise_distances_batched(
        self, points: Tensor, indices_i: Tensor, indices_j: Tensor
    ) -> Tensor:
        """
        Compute hyperbolic distances for batched pairs using hyperboloid model.

        Parameters
        ----------
        points : Tensor, shape (n_points, embed_dim+1)
        indices_i : Tensor, shape (batch_size,)
        indices_j : Tensor, shape (batch_size,)

        Returns
        -------
        Tensor, shape (batch_size,)
        """
        radius = self.radius
        radius_squared = self.radius_squared

        # Lorentzian inner product with signature (-, +, +, ...)
        time_i = points[indices_i, 0:1]
        time_j = points[indices_j, 0:1]
        spatial_i = points[indices_i, 1:]
        spatial_j = points[indices_j, 1:]

        lorentz_prod = -time_i * time_j + (spatial_i * spatial_j).sum(
            dim=1, keepdim=True
        )
        lorentz_prod_normalized = lorentz_prod / radius_squared

        # Add small epsilon to avoid infinite gradient at x=1
        eps = 1e-7
        input_to_acosh = torch.clamp(-lorentz_prod_normalized, min=1.0 + eps)
        distances = radius * torch.acosh(input_to_acosh)

        return distances.squeeze(1)

    def project_to_tangent(self, points: Tensor, grad: Tensor) -> Tensor:
        """
        Project gradient to tangent space of hyperboloid.

        Apply inverse metric tensor: g^{-1} = diag(-1, 1, ..., 1)
        Then project: proj^H_x(h) = h + ⟨x, h⟩_L * x
        """
        # Apply inverse metric tensor: g^{-1} = diag(-1, 1, ..., 1)
        h = grad.clone()
        h[..., 0:1] = -h[..., 0:1]

        # Project to tangent space: proj^H_x(h) = h + ⟨x, h⟩_L * x
        # Lorentzian inner product: ⟨x, h⟩_L = -x_0 * h_0 + x_1 * h_1 + ...
        lorentz_product = -points[..., 0:1] * h[..., 0:1] + (
            points[..., 1:] * h[..., 1:]
        ).sum(dim=-1, keepdim=True)
        v = h + lorentz_product * points

        return v

    def exp_map(self, points: Tensor, tangent_vec: Tensor) -> Tensor:
        """
        Exponential map on hyperboloid.

        exp_x(v) = cosh(||v||_L/r) * x + sinh(||v||_L/r) * (r * v/||v||_L)
        """
        radius = self.radius
        radius_squared = self.radius_squared

        # Compute Lorentzian norm: ||v||_L^2 = ⟨v, v⟩_L = -v0^2 + ||v_spatial||^2
        v_lorentz_sq = -(tangent_vec[..., 0:1] ** 2) + (tangent_vec[..., 1:] ** 2).sum(
            dim=-1, keepdim=True
        )

        # Clamp to ensure positive (for spacelike vectors, this should be positive)
        v_lorentz_sq = torch.clamp(v_lorentz_sq, min=1e-15)
        v_norm = torch.sqrt(v_lorentz_sq)

        # Clip the norm to prevent numerical overflow in cosh/sinh
        # When v_norm/radius becomes large (>10), cosh and sinh explode
        max_norm = 10.0 * radius
        scale_factor = torch.where(
            v_norm > max_norm, max_norm / v_norm, torch.ones_like(v_norm)
        )
        tangent_vec_clipped = tangent_vec * scale_factor
        v_norm_clipped = v_norm * scale_factor

        # Exponential map: exp_x(v) = cosh(||v||_L/r) x + sinh(||v||_L/r) (r * v/||v||_L)
        x_new = (
            torch.cosh(v_norm_clipped / radius) * points
            + torch.sinh(v_norm_clipped / radius)
            * tangent_vec_clipped
            / v_norm_clipped
            * radius
        )

        # Normalize to maintain hyperboloid constraint exactly
        # For hyperboloid: -x0^2 + ||spatial||^2 = -r^2
        spatial = x_new[..., 1:]
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(radius_squared + spatial_norm_sq)
        x_new = torch.cat([x0, spatial], dim=-1)

        return x_new

    def ambient_dim_for_embed_dim(self, embed_dim: int) -> int:
        """Ambient dimension is embed_dim + 1."""
        return embed_dim + 1
