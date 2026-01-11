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
        self, n_points: int, embed_dim: int, init_scale: float, device: torch.device
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
    def project_to_manifold(self, points: Tensor) -> Tensor:
        """
        Project points back onto the manifold (for numerical stability).

        Parameters
        ----------
        points : Tensor
            Approximate points, shape (..., ambient_dim)

        Returns
        -------
        Tensor
            Points projected onto manifold, shape (..., ambient_dim)
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

    def init_points(
        self, n_points: int, embed_dim: int, init_scale: float, device: torch.device
    ) -> Tensor:
        """Initialize points in R^d."""
        return torch.randn(n_points, embed_dim, device=device) * init_scale

    def pairwise_distances(self, points: Tensor) -> Tensor:
        """Compute Euclidean distances."""
        diff = points.unsqueeze(0) - points.unsqueeze(1)
        return torch.norm(diff, dim=2)

    def project_to_tangent(self, points: Tensor, grad: Tensor) -> Tensor:
        """Tangent space is entire space, so projection is identity."""
        return grad

    def exp_map(self, points: Tensor, tangent_vec: Tensor) -> Tensor:
        """Exponential map is just addition."""
        return points + tangent_vec

    def project_to_manifold(self, points: Tensor) -> Tensor:
        """No projection needed for Euclidean space."""
        return points

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

    def init_points(
        self, n_points: int, embed_dim: int, init_scale: float, device: torch.device
    ) -> Tensor:
        """
        Initialize points on sphere in R^(d+1).

        Uses constraint: ||x||^2 = r^2 where r = 1/sqrt(k)
        """
        # Initialize random spatial coordinates
        spatial_init = torch.randn(n_points, embed_dim, device=device) * init_scale

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

    def project_to_manifold(self, points: Tensor) -> Tensor:
        """Project approximate points back to sphere."""
        radius_squared = self.radius_squared
        x_new_norm = torch.norm(points, dim=-1, keepdim=True)
        return points / x_new_norm * (radius_squared**0.5)

    def ambient_dim_for_embed_dim(self, embed_dim: int) -> int:
        """Ambient dimension is embed_dim + 1."""
        return embed_dim + 1


class Hyperboloid(Manifold):
    """Hyperbolic manifold (k<0), hyperboloid model in R^(d+1)."""

    def __init__(self, curvature: float):
        """
        Initialize hyperbolic manifold.

        Parameters
        ----------
        curvature : float
            Curvature k < 0. Points lie on hyperboloid in R^(d+1)

        Notes
        -----
        Uses the hyperboloid model with Lorentzian metric (-, +, +, ...).
        Points satisfy: -x0^2 + x1^2 + ... + xd^2 = -r^2, x0 > 0
        """
        if curvature >= 0:
            raise ValueError("Hyperbolic manifold requires curvature < 0")
        super().__init__(curvature)

    def init_points(
        self, n_points: int, embed_dim: int, init_scale: float, device: torch.device
    ) -> Tensor:
        """
        Initialize points on hyperboloid in R^(d+1).

        Uses constraint: -x0^2 + ||spatial||^2 = -r^2, x0 > 0
        """
        # Initialize random spatial coordinates
        spatial_init = torch.randn(n_points, embed_dim, device=device) * init_scale

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

        # Exponential map: exp_x(v) = cosh(||v||_L/r) x + sinh(||v||_L/r) (r * v/||v||_L)
        x_new = (
            torch.cosh(v_norm / radius) * points
            + torch.sinh(v_norm / radius) * tangent_vec / v_norm * radius
        )

        # Normalize to maintain hyperboloid constraint exactly
        # For hyperboloid: -x0^2 + ||spatial||^2 = -r^2
        spatial = x_new[..., 1:]
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(radius_squared + spatial_norm_sq)
        x_new = torch.cat([x0, spatial], dim=-1)

        return x_new

    def project_to_manifold(self, points: Tensor) -> Tensor:
        """Project approximate points back to hyperboloid."""
        radius_squared = self.radius_squared
        spatial = points[..., 1:]
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(radius_squared + spatial_norm_sq)
        return torch.cat([x0, spatial], dim=-1)

    def ambient_dim_for_embed_dim(self, embed_dim: int) -> int:
        """Ambient dimension is embed_dim + 1."""
        return embed_dim + 1
