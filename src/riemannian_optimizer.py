"""
Riemannian optimizers for constant curvature spaces.

Implementation following:
- Gu et al. (2019): "Learning Mixed-Curvature Representations in Products of Model Spaces"
  Algorithm 1: R-SGD in products (page 5)
- Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class RiemannianSGD(Optimizer):
    """
    Riemannian Stochastic Gradient Descent for constant curvature spaces.

    Follows Algorithm 1 from Gu et al. (2019), which provides a unified RSGD
    for spherical, Euclidean, and hyperbolic geometries.

    The algorithm:
    1. Compute Euclidean gradient h ← ∇L(x^(t))
    2. For spherical components: Project to tangent space
    3. For hyperbolic components: Project to tangent space and apply metric tensor
    4. For Euclidean components: Use gradient as-is
    5. Update via exponential map in each component

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float
        Learning rate (η in the paper)
    curvature : float
        Curvature of the space (k > 0: sphere, k = 0: Euclidean, k < 0: hyperbolic)
    """

    def __init__(self, params, lr: float = 0.01, curvature: float = -1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, curvature=curvature)
        super(RiemannianSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore
        """
        Perform a single optimization step following Algorithm 1 from Gu et al. (2019).

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            curvature = group["curvature"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get Euclidean gradient h
                h = p.grad.data

                if curvature > 0:
                    # Spherical case
                    # Project gradient to tangent space and update via exponential map
                    radius_sq = 1.0 / abs(curvature)
                    v = self._project_sphere(p.data, h, radius_sq)
                    p.data = self._exp_map_sphere(p.data, -lr * v, radius_sq)

                elif curvature < 0:
                    # Hyperbolic case (lines 7-9 in Algorithm 1)
                    # Project gradient to tangent space and update via exponential map
                    radius_sq = 1.0 / abs(curvature)
                    v = self._riemannian_grad_hyperbolic(p.data, h, radius_sq)
                    p.data = self._exp_map_hyperbolic(p.data, -lr * v, radius_sq)

                else:  # curvature == 0
                    # Euclidean case (line 10 in Algorithm 1)
                    # Standard gradient descent
                    p.data = p.data - lr * h

        return loss

    def _project_sphere(
        self, points: Tensor, grad_points: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Project gradient to tangent space of sphere.

        Parameters are full ambient coordinates (x0, x1, ..., xd).
        """
        # Project to tangent space: proj_x(h) = h - ⟨h, x⟩x
        inner_prod = (grad_points * points).sum(dim=-1, keepdim=True)
        v = grad_points - inner_prod * points

        return v

    def _exp_map_sphere(
        self, points: Tensor, v_points: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Exponential map on sphere.

        Parameters and updates are in full ambient coordinates.
        """
        radius = torch.sqrt(torch.tensor(radius_sq, device=points.device))

        # Compute norm of tangent vector
        v_norm = torch.norm(v_points, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=1e-10)

        # Exponential map: exp_x(v) = cos(||v||/r) x + sin(||v||/r) (r * v/||v||)
        x_new = (
            torch.cos(v_norm / radius) * points
            + torch.sin(v_norm / radius) * v_points / v_norm * radius
        )

        # Normalize to maintain constraint exactly
        x_new_norm = torch.norm(x_new, dim=-1, keepdim=True)
        x_new = (
            x_new
            / x_new_norm
            * torch.sqrt(torch.tensor(radius_sq, device=points.device))
        )

        return x_new

    def _riemannian_grad_hyperbolic(
        self, points: Tensor, grad_points: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Compute Riemannian gradient for hyperbolic space (hyperboloid model).

        Parameters are full ambient coordinates (x0, x1, ..., xd).
        """
        # Apply inverse metric tensor: g^{-1} = diag(-1, 1, ..., 1)
        h = grad_points.clone()
        h[..., 0:1] = -h[..., 0:1]

        # Project to tangent space: proj^H_x(h) = h + ⟨x, h⟩_L * x
        # Lorentzian inner product: ⟨x, h⟩_L = -x_0 * h_0 + x_1 * h_1 + ...
        lorentz_product = -points[..., 0:1] * h[..., 0:1] + (
            points[..., 1:] * h[..., 1:]
        ).sum(dim=-1, keepdim=True)
        v = h + lorentz_product * points

        return v

    def _exp_map_hyperbolic(
        self, points: Tensor, v_points: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Exponential map on hyperboloid.

        Parameters and updates are in full ambient coordinates.
        """
        radius = torch.sqrt(torch.tensor(radius_sq, device=points.device))

        # Compute Lorentzian norm: ||v||_L^2 = ⟨v, v⟩_L = -v0^2 + ||v_spatial||^2
        v_lorentz_sq = -(v_points[..., 0:1] ** 2) + (v_points[..., 1:] ** 2).sum(
            dim=-1, keepdim=True
        )

        # Clamp to ensure positive (for spacelike vectors, this should be positive)
        v_lorentz_sq = torch.clamp(v_lorentz_sq, min=1e-15)
        v_norm = torch.sqrt(v_lorentz_sq)

        # Exponential map: exp_x(v) = cosh(||v||_L/r) x + sinh(||v||_L/r) (r * v/||v||_L)
        x_new = (
            torch.cosh(v_norm / radius) * points
            + torch.sinh(v_norm / radius) * v_points / v_norm * radius
        )

        # Normalize to maintain hyperboloid constraint exactly
        # For hyperboloid: -x0^2 + ||spatial||^2 = -r^2
        spatial = x_new[..., 1:]
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(radius_sq + spatial_norm_sq)
        x_new = torch.cat([x0, spatial], dim=-1)

        return x_new
