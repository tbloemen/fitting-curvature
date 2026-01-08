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
                    # Project gradient to tangent space: proj^S_x(h) = h - ⟨h, x⟩x
                    radius_sq = 1.0 / abs(curvature)
                    v = self._project_sphere(p.data, h, radius_sq)
                    # Update via exponential map
                    p.data = self._exp_map_sphere(p.data, -lr * v, radius_sq)

                elif curvature < 0:
                    # Hyperbolic case (lines 7-9 in Algorithm 1)
                    # For hyperboloid model with parameterization of spatial coords only
                    radius_sq = 1.0 / abs(curvature)
                    v = self._riemannian_grad_hyperbolic(p.data, h, radius_sq)
                    # Update via exponential map
                    p.data = self._exp_map_hyperbolic(p.data, -lr * v, radius_sq)

                else:  # curvature == 0
                    # Euclidean case (line 10 in Algorithm 1)
                    # Standard gradient descent
                    p.data = p.data - lr * h

        return loss

    def _project_sphere(
        self, spatial: Tensor, grad_spatial: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Project gradient to tangent space of sphere.

        Parameters are spatial coordinates (x1, ..., xd).
        x0 is computed from constraint: x0^2 + ||spatial||^2 = r^2
        """
        # Reconstruct full point on sphere
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        spatial_norm_sq = torch.clamp(spatial_norm_sq, max=radius_sq - 1e-7)
        x0 = torch.sqrt(radius_sq - spatial_norm_sq)
        x_full = torch.cat([x0, spatial], dim=-1)

        # Lift gradient to full space (gradient w.r.t. x0 is implicitly zero)
        grad_x0 = torch.zeros_like(x0)
        grad_full = torch.cat([grad_x0, grad_spatial], dim=-1)

        # Project to tangent space: proj_x(h) = h - ⟨h, x⟩x
        inner_prod = (grad_full * x_full).sum(dim=-1, keepdim=True)
        v_full = grad_full - inner_prod * x_full

        # Return only spatial part
        return v_full[..., 1:]

    def _exp_map_sphere(
        self, spatial: Tensor, v_spatial: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Exponential map on sphere.

        Parameters and updates are in spatial coordinates only.
        """
        radius = torch.sqrt(torch.tensor(radius_sq, device=spatial.device))

        # Reconstruct full point
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        spatial_norm_sq = torch.clamp(spatial_norm_sq, max=radius_sq - 1e-7)
        x0 = torch.sqrt(radius_sq - spatial_norm_sq)
        x_full = torch.cat([x0, spatial], dim=-1)

        # Lift update vector (x0 component computed from tangency constraint)
        # For tangent vector, we need: ⟨x, v⟩ = 0, so v_0 = -(spatial · v_spatial) / x0
        v0 = -(spatial * v_spatial).sum(dim=-1, keepdim=True) / x0
        v_full = torch.cat([v0, v_spatial], dim=-1)

        # Compute norm
        v_norm = torch.norm(v_full, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=1e-10)

        # Exponential map: exp_x(v) = cos(||v||/r) x + sin(||v||/r) (r * v/||v||)
        x_new_full = (
            torch.cos(v_norm / radius) * x_full
            + torch.sin(v_norm / radius) * v_full / v_norm * radius
        )

        # Return spatial coordinates
        return x_new_full[..., 1:]

    def _riemannian_grad_hyperbolic(
        self, spatial: Tensor, grad_spatial: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Compute Riemannian gradient for hyperbolic space (hyperboloid model).

        Parameters are spatial coordinates (x1, ..., xd).
        x0 is computed from constraint: x0^2 - ||spatial||^2 = r^2
        """
        # Reconstruct full hyperboloid point
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(radius_sq + spatial_norm_sq)
        x_full = torch.cat([x0, spatial], dim=-1)

        # Lift gradient to full space (gradient w.r.t. x0 is implicitly zero)
        grad_x0 = torch.zeros_like(x0)
        grad_full = torch.cat([grad_x0, grad_spatial], dim=-1)

        # Apply inverse metric tensor: g^{-1} = diag(-1, 1, ..., 1)
        h = grad_full.clone()
        h[..., 0:1] = -h[..., 0:1]

        # Project to tangent space: proj^H_x(h) = h + ⟨x, h⟩_L * x
        # Lorentzian inner product: ⟨x, h⟩_L = -x_0 * h_0 + x_1 * h_1 + ...
        lorentz_product = -x_full[..., 0:1] * h[..., 0:1] + (
            x_full[..., 1:] * h[..., 1:]
        ).sum(dim=-1, keepdim=True)
        v_full = h + lorentz_product * x_full

        # Return only spatial part
        return v_full[..., 1:]

    def _exp_map_hyperbolic(
        self, spatial: Tensor, v_spatial: Tensor, radius_sq: float = 1.0
    ) -> Tensor:
        """
        Exponential map on hyperboloid.

        Parameters and updates are in spatial coordinates only.
        """
        radius = torch.sqrt(torch.tensor(radius_sq, device=spatial.device))

        # Reconstruct full point
        spatial_norm_sq = (spatial**2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(radius_sq + spatial_norm_sq)
        x_full = torch.cat([x0, spatial], dim=-1)

        # Lift update vector (x0 component computed from tangency constraint)
        # For tangent vector, we need: ⟨x, v⟩_L = 0, so -x0*v0 + spatial·v_spatial = 0
        # Therefore: v_0 = (spatial · v_spatial) / x0
        v0 = (spatial * v_spatial).sum(dim=-1, keepdim=True) / x0
        v_full = torch.cat([v0, v_spatial], dim=-1)

        # Compute Lorentzian norm: ||v||_L^2 = ⟨v, v⟩_L = -v0^2 + ||v_spatial||^2
        v_lorentz_sq = -(v_full[..., 0:1] ** 2) + (v_full[..., 1:] ** 2).sum(
            dim=-1, keepdim=True
        )

        # Clamp to ensure positive (for spacelike vectors, this should be positive)
        v_lorentz_sq = torch.clamp(v_lorentz_sq, min=1e-15)
        v_norm = torch.sqrt(v_lorentz_sq)

        # Exponential map: exp_x(v) = cosh(||v||_L/r) x + sinh(||v||_L/r) (r * v/||v||_L)
        x_new_full = (
            torch.cosh(v_norm / radius) * x_full
            + torch.sinh(v_norm / radius) * v_full / v_norm * radius
        )

        # The exponential map should produce a point on the hyperboloid, but due to
        # numerical errors, we explicitly project back to ensure the constraint is satisfied.
        #
        # For the hyperboloid -x0^2 + ||spatial||^2 = -r^2, we have x0 = sqrt(r^2 + ||spatial||^2).
        # We normalize the result to satisfy this exactly.

        # Extract new spatial coordinates
        new_spatial = x_new_full[..., 1:]

        # Return only spatial coordinates (x0 will be recomputed in project_to_manifold)
        return new_spatial
