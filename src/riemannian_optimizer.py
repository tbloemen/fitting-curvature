"""
Riemannian optimizers for constant curvature spaces.

Implementation following:
- Gu et al. (2019): "Learning Mixed-Curvature Representations in Products of Model Spaces"
  Algorithm 1: R-SGD in products (page 5)
- Nickel & Kiela (2018): "Learning Continuous Hierarchies in the Lorentz Model"
"""

import torch
from torch.optim.optimizer import Optimizer

from src.manifolds import Euclidean, Hyperboloid, Manifold, Sphere


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

        # Create manifold for this optimizer
        self.manifold = self._create_manifold(curvature)

    def _create_manifold(self, curvature: float) -> Manifold:
        """Factory method to create appropriate manifold."""
        if curvature > 0:
            return Sphere(curvature)
        elif curvature == 0:
            return Euclidean(curvature)
        else:
            return Hyperboloid(curvature)

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

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get Euclidean gradient
                h = p.grad.data

                # Project to tangent space
                v = self.manifold.project_to_tangent(p.data, h)

                # Update via exponential map
                p.data = self.manifold.exp_map(p.data, -lr * v)

        return loss


class RiemannianSGDMomentum(RiemannianSGD):
    """
    Riemannian SGD with momentum for t-SNE optimization.

    Extends RiemannianSGD with velocity state and parallel transport
    for momentum on curved manifolds.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float
        Learning rate
    curvature : float
        Curvature of the space
    momentum : float
        Momentum factor (default: 0.5)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        curvature: float = -1.0,
        momentum: float = 0.5,
    ):
        super().__init__(params, lr=lr, curvature=curvature)
        self.momentum = momentum
        self.state_initialized = False

    def set_momentum(self, momentum: float) -> None:
        """
        Set momentum value (used for phase transitions in t-SNE).

        Parameters
        ----------
        momentum : float
            New momentum value
        """
        self.momentum = momentum

    def _parallel_transport_hyperboloid(
        self, x_old: torch.Tensor, x_new: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport velocity from tangent space at x_old to x_new (hyperboloid).

        Uses the closed-form formula for parallel transport along geodesics
        on the hyperboloid with Lorentzian metric.

        Formula: v_transported = v + <x_old, v>_L * (x_old + x_new) / (1 - <x_old, x_new>_L / r²)
        """
        r_sq = self.manifold.radius_squared

        # Lorentzian inner product: <a, b>_L = -a_0*b_0 + a_spatial · b_spatial
        inner_old_v = -x_old[..., 0:1] * v[..., 0:1] + (
            x_old[..., 1:] * v[..., 1:]
        ).sum(dim=-1, keepdim=True)

        inner_old_new = -x_old[..., 0:1] * x_new[..., 0:1] + (
            x_old[..., 1:] * x_new[..., 1:]
        ).sum(dim=-1, keepdim=True)

        # Denominator: 1 - <x_old, x_new>_L / r²
        # For points on hyperboloid, <x_old, x_new>_L <= -r², so this is >= 2
        denom = 1.0 - inner_old_new / r_sq
        denom = torch.clamp(denom, min=1e-8)  # Numerical stability

        # Parallel transport formula
        v_transported = v + inner_old_v * (x_old + x_new) / denom

        # Project to tangent space at x_new for numerical stability
        inner_new_v = -x_new[..., 0:1] * v_transported[..., 0:1] + (
            x_new[..., 1:] * v_transported[..., 1:]
        ).sum(dim=-1, keepdim=True)
        v_transported = v_transported + inner_new_v / r_sq * x_new

        return v_transported

    def _parallel_transport_sphere(
        self, x_old: torch.Tensor, x_new: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport velocity from tangent space at x_old to x_new (sphere).

        Uses the closed-form formula for parallel transport along geodesics
        on the sphere.

        Formula: v_transported = v - (<x_old, v> + <x_new, v>) / (1 + <x_old, x_new> / r²) * (x_old + x_new)
        """
        r_sq = self.manifold.radius_squared

        # Standard inner products
        inner_old_new = (x_old * x_new).sum(dim=-1, keepdim=True)
        inner_old_v = (x_old * v).sum(dim=-1, keepdim=True)
        inner_new_v = (x_new * v).sum(dim=-1, keepdim=True)

        # Denominator: 1 + <x_old, x_new> / r²
        # For points on sphere, -r² <= <x_old, x_new> <= r², so this is in [0, 2]
        denom = 1.0 + inner_old_new / r_sq
        denom = torch.clamp(denom, min=1e-8)  # Numerical stability

        # Parallel transport formula
        v_transported = v - (inner_old_v + inner_new_v) / denom * (x_old + x_new)

        # Project to tangent space at x_new for numerical stability
        inner_new_transported = (x_new * v_transported).sum(dim=-1, keepdim=True)
        v_transported = v_transported - inner_new_transported / r_sq * x_new

        return v_transported

    def _parallel_transport(
        self, x_old: torch.Tensor, x_new: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport velocity from tangent space at x_old to x_new.

        Parameters
        ----------
        x_old : Tensor
            Previous point on manifold
        x_new : Tensor
            New point on manifold
        v : Tensor
            Velocity in tangent space at x_old

        Returns
        -------
        Tensor
            Transported velocity in tangent space at x_new
        """
        if self.manifold.curvature == 0:
            # Euclidean: parallel transport is identity
            return v
        elif self.manifold.curvature > 0:
            return self._parallel_transport_sphere(x_old, x_new, v)
        else:
            return self._parallel_transport_hyperboloid(x_old, x_new, v)

    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore
        """
        Perform a single optimization step with momentum.

        Implements Riemannian SGD with momentum using parallel transport
        to move velocity vectors between tangent spaces.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get Euclidean gradient
                h = p.grad.data

                # Project to tangent space to get Riemannian gradient
                grad = self.manifold.project_to_tangent(p.data, h)

                # Initialize velocity state if needed
                param_state = self.state[p]
                if "velocity" not in param_state:
                    param_state["velocity"] = torch.zeros_like(p.data)
                    param_state["prev_point"] = p.data.clone()

                velocity = param_state["velocity"]
                prev_point = param_state["prev_point"]

                # Parallel transport velocity from previous tangent space to current
                if self.state_initialized:
                    velocity = self._parallel_transport(prev_point, p.data, velocity)

                # Update velocity
                velocity = self.momentum * velocity - lr * grad

                # Store current point before update
                param_state["prev_point"] = p.data.clone()

                # Update via exponential map
                p.data = self.manifold.exp_map(p.data, velocity)

                # Store velocity for next iteration
                param_state["velocity"] = velocity

        self.state_initialized = True
        return loss
