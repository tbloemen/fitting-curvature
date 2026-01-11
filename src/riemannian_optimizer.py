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
