import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.affinities import compute_perplexity_affinities
from src.kernels import compute_q_matrix
from src.manifolds import Euclidean, Hyperboloid, Manifold, Sphere
from src.riemannian_optimizer import RiemannianSGDMomentum
from src.types import InitMethod


def compute_tsne_kl_loss(Q: Tensor, V: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute KL divergence loss for t-SNE.

    KL(V || Q) = sum_i sum_j V_ij * log(V_ij / Q_ij)
               = sum(V * log(V)) - sum(V * log(Q))

    Since we're optimizing, we only need: -sum(V * log(Q))
    (the V * log(V) term is constant w.r.t. embedding)

    Parameters
    ----------
    Q : Tensor, shape (n_points, n_points)
        Low-dimensional affinity matrix from t-distribution kernel
    V : Tensor, shape (n_points, n_points)
        High-dimensional affinity matrix from perplexity computation
    eps : float
        Small constant to avoid log(0)

    Returns
    -------
    Tensor
        Scalar KL divergence loss
    """
    # Only compute the part that depends on Q
    # Mask out diagonal (self-affinities should be 0)
    mask = ~torch.eye(Q.shape[0], dtype=torch.bool, device=Q.device)
    V_masked = V[mask]
    Q_masked = Q[mask]

    loss = -torch.sum(V_masked * torch.log(Q_masked + eps))
    return loss


class ConstantCurvatureEmbedding(nn.Module):
    """
    Learn embeddings in constant curvature spaces using t-SNE.

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
        init_scale: float,
        device: torch.device,
        init_method: InitMethod = InitMethod.RANDOM,
        data: Tensor | None = None,
    ):
        super().__init__()
        self.n_points = n_points
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.device = device

        # Create appropriate manifold
        self.manifold = self._create_manifold(curvature)

        # Initialize points on manifold
        points_init = self.manifold.init_points(
            n_points, embed_dim, init_scale, self.device, init_method, data
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
    data: Tensor,
    embed_dim: int,
    curvature: float,
    device: torch.device,
    perplexity: float = 30.0,
    n_iterations: int = 1000,
    early_exaggeration_iterations: int = 250,
    early_exaggeration_factor: float = 12.0,
    learning_rate: float = 200.0,
    momentum_early: float = 0.5,
    momentum_main: float = 0.8,
    init_method: InitMethod = InitMethod.PCA,
    init_scale: float = 0.0001,
    verbose: bool = True,
) -> ConstantCurvatureEmbedding:
    """
    Fit a t-SNE embedding in constant curvature space.

    t-SNE (t-distributed Stochastic Neighbor Embedding) uses probability
    distributions to preserve local neighborhood structure. This implementation
    extends t-SNE to curved spaces (hyperbolic and spherical).

    The algorithm has two phases:
    1. Early exaggeration (default 250 iterations): V is multiplied by 12,
       momentum=0.5. This encourages clusters to form.
    2. Main optimization (remaining iterations): Normal V, momentum=0.8.
       Fine-tunes the embedding.

    Parameters
    ----------
    data : Tensor, shape (N, D)
        Input data points in original space
    embed_dim : int
        Embedding dimensionality
    curvature : float
        Curvature of the space (k > 0: sphere, k = 0: Euclidean, k < 0: hyperbolic)
    device : torch.device
        Device to use for computation
    perplexity : float
        Perplexity parameter (effective number of neighbors). Default: 30.
    n_iterations : int
        Total number of optimization iterations. Default: 1000.
    early_exaggeration_iterations : int
        Number of iterations for early exaggeration phase. Default: 250.
    early_exaggeration_factor : float
        Factor to multiply V by during early exaggeration. Default: 12.0.
    learning_rate : float
        Learning rate for optimization. Default: 200.0 (standard t-SNE value).
    momentum_early : float
        Momentum during early exaggeration. Default: 0.5.
    momentum_main : float
        Momentum during main optimization. Default: 0.8.
    init_method : InitMethod
        Initialization method. Default: PCA (recommended for t-SNE).
    init_scale : float
        Initialization scale for random init. Default: 0.0001.
    verbose : bool
        Print progress information. Default: True.

    Returns
    -------
    ConstantCurvatureEmbedding
        Fitted t-SNE embedding model

    References
    ----------
    - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
    """
    # Move data to device
    data = data.to(device)
    n_points = data.shape[0]

    if verbose:
        manifold_name = (
            "Euclidean"
            if curvature == 0
            else ("hyperbolic" if curvature < 0 else "spherical")
        )
        print(f"Fitting t-SNE embedding in {manifold_name} space (k={curvature})")
        print(f"  {n_points} points, embed_dim={embed_dim}, perplexity={perplexity}")

    # Step 1: Compute high-dimensional affinities
    if verbose:
        print("Computing high-dimensional affinities...")
    V = compute_perplexity_affinities(data, perplexity=perplexity, verbose=verbose)
    V = V.to(device)

    # Step 2: Initialize embedding
    model = ConstantCurvatureEmbedding(
        n_points=n_points,
        embed_dim=embed_dim,
        curvature=curvature,
        init_scale=init_scale,
        device=device,
        init_method=init_method,
        data=data,
    )

    # Step 3: Create optimizer with momentum
    optimizer = RiemannianSGDMomentum(
        model.parameters(),
        lr=learning_rate,
        curvature=curvature,
        momentum=momentum_early,
    )

    if verbose:
        print(f"Starting optimization: {n_iterations} iterations")
        print(
            f"  Early exaggeration: {early_exaggeration_iterations} iterations, "
            f"factor={early_exaggeration_factor}, momentum={momentum_early}"
        )
        print(
            f"  Main phase: {n_iterations - early_exaggeration_iterations} iterations, "
            f"momentum={momentum_main}"
        )

    # Training loop
    pbar = tqdm(range(n_iterations), disable=(not verbose), desc="t-SNE")

    for iteration in pbar:
        # Phase transition
        if iteration == early_exaggeration_iterations:
            optimizer.set_momentum(momentum_main)
            if verbose:
                pbar.set_description("t-SNE (main)")

        optimizer.zero_grad()

        # Determine V to use (exaggerated or normal)
        if iteration < early_exaggeration_iterations:
            V_current = V * early_exaggeration_factor
        else:
            V_current = V

        # Compute Q matrix using geodesic distances
        Q = compute_q_matrix(model.manifold, model.points, dof=1.0)

        # Compute KL divergence loss
        loss = compute_tsne_kl_loss(Q, V_current)

        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return model
