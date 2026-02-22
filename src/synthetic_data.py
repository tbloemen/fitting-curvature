"""
Synthetic dataset generators with known intrinsic curvature.

Each generator returns (X, y, D) where:
- X: ambient coordinates (Tensor, shape (n, d))
- y: integer labels (Tensor, shape (n,))
- D: precomputed geodesic distance matrix (Tensor, shape (n, n)) or None

Euclidean generators return D=None since Euclidean distances from coordinates are correct.
Spherical/hyperbolic generators return D with true geodesic distances.
"""

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Euclidean generators (D=None)
# ---------------------------------------------------------------------------


def generate_uniform_grid(n_samples: int) -> tuple[Tensor, Tensor, None]:
    """Uniform random samples in [0,1]^2, labels by quadrant (0-3)."""
    X = torch.rand(n_samples, 2)
    # Labels by quadrant
    y = (X[:, 0] >= 0.5).long() * 2 + (X[:, 1] >= 0.5).long()
    return X, y, None


def generate_gaussian_blob(n_samples: int) -> tuple[Tensor, Tensor, None]:
    """N(0, I) in R^2, labels by median radius (0=inner, 1=outer)."""
    X = torch.randn(n_samples, 2)
    radii = torch.norm(X, dim=1)
    median_r = radii.median()
    y = (radii >= median_r).long()
    return X, y, None


def generate_concentric_circles(n_samples: int) -> tuple[Tensor, Tensor, None]:
    """Two rings at r=1, r=2 with noise, labels by ring (0, 1)."""
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    angles_inner = torch.rand(n_inner) * 2 * torch.pi
    r_inner = 1.0 + 0.1 * torch.randn(n_inner)
    inner = torch.stack(
        [r_inner * torch.cos(angles_inner), r_inner * torch.sin(angles_inner)], dim=1
    )

    angles_outer = torch.rand(n_outer) * 2 * torch.pi
    r_outer = 2.0 + 0.1 * torch.randn(n_outer)
    outer = torch.stack(
        [r_outer * torch.cos(angles_outer), r_outer * torch.sin(angles_outer)], dim=1
    )

    X = torch.cat([inner, outer], dim=0)
    y = torch.cat(
        [torch.zeros(n_inner, dtype=torch.long), torch.ones(n_outer, dtype=torch.long)]
    )
    return X, y, None


# ---------------------------------------------------------------------------
# Spherical generators (D = great-circle distances)
# ---------------------------------------------------------------------------


def _spherical_distances(X: Tensor) -> Tensor:
    """Compute pairwise great-circle distances on the unit sphere."""
    # X is (n, 3) with ||x|| = 1
    # d(x, y) = arccos(clamp(x . y, -1, 1))
    dots = X @ X.t()
    dots = torch.clamp(dots, -1.0, 1.0)
    D = torch.acos(dots)
    D.fill_diagonal_(0.0)
    return D


def generate_uniform_sphere(n_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    """Uniform on S^2 via Marsaglia method, labels by hemisphere (0=south, 1=north)."""
    points = []
    while len(points) < n_samples:
        # Marsaglia: sample (u1, u2) uniform in unit disk, project to sphere
        u = torch.rand(n_samples * 2, 2) * 2 - 1
        s = (u**2).sum(dim=1)
        valid = s < 1.0
        u = u[valid]
        s = s[valid]
        x = 2 * u[:, 0] * torch.sqrt(1 - s)
        y = 2 * u[:, 1] * torch.sqrt(1 - s)
        z = 1 - 2 * s
        batch = torch.stack([x, y, z], dim=1)
        points.append(batch)

    X = torch.cat(points, dim=0)[:n_samples]
    # Normalize for numerical safety
    X = X / X.norm(dim=1, keepdim=True)
    labels = (X[:, 2] >= 0).long()  # hemisphere
    D = _spherical_distances(X)
    return X, labels, D


def _sample_vmf(mu: Tensor, kappa: float, n: int) -> Tensor:
    """Sample from von Mises-Fisher distribution on S^2.

    Uses the Wood (1994) rejection sampling algorithm.
    """
    dim = mu.shape[0]  # 3 for S^2
    m = dim - 1  # 2

    # Step 1: sample w from the marginal distribution
    b = (-2 * kappa + (4 * kappa**2 + m**2) ** 0.5) / m
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + m * torch.log(torch.tensor(1 - x0**2))

    ws = []
    while len(ws) < n:
        batch_size = n * 3
        z = torch.distributions.Beta(m / 2, m / 2).sample((batch_size,))
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = torch.rand(batch_size)
        accept = kappa * w + m * torch.log(1 - x0 * w) - c >= torch.log(u)
        ws.append(w[accept])

    w = torch.cat(ws)[:n]

    # Step 2: sample v uniformly on S^(m-1) (unit circle for m=2)
    angles = torch.rand(n) * 2 * torch.pi
    v = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    # Step 3: combine: x = sqrt(1 - w^2) * v + w * e_1 (in the frame of mu)
    # We need to rotate so that e_1 (north pole [0,0,1]) maps to mu
    sqrt_term = torch.sqrt(torch.clamp(1 - w**2, min=0.0))
    # Points in the coordinate system where mu = [0, 0, 1]
    samples_local = torch.zeros(n, 3)
    samples_local[:, 0] = sqrt_term * v[:, 0]
    samples_local[:, 1] = sqrt_term * v[:, 1]
    samples_local[:, 2] = w

    # Rotate to align [0, 0, 1] -> mu
    mu = mu / mu.norm()
    if torch.allclose(mu, torch.tensor([0.0, 0.0, 1.0])):
        return samples_local / samples_local.norm(dim=1, keepdim=True)
    elif torch.allclose(mu, torch.tensor([0.0, 0.0, -1.0])):
        samples_local[:, 2] = -samples_local[:, 2]
        return samples_local / samples_local.norm(dim=1, keepdim=True)

    # General rotation via Rodrigues' formula
    e3 = torch.tensor([0.0, 0.0, 1.0])
    axis = torch.cross(e3, mu)
    axis = axis / axis.norm()
    cos_angle = mu[2]  # e3 . mu
    sin_angle = torch.sqrt(1 - cos_angle**2)

    # Rotation matrix: R = I*cos + (1-cos)*kk^T + sin*K
    K = torch.tensor(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = (
        cos_angle * torch.eye(3)
        + (1 - cos_angle) * torch.outer(axis, axis)
        + sin_angle * K
    )

    samples = (R @ samples_local.t()).t()
    return samples / samples.norm(dim=1, keepdim=True)


def generate_von_mises_fisher(n_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    """vMF distribution (kappa=10) around north pole, labels by distance from pole."""
    mu = torch.tensor([0.0, 0.0, 1.0])
    X = _sample_vmf(mu, kappa=10.0, n=n_samples)
    # Label by distance from pole: 0=close, 1=far (split at median)
    dist_from_pole = torch.acos(torch.clamp(X[:, 2], -1.0, 1.0))
    median_dist = dist_from_pole.median()
    y = (dist_from_pole >= median_dist).long()
    D = _spherical_distances(X)
    return X, y, D


def generate_antipodal_clusters(n_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    """Two vMF clusters at north and south poles (kappa=10), labels by cluster."""
    n_north = n_samples // 2
    n_south = n_samples - n_north

    north = _sample_vmf(torch.tensor([0.0, 0.0, 1.0]), kappa=10.0, n=n_north)
    south = _sample_vmf(torch.tensor([0.0, 0.0, -1.0]), kappa=10.0, n=n_south)

    X = torch.cat([north, south], dim=0)
    y = torch.cat(
        [torch.zeros(n_north, dtype=torch.long), torch.ones(n_south, dtype=torch.long)]
    )
    D = _spherical_distances(X)
    return X, y, D


# ---------------------------------------------------------------------------
# Hyperbolic generators (D = hyperboloid distances)
# ---------------------------------------------------------------------------


def _poincare_to_hyperboloid(p: Tensor) -> Tensor:
    """Convert Poincaré disk coordinates (n, 2) to hyperboloid (n, 3).

    Hyperboloid model: -x0^2 + x1^2 + x2^2 = -1, x0 > 0
    From Poincaré disk (p1, p2) with ||p||^2 < 1:
        x0 = (1 + ||p||^2) / (1 - ||p||^2)
        xi = 2*pi / (1 - ||p||^2)
    """
    sq_norm = (p**2).sum(dim=1, keepdim=True)  # (n, 1)
    denom = 1 - sq_norm  # (n, 1)
    x0 = (1 + sq_norm) / denom  # (n, 1)
    spatial = 2 * p / denom  # (n, 2)
    return torch.cat([x0, spatial], dim=1)  # (n, 3)


def _hyperboloid_distances(X: Tensor) -> Tensor:
    """Compute pairwise hyperboloid distances.

    d(x, y) = acosh(-<x, y>_L) where <x, y>_L = -x0*y0 + x1*y1 + x2*y2
    """
    # Lorentzian inner product
    L = X.clone()
    L[:, 0] = -L[:, 0]
    inner = X @ L.t()  # this gives -x0*y0 + x1*y1 + x2*y2
    # d = acosh(-inner) but inner is already the Lorentzian product
    # <x,y>_L = -x0*y0 + sum(xi*yi) which is what we computed
    # For unit hyperboloid, <x,x>_L = -1, so -<x,y>_L >= 1
    minus_inner = -inner
    minus_inner = torch.clamp(minus_inner, min=1.0)
    D = torch.acosh(minus_inner)
    D.fill_diagonal_(0.0)
    return D


def generate_uniform_hyperbolic(n_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    """Proper sinh-weighted radial sampling in Poincaré disk, labels by radius bins."""
    # In the Poincaré disk, the area element is (2/(1-r^2))^2 r dr dtheta
    # For uniform sampling by hyperbolic area, sample hyperbolic radius rho
    # uniformly by area: P(rho < R) prop sinh^2(R/2) for curvature K=-1
    # Then convert: poincare_r = tanh(rho / 2)
    max_rho = 3.0  # max hyperbolic radius
    # Sample rho: CDF proportional to cosh(rho) - 1
    u = torch.rand(n_samples)
    # Inverse CDF: cosh(rho) = 1 + u*(cosh(max_rho) - 1)
    cosh_rho = 1 + u * (torch.cosh(torch.tensor(max_rho)) - 1)
    rho = torch.acosh(cosh_rho)

    # Convert to Poincaré disk radius
    poincare_r = torch.tanh(rho / 2)

    angles = torch.rand(n_samples) * 2 * torch.pi
    p = torch.stack(
        [poincare_r * torch.cos(angles), poincare_r * torch.sin(angles)], dim=1
    )

    X = _poincare_to_hyperboloid(p)
    # Labels by radius bins (3 bins)
    y = torch.clamp((rho / max_rho * 3).long(), max=2)
    D = _hyperboloid_distances(X)
    return X, y, D


def generate_tree_structured(n_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    """Regular branching tree embedded in hyperbolic space, labels by depth."""
    # Build a binary tree in the Poincaré disk
    # Each level has 2^depth nodes, placed radially
    import math

    max_depth = max(2, int(math.log2(max(n_samples, 2))))

    points = []
    labels = []

    # Root at origin
    points.append(torch.tensor([0.0, 0.0]))
    labels.append(0)

    for depth in range(1, max_depth + 1):
        n_at_depth = 2**depth
        r = torch.tanh(torch.tensor(depth * 0.8) / 2)
        for i in range(n_at_depth):
            angle = 2 * torch.pi * i / n_at_depth + depth * 0.3  # offset per depth
            p = r * torch.tensor(
                [torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))]
            )
            points.append(p)
            labels.append(min(depth, 4))  # cap labels at 4

            if len(points) >= n_samples:
                break
        if len(points) >= n_samples:
            break

    # Trim or pad to n_samples
    if len(points) > n_samples:
        points = points[:n_samples]
        labels = labels[:n_samples]
    elif len(points) < n_samples:
        # Fill remaining with random points at various depths
        for _ in range(n_samples - len(points)):
            depth = torch.randint(1, max_depth + 1, (1,)).item()
            r = torch.tanh(torch.tensor(depth * 0.8) / 2)
            angle = torch.rand(1).item() * 2 * torch.pi
            p = r * torch.tensor([math.cos(angle), math.sin(angle)])
            points.append(p)
            labels.append(min(depth, 4))

    p_disk = torch.stack(points)
    # Clamp to stay inside disk
    norms = p_disk.norm(dim=1, keepdim=True)
    p_disk = torch.where(norms >= 1.0, p_disk / norms * 0.99, p_disk)

    X = _poincare_to_hyperboloid(p_disk)
    y = torch.tensor(labels, dtype=torch.long)
    D = _hyperboloid_distances(X)
    return X, y, D


def generate_hyperbolic_shells(n_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    """Concentric rings at fixed hyperbolic radii, labels by shell (0, 1, 2)."""
    n_per_shell = n_samples // 3
    n_last = n_samples - 2 * n_per_shell

    shells = []
    labels = []
    for i, (n_pts, rho) in enumerate(
        [(n_per_shell, 0.5), (n_per_shell, 1.5), (n_last, 2.5)]
    ):
        poincare_r = torch.tanh(torch.tensor(rho) / 2)
        noise = 0.05 * torch.randn(n_pts)
        r = torch.clamp(poincare_r + noise, min=0.01, max=0.99)
        angles = torch.rand(n_pts) * 2 * torch.pi
        p = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1)
        shells.append(p)
        labels.append(torch.full((n_pts,), i, dtype=torch.long))

    p_disk = torch.cat(shells, dim=0)
    X = _poincare_to_hyperboloid(p_disk)
    y = torch.cat(labels)
    D = _hyperboloid_distances(X)
    return X, y, D


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

SYNTHETIC_DATASETS = {
    "uniform_grid": generate_uniform_grid,
    "gaussian_blob": generate_gaussian_blob,
    "concentric_circles": generate_concentric_circles,
    "uniform_sphere": generate_uniform_sphere,
    "von_mises_fisher": generate_von_mises_fisher,
    "antipodal_clusters": generate_antipodal_clusters,
    "uniform_hyperbolic": generate_uniform_hyperbolic,
    "tree_structured": generate_tree_structured,
    "hyperbolic_shells": generate_hyperbolic_shells,
}


def load_synthetic(
    dataset: str, n_samples: int = 500
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Load a synthetic dataset by name.

    Parameters
    ----------
    dataset : str
        Dataset name (one of SYNTHETIC_DATASETS keys)
    n_samples : int
        Number of samples to generate

    Returns
    -------
    tuple[Tensor, Tensor, Tensor | None]
        (X, y, D) where D is precomputed geodesic distances or None
    """
    if dataset not in SYNTHETIC_DATASETS:
        raise ValueError(
            f"Unknown synthetic dataset: {dataset}. "
            f"Available: {list(SYNTHETIC_DATASETS.keys())}"
        )
    return SYNTHETIC_DATASETS[dataset](n_samples)
