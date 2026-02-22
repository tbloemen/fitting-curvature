import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def _align_sphere_to_centroid(X: np.ndarray) -> np.ndarray:
    """Rotate spherical points so the data centroid aligns with -e_0 (south pole).

    This ensures projections are centered on the data's center of mass,
    regardless of the original coordinate orientation. Uses a Householder
    reflection to map the centroid direction to -e_0.

    Parameters
    ----------
    X : array, shape (N, d)
        Points on a sphere in R^d.

    Returns
    -------
    X_rotated : array, shape (N, d)
        Rotated points with centroid direction at -e_0.
    """
    d = X.shape[1]
    centroid = X.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-10:
        # Data is roughly uniform/centered — no preferred direction.
        # Fall back to axis with highest variance.
        variances = X.var(axis=0)
        # Use the axis with LOWEST variance as the pole (highest variance
        # axes become the spatial/visible axes).
        pole_idx = int(np.argmin(variances))
        if pole_idx == 0:
            return X  # already aligned
        # Swap pole_idx with axis 0
        X_swapped = X.copy()
        X_swapped[:, 0] = X[:, pole_idx]
        X_swapped[:, pole_idx] = X[:, 0]
        return X_swapped

    c = centroid / centroid_norm

    # Householder reflection: H maps c -> -e_0
    # H = I - 2 * vv^T / (v^T v)  where v = c + e_0
    e0 = np.zeros(d)
    e0[0] = 1.0
    v = c + e0
    v_dot = np.dot(v, v)
    if v_dot < 1e-15:
        # c ≈ -e_0 already, no rotation needed
        return X

    # Apply Householder: H @ x = x - 2 * v * (v^T x) / (v^T v)
    vTX = X @ v  # (N,)
    X_rotated = X - 2.0 * np.outer(vTX, v) / v_dot

    return X_rotated


def project_to_2d(
    X: np.ndarray,
    k: float,
    i: int = 0,
    j: int = 1,
    pole_axis: int | None = None,
    projection: str = "stereographic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project F-dimensional constant-curvature points onto 2D using two coordinates.

    Parameters
    ----------
    X : array, shape (N, F)
        Input points in either Euclidean, spherical, or hyperbolic form.
    i, j : int
        Indices of the two spatial features to visualise (0-indexed, excluding pole).
    k : float
        Curvature. >0 spherical, 0 Euclidean, <0 hyperbolic.
    pole_axis : int or None
        For spherical projections: the coordinate chosen as the projection pole.
        If None (default), uses data-driven alignment: rotates points so the
        centroid direction becomes the pole axis, centering the projection on
        where the data actually lives.
    projection : str
        Projection method for spherical embeddings (k > 0). Options:
        - "stereographic": Conformal projection (default)
        - "azimuthal_equidistant": Preserves distances from center point
        - "orthographic": Globe-like view (shows one hemisphere)
        Ignored for k <= 0.

    Returns
    -------
    (x, y) : array
        2D projected coordinates, scaled to fit within the unit circle.

    Notes
    -----
    For k > 0 (spherical): Multiple projection methods available.
        When pole_axis is None, the data is rotated so that the centroid
        direction maps to the south pole (-e_0). This means:
        - Stereographic projects from the antipodal point of the centroid,
          centering the view on the data.
        - Azimuthal equidistant preserves distances from the data center.
        - Orthographic views the sphere from the centroid direction.

    For k = 0 (Euclidean): Directly uses two coordinates, rescaled to fit within the
        unit circle.

    For k < 0 (hyperbolic): Uses projection from hyperboloid to Poincaré disk, which
        naturally maps to the unit disk.
    """

    if k > 0:
        # Spherical projections
        # Points live on sphere of radius r = 1/sqrt(k) in R^(d+1)
        r = 1.0 / np.sqrt(k)  # sphere radius

        if pole_axis is None:
            # Data-driven: rotate so centroid -> -e_0 (south pole)
            X = _align_sphere_to_centroid(X)
            pole_axis = 0

        spatial_axes = [a for a in range(X.shape[1]) if a != pole_axis]
        x_proj = X[:, spatial_axes[i]]
        y_proj = X[:, spatial_axes[j]]

        if projection == "stereographic":
            # Stereographic projection from pole (pole_axis = +r)
            # Centroid (at -e_0 after rotation) maps to origin
            z = X[:, pole_axis]  # coord used as pole
            denom = r - z
            # Avoid numerical blowups:
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            scale = r / denom

            # Spatial coordinates (all axes except pole)
            x_proj = scale * x_proj
            y_proj = scale * y_proj

            # Rescale to fit in unit circle
            max_dist = np.sqrt(x_proj**2 + y_proj**2).max()
            if max_dist > 0:
                scale_factor = 0.95 / max_dist
                x_proj *= scale_factor
                y_proj *= scale_factor

        elif projection == "azimuthal_equidistant":
            # Azimuthal equidistant: preserves distances from south pole
            # After rotation, south pole = centroid -> projection centered on data
            x0 = X[:, pole_axis]
            theta = np.arccos(np.clip(-x0 / r, -1.0, 1.0))  # angle from south pole
            phi = np.arctan2(y_proj, x_proj)

            # Project: radius proportional to angular distance
            scale_factor = 0.95 / (np.pi * r)
            x_proj = theta * np.cos(phi) * scale_factor * r
            y_proj = theta * np.sin(phi) * scale_factor * r

        elif projection == "orthographic":
            # Orthographic: globe-like view looking along pole axis
            # Scale to fit in unit circle
            max_dist = np.sqrt(x_proj**2 + y_proj**2).max()
            if max_dist > 0:
                scale_factor = 0.95 / max_dist
                x_proj *= scale_factor
                y_proj *= scale_factor

        else:
            raise ValueError(
                f"Unknown projection '{projection}'. "
                f"Choose from: stereographic, azimuthal_equidistant, orthographic"
            )

        return x_proj, y_proj

    elif k == 0:
        # Euclidean projection: just take two coordinates
        x_proj = X[:, i].astype(float)
        y_proj = X[:, j].astype(float)

        # Rescale to fit in unit circle
        max_dist = np.sqrt(x_proj**2 + y_proj**2).max()
        if max_dist > 0:
            scale_factor = 0.95 / max_dist  # 0.95 to leave small margin
            x_proj *= scale_factor
            y_proj *= scale_factor

        return x_proj, y_proj

    else:
        # Hyperbolic (hyperboloid -> Poincaré disk)
        # X[:,0] must be the time-like coordinate
        # This projection naturally maps to the unit disk
        r = 1.0 / np.sqrt(-k)
        x0 = X[:, 0]

        denom = x0 + r
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        return X[:, i + 1] / denom, X[:, j + 1] / denom


def default_plot(x, y, labels=None):
    """
    Create a scatter plot with optional label-based coloring.

    Parameters
    ----------
    x, y : array
        2D coordinates to plot
    labels : array or None
        Optional labels for coloring points. If provided, points are colored by label.
        For MNIST, labels should be 0-9 for the 10 digit classes.

    Returns
    -------
    fig : matplotlib figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if labels is not None:
        # Define a colormap for 10 MNIST digit classes
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.arange(10))

        # Plot each label separately to get proper legend
        unique_labels = np.unique(labels)
        for label in sorted(unique_labels):
            mask = labels == label
            ax.scatter(
                x[mask],
                y[mask],
                c=[colors[int(label)]],
                label=f"Digit {int(label)}",
                marker=".",
                s=1,
                alpha=0.7,
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(x, y, marker=".", s=1)

    ax.add_patch(Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    return fig
