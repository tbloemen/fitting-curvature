import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


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
        Indices of the two features to visualise.
    k : float
        Curvature. >0 spherical, 0 Euclidean, <0 hyperbolic.
    pole_axis : int or None
        For spherical stereographic projection: the coordinate chosen as the projection pole.
        If None, defaults to the first axis (0) for spherical.
    projection : str
        Projection method for spherical embeddings (k > 0). Options:
        - "stereographic": Conformal projection (default)
        - "azimuthal_equidistant": Preserves distances from center point
        - "orthographic": Globe-like view (shows one hemisphere)
        - "direct": Simply use spatial coordinates (simplest, bounded by radius)
        Ignored for k <= 0.

    Returns
    -------
    (x, y) : array
        2D projected coordinates, scaled to fit within the unit circle.

    Notes
    -----
    For k > 0 (spherical): Multiple projection methods available.
        - Stereographic: Conformal but distorts distances near pole
        - Azimuthal equidistant: Preserves radial distances from center
        - Orthographic: Natural globe view, may hide back hemisphere
        - Direct: Just uses spatial coordinates, simplest option

    For k = 0 (Euclidean): Directly uses two coordinates, rescaled to fit within the
        unit circle.

    For k < 0 (hyperbolic): Uses projection from hyperboloid to Poincaré disk, which
        naturally maps to the unit disk.
    """

    if k > 0:
        # Spherical projections
        # Points live on sphere of radius r = 1/sqrt(k) in R^(d+1)
        r = 1.0 / np.sqrt(k)  # sphere radius

        if projection == "stereographic":
            # Stereographic projection: conformal, maps to infinite plane
            if pole_axis is None:
                pole_axis = 0
            z = X[:, pole_axis]  # coord used as pole
            denom = r - z
            # Avoid numerical blowups:
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            scale = r / denom

            x_proj = scale * X[:, i + 1]
            y_proj = scale * X[:, j + 1]

            # Rescale to fit in unit circle
            max_dist = np.sqrt(x_proj**2 + y_proj**2).max()
            if max_dist > 0:
                scale_factor = 0.95 / max_dist
                x_proj *= scale_factor
                y_proj *= scale_factor

        elif projection == "azimuthal_equidistant":
            # Azimuthal equidistant: preserves distances from center point
            # Center at "south pole" (x_0 = -r)
            if pole_axis is None:
                pole_axis = 0

            # Angular distance from south pole
            x0 = X[:, pole_axis]
            theta = np.arccos(np.clip(-x0 / r, -1.0, 1.0))  # angle from south pole

            # Get spatial coordinates for azimuthal angle
            spatial_i = (
                X[:, i + 1]
                if i + 1 != pole_axis
                else X[:, i + 2 if i + 2 < X.shape[1] else 1]
            )
            spatial_j = (
                X[:, j + 1]
                if j + 1 != pole_axis
                else X[:, j + 2 if j + 2 < X.shape[1] else 1]
            )

            phi = np.arctan2(spatial_j, spatial_i)

            # Project: radius is proportional to angular distance
            # Maximum theta is π (north pole), so max radius is πr
            # Scale to fit in unit circle
            scale_factor = 0.95 / (np.pi * r)
            x_proj = theta * np.cos(phi) * scale_factor * r
            y_proj = theta * np.sin(phi) * scale_factor * r

        elif projection == "orthographic":
            # Orthographic: globe-like view, shows one hemisphere
            # Simply use two spatial coordinates, points naturally bounded by r
            x_proj = X[:, i + 1]
            y_proj = X[:, j + 1]

            # Optionally fade out points on back hemisphere
            # (where x_pole_axis is positive, assuming viewing from negative side)
            if pole_axis is None:
                pole_axis = 0

            # Scale to fit in unit circle
            max_dist = np.sqrt(x_proj**2 + y_proj**2).max()
            if max_dist > 0:
                scale_factor = 0.95 / max_dist
                x_proj *= scale_factor
                y_proj *= scale_factor

        elif projection == "direct":
            # Direct spatial coordinates: simplest option, naturally bounded
            # Just use two spatial coordinates directly
            x_proj = X[:, i + 1]
            y_proj = X[:, j + 1]

            # Points on sphere of radius r satisfy: x_0^2 + x_1^2 + ... = r^2
            # So each coordinate is bounded by [-r, r]
            # Scale to unit circle
            max_dist = np.sqrt(x_proj**2 + y_proj**2).max()
            if max_dist > 0:
                scale_factor = 0.95 / max_dist
                x_proj *= scale_factor
                y_proj *= scale_factor

        else:
            raise ValueError(
                f"Unknown projection '{projection}'. "
                f"Choose from: stereographic, azimuthal_equidistant, orthographic, direct"
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
