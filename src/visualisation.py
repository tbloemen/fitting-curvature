import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def project_to_2d(
    X: np.ndarray, k: float, i: int = 0, j: int = 1, pole_axis: int | None = None
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
        If None, defaults to the last axis (F-1).

    Returns
    -------
    (x, y) : array
        2D projected coordinates.
    """

    if k > 0:
        # Spherical stereographic
        r = 1.0 / np.sqrt(k)  # sphere radius
        if pole_axis is None:
            pole_axis = 0
        z = X[:, pole_axis]  # coord used as pole
        denom = r - z
        # Avoid numerical blowups:
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        scale = r / denom

        return scale * X[:, i + 1], scale * X[:, j + 1]

    elif k == 0:
        # Euclidean projection
        return X[:, i].astype(float), X[:, j].astype(float)

    else:
        # Hyperbolic (hyperboloid -> Poincaré disk)
        # X[:,0] must be the time-like coordinate
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
