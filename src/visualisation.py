import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def project_to_2d(
    X: np.ndarray, i: int, j: int, k: float, pole_axis: int | None = None
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

    if pole_axis is None:
        pole_axis = X.shape[1] - 1

    if k > 0:
        # Spherical stereographic
        r = 1.0 / np.sqrt(k)  # sphere radius
        z = X[:, pole_axis]  # coord used as pole
        denom = r - z
        # Avoid numerical blowups:
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        scale = r / denom

        return scale * X[:, i], scale * X[:, j]

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

        return X[:, i] / denom, X[:, j] / denom


def default_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker=".", s=1)
    ax.add_patch(Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    return fig
