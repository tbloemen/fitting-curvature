from matplotlib import pyplot as plt

from src.load_data import load_raw_data
from src.visualisation import default_plot, project_to_2d


def main():
    print("Hello from fitting-curvature!")
    X, _ = load_raw_data("mnist")
    X_numpy = X.numpy()
    for k in [-1, 0, 1]:
        x, y = project_to_2d(X_numpy, 0, 1, k)
        default_plot(x, y)
        plt.savefig(f"plots/plot_{k}.png")


if __name__ == "__main__":
    main()
