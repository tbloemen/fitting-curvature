import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/experiment_1.csv", sep=";")

curvature_labels = {1: "Spherical (k=1)", 0: "Euclidean (k=0)", -1: "Hyperbolic (k=-1)"}
df["Curvature Label"] = df["Curvature"].map(curvature_labels)

datasets = ["uniform_sphere", "uniform_grid", "uniform_hyperbolic"]
curvatures = [1, 0, -1]

all_metrics = [
    "Trustworthiness",
    "Continuity",
    "K-NN Overlap",
    "Geodesic distortion GU",
    "Geodesic distortion MSE",
    "Area Utilisation",
    "Radial Distribution",
    "Silhouette score",
    "Davies-Bouldin Index",
    "Dunn Index",
]
log_scale_metrics = {"Geodesic distortion GU", "Geodesic distortion MSE"}

x = np.arange(len(curvatures))
width = 0.25

for metric in all_metrics:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, dataset in enumerate(datasets):
        subset = df[df["Dataset"] == dataset]
        values = [
            subset[subset["Curvature"] == k][metric].values[0] for k in curvatures
        ]
        ax.bar(x + i * width, values, width, label=dataset)
    ax.set_title(metric, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [curvature_labels[k] for k in curvatures], rotation=20, ha="right"
    )
    if metric in log_scale_metrics:
        ax.set_yscale("log")
    if metric == "Trustworthiness" or metric == "Continuity":
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.legend(fontsize=8)
    fig.tight_layout()
    safe_name = metric.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(f"data/experiment_1_{safe_name}.svg", dpi=150)

plt.show()
print("Plots saved to data/experiment_1_*.svg")
