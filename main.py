import tomllib
import torch

from src.embedding import fit_embedding
from src.load_data import load_raw_data
from src.matrices import get_default_init_scale, normalize_data
from src.metrics import evaluate_embedding
from src.types import InitMethod
from src.visualisation import default_plot, project_to_2d


def load_config(config_path: str = "config.toml") -> dict:
    """
    Load configuration from TOML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def main():
    config = load_config()

    data_config = config["data"]
    embedding_config = config["embedding"]
    hyperparam_config = config["hyperparameters"]
    experiment_config = config["experiments"]
    evaluation_config = config["evaluation"]

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    n_samples = data_config["n_samples"]
    X, y, D = load_raw_data(data_config["dataset"], n_samples=n_samples)
    if n_samples > 0 and n_samples < len(X):
        X = X[:n_samples]
        y = y[:n_samples]
        if D is not None:
            D = D[:n_samples, :n_samples]
    X = X.to(device)
    print(f"\nLoaded {len(X)} samples with {X.shape[1]} features")

    embed_dim = embedding_config["embed_dim"]
    n_iterations = embedding_config["n_iterations"]
    init_method = InitMethod(embedding_config.get("init_method", "pca"))
    perplexity = embedding_config.get("perplexity", 30.0)
    early_exaggeration_iterations = embedding_config.get(
        "early_exaggeration_iterations", 250
    )
    early_exaggeration_factor = embedding_config.get("early_exaggeration_factor", 12.0)
    momentum_early = embedding_config.get("momentum_early", 0.5)
    momentum_main = embedding_config.get("momentum_main", 0.8)

    # Normalize data so mean pairwise distance = 1
    print("\nNormalizing data...")
    if D is not None:
        mask = ~torch.eye(D.shape[0], dtype=torch.bool)
        mean_d = D[mask].mean()
        D = D / mean_d
        print(f"Normalized precomputed distances (divided by {mean_d:.4f})")
    else:
        X = normalize_data(X, verbose=True)

    # Get initialization scale
    init_scale_value = hyperparam_config["init_scale"]
    if init_scale_value == "auto":
        init_scale = get_default_init_scale(embed_dim)
        print(f"Using default init_scale: {init_scale:.4f}")
    else:
        init_scale = init_scale_value
        print(f"Using manual init_scale: {init_scale}")

    curvatures = experiment_config["curvatures"]
    learning_rates = hyperparam_config["learning_rates"]
    n_neighbors = evaluation_config["n_neighbors"]

    # Get visualization settings
    viz_config = config.get("visualization", {})
    spherical_projection = viz_config.get("spherical_projection", "orthographic")

    for k in curvatures:
        print(f"\n{'=' * 60}")
        print(f"Training t-SNE embedding with curvature k = {k}")
        print(f"Perplexity: {perplexity}")
        print(f"Initialization method: {init_method.value}")
        print(f"{'=' * 60}")

        # Get learning rate for this curvature
        lr_key = str(k)
        if lr_key in learning_rates:
            lr = learning_rates[lr_key]
        else:
            lr = learning_rates["k"]
            print(f"Using default learning rate: {lr}")

        # Train the embedding
        model = fit_embedding(
            data=X,
            embed_dim=embed_dim,
            curvature=k,
            device=device,
            perplexity=perplexity,
            n_iterations=n_iterations,
            early_exaggeration_iterations=early_exaggeration_iterations,
            early_exaggeration_factor=early_exaggeration_factor,
            learning_rate=lr,
            momentum_early=momentum_early,
            momentum_main=momentum_main,
            init_method=init_method,
            init_scale=init_scale,
            verbose=True,
            precomputed_distances=D,
        )

        # Get the learned embeddings (move to CPU for visualization)
        embeddings = model.get_embeddings().detach().cpu().numpy()

        # Calculate embedding quality metrics
        X_cpu = X.detach().cpu().numpy()
        trust_score, continuity_score = evaluate_embedding(
            X_cpu, embeddings, n_neighbors=n_neighbors
        )
        print(f"Trustworthiness: {trust_score:.4f}")
        print(f"Continuity: {continuity_score:.4f}")

        # Visualize the first two dimensions
        x_proj, y_proj = project_to_2d(
            embeddings,
            k=k,
            projection=spherical_projection if k > 0 else "stereographic",
        )
        y_labels = y.detach().cpu().numpy() if hasattr(y, "detach") else y
        fig = default_plot(x_proj, y_proj, labels=y_labels)

        # Add projection info to title for spherical embeddings
        proj_info = f" ({spherical_projection} projection)" if k > 0 else ""
        fig.suptitle(f"Embedding with curvature k = {k}{proj_info}")
        fig.savefig(f"plots/plot_{k}.png", dpi=150, bbox_inches="tight")
        print(f"Saved visualization to plots/plot_{k}.png")


if __name__ == "__main__":
    main()
