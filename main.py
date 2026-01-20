import tomllib

import torch

from src.embedding import InitMethod, LossType, fit_embedding
from src.load_data import load_raw_data
from src.matrices import get_default_init_scale, normalize_data
from src.metrics import evaluate_embedding
from src.samplers import SamplerType
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
    batching_config = config.get("batching", {})

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    X, y = load_raw_data(data_config["dataset"])
    n_samples = data_config["n_samples"]
    if n_samples > 0:
        X = X[:n_samples]
        y = y[:n_samples]
    X = X.to(device)
    print(f"\nLoaded {len(X)} samples with {X.shape[1]} features")

    embed_dim = embedding_config["embed_dim"]
    n_iterations = embedding_config["n_iterations"]
    loss_type = LossType(embedding_config["loss_type"])
    init_method = InitMethod(embedding_config.get("init_method", "random"))

    # Normalize data so mean pairwise distance = 1
    print("\nNormalizing data...")
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

    # Extract batching parameters
    batch_size = batching_config.get("batch_size", 4096)
    sampler_type = SamplerType(batching_config.get("sampler_type", "random"))
    sampler_params = batching_config.get("sampler_params", {})

    for k in curvatures:
        print(f"\n{'=' * 60}")
        print(f"Training embedding with curvature k = {k}")
        print(f"Loss function: {loss_type.value}")
        print(f"Initialization method: {init_method.value}")
        print(f"Batch size: {batch_size}, sampler: {sampler_type.value}")
        print(f"{'=' * 60}")

        # Get learning rate for this curvature
        lr_key = str(k)
        if lr_key in learning_rates:
            lr = learning_rates[lr_key]
        else:
            lr = learning_rates["k"]
            print(f"Using default learning rate: {lr}")

        # Train the embedding (distances computed on-the-fly from raw data)
        model = fit_embedding(
            data=X,
            embed_dim=embed_dim,
            curvature=k,
            n_iterations=n_iterations,
            device=device,
            lr=lr,
            verbose=True,
            init_scale=init_scale,
            loss_type=loss_type,
            sampler_type=sampler_type,
            batch_size=batch_size,
            sampler_kwargs=sampler_params,
            init_method=init_method,
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
        x_proj, y_proj = project_to_2d(embeddings, k=k)
        y_labels = y.detach().cpu().numpy() if hasattr(y, "detach") else y
        fig = default_plot(x_proj, y_proj, labels=y_labels)
        fig.suptitle(f"Embedding with curvature k = {k}")
        fig.savefig(f"plots/plot_{k}.png", dpi=150, bbox_inches="tight")
        print(f"Saved visualization to plots/plot_{k}.png")


if __name__ == "__main__":
    main()
