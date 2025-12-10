import torch

from src.embedding import fit_embedding
from src.load_data import load_raw_data
from src.matrices import calculate_distance_matrix
from src.visualisation import default_plot, project_to_2d


def main():
    print("Hello from fitting-curvature!")

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    X, _ = load_raw_data("mnist")
    X = X[:1000]  # Use first 1000 samples to fit in GPU memory
    X = X.to(device)

    embed_dim = 10  # Embedding dimension
    n_iterations = 500

    # Calculate distance matrix in original space (stays on device)
    print("\nCalculating distance matrix...")
    A = calculate_distance_matrix(X)
    print(f"Distance matrix shape: {A.shape}, device: {A.device}")

    for k in [-1, 0, 1]:
        print(f"\n{'=' * 60}")
        print(f"Training embedding with curvature k = {k}")
        print(f"{'=' * 60}")

        # Train the embedding
        model = fit_embedding(
            distance_matrix=A,
            embed_dim=embed_dim,
            curvature=k,
            n_iterations=n_iterations,
            lr=0.01,
            verbose=True,
        )

        # Get the learned embeddings (move to CPU for visualization)
        embeddings = model.get_embeddings().detach().cpu().numpy()

        # Visualize the first two dimensions
        x_proj, y_proj = project_to_2d(embeddings, i=1, j=2, k=k)
        fig = default_plot(x_proj, y_proj)
        fig.suptitle(f"Embedding with curvature k = {k}")
        fig.savefig(f"plots/plot_{k}.png", dpi=150)
        print(f"Saved visualization to plots/plot_{k}.png")


if __name__ == "__main__":
    main()
