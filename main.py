from src.embedding import fit_embedding
from src.load_data import load_raw_data
from src.matrices import calculate_distance_matrix
from src.visualisation import default_plot, project_to_2d


def main():
    print("Hello from fitting-curvature!")
    X, _ = load_raw_data("mnist")
    X = X[:1000]

    embed_dim = 10  # Embedding dimension
    n_iterations = 500

    for k in [-1, 0, 1]:
        print(f"\n{'=' * 60}")
        print(f"Training embedding with curvature k = {k}")
        print(f"{'=' * 60}")

        # Calculate distance matrix in original space
        A = calculate_distance_matrix(X)

        # Train the embedding
        model = fit_embedding(
            distance_matrix=A,
            embed_dim=embed_dim,
            curvature=k,
            n_iterations=n_iterations,
            lr=0.01,
            verbose=True,
        )

        # Get the learned embeddings
        embeddings = model.get_embeddings().detach().numpy()

        # Visualize the first two dimensions
        x_proj, y_proj = project_to_2d(embeddings, i=1, j=2, k=k)
        fig = default_plot(x_proj, y_proj)
        fig.suptitle(f"Embedding with curvature k = {k}")
        fig.savefig(f"plots/plot_{k}.png", dpi=150)
        print(f"Saved visualization to plots/plot_{k}.png")


if __name__ == "__main__":
    main()
