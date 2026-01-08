import torch
from torch import Tensor


def calculate_distance_matrix(
    X: Tensor, sparsity_threshold: float = 1.0, chunk_size: int = 1000
) -> Tensor:
    """
    Compute pairwise Euclidean distance matrix efficiently using chunked operations.

    This uses the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>

    For large datasets, the computation is performed in chunks to avoid memory overflow.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points where N is number of points and D is dimensionality
    sparsity_threshold : float, default=1.0
        Target percentage of distances to keep (0.0 to 1.0).
        For example, 0.1 keeps only the 10% smallest distances, zeros out the rest.
        A value of 1.0 keeps all distances (no sparsification).
    chunk_size : int, default=1000
        Number of points to process at a time. Adjust based on available GPU memory.

    Returns
    -------
    Tensor, shape (N, N)
        Pairwise distance matrix on the same device as X, potentially sparse.
        If sparsity_threshold < 1.0, returns a sparse COO tensor.
    """
    n_points = X.shape[0]
    device = X.device

    # Compute squared norms for each point: ||x||^2
    # Shape: (N, 1)
    squared_norms = (X**2).sum(dim=1, keepdim=True)

    # Decide computation strategy based on size and sparsity
    if sparsity_threshold < 1.0:
        # For sparse matrices, use chunked sparse computation
        distances = _compute_distance_matrix_sparse_chunked(
            X, squared_norms, sparsity_threshold, chunk_size, device
        )
    elif n_points <= chunk_size:
        # For small dense datasets, compute distance matrix directly
        # Compute dot products: X @ X^T
        # Shape: (N, N)
        dot_products = torch.mm(X, X.t())

        # Use the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        # Broadcasting: (N, 1) + (1, N) - (N, N) = (N, N)
        squared_distances = squared_norms + squared_norms.t() - 2 * dot_products

        # Clamp to avoid negative values due to numerical errors
        squared_distances = torch.clamp(squared_distances, min=0.0)

        # Take square root to get Euclidean distances
        distances = torch.sqrt(squared_distances)
    else:
        # For large dense datasets, compute in chunks to avoid OOM
        distances = _compute_distance_matrix_chunked(
            X, squared_norms, chunk_size, device
        )

    return distances


def _compute_distance_matrix_chunked(
    X: Tensor, squared_norms: Tensor, chunk_size: int, device: torch.device
) -> Tensor:
    """
    Compute distance matrix in chunks to avoid memory overflow.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points
    squared_norms : Tensor, shape (N, 1)
        Pre-computed squared norms
    chunk_size : int
        Number of points to process at a time
    device : torch.device
        Device to use for computation

    Returns
    -------
    Tensor, shape (N, N)
        Pairwise distance matrix
    """
    n_points = X.shape[0]
    distances = torch.zeros(n_points, n_points, device=device)

    # Process rows in chunks
    for i in range(0, n_points, chunk_size):
        end_i = min(i + chunk_size, n_points)
        chunk_i = X[i:end_i]  # shape: (chunk_size, D)

        # Compute dot products for this chunk
        # chunk_i @ X^T gives shape (chunk_size, N)
        dot_products_chunk = torch.mm(chunk_i, X.t())

        # Use the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        # squared_norms[i:end_i] has shape (chunk_size, 1)
        # squared_norms.t() has shape (1, N)
        # dot_products_chunk has shape (chunk_size, N)
        squared_distances_chunk = (
            squared_norms[i:end_i] + squared_norms.t() - 2 * dot_products_chunk
        )

        # Clamp and take square root
        squared_distances_chunk = torch.clamp(squared_distances_chunk, min=0.0)
        distances[i:end_i] = torch.sqrt(squared_distances_chunk)

    return distances


def _compute_distance_matrix_sparse_chunked(
    X: Tensor,
    squared_norms: Tensor,
    sparsity_threshold: float,
    chunk_size: int,
    device: torch.device,
) -> Tensor:
    """
    Compute sparse distance matrix in chunks to avoid memory overflow.

    Uses a 2D chunking strategy with incremental sparsification to minimize
    peak memory usage by never creating large intermediate tensors.

    Parameters
    ----------
    X : Tensor, shape (N, D)
        Input data points
    squared_norms : Tensor, shape (N, 1)
        Pre-computed squared norms
    sparsity_threshold : float
        Percentage of smallest distances to keep per row
    chunk_size : int
        Number of points to process at a time
    device : torch.device
        Device to use for computation

    Returns
    -------
    Tensor
        Sparse COO tensor with shape (N, N)
    """
    n_points = X.shape[0]

    # Calculate the number of elements to keep per row
    elements_per_row = max(1, int(n_points * sparsity_threshold))
    k = min(elements_per_row, n_points)

    # Storage for sparse tensor indices and values
    # Use CPU for accumulation to save GPU memory
    all_row_indices = []
    all_col_indices = []
    all_values = []

    # Process rows in chunks
    print(
        f"Computing sparse distance matrix ({n_points} points, {k} neighbors per point)..."
    )
    for row_idx, i in enumerate(range(0, n_points, chunk_size)):
        end_i = min(i + chunk_size, n_points)
        chunk_size_actual = end_i - i
        chunk_i = X[i:end_i]  # shape: (chunk_size, D)
        chunk_norms_i = squared_norms[i:end_i]  # shape: (chunk_size, 1)

        # For each row in this chunk, we'll incrementally find k nearest neighbors
        # Initialize with very large distances
        top_distances = torch.full(
            (chunk_size_actual, k), float("inf"), device=device, dtype=X.dtype
        )
        top_indices = torch.zeros(
            (chunk_size_actual, k), device=device, dtype=torch.long
        )

        # Process columns in chunks to avoid large intermediate tensors
        for j in range(0, n_points, chunk_size):
            end_j = min(j + chunk_size, n_points)
            chunk_j = X[j:end_j]  # shape: (col_chunk_size, D)
            chunk_norms_j = squared_norms[j:end_j]  # shape: (col_chunk_size, 1)

            # Compute dot products for this sub-block
            # Shape: (chunk_size_actual, col_chunk_size)
            dot_products_block = torch.mm(chunk_i, chunk_j.t())

            # Compute squared distances for this block
            # Broadcasting: (chunk_size, 1) + (1, col_chunk_size) - (chunk_size, col_chunk_size)
            squared_distances_block = (
                chunk_norms_i + chunk_norms_j.t() - 2 * dot_products_block
            )

            # Clamp and take square root
            squared_distances_block = torch.clamp(squared_distances_block, min=0.0)
            distances_block = torch.sqrt(squared_distances_block)

            # Create column indices for this block
            col_indices_block = (
                torch.arange(j, end_j, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(chunk_size_actual, -1)
            )

            # Merge with existing top-k values
            # For each row, we combine existing top-k with new block values
            combined_distances = torch.cat([top_distances, distances_block], dim=1)
            combined_indices = torch.cat([top_indices, col_indices_block], dim=1)

            # Keep only the k smallest
            new_top_distances, new_top_order = torch.topk(
                combined_distances, k, dim=1, largest=False
            )

            # Reorder the column indices to match the new top-k
            new_top_indices = torch.gather(combined_indices, 1, new_top_order)

            top_distances = new_top_distances
            top_indices = new_top_indices

        # Create row indices for this chunk
        chunk_row_indices = (
            torch.arange(i, end_i, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, k)
        )

        # Flatten and move to CPU to save GPU memory
        all_row_indices.append(chunk_row_indices.reshape(-1).cpu())
        all_col_indices.append(top_indices.reshape(-1).cpu())
        all_values.append(top_distances.reshape(-1).cpu())

        if (row_idx + 1) % max(1, (n_points // chunk_size) // 10) == 0:
            progress = 100 * (min(end_i, n_points)) / n_points
            print(f"  Progress: {progress:.1f}%")

    print("  Building sparse tensor...")

    # Concatenate all indices and values (on CPU)
    row_indices = torch.cat(all_row_indices)
    col_indices = torch.cat(all_col_indices)
    values = torch.cat(all_values)

    # Move to device and create sparse tensor
    indices_sparse = torch.stack([row_indices.to(device), col_indices.to(device)])
    values = values.to(device)

    sparse_distances = torch.sparse_coo_tensor(
        indices_sparse,
        values,
        (n_points, n_points),
        device=device,
    )

    print(f"  Sparse matrix created: {sparse_distances._nnz()} non-zero elements")

    return sparse_distances


def get_init_scale(distance_matrix: Tensor, embed_dim: int, verbose=True) -> float:
    n_points = distance_matrix.shape[0]

    # Compute statistics from the distance matrix to inform initialization
    # For sparse matrices, compute statistics directly from the non-zero values
    # (excluding diagonal elements which should be zero anyway)

    if distance_matrix.is_sparse:
        # For sparse tensors, get statistics from non-zero values only
        # Move to CPU to save GPU memory during coalesce
        sparse = distance_matrix.cpu().coalesce()
        values = sparse.values()
        indices = sparse.indices()

        # Filter out diagonal elements (where row == col)
        non_diag_mask = indices[0] != indices[1]
        distances_no_diag = values[non_diag_mask]

        if len(distances_no_diag) == 0:
            # Fallback if all sparse values are on diagonal
            distances_no_diag = values

        n_nonzero = sparse._nnz()
        total_elements = n_points * n_points
        sparsity = 100 * (1 - n_nonzero / total_elements)
    else:
        # For dense tensors, use masking approach
        mask = ~torch.eye(n_points, dtype=torch.bool, device=distance_matrix.device)
        distances_no_diag = distance_matrix[mask]
        sparsity = 0.0

    mean_distance = distances_no_diag.mean().item()
    std_distance = distances_no_diag.std().item()

    # Use a scale that produces reasonable initial distances in embedding space
    # We want initial random distances to be on the same order as target distances
    init_scale = (
        mean_distance
        / (2 * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))).item()
    )

    if verbose:
        sparsity_info = ""
        if distance_matrix.is_sparse:
            sparsity_info = f", sparsity={sparsity:.2f}%"
        print(
            f"Distance statistics: mean={mean_distance:.4f}, std={std_distance:.4f}{sparsity_info}"
        )
        print(f"Initialization scale: {init_scale:.4f}")

    return init_scale
