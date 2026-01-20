"""Performance tests for constant curvature embeddings."""

import time

import pytest
import torch

from src.embedding import ConstantCurvatureEmbedding


@pytest.fixture
def perf_model():
    """Create a model for performance testing."""
    n_points = 100
    embed_dim = 10
    curvature = -1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, device=device, init_scale=0.1
    )


def test_forward_pass_performance(perf_model):
    """Test forward pass performance."""
    # Warm up
    for _ in range(10):
        _ = perf_model()

    # Timing test
    n_iterations = 100
    start = time.time()
    for _ in range(n_iterations):
        _ = perf_model()
    elapsed = time.time() - start

    elapsed_ms = elapsed * 1000  # Convert to ms
    per_iteration = elapsed_ms / n_iterations

    # Performance assertions
    assert (
        per_iteration < 10.0
    ), f"Forward pass too slow: {per_iteration:.4f}ms per iteration"

    # Log performance metrics
    print("\nForward pass performance:")
    print(f"  Total time for {n_iterations} iterations: {elapsed_ms:.2f} ms")
    print(f"  Time per iteration: {per_iteration:.4f} ms")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_performance_faster_than_cpu():
    """Test that GPU is faster than CPU for forward passes."""
    n_points = 200
    embed_dim = 10
    curvature = -1.0
    n_iterations = 50

    # CPU model
    model_cpu = ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, init_scale=0.1, device=torch.device("cpu")
    )

    # GPU model
    model_gpu = ConstantCurvatureEmbedding(
        n_points, embed_dim, curvature, init_scale=0.1, device=torch.device("cuda")
    )

    # Warm up
    for _ in range(10):
        _ = model_cpu()
        _ = model_gpu()

    # Time CPU
    start = time.time()
    for _ in range(n_iterations):
        _ = model_cpu()
    cpu_time = time.time() - start

    # Time GPU (with synchronization)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iterations):
        _ = model_gpu()
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time

    print("\nGPU vs CPU performance:")
    print(f"  CPU time: {cpu_time:.4f}s")
    print(f"  GPU time: {gpu_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # GPU should be at least a bit faster (allowing for small models where overhead dominates)
    assert (
        gpu_time < cpu_time
    ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"


@pytest.mark.parametrize(
    "n_points,embed_dim",
    [
        (50, 5),
        (100, 10),
        (200, 20),
    ],
)
def test_performance_scales_reasonably(n_points, embed_dim):
    """Test that performance scales reasonably with size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConstantCurvatureEmbedding(
        n_points, embed_dim, -1.0, device=device, init_scale=0.1
    )

    # Warm up
    for _ in range(5):
        _ = model()

    # Time single forward pass
    start = time.time()
    _ = model()
    elapsed = time.time() - start

    # Should complete in reasonable time (< 1 second for these sizes)
    assert (
        elapsed < 1.0
    ), f"Forward pass too slow for {n_points} points, {embed_dim}D: {elapsed:.4f}s"

    print(f"\n{n_points} points, {embed_dim}D: {elapsed * 1000:.2f}ms")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_memory_leak():
    """Test that repeated forward passes don't leak memory."""
    import gc

    model = ConstantCurvatureEmbedding(
        100, 10, -1.0, init_scale=0.1, device=torch.device("cuda")
    )

    # Get initial memory
    gc.collect()
    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated()

    # Run many iterations
    for _ in range(100):
        distances = model()
        del distances

    # Check memory hasn't grown significantly
    gc.collect()
    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated()
    mem_growth = final_mem - initial_mem

    # Allow some growth but not excessive
    assert (
        mem_growth < 10 * 1024 * 1024
    ), f"Memory leaked: {mem_growth / 1024 / 1024:.2f}MB"
