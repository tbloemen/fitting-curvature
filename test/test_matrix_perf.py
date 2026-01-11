"""Performance tests for distance matrix calculation."""

import time

import pytest
import torch

from src.matrices import calculate_distance_matrix


@pytest.mark.parametrize("n_points", [100, 500, 1000])
def test_distance_matrix_cpu_performance(n_points):
    """Test distance matrix calculation performance on CPU."""
    X = torch.randn(n_points, 784)

    start = time.time()
    dist = calculate_distance_matrix(X)
    elapsed = time.time() - start

    # Should be reasonably fast
    assert elapsed < 0.1, (
        f"Distance matrix calculation too slow for {n_points} points: {elapsed:.4f}s"
    )

    # Check output shape
    assert dist.shape == (n_points, n_points)

    print(f"\n{n_points} points, CPU: {elapsed:.4f}s")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n_points", [100, 500, 1000])
def test_distance_matrix_gpu_performance(n_points):
    """Test distance matrix calculation performance on GPU."""
    X = torch.randn(n_points, 784).cuda()

    # Warm up
    _ = calculate_distance_matrix(X)
    torch.cuda.synchronize()

    # Time it
    start = time.time()
    dist = calculate_distance_matrix(X)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # GPU should be very fast
    assert elapsed < 0.01, (
        f"GPU distance matrix calculation too slow for {n_points} points: {elapsed:.4f}s"
    )

    # Check output shape and device
    assert dist.shape == (n_points, n_points)
    assert dist.device.type == "cuda"

    print(f"\n{n_points} points, GPU: {elapsed:.4f}s")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n_points", [100, 500, 1000])
def test_gpu_faster_than_cpu(n_points):
    """Test that GPU is faster than CPU for distance matrix calculation."""
    X_cpu = torch.randn(n_points, 784)
    X_gpu = X_cpu.cuda()

    # Time CPU
    start = time.time()
    _ = calculate_distance_matrix(X_cpu)
    cpu_time = time.time() - start

    # Time GPU (with warm-up)
    _ = calculate_distance_matrix(X_gpu)
    torch.cuda.synchronize()

    start = time.time()
    _ = calculate_distance_matrix(X_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time

    print(
        f"\n{n_points} points - CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x"
    )

    # GPU should be faster
    assert gpu_time < cpu_time, "GPU should be faster than CPU"


def test_distance_matrix_correctness():
    """Test that distance matrix calculation is correct."""
    # Create simple test case
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    dist = calculate_distance_matrix(X)

    # Check known distances
    assert torch.allclose(dist[0, 1], torch.tensor(1.0), atol=1e-5), (
        "Distance (0,0) to (1,0) should be 1"
    )
    assert torch.allclose(dist[0, 2], torch.tensor(1.0), atol=1e-5), (
        "Distance (0,0) to (0,1) should be 1"
    )
    assert torch.allclose(dist[1, 2], torch.sqrt(torch.tensor(2.0)), atol=1e-5), (
        "Distance (1,0) to (0,1) should be sqrt(2)"
    )

    # Check diagonal is zero
    assert torch.allclose(torch.diag(dist), torch.zeros(3), atol=1e-6), (
        "Diagonal should be zero"
    )


def test_distance_matrix_symmetric():
    """Test that distance matrix is symmetric."""
    X = torch.randn(50, 10)
    dist = calculate_distance_matrix(X)

    assert torch.allclose(dist, dist.t(), atol=1e-5), (
        "Distance matrix should be symmetric"
    )


def test_distance_matrix_non_negative():
    """Test that all distances are non-negative."""
    X = torch.randn(30, 20)
    dist = calculate_distance_matrix(X)

    assert (dist >= 0).all(), "All distances should be non-negative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_results_match():
    """Test that CPU and GPU produce the same results."""
    X_cpu = torch.randn(100, 784)
    X_gpu = X_cpu.cuda()

    dist_cpu = calculate_distance_matrix(X_cpu)
    dist_gpu = calculate_distance_matrix(X_gpu)

    # Move GPU result to CPU for comparison
    dist_gpu_cpu = dist_gpu.cpu()

    # Results should match within numerical precision
    max_diff = (dist_cpu - dist_gpu_cpu).abs().max().item()
    assert max_diff < 0.1, (
        f"CPU and GPU results differ too much: max diff = {max_diff:.2e}"
    )


def test_distance_matrix_preserves_device():
    """Test that output is on the same device as input."""
    X_cpu = torch.randn(20, 10)
    dist_cpu = calculate_distance_matrix(X_cpu)
    assert dist_cpu.device.type == "cpu", "CPU input should produce CPU output"

    if torch.cuda.is_available():
        X_gpu = torch.randn(20, 10).cuda()
        dist_gpu = calculate_distance_matrix(X_gpu)
        assert dist_gpu.device.type == "cuda", "GPU input should produce GPU output"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distance_matrix_dtype_preservation(dtype):
    """Test that distance matrix preserves dtype."""
    X = torch.randn(20, 10, dtype=dtype)
    dist = calculate_distance_matrix(X)

    assert dist.dtype == dtype, (
        f"Output dtype {dist.dtype} should match input dtype {dtype}"
    )
