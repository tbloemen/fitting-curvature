# Tests

This directory contains pytest-based tests for the fitting-curvature project.

## Running Tests

Run all tests:
```bash
uv run pytest test/
```

Run with verbose output:
```bash
uv run pytest test/ -v
```

Run with print statements visible:
```bash
uv run pytest test/ -v -s
```

Run specific test file:
```bash
uv run pytest test/test_grad.py
```

Run specific test:
```bash
uv run pytest test/test_grad.py::test_gradient_flow_hyperbolic
```

## Test Structure

### test_grad.py
Tests for gradient flow and correctness of embeddings:
- Model initialization
- Manifold projection
- Gradient flow for all curvatures (hyperbolic, Euclidean, spherical)
- GPU support
- Distance properties (symmetry, non-negativity)

### test_perf.py
Performance tests for embedding operations:
- Forward pass speed
- GPU vs CPU performance comparison
- Scaling with model size
- Memory leak detection

### test_matrix_perf.py
Performance and correctness tests for distance matrix calculation:
- CPU and GPU performance
- Speedup measurements
- Correctness verification
- Symmetry and non-negativity checks
- Device and dtype preservation

## Test Coverage

- ✅ 32 tests total
- ✅ All curvatures tested (k = -1, 0, 1)
- ✅ CPU and GPU support
- ✅ Performance benchmarks
- ✅ Correctness verification
- ✅ Memory leak detection

## Pytest Configuration

The `conftest.py` file configures the Python path to allow importing from the `src/` directory.
