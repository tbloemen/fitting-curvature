"""
Tests for web interface components.

Tests configuration management and training manager functionality.
"""

import copy
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from web.config_manager import (
    DEFAULT_CONFIG,
    get_default_config,
    load_config,
    save_config,
    validate_config,
)
from web.training_manager import TrainingManager, TrainingState, TrainingStatus


# --- Config Manager Tests ---


def test_get_default_config():
    """Test getting default configuration."""
    config = copy.deepcopy(get_default_config())

    assert "data" in config
    assert "embedding" in config
    assert "hyperparameters" in config
    assert "experiments" in config

    # Note: get_default_config returns a shallow copy, so nested dicts are shared
    # This test just verifies the structure is present
    assert config["data"]["dataset"] == "mnist"


def test_validate_config_valid():
    """Test validation with valid configuration."""
    config = copy.deepcopy(get_default_config())
    is_valid, error_msg = validate_config(config)

    assert is_valid
    assert error_msg == ""


def test_validate_config_missing_sections():
    """Test validation with missing required sections."""
    # Missing data section
    config = {"embedding": {}, "hyperparameters": {}, "experiments": {}}
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "Missing 'data' section" in error_msg

    # Missing embedding section (need valid data section to get past data checks)
    config = {
        "data": {"dataset": "mnist", "n_samples": 100},
        "hyperparameters": {"learning_rates": {"k": 100.0}},
        "experiments": {"curvatures": [0.0]},
    }
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "Missing 'embedding' section" in error_msg


def test_validate_config_invalid_dataset():
    """Test validation with invalid dataset."""
    config = copy.deepcopy(get_default_config())
    config["data"]["dataset"] = "invalid_dataset"

    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "Invalid dataset" in error_msg


def test_validate_config_invalid_n_samples():
    """Test validation with invalid n_samples values."""
    config = copy.deepcopy(get_default_config())

    # Negative value (not -1)
    config["data"]["n_samples"] = -5
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "n_samples" in error_msg

    # Zero
    config["data"]["n_samples"] = 0
    is_valid, error_msg = validate_config(config)
    assert not is_valid

    # Valid: positive integer
    config["data"]["n_samples"] = 100
    is_valid, _ = validate_config(config)
    assert is_valid

    # Valid: -1 for all samples
    config["data"]["n_samples"] = -1
    is_valid, _ = validate_config(config)
    assert is_valid


def test_validate_config_invalid_embed_dim():
    """Test validation with invalid embed_dim."""
    config = copy.deepcopy(get_default_config())

    # Zero
    config["embedding"]["embed_dim"] = 0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "embed_dim" in error_msg

    # Negative
    config["embedding"]["embed_dim"] = -2
    is_valid, error_msg = validate_config(config)
    assert not is_valid


def test_validate_config_invalid_iterations():
    """Test validation with invalid n_iterations."""
    config = copy.deepcopy(get_default_config())

    config["embedding"]["n_iterations"] = 0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "n_iterations" in error_msg

    config["embedding"]["n_iterations"] = -100
    is_valid, error_msg = validate_config(config)
    assert not is_valid


def test_validate_config_invalid_init_method():
    """Test validation with invalid init_method."""
    config = copy.deepcopy(get_default_config())
    config["embedding"]["init_method"] = "invalid_method"

    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "init_method" in error_msg


def test_validate_config_invalid_perplexity():
    """Test validation with invalid perplexity."""
    config = copy.deepcopy(get_default_config())

    # Zero
    config["embedding"]["perplexity"] = 0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "perplexity" in error_msg

    # Negative
    config["embedding"]["perplexity"] = -5.0
    is_valid, error_msg = validate_config(config)
    assert not is_valid


def test_validate_config_invalid_early_exaggeration():
    """Test validation with invalid early exaggeration parameters."""
    config = copy.deepcopy(get_default_config())

    # Negative iterations
    config["embedding"]["early_exaggeration_iterations"] = -1
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "early_exaggeration_iterations" in error_msg

    # Early exaggeration iterations >= total iterations
    config["embedding"]["early_exaggeration_iterations"] = 1000
    config["embedding"]["n_iterations"] = 1000
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "early_exaggeration_iterations must be less than n_iterations" in error_msg

    # Invalid factor
    config["embedding"]["early_exaggeration_iterations"] = 100
    config["embedding"]["early_exaggeration_factor"] = 0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "early_exaggeration_factor" in error_msg


def test_validate_config_invalid_momentum():
    """Test validation with invalid momentum values."""
    config = copy.deepcopy(get_default_config())

    # momentum_early out of range
    config["embedding"]["momentum_early"] = 1.5
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "momentum_early" in error_msg

    config["embedding"]["momentum_early"] = -0.1
    is_valid, error_msg = validate_config(config)
    assert not is_valid

    # momentum_main out of range
    config["embedding"]["momentum_early"] = 0.5
    config["embedding"]["momentum_main"] = 2.0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "momentum_main" in error_msg


def test_validate_config_invalid_learning_rates():
    """Test validation with invalid learning rates."""
    config = copy.deepcopy(get_default_config())

    # Not a dictionary
    config["hyperparameters"]["learning_rates"] = 100.0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "learning_rates must be a dictionary" in error_msg

    # Negative learning rate
    config["hyperparameters"]["learning_rates"] = {"k": -10.0}
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "learning_rates" in error_msg


def test_validate_config_invalid_curvatures():
    """Test validation with invalid curvatures."""
    config = copy.deepcopy(get_default_config())

    # Not a list
    config["experiments"]["curvatures"] = 1.0
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "curvatures must be a list" in error_msg

    # Empty list
    config["experiments"]["curvatures"] = []
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "curvatures list cannot be empty" in error_msg

    # Invalid curvature value
    config["experiments"]["curvatures"] = [1.0, "invalid"]
    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "Invalid curvature value" in error_msg


def test_validate_config_invalid_projection():
    """Test validation with invalid spherical projection."""
    config = copy.deepcopy(get_default_config())
    config["visualization"]["spherical_projection"] = "invalid_projection"

    is_valid, error_msg = validate_config(config)
    assert not is_valid
    assert "spherical_projection" in error_msg


def test_save_and_load_config():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.toml"

        # Patch get_config_path to use temp directory
        with patch("web.config_manager.get_config_path", return_value=config_path):
            # Create custom config
            test_config = get_default_config()
            test_config["data"]["n_samples"] = 500
            test_config["embedding"]["embed_dim"] = 3

            # Save config
            save_config(test_config)

            # Verify file was created
            assert config_path.exists()

            # Load config
            loaded_config = load_config()

            # Verify loaded config matches
            assert loaded_config["data"]["n_samples"] == 500
            assert loaded_config["embedding"]["embed_dim"] == 3


def test_load_config_creates_default():
    """Test that load_config creates default config if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "nonexistent_config.toml"

        with patch("web.config_manager.get_config_path", return_value=config_path):
            # Load config when file doesn't exist
            loaded_config = load_config()

            # Should return default config
            assert loaded_config == DEFAULT_CONFIG

            # File should have been created
            assert config_path.exists()


# --- Training Manager Tests ---


@pytest.fixture
def training_manager():
    """Create a training manager for testing."""
    return TrainingManager()


def test_training_manager_initial_state(training_manager):
    """Test initial state of training manager."""
    state = training_manager.get_state()

    assert state.status == TrainingStatus.IDLE
    assert state.iteration == 0
    assert state.max_iterations == 0
    assert state.loss == 0.0
    assert state.embeddings is None
    assert state.labels is None
    assert state.error_message == ""
    assert len(state.loss_history) == 0


def test_training_state_creation():
    """Test TrainingState dataclass creation."""
    state = TrainingState()

    assert state.status == TrainingStatus.IDLE
    assert state.iteration == 0
    assert state.version == 0
    assert state.loss_history == []


def test_training_manager_observer_pattern(training_manager):
    """Test observer pattern for state updates."""
    callback_count = [0]

    def observer_callback():
        callback_count[0] += 1

    # Add observer
    training_manager.add_observer(observer_callback)

    # Notify observers
    training_manager._notify_observers()

    assert callback_count[0] == 1

    # Remove observer
    training_manager.remove_observer(observer_callback)
    training_manager._notify_observers()

    # Count should not increase
    assert callback_count[0] == 1


def test_training_manager_multiple_observers(training_manager):
    """Test multiple observers can be registered."""
    counts = {"obs1": 0, "obs2": 0}

    def obs1():
        counts["obs1"] += 1

    def obs2():
        counts["obs2"] += 1

    training_manager.add_observer(obs1)
    training_manager.add_observer(obs2)

    training_manager._notify_observers()

    assert counts["obs1"] == 1
    assert counts["obs2"] == 1


def test_training_manager_observer_duplicate(training_manager):
    """Test that adding the same observer twice doesn't duplicate it."""
    count = [0]

    def observer():
        count[0] += 1

    training_manager.add_observer(observer)
    training_manager.add_observer(observer)  # Add again

    training_manager._notify_observers()

    assert count[0] == 1  # Should only be called once


def test_training_manager_observer_error_handling(training_manager):
    """Test that observer errors don't break notification."""
    counts = {"good": 0}

    def bad_observer():
        raise RuntimeError("Observer error")

    def good_observer():
        counts["good"] += 1

    training_manager.add_observer(bad_observer)
    training_manager.add_observer(good_observer)

    # Should not raise exception
    training_manager._notify_observers()

    # Good observer should still be called
    assert counts["good"] == 1


def test_training_manager_load_data(training_manager):
    """Test data loading and caching."""
    # Mock the data loading function
    mock_data = torch.randn(100, 784)
    mock_labels = torch.randint(0, 10, (100,))

    with patch("web.training_manager.load_raw_data", return_value=(mock_data, mock_labels)):
        with patch("web.training_manager.normalize_data", side_effect=lambda x: x):
            # Load data
            data, labels = training_manager.load_data("mnist", n_samples=-1)

            assert data.shape == (100, 784)
            assert labels.shape == (100,)

            # Load again - should use cache
            data2, labels2 = training_manager.load_data("mnist", n_samples=-1)

            assert torch.equal(data, data2)
            assert torch.equal(labels, labels2)


def test_training_manager_load_data_with_sampling(training_manager):
    """Test data loading with sample limit."""
    mock_data = torch.randn(1000, 784)
    mock_labels = torch.randint(0, 10, (1000,))

    with patch("web.training_manager.load_raw_data", return_value=(mock_data, mock_labels)):
        with patch("web.training_manager.normalize_data", side_effect=lambda x: x):
            # Load with sampling
            data, labels = training_manager.load_data("mnist", n_samples=100)

            assert data.shape == (100, 784)
            assert labels.shape == (100,)


def test_training_manager_clear_cache(training_manager):
    """Test clearing data cache."""
    mock_data = torch.randn(100, 784)
    mock_labels = torch.randint(0, 10, (100,))

    with patch("web.training_manager.load_raw_data", return_value=(mock_data, mock_labels)):
        with patch("web.training_manager.normalize_data", side_effect=lambda x: x):
            # Load data to populate cache
            training_manager.load_data("mnist")

            assert training_manager._data_cache is not None

            # Clear cache
            training_manager.clear_cache()

            assert training_manager._data_cache is None


def test_training_manager_callback(training_manager):
    """Test training callback updates state correctly."""
    # Create a mock model
    mock_model = Mock()
    mock_embeddings = torch.randn(50, 3)
    mock_model.points = mock_embeddings

    # Call callback
    should_continue = training_manager._training_callback(
        iteration=10, loss=0.5, model=mock_model, phase="early"
    )

    assert should_continue is True

    state = training_manager.get_state()
    assert state.iteration == 10
    assert state.loss == 0.5
    assert state.phase == "early"
    assert state.embeddings.shape == (50, 3)
    assert len(state.loss_history) == 1
    assert state.loss_history[0] == (10, 0.5)
    assert state.version == 1


def test_training_manager_callback_stop_requested(training_manager):
    """Test callback returns False when stop is requested."""
    mock_model = Mock()
    mock_model.points = torch.randn(50, 3)

    # Request stop
    training_manager._stop_requested = True

    # Call callback
    should_continue = training_manager._training_callback(
        iteration=10, loss=0.5, model=mock_model, phase="main"
    )

    assert should_continue is False


def test_training_manager_callback_increments_version(training_manager):
    """Test callback increments version for UI updates."""
    mock_model = Mock()
    mock_model.points = torch.randn(50, 3)

    initial_version = training_manager.state.version

    # Call callback multiple times
    for i in range(5):
        training_manager._training_callback(
            iteration=i, loss=0.5, model=mock_model, phase="main"
        )

    state = training_manager.get_state()
    assert state.version == initial_version + 5


def test_training_manager_reset(training_manager):
    """Test resetting training manager state."""
    # Modify state
    training_manager.state.iteration = 100
    training_manager.state.loss = 0.5
    training_manager.state.status = TrainingStatus.COMPLETED
    training_manager._stop_requested = True

    # Reset
    training_manager.reset()

    state = training_manager.get_state()
    assert state.status == TrainingStatus.IDLE
    assert state.iteration == 0
    assert state.loss == 0.0
    assert training_manager._stop_requested is False


def test_training_manager_reset_while_running_raises_error(training_manager):
    """Test that reset raises error if training is running."""
    training_manager.state.status = TrainingStatus.RUNNING

    with pytest.raises(RuntimeError, match="Cannot reset while training is running"):
        training_manager.reset()


def test_training_manager_start_training_already_running(training_manager):
    """Test that starting training while already running raises error."""
    training_manager.state.status = TrainingStatus.RUNNING

    config = copy.deepcopy(get_default_config())

    with pytest.raises(RuntimeError, match="Training is already running"):
        training_manager.start_training(config)


def test_training_manager_stop_training(training_manager):
    """Test requesting training to stop."""
    training_manager.state.status = TrainingStatus.RUNNING

    training_manager.stop_training()

    assert training_manager._stop_requested is True


def test_training_manager_thread_safety(training_manager):
    """Test thread-safe state access."""
    mock_model = Mock()
    mock_model.points = torch.randn(50, 3)

    # Simulate concurrent access
    def update_state():
        for i in range(100):
            training_manager._training_callback(
                iteration=i, loss=0.5, model=mock_model, phase="main"
            )

    def read_state():
        for _ in range(100):
            training_manager.get_state()

    thread1 = threading.Thread(target=update_state)
    thread2 = threading.Thread(target=read_state)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Should complete without errors
    assert True


def test_training_status_enum():
    """Test TrainingStatus enum values."""
    assert TrainingStatus.IDLE == "idle"
    assert TrainingStatus.PRECOMPUTING == "precomputing"
    assert TrainingStatus.RUNNING == "running"
    assert TrainingStatus.COMPLETED == "completed"
    assert TrainingStatus.STOPPED == "stopped"
    assert TrainingStatus.ERROR == "error"


def test_training_manager_loss_history_accumulation(training_manager):
    """Test that loss history accumulates correctly."""
    mock_model = Mock()
    mock_model.points = torch.randn(50, 3)

    loss_values = [1.0, 0.8, 0.6, 0.4, 0.2]

    for i, loss in enumerate(loss_values):
        training_manager._training_callback(
            iteration=i * 10, loss=loss, model=mock_model, phase="main"
        )

    state = training_manager.get_state()
    assert len(state.loss_history) == 5

    for i, (iteration, loss) in enumerate(state.loss_history):
        assert iteration == i * 10
        assert loss == loss_values[i]


def test_training_manager_embeddings_on_cpu(training_manager):
    """Test that embeddings are moved to CPU in callback."""
    mock_model = Mock()

    # Create GPU tensor if CUDA is available, otherwise CPU
    if torch.cuda.is_available():
        mock_embeddings = torch.randn(50, 3).cuda()
    else:
        mock_embeddings = torch.randn(50, 3)

    mock_model.points = mock_embeddings

    training_manager._training_callback(
        iteration=1, loss=0.5, model=mock_model, phase="main"
    )

    state = training_manager.get_state()

    # Embeddings should be numpy array (converted from CPU tensor)
    assert isinstance(state.embeddings, np.ndarray)
    assert state.embeddings.shape == (50, 3)
