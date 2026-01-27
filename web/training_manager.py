"""Training manager for running embeddings with callbacks."""

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.embedding import ConstantCurvatureEmbedding, fit_embedding
from src.load_data import load_raw_data
from src.matrices import get_default_init_scale, normalize_data
from src.types import InitMethod


class TrainingStatus(str, Enum):
    """Training status enum."""

    IDLE = "idle"
    PRECOMPUTING = "precomputing"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TrainingState:
    """Current state of training."""

    status: TrainingStatus = TrainingStatus.IDLE
    iteration: int = 0
    max_iterations: int = 0
    loss: float = 0.0
    embeddings: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    curvature: float = 0.0
    phase: str = "idle"
    error_message: str = ""
    loss_history: list = field(default_factory=list)
    model: Optional[ConstantCurvatureEmbedding] = None
    version: int = 0  # Incremented each time embeddings are updated


class TrainingManager:
    """Manages training execution with callbacks."""

    def __init__(self):
        self.state = TrainingState()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._stop_requested = False
        self._training_thread: Optional[threading.Thread] = None
        self._data_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._lock = threading.Lock()
        self._observers: list = []  # Callbacks to notify on state updates

    def load_data(
        self, dataset: str, n_samples: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and cache dataset.

        Parameters
        ----------
        dataset : str
            Dataset name (e.g., "mnist")
        n_samples : int
            Number of samples to use (-1 for all)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (data, labels)
        """
        if self._data_cache is None:
            data, labels = load_raw_data(dataset)
            data = normalize_data(data)

            if n_samples > 0 and n_samples < len(data):
                indices = torch.randperm(len(data))[:n_samples]
                data = data[indices]
                labels = labels[indices]

            self._data_cache = (data, labels)

        return self._data_cache

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache = None

    def add_observer(self, callback):
        """
        Add an observer callback that will be notified on state updates.

        Parameters
        ----------
        callback : callable
            Function to call when state is updated (no arguments)
        """
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback):
        """
        Remove an observer callback.

        Parameters
        ----------
        callback : callable
            Function to remove from observers
        """
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self):
        """Notify all registered observers of state update."""
        for callback in self._observers:
            try:
                callback()
            except Exception as e:
                print(f"Error in observer callback: {e}")

    def _training_callback(
        self, iteration: int, loss: float, model: ConstantCurvatureEmbedding, phase: str
    ) -> bool:
        """
        Callback called during training.

        Parameters
        ----------
        iteration : int
            Current iteration
        loss : float
            Current loss value
        model : ConstantCurvatureEmbedding
            Current embedding model
        phase : str
            Current phase ("early" or "main")

        Returns
        -------
        bool
            True to continue training, False to stop
        """
        with self._lock:
            # Check if stop was requested
            if self._stop_requested:
                return False

            # Update state
            self.state.iteration = iteration
            self.state.loss = loss
            self.state.phase = phase
            self.state.model = model

            # Store embeddings (move to CPU to save GPU memory)
            embeddings = model.points.detach().cpu().numpy()
            self.state.embeddings = embeddings

            # Track loss history
            self.state.loss_history.append((iteration, loss))

            # Increment version to signal UI update
            self.state.version += 1

        # Notify observers outside the lock to avoid potential deadlocks
        self._notify_observers()

        return True

    def start_training(self, config: Dict[str, Any]) -> None:
        """
        Start training in background thread.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        if self.state.status == TrainingStatus.RUNNING:
            raise RuntimeError("Training is already running")

        # Reset state
        with self._lock:
            self._stop_requested = False
            self.state.status = TrainingStatus.RUNNING
            self.state.iteration = 0
            self.state.loss = 0.0
            self.state.error_message = ""
            self.state.loss_history = []
            self.state.max_iterations = config["embedding"]["n_iterations"]

        # Start training in executor
        self._training_thread = threading.Thread(
            target=self._run_training, args=(config,), daemon=True
        )
        self._training_thread.start()

    def _run_training(self, config: Dict[str, Any]) -> None:
        """
        Run training (executed in background thread).

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        try:
            # Set precomputing status
            with self._lock:
                self.state.status = TrainingStatus.PRECOMPUTING
                self.state.phase = "precomputing"
            self._notify_observers()

            # Load data
            dataset = config["data"]["dataset"]
            n_samples = config["data"]["n_samples"]
            data, labels = self.load_data(dataset, n_samples)

            # Store labels for visualization
            with self._lock:
                self.state.labels = labels.cpu().numpy()

            # Get configuration
            embed_dim = config["embedding"]["embed_dim"]
            n_iterations = config["embedding"]["n_iterations"]
            init_method_str = config["embedding"]["init_method"]
            init_method = (
                InitMethod.PCA if init_method_str == "pca" else InitMethod.RANDOM
            )
            perplexity = config["embedding"]["perplexity"]
            early_exag_iters = config["embedding"]["early_exaggeration_iterations"]
            early_exag_factor = config["embedding"]["early_exaggeration_factor"]
            momentum_early = config["embedding"]["momentum_early"]
            momentum_main = config["embedding"]["momentum_main"]

            # Get single curvature (web interface runs one at a time)
            curvatures = config["experiments"]["curvatures"]
            if len(curvatures) == 0:
                raise ValueError("No curvatures specified")
            curvature = curvatures[0]

            with self._lock:
                self.state.curvature = curvature

            # Get learning rate
            learning_rates = config["hyperparameters"]["learning_rates"]
            if str(curvature) in learning_rates:
                learning_rate = learning_rates[str(curvature)]
            elif "k" in learning_rates:
                learning_rate = learning_rates["k"]
            else:
                learning_rate = 200.0

            # Get init scale
            init_scale_config = config["hyperparameters"]["init_scale"]
            if init_scale_config == "auto":
                init_scale = get_default_init_scale(embed_dim)
            else:
                init_scale = float(init_scale_config)

            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Update status to running after precomputation
            with self._lock:
                self.state.status = TrainingStatus.RUNNING
                self.state.phase = "training"
            self._notify_observers()

            # Run embedding with callback
            model = fit_embedding(
                data=data,
                embed_dim=embed_dim,
                curvature=curvature,
                device=device,
                perplexity=perplexity,
                n_iterations=n_iterations,
                early_exaggeration_iterations=early_exag_iters,
                early_exaggeration_factor=early_exag_factor,
                learning_rate=learning_rate,
                momentum_early=momentum_early,
                momentum_main=momentum_main,
                init_method=init_method,
                init_scale=init_scale,
                verbose=False,  # Disable console output
                callback=self._training_callback,
            )

            # Training completed
            with self._lock:
                if self._stop_requested:
                    self.state.status = TrainingStatus.STOPPED
                else:
                    self.state.status = TrainingStatus.COMPLETED
                    self.state.model = model
                    # Store final embeddings
                    self.state.embeddings = model.points.detach().cpu().numpy()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            # Handle errors
            with self._lock:
                self.state.status = TrainingStatus.ERROR
                self.state.error_message = str(e)
            print(f"Training error: {e}")
            traceback.print_exc()

    def stop_training(self) -> None:
        """Request training to stop."""
        with self._lock:
            if self.state.status == TrainingStatus.RUNNING:
                self._stop_requested = True

    def get_state(self) -> TrainingState:
        """Get current training state (thread-safe)."""
        with self._lock:
            return self.state

    def reset(self) -> None:
        """Reset to idle state."""
        with self._lock:
            if self.state.status == TrainingStatus.RUNNING:
                raise RuntimeError("Cannot reset while training is running")
            self.state = TrainingState()
            self._stop_requested = False
