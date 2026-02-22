"""Training manager for running embeddings with callbacks."""

import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from src.embedding import ConstantCurvatureEmbedding, fit_embedding
from src.load_data import load_raw_data
from src.manifolds import Euclidean, Hyperboloid, Sphere
from src.matrices import get_default_init_scale, normalize_data
from src.metrics import compute_all_metrics
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
    high_dim_data: Optional[np.ndarray] = None
    high_dim_distances: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, Optional[float]]] = None
    projection: str = "stereographic"


class TrainingManager:
    """Manages training execution with callbacks."""

    def __init__(self):
        self.state = TrainingState()
        self._stop_requested = False
        self._training_thread: Optional[threading.Thread] = None
        self._data_cache: Optional[
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = None
        self._lock = threading.Lock()
        self._update_callback: Optional[Callable] = None
        self._status_callback: Optional[Callable] = None

    def load_data(
        self, dataset: str, n_samples: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Load and cache dataset."""
        if self._data_cache is None:
            data, labels, D = load_raw_data(dataset, n_samples=n_samples)

            if n_samples > 0 and n_samples < len(data):
                indices = torch.randperm(len(data))[:n_samples]
                data = data[indices]
                labels = labels[indices]
                if D is not None:
                    D = D[indices][:, indices]

            # Normalize
            if D is not None:
                mask = ~torch.eye(D.shape[0], dtype=torch.bool)
                mean_d = D[mask].mean()
                D = D / mean_d
            else:
                data = normalize_data(data)

            self._data_cache = (data, labels, D)

        return self._data_cache

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache = None

    def _training_callback(
        self, iteration: int, loss: float, model: ConstantCurvatureEmbedding, phase: str
    ) -> bool:
        """Callback called during training."""
        with self._lock:
            if self._stop_requested:
                return False

            self.state.iteration = iteration
            self.state.loss = loss
            self.state.phase = phase
            self.state.model = model

            # Store embeddings (move to CPU)
            embeddings = model.points.detach().cpu().numpy()
            self.state.embeddings = embeddings

            self.state.loss_history.append((iteration, loss))

        # Push update to WebSocket clients
        if self._update_callback:
            try:
                self._update_callback(iteration, loss, model, phase, self.state)
            except Exception as e:
                print(f"Error in update callback: {e}")

        return True

    def start_training(
        self,
        config: Dict[str, Any],
        update_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
    ) -> None:
        """Start training in background thread."""
        if self.state.status == TrainingStatus.RUNNING:
            raise RuntimeError("Training is already running")

        self._update_callback = update_callback
        self._status_callback = status_callback

        # Reset state
        with self._lock:
            self._stop_requested = False
            self.state.status = TrainingStatus.RUNNING
            self.state.iteration = 0
            self.state.loss = 0.0
            self.state.error_message = ""
            self.state.loss_history = []
            self.state.max_iterations = config["embedding"]["n_iterations"]
            self.state.projection = config.get("visualization", {}).get(
                "spherical_projection", "stereographic"
            )

        self._training_thread = threading.Thread(
            target=self._run_training, args=(config,), daemon=True
        )
        self._training_thread.start()

    def _notify_status(self, status: str, error_message: str = ""):
        """Notify status change via callback."""
        if self._status_callback:
            try:
                self._status_callback(status, error_message)
            except Exception as e:
                print(f"Error in status callback: {e}")

    def _run_training(self, config: Dict[str, Any]) -> None:
        """Run training (executed in background thread)."""
        try:
            with self._lock:
                self.state.status = TrainingStatus.PRECOMPUTING
                self.state.phase = "precomputing"
            self._notify_status("precomputing")

            # Load data
            dataset = config["data"]["dataset"]
            n_samples = config["data"]["n_samples"]
            data, labels, precomputed_distances = self.load_data(dataset, n_samples)

            with self._lock:
                self.state.labels = labels.cpu().numpy()
                self.state.high_dim_data = data.cpu().numpy()
                if precomputed_distances is not None:
                    self.state.high_dim_distances = precomputed_distances.cpu().numpy()
                else:
                    from scipy.spatial.distance import pdist, squareform

                    self.state.high_dim_distances = squareform(
                        pdist(self.state.high_dim_data)
                    )

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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with self._lock:
                self.state.status = TrainingStatus.RUNNING
                self.state.phase = "training"
            self._notify_status("running")

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
                verbose=False,
                callback=self._training_callback,
                precomputed_distances=precomputed_distances,
            )

            with self._lock:
                if self._stop_requested:
                    self.state.status = TrainingStatus.STOPPED
                    self._notify_status("stopped")
                else:
                    self.state.status = TrainingStatus.COMPLETED
                    self.state.model = model
                    self.state.embeddings = model.points.detach().cpu().numpy()
                    self._compute_metrics(curvature, embed_dim)
                    self._notify_status("completed")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            with self._lock:
                self.state.status = TrainingStatus.ERROR
                self.state.error_message = str(e)
            self._notify_status("error", str(e))
            print(f"Training error: {e}")
            traceback.print_exc()

    def _compute_metrics(self, curvature: float, embed_dim: int) -> None:
        """Compute all quality metrics on the final embedding."""
        try:
            embeddings = self.state.embeddings
            if embeddings is None:
                return

            # Compute embedded pairwise distances using the manifold
            points_tensor = torch.tensor(embeddings, dtype=torch.float32)
            if curvature > 0:
                manifold = Sphere(curvature)
            elif curvature < 0:
                manifold = Hyperboloid(curvature)
            else:
                manifold = Euclidean()

            with torch.no_grad():
                embedded_distances = manifold.pairwise_distances(points_tensor).numpy()
            np.fill_diagonal(embedded_distances, 0.0)

            self.state.metrics = compute_all_metrics(
                embedded_distances=embedded_distances,
                embeddings=embeddings,
                high_dim_data=self.state.high_dim_data,
                high_dim_distances=self.state.high_dim_distances,
                labels=self.state.labels,
            )
        except Exception as e:
            print(f"Error computing metrics: {e}")
            self.state.metrics = None

    def stop_training(self) -> None:
        """Request training to stop."""
        with self._lock:
            if self.state.status in (
                TrainingStatus.RUNNING,
                TrainingStatus.PRECOMPUTING,
            ):
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
