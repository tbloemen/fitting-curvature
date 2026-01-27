"""Three.js-based embedding visualization for high-performance rendering."""

import json
import uuid
from pathlib import Path

import numpy as np
from nicegui import ui

from src.visualisation import project_to_2d
from web.training_manager import TrainingManager, TrainingStatus


class ThreeJSEmbeddingPlot:
    """High-performance embedding plot using Three.js WebGL renderer."""

    # Class variables for JavaScript template (shared across all instances)
    _js_template_path = Path(__file__).parent.parent / "static" / "threejs_plot.js"
    _js_template_cache = None

    def __init__(self, training_manager: TrainingManager, projection: str = "direct"):
        """
        Initialize Three.js embedding plot.

        Parameters
        ----------
        training_manager : TrainingManager
            Training manager instance
        projection : str
            Projection method for spherical embeddings
        """
        self.training_manager = training_manager
        self.projection = projection
        self.plot_id = f"plot_{uuid.uuid4().hex[:8]}"
        self.container = None
        self.last_iteration = -1
        self._update_pending = False
        self._ui_timer = None

        # Register as observer to set update flag (thread-safe)
        self.training_manager.add_observer(self._on_data_updated)

        # Color palette (tab10 from matplotlib)
        self.colors_hex = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    def _hex_to_rgb_normalized(self, hex_color: str) -> tuple[float, float, float]:
        """Convert hex color to normalized RGB (0-1 range)."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def _labels_to_rgb(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert labels to RGB color array.

        Parameters
        ----------
        labels : np.ndarray
            Label array (e.g., MNIST digits 0-9)

        Returns
        -------
        np.ndarray
            RGB colors array, shape (n_points, 3), values in [0, 1]
        """
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        unique_labels = np.unique(labels)

        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            hex_color = self.colors_hex[label_idx % len(self.colors_hex)]
            rgb = self._hex_to_rgb_normalized(hex_color)
            colors[mask] = rgb

        return colors

    def _get_boundary_circle(self, curvature: float) -> list | None:
        """
        Get boundary circle points for curved spaces.

        Parameters
        ----------
        curvature : float
            Space curvature

        Returns
        -------
        list or None
            List of (x, y, z) points forming a circle, or None if no boundary
        """
        if curvature == 0:
            return None

        # Generate circle points
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        circle_z = np.zeros_like(circle_x)

        # Return as flat list for Three.js
        points = []
        for x, y, z in zip(circle_x, circle_y, circle_z):
            points.extend([float(x), float(y), float(z)])

        return points

    def _on_data_updated(self):
        """
        Called by training manager when new data is available (from background thread).

        Sets a flag that will be checked by the UI timer to trigger actual update.
        This is thread-safe since it only sets a boolean flag.
        """
        self._update_pending = True

    def _get_init_script(self) -> str:
        """
        Generate the Three.js initialization script.

        Loads the JavaScript template from disk (cached after first load)
        and replaces the plot ID placeholder.

        Returns
        -------
        str
            JavaScript code ready to execute
        """
        # Load template from disk (only once per class)
        if self.__class__._js_template_cache is None:
            self.__class__._js_template_cache = (
                self.__class__._js_template_path.read_text()
            )

        # Replace placeholder with actual plot ID
        return self.__class__._js_template_cache.replace("{{PLOT_ID}}", self.plot_id)

    def create(self):
        """Create the plot UI element."""
        # Load Three.js from CDN (only once per page)
        ui.add_head_html(
            '<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>'
        )

        # Create HTML container div with visible border for debugging
        container_html = f'<div id="container_{self.plot_id}" style="width: 100%; height: 600px; position: relative; border: 1px solid #ddd; background: #fafafa;"></div>'
        self.container = ui.html(container_html, sanitize=False)

        # Initialize Three.js scene after a short delay to ensure container is rendered
        ui.timer(
            0.5,
            lambda: ui.run_javascript(self._get_init_script()),
            once=True,
        )

        # Start fast timer to check for pending updates
        # This runs in the UI thread, so it's safe to call ui.run_javascript()
        # The observer pattern sets a flag, and this timer acts on it immediately
        self._ui_timer = ui.timer(0.016, self._check_and_update)  # ~60 FPS

        return self.container

    def _check_and_update(self):
        """Check if update is pending and perform it (runs in UI thread)."""
        if self._update_pending:
            self._update_pending = False
            self.update()

    def update(self):
        """Update the plot with current state."""
        state = self.training_manager.get_state()

        # Only update if training is running or just completed (not during precomputing)
        if state.status not in [TrainingStatus.RUNNING, TrainingStatus.COMPLETED]:
            return

        # Only update every 10 iterations (or on first update, or on completion)
        if state.iteration == self.last_iteration:
            return

        self.last_iteration = state.iteration

        # Check if we have data
        if state.embeddings is None or state.labels is None:
            return

        try:
            # Project to 2D
            embeddings = state.embeddings
            k = state.curvature

            x, y = project_to_2d(
                embeddings,
                k=k,
                i=0,
                j=1,
                projection=self.projection if k > 0 else "direct",
            )

            # Convert to Three.js format (z=0 for 2D)
            positions = np.column_stack([x, y, np.zeros(len(x))]).astype(np.float32)

            # Map labels to colors
            colors = self._labels_to_rgb(state.labels)

            # Calculate boundary
            boundary_points = self._get_boundary_circle(k)

            # Generate title
            geometry = (
                "Euclidean" if k == 0 else ("Spherical" if k > 0 else "Hyperbolic")
            )
            title = f"Embedding Visualization - {geometry} (k={k:.4f})\\nIteration {state.iteration}/{state.max_iterations}"

            # Update via JavaScript
            ui.run_javascript(
                f"""
                updateEmbedding_{self.plot_id}(
                    {json.dumps(positions.flatten().tolist())},
                    {json.dumps(colors.flatten().tolist())},
                    {json.dumps(boundary_points) if boundary_points else 'null'},
                    {json.dumps(title)}
                );
                """
            )

        except Exception as e:
            print(f"Update error: {e}")
            import traceback

            traceback.print_exc()

    def set_projection(self, projection: str):
        """Change projection method."""
        self.projection = projection
        # Force immediate update by resetting iteration
        self.last_iteration = -1
        self.update()

    def cleanup(self):
        """Clean up resources and unregister observer."""
        self.training_manager.remove_observer(self._on_data_updated)
        if self._ui_timer is not None:
            self._ui_timer.deactivate()
