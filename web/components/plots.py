"""Plotly visualization components for embeddings."""

import numpy as np
import plotly.graph_objects as go
from nicegui import ui

from src.visualisation import project_to_2d
from web.training_manager import TrainingState, TrainingStatus


def create_embedding_plot(
    state: TrainingState, projection: str = "direct"
) -> go.Figure:
    """
    Create a Plotly scatter plot of the current embedding.

    Parameters
    ----------
    state : TrainingState
        Current training state
    projection : str
        Projection method for spherical embeddings

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    if state.embeddings is None or state.labels is None:
        # Empty plot
        fig.update_layout(
            title="Embedding Visualization",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=700,
            height=600,
            template="plotly_white",
        )
        return fig

    # Project to 2D
    embeddings = state.embeddings
    k = state.curvature

    try:
        x, y = project_to_2d(
            embeddings,
            k=k,
            i=0,
            j=1,
            projection=projection if k > 0 else "direct",
        )
    except Exception as e:
        print(f"Projection error: {e}")
        # Fallback to direct coordinates
        if embeddings.shape[1] >= 2:
            x = embeddings[:, 0]
            y = embeddings[:, 1]
        else:
            x = embeddings[:, 0]
            y = np.zeros_like(x)

    # Color by digit class
    labels = state.labels
    unique_labels = np.unique(labels)

    # Create color palette
    colors = [
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

    for label_idx, label in enumerate(unique_labels):
        mask = labels == label
        color = colors[label_idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=f"Digit {int(label)}",
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=f"Digit {int(label)}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
            )
        )

    # Add unit circle/boundary for curved spaces
    if k != 0:
        theta = np.linspace(0, 2 * np.pi, 100)
        if k > 0:
            # For spherical, the boundary depends on projection
            if projection == "stereographic":
                # Stereographic can extend beyond unit circle
                pass
            else:
                # Unit circle boundary
                circle_x = np.cos(theta)
                circle_y = np.sin(theta)
                fig.add_trace(
                    go.Scatter(
                        x=circle_x,
                        y=circle_y,
                        mode="lines",
                        name="Boundary",
                        line=dict(color="black", dash="dash", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        else:
            # Hyperbolic (Poincaré disk boundary)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    name="Boundary",
                    line=dict(color="black", dash="dash", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Update layout
    geometry = "Euclidean" if k == 0 else ("Spherical" if k > 0 else "Hyperbolic")
    title = f"Embedding Visualization - {geometry} (k={k})<br>Iteration {state.iteration}/{state.max_iterations}"

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=700,
        height=600,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    # Equal aspect ratio
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    return fig


def create_loss_chart(state: TrainingState) -> go.Figure:
    """
    Create a Plotly line chart of loss history.

    Parameters
    ----------
    state : TrainingState
        Current training state

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    if len(state.loss_history) == 0:
        # Empty plot
        fig.update_layout(
            title="Loss History",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            width=700,
            height=300,
            template="plotly_white",
        )
        return fig

    # Extract iterations and losses
    iterations = [item[0] for item in state.loss_history]
    losses = [item[1] for item in state.loss_history]

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=losses,
            mode="lines+markers",
            name="Loss",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="Iteration: %{x}<br>Loss: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Loss History",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log",  # Log scale for loss
        width=700,
        height=300,
        template="plotly_white",
        hovermode="closest",
    )

    return fig


class EmbeddingPlot:
    """Interactive embedding plot component."""

    def __init__(self, training_manager, projection: str = "direct"):
        """
        Initialize embedding plot.

        Parameters
        ----------
        training_manager : TrainingManager
            Training manager instance
        projection : str
            Projection method for spherical embeddings
        """
        self.training_manager = training_manager
        self.projection = projection
        self.plot = None
        self.timer = None

    def create(self) -> ui.plotly:
        """Create the plot UI element."""
        state = self.training_manager.get_state()
        fig = create_embedding_plot(state, self.projection)
        self.plot = ui.plotly(fig).classes("w-full")

        # Set up auto-update timer
        self.timer = ui.timer(1.0, self.update)

        return self.plot

    def update(self):
        """Update the plot with current state."""
        state = self.training_manager.get_state()

        # Only update if training is running or just completed
        if state.status in [TrainingStatus.RUNNING, TrainingStatus.COMPLETED]:
            fig = create_embedding_plot(state, self.projection)
            if self.plot is not None:
                self.plot.update_figure(fig)

    def set_projection(self, projection: str):
        """Change projection method."""
        self.projection = projection
        self.update()


class LossChart:
    """Interactive loss chart component."""

    def __init__(self, training_manager):
        """
        Initialize loss chart.

        Parameters
        ----------
        training_manager : TrainingManager
            Training manager instance
        """
        self.training_manager = training_manager
        self.plot = None
        self.timer = None

    def create(self) -> ui.plotly:
        """Create the chart UI element."""
        state = self.training_manager.get_state()
        fig = create_loss_chart(state)
        self.plot = ui.plotly(fig).classes("w-full")

        # Set up auto-update timer
        self.timer = ui.timer(1.0, self.update)

        return self.plot

    def update(self):
        """Update the chart with current state."""
        state = self.training_manager.get_state()

        # Only update if training is running or just completed
        if state.status in [TrainingStatus.RUNNING, TrainingStatus.COMPLETED]:
            fig = create_loss_chart(state)
            if self.plot is not None:
                self.plot.update_figure(fig)
