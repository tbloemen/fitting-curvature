"""Plotly visualization components for embeddings."""

import plotly.graph_objects as go
from nicegui import ui

from web.training_manager import TrainingManager, TrainingState, TrainingStatus


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


class LossChart:
    """Interactive loss chart component."""

    def __init__(self, training_manager: TrainingManager):
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

        # Set up auto-update timer (5 FPS for smooth updates)
        self.timer = ui.timer(0.2, self.update)

        return self.plot

    def update(self):
        """Update the chart with current state."""
        state = self.training_manager.get_state()

        # Only update if training is running or just completed
        if state.status in [TrainingStatus.RUNNING, TrainingStatus.COMPLETED]:
            fig = create_loss_chart(state)
            if self.plot is not None:
                self.plot.update_figure(fig)
