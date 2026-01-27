"""Training control UI component."""

from typing import TYPE_CHECKING, Optional

from nicegui import ui

if TYPE_CHECKING:
    from nicegui.elements.button import Button
    from nicegui.elements.label import Label
    from nicegui.elements.progress import LinearProgress
    from nicegui.elements.timer import Timer

from web.components.config_editor import ConfigEditor
from web.training_manager import TrainingManager, TrainingStatus


class TrainingControl:
    """Training control panel component."""

    def __init__(self, training_manager: TrainingManager, config_editor: ConfigEditor):
        """
        Initialize training control.

        Parameters
        ----------
        training_manager : TrainingManager
            Training manager instance
        config_editor : ConfigEditor
            Config editor instance
        """
        self.training_manager = training_manager
        self.config_editor = config_editor
        self.start_button: Optional[Button] = None
        self.stop_button: Optional[Button] = None
        self.progress_bar: Optional[LinearProgress] = None
        self.status_label: Optional[Label] = None
        self.iteration_label: Optional[Label] = None
        self.loss_label: Optional[Label] = None
        self.phase_label: Optional[Label] = None
        self.timer: Optional[Timer] = None

    def create(self) -> ui.column:
        """
        Create the training control UI.

        Returns
        -------
        ui.column
            NiceGUI column element containing the controls
        """
        with ui.column().classes("w-full gap-4") as container:
            # Header
            ui.label("Training Control").classes("text-h5 font-bold")

            # Status indicator
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Status:").classes("font-bold")
                self.status_label = ui.label("Idle").classes("text-lg")
                self.status_label.style("color: gray")

            # Control buttons
            with ui.row().classes("w-full gap-2"):
                self.start_button = (
                    ui.button(
                        "Start Training",
                        on_click=self._start_training,
                        icon="play_arrow",
                    )
                    .classes("flex-1")
                    .props("color=positive")
                )

                self.stop_button = (
                    ui.button(
                        "Stop Training", on_click=self._stop_training, icon="stop"
                    )
                    .classes("flex-1")
                    .props("color=negative")
                )
                self.stop_button.disable()

            # Progress section
            with ui.column().classes("w-full gap-2"):
                ui.label("Progress:").classes("font-bold")

                with ui.row().classes("w-full items-center gap-2"):
                    self.iteration_label = ui.label("Iteration: 0 / 0")
                    self.phase_label = ui.label("")

                self.progress_bar = ui.linear_progress(value=0).classes("w-full")

                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("Current Loss:").classes("font-bold")
                    self.loss_label = ui.label("N/A")

            # Set up auto-update timer
            self.timer = ui.timer(0.5, self._update_status)

        return container

    def _start_training(self) -> None:
        """Start training."""
        assert self.start_button is not None
        assert self.stop_button is not None

        try:
            # Get current config
            config = self.config_editor.get_config()

            # Validate
            from web.config_manager import validate_config

            is_valid, error_message = validate_config(config)
            if not is_valid:
                ui.notify(f"Invalid configuration: {error_message}", type="negative")
                return

            # Start training
            self.training_manager.start_training(config)

            # Update button states
            self.start_button.disable()
            self.stop_button.enable()

            ui.notify("Training started", type="positive")

        except Exception as e:
            ui.notify(f"Error starting training: {str(e)}", type="negative")

    def _stop_training(self) -> None:
        """Stop training."""
        assert self.stop_button is not None

        self.training_manager.stop_training()
        self.stop_button.disable()
        ui.notify("Stopping training...", type="info")

    def _update_status(self) -> None:
        """Update status display (called by timer)."""
        assert self.status_label is not None
        assert self.progress_bar is not None
        assert self.iteration_label is not None
        assert self.phase_label is not None
        assert self.loss_label is not None
        assert self.start_button is not None
        assert self.stop_button is not None

        state = self.training_manager.get_state()

        # Update status label
        status_colors = {
            TrainingStatus.IDLE: ("gray", "Idle"),
            TrainingStatus.PRECOMPUTING: ("purple", "Precomputing..."),
            TrainingStatus.RUNNING: ("green", "Running"),
            TrainingStatus.COMPLETED: ("blue", "Completed"),
            TrainingStatus.STOPPED: ("orange", "Stopped"),
            TrainingStatus.ERROR: ("red", "Error"),
        }

        color, text = status_colors.get(state.status, ("gray", "Unknown"))
        self.status_label.text = text
        self.status_label.style(f"color: {color}")

        # Update progress (iterations are 0-indexed, so add 1 for correct percentage)
        if state.max_iterations > 0:
            progress = (state.iteration + 1) / state.max_iterations
            self.progress_bar.value = progress
        else:
            self.progress_bar.value = 0

        # Update iteration label (iterations are 0-indexed, display as 1-indexed count)
        iteration_count = state.iteration + 1 if state.max_iterations > 0 else 0
        self.iteration_label.text = (
            f"Iteration: {iteration_count} / {state.max_iterations}"
        )

        # Update phase label
        if state.phase == "precomputing":
            self.phase_label.text = "(Computing affinities)"
            self.phase_label.style("color: purple")
        elif state.phase == "early":
            self.phase_label.text = "(Early Exaggeration)"
            self.phase_label.style("color: orange")
        elif state.phase == "main":
            self.phase_label.text = "(Main Phase)"
            self.phase_label.style("color: green")
        else:
            self.phase_label.text = ""

        # Update loss label
        if state.loss > 0:
            self.loss_label.text = f"{state.loss:.6f}"
        else:
            self.loss_label.text = "N/A"

        # Update button states based on status
        if state.status in [TrainingStatus.RUNNING, TrainingStatus.PRECOMPUTING]:
            self.start_button.disable()
            self.stop_button.enable()
        else:
            self.start_button.enable()
            self.stop_button.disable()

        # Show error message if error occurred
        if state.status == TrainingStatus.ERROR and state.error_message:
            ui.notify(f"Training error: {state.error_message}", type="negative")
            # Clear error message after showing
            state.error_message = ""
