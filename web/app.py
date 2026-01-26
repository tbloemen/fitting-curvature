"""Main web application for constant curvature embeddings."""

from nicegui import ui

from web.components.config_editor import ConfigEditor
from web.components.plots import LossChart
from web.components.threejs_plot import ThreeJSEmbeddingPlot
from web.components.training_control import TrainingControl
from web.training_manager import TrainingManager


def create_app():
    """Create and configure the NiceGUI application."""

    # Initialize managers
    training_manager = TrainingManager()

    # Create main page
    @ui.page("/")
    def main_page():
        """Main application page."""

        # Header
        with ui.header().classes("items-center justify-between"):
            ui.label("Constant Curvature Embeddings").classes("text-h4 font-bold")
            ui.label("Interactive t-SNE Visualization").classes("text-subtitle1")

        # Main layout with splitter
        with ui.splitter(value=30).classes("w-full h-screen") as splitter:
            # Left panel: Configuration editor
            with splitter.before:
                with ui.scroll_area().classes("w-full h-full p-4"):
                    config_editor = ConfigEditor(
                        on_save=lambda cfg: on_config_save(cfg)
                    )
                    config_editor.create()

            # Right panel: Training controls and visualizations
            with splitter.after:
                with ui.scroll_area().classes("w-full h-full p-4"):
                    with ui.column().classes("w-full gap-6"):
                        # Training control panel
                        training_control = TrainingControl(
                            training_manager, config_editor
                        )
                        training_control.create()

                        ui.separator()

                        # Visualization section
                        ui.label("Visualizations").classes("text-h5 font-bold")

                        # Embedding plot and loss chart side by side
                        with ui.row().classes("w-full gap-4").style("flex-wrap: nowrap;"):
                            # Embedding plot (left side) - fixed width
                            with ui.card().style("flex-shrink: 0;"):
                                ui.label("Embedding Plot").classes("text-h6 font-bold mb-2")

                                embedding_plot = ThreeJSEmbeddingPlot(
                                    training_manager, projection="direct"
                                )
                                embedding_plot.create()

                            # Loss chart (right side) - takes remaining space
                            with ui.card().classes("flex-1").style("min-width: 0;"):
                                ui.label("Loss History").classes("text-h6 font-bold mb-2")
                                loss_chart = LossChart(training_manager)
                                loss_chart.create()

                        # Connect config editor projection changes to embedding plot
                        config_editor.inputs["spherical_projection"].on(
                            "update:model-value",
                            lambda e: embedding_plot.set_projection(e.args),
                        )

                        # Instructions
                        with ui.expansion("Instructions", icon="help").classes(
                            "w-full mt-4"
                        ):
                            with ui.column().classes("gap-2 p-2"):
                                ui.markdown(
                                    """
                                ### How to Use

                                1. **Configure Settings**: Use the left panel to adjust embedding parameters
                                   - Select dataset and number of samples
                                   - Choose embedding dimension (typically 2 for visualization)
                                   - Set number of iterations
                                   - Configure perplexity and momentum parameters
                                   - Select a curvature value (k < 0 for hyperbolic, k = 0 for Euclidean, k > 0 for spherical)

                                2. **Save Configuration**: Click "Save Configuration" to persist your settings

                                3. **Start Training**: Click "Start Training" to begin the embedding process
                                   - The visualization updates every 10 iterations
                                   - Progress bar shows current iteration
                                   - Loss chart tracks convergence

                                4. **Monitor Progress**:
                                   - Watch the embedding plot evolve as clusters form
                                   - Check the loss chart to verify convergence
                                   - Phase indicator shows "Early Exaggeration" or "Main Phase"

                                5. **Stop Training**: Click "Stop Training" to halt the process early if needed

                                ### Tips

                                - **Quick Test**: Use n_samples=100 and n_iterations=100 for fast results
                                - **Production**: Use n_samples=1000-5000 and n_iterations=1000 for quality embeddings
                                - **Perplexity**: Typical values are 5-50, with 30 being a good default
                                - **Curvature**: Start with k=0 (Euclidean), then try k=-1 (hyperbolic) or k=1 (spherical)
                                - **Projection**: For spherical embeddings (k > 0), try different projections to see which visualizes best

                                ### Geometry Types

                                - **Hyperbolic (k < 0)**: Good for hierarchical data, displayed in Poincaré disk
                                - **Euclidean (k = 0)**: Standard t-SNE, no curvature
                                - **Spherical (k > 0)**: Good for circular/periodic data, displayed on sphere surface
                                """
                                )

        # Footer
        with ui.footer():
            ui.label("Constant Curvature Embeddings - Thesis Project").classes(
                "text-sm"
            )

        def on_config_save(config):
            """Handle configuration save."""
            # Clear data cache when config changes
            training_manager.clear_cache()


if __name__ in {"__main__", "__mp_main__"}:
    create_app()
    ui.run(
        title="Constant Curvature Embeddings",
        port=8080,
        reload=False,
        show=True,
    )
