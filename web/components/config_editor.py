"""Configuration editor UI component."""

from typing import Any, Callable, Dict

from nicegui import ui

from web.config_manager import (get_default_config, load_config, save_config,
                                validate_config)


class ConfigEditor:
    """Configuration editor component."""

    def __init__(self, on_save: Callable[[Dict[str, Any]], None]):
        """
        Initialize config editor.

        Parameters
        ----------
        on_save : Callable
            Callback function called when config is saved
        """
        self.on_save = on_save
        self.config = load_config()
        self.inputs = {}

    def create(self) -> ui.column:
        """
        Create the config editor UI.

        Returns
        -------
        ui.column
            NiceGUI column element containing the editor
        """
        with ui.column().classes("w-full gap-2") as container:
            ui.label("Configuration").classes("text-h5 font-bold mb-2")

            # Data section
            with ui.expansion("Data", icon="storage", value=True).classes("w-full"):
                with ui.column().classes("gap-2 p-2"):
                    self.inputs["dataset"] = ui.select(
                        label="Dataset",
                        options=["mnist"],
                        value=self.config["data"]["dataset"],
                    ).classes("w-full")

                    self.inputs["n_samples"] = ui.number(
                        label="Number of samples (-1 for all)",
                        value=self.config["data"]["n_samples"],
                        step=100,
                        min=-1,
                        format="%.0f",
                    ).classes("w-full")

            # Embedding section
            with ui.expansion(
                "Embedding Parameters", icon="settings", value=True
            ).classes("w-full"):
                with ui.column().classes("gap-2 p-2"):
                    self.inputs["embed_dim"] = ui.number(
                        label="Embedding dimension",
                        value=self.config["embedding"]["embed_dim"],
                        step=1,
                        min=1,
                        max=10,
                        format="%.0f",
                    ).classes("w-full")

                    self.inputs["n_iterations"] = ui.number(
                        label="Number of iterations",
                        value=self.config["embedding"]["n_iterations"],
                        step=100,
                        min=1,
                        format="%.0f",
                    ).classes("w-full")

                    self.inputs["init_method"] = ui.select(
                        label="Initialization method",
                        options=["pca", "random"],
                        value=self.config["embedding"]["init_method"],
                    ).classes("w-full")

                    self.inputs["perplexity"] = ui.number(
                        label="Perplexity",
                        value=self.config["embedding"]["perplexity"],
                        step=5,
                        min=5,
                        max=100,
                    ).classes("w-full")

                    self.inputs["early_exaggeration_iterations"] = ui.number(
                        label="Early exaggeration iterations",
                        value=self.config["embedding"]["early_exaggeration_iterations"],
                        step=50,
                        min=0,
                        format="%.0f",
                    ).classes("w-full")

                    self.inputs["early_exaggeration_factor"] = ui.number(
                        label="Early exaggeration factor",
                        value=self.config["embedding"]["early_exaggeration_factor"],
                        step=1,
                        min=1,
                    ).classes("w-full")

                    self.inputs["momentum_early"] = ui.number(
                        label="Momentum (early phase)",
                        value=self.config["embedding"]["momentum_early"],
                        step=0.1,
                        min=0,
                        max=1,
                    ).classes("w-full")

                    self.inputs["momentum_main"] = ui.number(
                        label="Momentum (main phase)",
                        value=self.config["embedding"]["momentum_main"],
                        step=0.1,
                        min=0,
                        max=1,
                    ).classes("w-full")

            # Hyperparameters section
            with ui.expansion("Hyperparameters", icon="tune", value=False).classes(
                "w-full"
            ):
                with ui.column().classes("gap-2 p-2"):
                    # Learning rate - simplified to single value
                    learning_rates = self.config["hyperparameters"]["learning_rates"]
                    lr_value = learning_rates.get("k", 200.0)

                    self.inputs["learning_rate"] = ui.number(
                        label="Learning rate",
                        value=lr_value,
                        step=10,
                        min=0.1,
                    ).classes("w-full")

                    # Init scale
                    init_scale = self.config["hyperparameters"]["init_scale"]
                    if init_scale == "auto":
                        init_scale_value = "auto"
                    else:
                        init_scale_value = str(init_scale)

                    self.inputs["init_scale"] = ui.input(
                        label='Init scale (use "auto" or a number)',
                        value=init_scale_value,
                    ).classes("w-full")

            # Experiments section
            with ui.expansion("Experiments", icon="science", value=True).classes(
                "w-full"
            ):
                with ui.column().classes("gap-2 p-2"):
                    curvatures = self.config["experiments"]["curvatures"]
                    current_curvature = curvatures[0] if curvatures else 0

                    self.inputs["curvature"] = ui.number(
                        label="Curvature",
                        value=current_curvature,
                        step=0.1,
                    ).classes("w-full")
                    ui.label("k < 0: Hyperbolic | k = 0: Euclidean | k > 0: Spherical").classes("text-xs text-gray-500")

            # Visualization section
            with ui.expansion("Visualization", icon="visibility", value=True).classes(
                "w-full"
            ):
                with ui.column().classes("gap-2 p-2"):
                    self.inputs["spherical_projection"] = ui.select(
                        label="Projection (for spherical k > 0)",
                        options={
                            "direct": "Direct",
                            "stereographic": "Stereographic",
                            "azimuthal_equidistant": "Azimuthal Equidistant",
                            "orthographic": "Orthographic",
                        },
                        value=self.config["visualization"]["spherical_projection"],
                    ).classes("w-full")
                    ui.label("Note: This setting only affects spherical embeddings (k > 0)").classes("text-xs text-gray-500")

            # Action buttons
            with ui.row().classes("w-full gap-2 mt-4"):
                ui.button(
                    "Save Configuration", on_click=self._save_config, icon="save"
                ).classes("flex-1").props("color=primary")
                ui.button(
                    "Reset to Defaults", on_click=self._reset_config, icon="refresh"
                ).classes("flex-1").props("color=secondary")

        return container

    def _save_config(self):
        """Save configuration from UI inputs."""
        try:
            # Build config from inputs
            config = {
                "data": {
                    "dataset": self.inputs["dataset"].value,
                    "n_samples": int(self.inputs["n_samples"].value),
                },
                "embedding": {
                    "embed_dim": int(self.inputs["embed_dim"].value),
                    "n_iterations": int(self.inputs["n_iterations"].value),
                    "init_method": self.inputs["init_method"].value,
                    "perplexity": float(self.inputs["perplexity"].value),
                    "early_exaggeration_iterations": int(
                        self.inputs["early_exaggeration_iterations"].value
                    ),
                    "early_exaggeration_factor": float(
                        self.inputs["early_exaggeration_factor"].value
                    ),
                    "momentum_early": float(self.inputs["momentum_early"].value),
                    "momentum_main": float(self.inputs["momentum_main"].value),
                },
                "hyperparameters": {
                    "learning_rates": {"k": float(self.inputs["learning_rate"].value)},
                    "init_scale": (
                        self.inputs["init_scale"].value
                        if self.inputs["init_scale"].value == "auto"
                        else float(self.inputs["init_scale"].value)
                    ),
                },
                "experiments": {
                    "curvatures": [self.inputs["curvature"].value],
                },
                "evaluation": self.config.get("evaluation", {"n_neighbors": 5}),
                "visualization": {
                    "spherical_projection": self.inputs["spherical_projection"].value,
                },
            }

            # Validate
            is_valid, error_message = validate_config(config)
            if not is_valid:
                ui.notify(f"Invalid configuration: {error_message}", type="negative")
                return

            # Save
            save_config(config)
            self.config = config

            ui.notify("Configuration saved successfully!", type="positive")

            # Call callback
            if self.on_save:
                self.on_save(config)

        except Exception as e:
            ui.notify(f"Error saving configuration: {str(e)}", type="negative")

    def _reset_config(self):
        """Reset configuration to defaults."""
        self.config = get_default_config()

        # Update UI inputs
        self.inputs["dataset"].value = self.config["data"]["dataset"]
        self.inputs["n_samples"].value = self.config["data"]["n_samples"]
        self.inputs["embed_dim"].value = self.config["embedding"]["embed_dim"]
        self.inputs["n_iterations"].value = self.config["embedding"]["n_iterations"]
        self.inputs["init_method"].value = self.config["embedding"]["init_method"]
        self.inputs["perplexity"].value = self.config["embedding"]["perplexity"]
        self.inputs["early_exaggeration_iterations"].value = self.config["embedding"][
            "early_exaggeration_iterations"
        ]
        self.inputs["early_exaggeration_factor"].value = self.config["embedding"][
            "early_exaggeration_factor"
        ]
        self.inputs["momentum_early"].value = self.config["embedding"]["momentum_early"]
        self.inputs["momentum_main"].value = self.config["embedding"]["momentum_main"]

        learning_rates = self.config["hyperparameters"]["learning_rates"]
        self.inputs["learning_rate"].value = learning_rates.get("k", 200.0)
        self.inputs["init_scale"].value = str(
            self.config["hyperparameters"]["init_scale"]
        )

        curvatures = self.config["experiments"]["curvatures"]
        self.inputs["curvature"].value = curvatures[0] if curvatures else 0

        self.inputs["spherical_projection"].value = self.config["visualization"][
            "spherical_projection"
        ]

        ui.notify("Configuration reset to defaults", type="info")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config

    def reload_config(self):
        """Reload configuration from file."""
        self.config = load_config()

    def get_projection(self) -> str:
        """Get current projection setting."""
        return self.inputs["spherical_projection"].value
