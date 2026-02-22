"""Configuration management for web interface."""

from pathlib import Path
from typing import Any, Dict

import toml
import tomllib

from src.load_data import VALID_DATASETS

DEFAULT_CONFIG = {
    "data": {
        "dataset": "mnist",
        "n_samples": 1000,
    },
    "embedding": {
        "embed_dim": 2,
        "n_iterations": 1000,
        "init_method": "pca",
        "perplexity": 30.0,
        "early_exaggeration_iterations": 250,
        "early_exaggeration_factor": 12.0,
        "momentum_early": 0.5,
        "momentum_main": 0.8,
    },
    "hyperparameters": {
        "learning_rates": {"k": 200.0},
        "init_scale": "auto",
    },
    "experiments": {
        "curvatures": [-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2],
    },
    "evaluation": {
        "n_neighbors": 5,
    },
    "visualization": {
        "spherical_projection": "direct",
    },
}


def get_config_path() -> Path:
    """Get the path to config.toml."""
    return Path(__file__).parent.parent / "config.toml"


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.toml.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config if it doesn't exist
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config


def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to config.toml.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save
    """
    config_path = get_config_path()

    with open(config_path, "w") as f:
        toml.dump(config, f)


def validate_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate configuration values.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    try:
        # Validate data section
        if "data" not in config:
            return False, "Missing 'data' section"

        if config["data"].get("dataset") not in VALID_DATASETS:
            return False, f"Invalid dataset. Must be one of {VALID_DATASETS}"

        n_samples = config["data"].get("n_samples", -1)
        if not isinstance(n_samples, int) or (n_samples < 1 and n_samples != -1):
            return False, "n_samples must be a positive integer or -1 for all samples"

        # Validate embedding section
        if "embedding" not in config:
            return False, "Missing 'embedding' section"

        embed_dim = config["embedding"].get("embed_dim")
        if not isinstance(embed_dim, int) or embed_dim < 1:
            return False, "embed_dim must be a positive integer"

        n_iterations = config["embedding"].get("n_iterations")
        if not isinstance(n_iterations, int) or n_iterations < 1:
            return False, "n_iterations must be a positive integer"

        if config["embedding"].get("init_method") not in ["random", "pca"]:
            return False, "init_method must be 'random' or 'pca'"

        perplexity = config["embedding"].get("perplexity", 30.0)
        if not isinstance(perplexity, (int, float)) or perplexity <= 0:
            return False, "perplexity must be a positive number"

        early_exag_iters = config["embedding"].get("early_exaggeration_iterations")
        if not isinstance(early_exag_iters, int) or early_exag_iters < 0:
            return False, "early_exaggeration_iterations must be a non-negative integer"

        if early_exag_iters >= n_iterations:
            return False, "early_exaggeration_iterations must be less than n_iterations"

        early_exag_factor = config["embedding"].get("early_exaggeration_factor")
        if not isinstance(early_exag_factor, (int, float)) or early_exag_factor <= 0:
            return False, "early_exaggeration_factor must be a positive number"

        momentum_early = config["embedding"].get("momentum_early")
        if not isinstance(momentum_early, (int, float)) or not (
            0 <= momentum_early <= 1
        ):
            return False, "momentum_early must be between 0 and 1"

        momentum_main = config["embedding"].get("momentum_main")
        if not isinstance(momentum_main, (int, float)) or not (0 <= momentum_main <= 1):
            return False, "momentum_main must be between 0 and 1"

        # Validate hyperparameters section
        if "hyperparameters" not in config:
            return False, "Missing 'hyperparameters' section"

        learning_rates = config["hyperparameters"].get("learning_rates")
        if not isinstance(learning_rates, dict):
            return False, "learning_rates must be a dictionary"

        for key, value in learning_rates.items():
            if not isinstance(value, (int, float)) or value <= 0:
                return False, f"learning_rates[{key}] must be a positive number"

        # Validate experiments section
        if "experiments" not in config:
            return False, "Missing 'experiments' section"

        curvatures = config["experiments"].get("curvatures")
        if not isinstance(curvatures, list):
            return False, "curvatures must be a list"

        if len(curvatures) == 0:
            return False, "curvatures list cannot be empty"

        for curv in curvatures:
            if not isinstance(curv, (int, float)):
                return False, f"Invalid curvature value: {curv}"

        # Validate visualization section
        if "visualization" in config:
            proj = config["visualization"].get("spherical_projection")
            valid_projections = [
                "stereographic",
                "azimuthal_equidistant",
                "orthographic",
            ]
            if proj not in valid_projections:
                return False, f"spherical_projection must be one of {valid_projections}"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_default_config() -> Dict[str, Any]:
    """
    Get a copy of the default configuration.

    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()
