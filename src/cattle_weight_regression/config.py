"""Central config loader — reads YAML files from the configs/ directory."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config file by stem name (e.g. 'data', 'features')."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def load_model_config(name: str) -> dict:
    """Load a model-specific YAML config (e.g. 'yolo11')."""
    path = CONFIG_DIR / "models" / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)
