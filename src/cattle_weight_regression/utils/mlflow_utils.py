"""Convenience wrappers around the MLflow client."""

from pathlib import Path

import mlflow


def setup_experiment(name: str, tracking_uri: str = "sqlite:///mlruns.db") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name)
    return mlflow.get_experiment_by_name(name).experiment_id


def log_config(config: dict, prefix: str = "") -> None:
    """Flatten and log a nested config dict as MLflow params."""
    flat = _flatten(config, prefix)
    mlflow.log_params(flat)


def _flatten(d: dict, prefix: str = "") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items


def log_artifacts_dir(directory: Path) -> None:
    mlflow.log_artifacts(str(directory))