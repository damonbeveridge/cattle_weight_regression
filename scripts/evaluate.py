"""Evaluate a trained model and log results to MLflow."""

import argparse
from pathlib import Path

import mlflow
import numpy as np

from cattle_weight_regression.config import load_config
from cattle_weight_regression.evaluation.metrics import compute_all
from cattle_weight_regression.evaluation.visualisation import plot_predictions, plot_residuals
from cattle_weight_regression.utils.logging import get_logger
from cattle_weight_regression.utils.mlflow_utils import setup_experiment

logger = get_logger(__name__)


def run(model_path: Path | None = None) -> None:
    eval_cfg = load_config("evaluation")
    output_dir = Path(eval_cfg["output_dir"])
    experiment_name = eval_cfg.get("experiment_name", "cattle_weight_regression")
    setup_experiment(experiment_name)

    # Placeholder: load predictions from the model
    y_true = np.array([])
    y_pred = np.array([])

    metrics = compute_all(y_true, y_pred)
    logger.info("Metrics: %s", metrics)

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        plot_predictions(y_true, y_pred, output_dir / "pred_vs_actual.png")
        plot_residuals(y_true, y_pred, output_dir / "residuals.png")
        mlflow.log_artifacts(str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    run(args.model_path)
