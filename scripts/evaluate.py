"""Evaluate a trained model on the test set and log results to MLflow.

Dispatches to CNN or YOLO evaluation based on the `type` field in the model config.
For the CNN path, predictions are averaged across the 4 views of each cow before
computing metrics — this reflects the actual task (predict weight per animal).

Usage:
    uv run python scripts/evaluate.py --model resnet50 --checkpoint outputs/models/resnet50_baseline/model.pth
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from cattle_weight_regression.config import load_config, load_config_from_dir, load_model_config
from cattle_weight_regression.evaluation.metrics import compute_all
from cattle_weight_regression.evaluation.visualisation import plot_predictions, plot_residuals
from cattle_weight_regression.utils.logging import get_logger
from cattle_weight_regression.utils.mlflow_utils import setup_experiment

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")


def _evaluate_cnn(model_cfg: dict, data_cfg: dict, features_cfg: dict, checkpoint: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from cattle_weight_regression.data.dataset import CattleWeightDataset
    from cattle_weight_regression.data.transforms import get_transforms_from_config
    from cattle_weight_regression.models.pytorch.cnn import CattleWeightCNN

    image_dir = Path(data_cfg["image_dir"])
    weight_col: str = data_cfg["weight_col"]
    sku_col: str = data_cfg.get("sku_col", "sku")
    batch_size: int = model_cfg.get("batch_size", 16)
    num_workers: int = model_cfg.get("num_workers", 0)

    test_path = PROCESSED_DIR / "labels_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found — run scripts/prepare_data.py first.")

    test_df = pd.read_csv(test_path)
    test_ds = CattleWeightDataset(test_df, image_dir, get_transforms_from_config(features_cfg, "val"), weight_col=weight_col)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CattleWeightCNN(backbone=model_cfg.get("backbone", "resnet50"), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds, all_skus, all_targets = [], [], []
    idx = 0
    with torch.no_grad():
        for images, weights in test_loader:
            batch_len = len(weights)
            preds = model(images.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_skus.extend(test_df[sku_col].iloc[idx : idx + batch_len].tolist())
            all_targets.extend(weights.numpy())
            idx += batch_len

    # Average the 4 per-view predictions for each cow
    results_df = pd.DataFrame({"sku": all_skus, "pred": all_preds, "true": all_targets})
    cow_df = results_df.groupby("sku").agg(pred=("pred", "mean"), true=("true", "first")).reset_index()
    logger.info("Evaluated %d images → %d cows", len(results_df), len(cow_df))

    y_true = cow_df["true"].to_numpy()
    y_pred = cow_df["pred"].to_numpy()
    return compute_all(y_true, y_pred), y_true, y_pred


def run(model_name: str, checkpoint: Path | None) -> None:
    # Load training-time configs from the model's snapshot dir when available,
    # so evaluation always uses the same preprocessing and architecture as training.
    config_dir = checkpoint.parent / "configs" if checkpoint is not None else None
    if config_dir is not None and config_dir.exists():
        logger.info("Loading configs from snapshot at %s", config_dir)
        data_cfg = load_config_from_dir("data", config_dir)
        model_cfg = load_config_from_dir("model", config_dir)
        features_cfg = load_config_from_dir("features", config_dir)
    else:
        if config_dir is not None:
            logger.warning(
                "No config snapshot found at %s — falling back to global configs/. "
                "Results may be incorrect if configs changed since training.",
                config_dir,
            )
        data_cfg = load_config("data")
        model_cfg = load_model_config(model_name)
        features_cfg = load_config("features")

    eval_cfg = load_config("evaluation")

    model_type: str = model_cfg.get("type", "")
    if not model_type:
        raise ValueError(f"Model config '{model_name}.yaml' is missing a 'type' field.")

    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_experiment(
        eval_cfg.get("experiment_name", "cattle_weight_regression"),
        tracking_uri=eval_cfg.get("tracking_uri", "sqlite:///mlruns.db"),
    )

    if model_type == "cnn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for CNN evaluation.")
        metrics, y_true, y_pred = _evaluate_cnn(model_cfg, data_cfg, features_cfg, checkpoint)
    elif model_type == "yolo":
        raise NotImplementedError("YOLO evaluation not yet implemented.")
    else:
        raise ValueError(f"Unknown model type '{model_type}' in {model_name}.yaml.")

    logger.info("Test metrics (per-cow): %s", metrics)

    plot_predictions(y_true, y_pred, output_dir / f"{model_name}_pred_vs_actual.png")
    plot_residuals(y_true, y_pred, output_dir / f"{model_name}_residuals.png")

    with mlflow.start_run():
        mlflow.log_params({"model": model_name, "checkpoint": str(checkpoint)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--checkpoint", type=Path, help="Path to model.pth (required for CNN)")
    args = parser.parse_args()
    run(args.model, args.checkpoint)