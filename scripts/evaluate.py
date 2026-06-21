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

    from cattle_weight_regression.data.dataset import CattleWeightDataset, MultiViewCattleDataset
    from cattle_weight_regression.data.transforms import get_transforms_from_config
    from cattle_weight_regression.models.pytorch.cnn import CattleWeightCNN, MultiViewCattleWeightCNN

    image_dir = Path(data_cfg["image_dir"])
    weight_col: str = data_cfg["weight_col"]
    sku_col: str = data_cfg.get("sku_col", "sku")
    batch_size: int = model_cfg.get("batch_size", 16)
    num_workers: int = model_cfg.get("num_workers", 0)
    architecture: str = model_cfg.get("architecture", "single-view")
    n_views: int = model_cfg.get("n_views", 4)

    test_path = PROCESSED_DIR / "labels_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found — run scripts/prepare_data.py first.")

    test_df = pd.read_csv(test_path)
    val_transform = get_transforms_from_config(features_cfg, "val")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if architecture == "multi-view":
        logger.info("Architecture: multi-view (%d views, late-fusion)", n_views)
        test_df_eval = test_df.drop_duplicates(subset=[sku_col]).reset_index(drop=True)
        test_ds = MultiViewCattleDataset(test_df, image_dir, n_views=n_views, transform=val_transform, sku_col=sku_col, weight_col=weight_col)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = MultiViewCattleWeightCNN(backbone=model_cfg.get("backbone", "resnet50"), n_views=n_views, pretrained=False)
    else:
        logger.info("Architecture: single-view")
        test_df_eval = test_df
        test_ds = CattleWeightDataset(test_df, image_dir, val_transform, weight_col=weight_col)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = CattleWeightCNN(backbone=model_cfg.get("backbone", "resnet50"), pretrained=False)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds, all_skus, all_targets = [], [], []
    idx = 0
    with torch.no_grad():
        for images, weights in test_loader:
            batch_len = len(weights)
            if isinstance(images, (list, tuple)) and isinstance(images[0], torch.Tensor):
                images = [v.to(device) for v in images]
            else:
                images = images.to(device)
            preds = model(images).cpu().numpy()
            all_preds.extend(preds)
            all_skus.extend(test_df_eval[sku_col].iloc[idx : idx + batch_len].tolist())
            all_targets.extend(weights.numpy())
            idx += batch_len

    output_mean = data_cfg.get("output_mean")
    output_std = data_cfg.get("output_std")
    if output_mean is not None:
        all_preds = [p * output_std + output_mean for p in all_preds]

    # Saves predictions for each angle
    # new_df = pd.DataFrame()
    # new_df["all_preds"] = all_preds
    # new_df["all_skus"] = all_skus
    # new_df["all_targets"] = all_targets
    # new_df.to_csv(f"./outputs/predictions/{model_cfg.get("name")}.csv")

    if architecture == "multi-view":
        # Model already outputs one prediction per cow — no averaging needed.
        cow_df = pd.DataFrame({"sku": all_skus, "pred": all_preds, "true": all_targets})
        logger.info("Evaluated %d cows", len(cow_df))
    else:
        # Average the per-view predictions for each cow.
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

    plot_predictions(y_true, y_pred, output_dir / f"{model_cfg.get("name", model_name)}-pred_vs_actual.png")
    plot_residuals(y_true, y_pred, output_dir / f"{model_cfg.get("name", model_name)}-residuals.png")

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