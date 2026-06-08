"""Train a weight estimation model.

Dispatches to the CNN or YOLO training path based on the `type` field
in the model config file.

Usage:
    uv run python scripts/train.py --model resnet50   # CNN path
    uv run python scripts/train.py --model yolo11     # YOLO path
"""

import argparse
from pathlib import Path

from cattle_weight_regression.config import load_config, load_model_config
from cattle_weight_regression.utils.logging import get_logger
from cattle_weight_regression.utils.mlflow_utils import setup_experiment

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")


def _train_cnn(model_cfg: dict, data_cfg: dict) -> None:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from cattle_weight_regression.data.dataset import CattleWeightDataset
    from cattle_weight_regression.data.transforms import get_transforms_from_config
    from cattle_weight_regression.models.pytorch.cnn import CattleWeightCNN
    from cattle_weight_regression.training.trainer import RegressionTrainer

    image_dir = Path(data_cfg["image_dir"])
    weight_col: str = data_cfg["weight_col"]
    features_cfg = load_config("features")
    batch_size: int = model_cfg.get("batch_size", 16)
    num_workers: int = model_cfg.get("num_workers", 0)

    for split in ("train", "val"):
        path = PROCESSED_DIR / f"labels_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run scripts/prepare_data.py first.")

    train_df = pd.read_csv(PROCESSED_DIR / "labels_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "labels_val.csv")
    logger.info("Loaded %d train / %d val images", len(train_df), len(val_df))

    pin = torch.cuda.is_available()
    train_ds = CattleWeightDataset(train_df, image_dir, get_transforms_from_config(features_cfg, "train"), weight_col=weight_col)
    val_ds = CattleWeightDataset(val_df, image_dir, get_transforms_from_config(features_cfg, "val"), weight_col=weight_col)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = CattleWeightCNN(
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=model_cfg.get("pretrained", True),
    )
    output_dir = Path("outputs/models") / model_cfg.get("name", "cnn_run")

    trainer = RegressionTrainer(model, train_loader, val_loader, lr=float(model_cfg.get("lr", 1e-4)))
    trainer.train(epochs=int(model_cfg.get("epochs", 50)), output_dir=output_dir)
    logger.info("Checkpoint saved to %s/model.pth", output_dir)


def _train_yolo(model_cfg: dict, data_cfg: dict) -> None:
    from cattle_weight_regression.models.yolo import YOLOWeightEstimator

    estimator = YOLOWeightEstimator(model_variant=model_cfg["model"])
    estimator.train(data_cfg, model_cfg)


def run(model_name: str = "resnet50") -> None:
    data_cfg = load_config("data")
    model_cfg = load_model_config(model_name)
    eval_cfg = load_config("evaluation")

    model_type: str = model_cfg.get("type", "")
    if not model_type:
        raise ValueError(f"Model config '{model_name}.yaml' is missing a 'type' field (expected 'cnn' or 'yolo').")

    setup_experiment(eval_cfg.get("experiment_name", "cattle_weight_regression"))
    logger.info("Training %s (type=%s)", model_name, model_type)

    if model_type == "cnn":
        _train_cnn(model_cfg, data_cfg)
    elif model_type == "yolo":
        _train_yolo(model_cfg, data_cfg)
    else:
        raise ValueError(f"Unknown model type '{model_type}' in {model_name}.yaml — expected 'cnn' or 'yolo'.")

    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", help="Model config stem (e.g. resnet50, yolo11)")
    args = parser.parse_args()
    run(args.model)