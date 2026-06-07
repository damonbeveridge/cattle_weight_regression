"""YOLO-based cattle weight estimator using Ultralytics."""

from pathlib import Path

import mlflow
from ultralytics import YOLO

from cattle_weight_regression.models.base import BaseWeightEstimator


class YOLOWeightEstimator(BaseWeightEstimator):
    def __init__(self, model_variant: str = "yolo11n.pt"):
        self.model_variant = model_variant
        self.model: YOLO | None = None

    def train(self, data_config: dict, model_config: dict) -> None:
        self.model = YOLO(self.model_variant)
        with mlflow.start_run():
            mlflow.log_params(model_config)
            results = self.model.train(**model_config)
            mlflow.log_metrics({
                "final_loss": results.results_dict.get("train/box_loss", float("nan")),
            })

    def predict(self, image_path: Path) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        results = self.model(str(image_path))
        # Placeholder: extract regression output from results
        raise NotImplementedError("Map YOLO output to weight prediction.")

    def evaluate(self, data_config: dict) -> dict:
        raise NotImplementedError

    def save(self, output_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir / "weights.pt"))

    def load(self, model_path: Path) -> None:
        self.model = YOLO(str(model_path))
