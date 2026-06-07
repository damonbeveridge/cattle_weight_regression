"""Train a weight estimation model using the config files."""

import argparse

from cattle_weight_regression.config import load_config, load_model_config
from cattle_weight_regression.models.yolo import YOLOWeightEstimator
from cattle_weight_regression.utils.logging import get_logger
from cattle_weight_regression.utils.mlflow_utils import setup_experiment

logger = get_logger(__name__)


def run(model: str = "yolo11") -> None:
    data_cfg = load_config("data")
    model_cfg = load_model_config(model)

    experiment_name = load_config("evaluation").get("experiment_name", "cattle_weight_regression")
    setup_experiment(experiment_name)

    logger.info("Training %s", model)
    estimator = YOLOWeightEstimator(model_variant=model_cfg["model"])
    estimator.train(data_cfg, model_cfg)
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11", help="Model config stem (e.g. yolo11, yolo26)")
    args = parser.parse_args()
    run(args.model)
