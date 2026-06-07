"""Abstract base class that all weight-estimation models must implement."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseWeightEstimator(ABC):
    @abstractmethod
    def train(self, data_config: dict, model_config: dict) -> None:
        """Run training and log results to MLflow."""
        ...

    @abstractmethod
    def predict(self, image_path: Path) -> float:
        """Return predicted weight in kg for a single image."""
        ...

    @abstractmethod
    def evaluate(self, data_config: dict) -> dict:
        """Return a dict of metric_name → value on the test set."""
        ...

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Persist the trained model to disk."""
        ...

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load a previously saved model from disk."""
        ...
