"""Training loop for PyTorch regression models (single-view and multi-view)."""

from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


class RegressionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        device: str | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.optimiser = AdamW(model.parameters(), lr=lr)
        self.model.to(self.device)

    def train(self, epochs: int, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with mlflow.start_run():
            mlflow.log_params({"epochs": epochs, "lr": self.optimiser.param_groups[0]["lr"]})
            for epoch in range(1, epochs + 1):
                train_loss = self._train_epoch()
                val_loss = self._val_epoch()
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                print(f"Epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
            torch.save(self.model.state_dict(), output_dir / "model.pth")
            mlflow.pytorch.log_model(self.model, "model")

    def _to_device(self, batch_input):
        """Move images to device, handling both single tensors and lists of tensors."""
        if isinstance(batch_input, (list, tuple)) and isinstance(batch_input[0], torch.Tensor):
            return [v.to(self.device) for v in batch_input]
        return batch_input.to(self.device)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for images, weights in self.train_loader:
            images = self._to_device(images)
            weights = weights.float().to(self.device)
            self.optimiser.zero_grad()
            loss = self.criterion(self.model(images), weights)
            loss.backward()
            self.optimiser.step()
            total += loss.item() * len(weights)
        return total / len(self.train_loader.dataset)

    def _val_epoch(self) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for images, weights in self.val_loader:
                images = self._to_device(images)
                weights = weights.float().to(self.device)
                total += self.criterion(self.model(images), weights).item() * len(weights)
        return total / len(self.val_loader.dataset)
