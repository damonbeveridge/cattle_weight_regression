"""Training loop for PyTorch regression models (single-view and multi-view)."""

from pathlib import Path
import time

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
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
        val_df: pd.DataFrame | None = None,
        sku_col: str = "sku",
        weight_col: str = "weight_kg",
    ):
        """
        Args:
            val_df:     The validation split DataFrame (same one passed to CattleWeightDataset).
                        Required for per-cow MAE tracking during training. If None, only
                        per-image val loss is logged — which does not match the test metric.
            sku_col:    Column name for the cow identifier in val_df.
            weight_col: Column name for the target weight in val_df.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.optimiser = AdamW(model.parameters(), lr=lr)
        self.val_df = val_df
        self.sku_col = sku_col
        self.weight_col = weight_col
        self.model.to(self.device)

    def train(self, epochs: int, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with mlflow.start_run():
            mlflow.log_params({"epochs": epochs, "lr": self.optimiser.param_groups[0]["lr"]})
            for epoch in range(1, epochs + 1):
                start_epoch = time.time()

                train_loss = self._train_epoch()
                val_loss = self._val_epoch()

                log = {"train_loss": train_loss, "val_loss": val_loss}

                if self.val_df is not None:
                    val_mae = self._val_mae_per_cow()
                    log["val_mae"] = val_mae
                    print(
                        f"Epoch {epoch}/{epochs}  "
                        f"Duration: {time.time()-start_epoch:.2f} seconds  "
                        f"train_loss={train_loss:.4f}  "
                        f"val_loss={val_loss:.4f}  "
                        f"val_mae={val_mae:.2f}kg"
                    )
                else:
                    print(f"Epoch {epoch}/{epochs}  Duration: {time.time()-start_epoch:.2f} seconds  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

                mlflow.log_metrics(log, step=epoch)

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
        """Per-image MSE loss — used for convergence tracking."""
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for images, weights in self.val_loader:
                images = self._to_device(images)
                weights = weights.float().to(self.device)
                total += self.criterion(self.model(images), weights).item() * len(weights)
        return total / len(self.val_loader.dataset)

    def _val_mae_per_cow(self) -> float:
        """Per-cow MAE — averages the 4 per-view predictions per cow before scoring.

        This is the metric that matches test evaluation and should be used for
        model selection and early stopping decisions.
        """
        self.model.eval()
        all_preds, all_skus, all_targets = [], [], []
        idx = 0
        with torch.no_grad():
            for images, weights in self.val_loader:
                batch_len = len(weights)
                preds = self.model(self._to_device(images)).cpu().numpy()
                all_preds.extend(preds)
                all_skus.extend(self.val_df[self.sku_col].iloc[idx : idx + batch_len].tolist())
                all_targets.extend(self.val_df[self.weight_col].iloc[idx : idx + batch_len].tolist())
                idx += batch_len

        cow_df = pd.DataFrame({"sku": all_skus, "pred": all_preds, "true": all_targets})
        per_cow = cow_df.groupby("sku").agg(pred=("pred", "mean"), true=("true", "first"))
        return float(np.mean(np.abs(per_cow["true"].values - per_cow["pred"].values)))