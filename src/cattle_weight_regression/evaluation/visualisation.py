"""Plotting utilities for evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Scatter plot of predicted vs actual weights with a perfect-fit line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect fit")
    ax.set_xlabel("Actual weight (kg)")
    ax.set_ylabel("Predicted weight (kg)")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Residual plot (error vs actual weight)."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_true, residuals, alpha=0.5, edgecolors="k", linewidths=0.4)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Actual weight (kg)")
    ax.set_ylabel("Residual (kg)")
    ax.set_title("Residuals")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.show()
