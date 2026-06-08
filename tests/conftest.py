"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cow_df() -> pd.DataFrame:
    """Per-cow DataFrame (10 cows) matching the real labels structure."""
    return pd.DataFrame({
        "sku": [f"BLF {1000 + i}" for i in range(10)],
        "weight_kg": [350.0, 420.5, 510.2, 280.0, 615.8, 390.0, 445.0, 502.5, 310.0, 580.0],
    })


@pytest.fixture
def sample_labels_csv(tmp_path: Path, sample_cow_df: pd.DataFrame) -> Path:
    """Write the per-cow DataFrame to a temporary CSV."""
    path = tmp_path / "dataset.csv"
    sample_cow_df.to_csv(path, index=False)
    return path


@pytest.fixture
def dummy_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Simple y_true / y_pred pair for metric tests."""
    y_true = np.array([350.0, 420.5, 510.2, 280.0, 615.8])
    y_pred = np.array([360.0, 410.0, 520.0, 275.0, 600.0])
    return y_true, y_pred
