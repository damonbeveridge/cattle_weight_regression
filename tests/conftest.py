"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_labels_csv(tmp_path: Path) -> Path:
    """A tiny labels CSV with 5 rows for use in tests."""
    df = pd.DataFrame({
        "image_path": [f"cow_{i:03d}.jpg" for i in range(5)],
        "weight_kg": [350.0, 420.5, 510.2, 280.0, 615.8],
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def dummy_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Simple y_true / y_pred pair for metric tests."""
    y_true = np.array([350.0, 420.5, 510.2, 280.0, 615.8])
    y_pred = np.array([360.0, 410.0, 520.0, 275.0, 600.0])
    return y_true, y_pred
