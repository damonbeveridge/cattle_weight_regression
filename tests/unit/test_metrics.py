"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from cattle_weight_regression.evaluation.metrics import compute_all, mae, r2, rmse


def test_mae_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_rmse_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_r2_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_mae_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    assert mae(y_true, y_pred) == pytest.approx(10.0)


def test_compute_all_returns_all_keys(dummy_predictions):
    y_true, y_pred = dummy_predictions
    result = compute_all(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "mape", "r2"}
