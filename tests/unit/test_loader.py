"""Unit tests for data loading utilities."""

import pytest

from cattle_weight_regression.data.loader import load_labels, split_dataframe


def test_load_labels_valid(sample_labels_csv):
    df = load_labels(sample_labels_csv)
    assert len(df) == 5
    assert "image_path" in df.columns
    assert "weight_kg" in df.columns


def test_load_labels_missing_column(tmp_path):
    import pandas as pd
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"image_path": ["a.jpg"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_labels(bad_csv)


def test_split_sizes(sample_labels_csv):
    from cattle_weight_regression.data.loader import load_labels
    df = load_labels(sample_labels_csv)
    train, val, test = split_dataframe(df, train=0.6, val=0.2)
    assert len(train) + len(val) + len(test) == len(df)
