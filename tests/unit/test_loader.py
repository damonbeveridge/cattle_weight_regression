"""Unit tests for data loading and splitting utilities."""

import pytest

from cattle_weight_regression.data.loader import (
    expand_to_images,
    load_labels,
    split_by_cow,
)


def test_load_labels_valid(sample_labels_csv):
    df = load_labels(sample_labels_csv)
    assert len(df) == 10
    assert "sku" in df.columns
    assert "weight_kg" in df.columns


def test_load_labels_missing_column(tmp_path):
    import pandas as pd
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"sku": ["BLF 1000"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_labels(bad_csv, sku_col="sku", weight_col="weight_kg")


def test_expand_to_images_row_count(sample_cow_df):
    expanded = expand_to_images(sample_cow_df, n_views=4)
    assert len(expanded) == len(sample_cow_df) * 4


def test_expand_to_images_path_format(sample_cow_df):
    expanded = expand_to_images(sample_cow_df, n_views=4)
    first_sku = sample_cow_df["sku"].iloc[0]
    paths = expanded[expanded["sku"] == first_sku]["image_path"].tolist()
    assert paths == [
        f"{first_sku}/{first_sku}_0.jpg",
        f"{first_sku}/{first_sku}_1.jpg",
        f"{first_sku}/{first_sku}_2.jpg",
        f"{first_sku}/{first_sku}_3.jpg",
    ]


def test_split_by_cow_no_leakage(sample_cow_df):
    """No cow's sku should appear in more than one split."""
    expanded = expand_to_images(sample_cow_df, n_views=4)
    train, val, test = split_by_cow(expanded)
    train_skus = set(train["sku"])
    val_skus   = set(val["sku"])
    test_skus  = set(test["sku"])
    assert train_skus.isdisjoint(val_skus)
    assert train_skus.isdisjoint(test_skus)
    assert val_skus.isdisjoint(test_skus)


def test_split_by_cow_covers_all_cows(sample_cow_df):
    train, val, test = split_by_cow(sample_cow_df)
    all_skus = set(train["sku"]) | set(val["sku"]) | set(test["sku"])
    assert all_skus == set(sample_cow_df["sku"])
