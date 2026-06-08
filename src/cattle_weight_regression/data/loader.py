"""Utilities for loading and preparing raw data."""

import tarfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def load_labels(
    csv_path: Path,
    sku_col: str = "sku",
    weight_col: str = "weight_kg",
) -> pd.DataFrame:
    """Load the labels CSV and perform basic validation."""
    df = pd.read_csv(csv_path)
    required = {sku_col, weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing columns: {missing}")
    return df


def expand_to_images(
    df: pd.DataFrame,
    n_views: int = 4,
    sku_col: str = "sku",
) -> pd.DataFrame:
    """Expand a per-cow DataFrame to one row per image view.

    Given a row with sku="BLF 2340", produces n_views rows with image_path set to
    "BLF 2340/BLF 2340_0.jpg", "BLF 2340/BLF 2340_1.jpg", etc.
    The original per-cow columns (including weight_kg) are preserved on every row.
    """
    rows = []
    for _, row in df.iterrows():
        sku = row[sku_col]
        for view_idx in range(n_views):
            new_row = row.to_dict()
            new_row["image_path"] = f"{sku}/{sku}_{view_idx}.jpg"
            new_row["view_idx"] = view_idx
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def split_by_cow(
    df: pd.DataFrame,
    sku_col: str = "sku",
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split at the cow level to prevent data leakage across views.

    All views of a given cow land in exactly one of train / val / test.
    df may be either the raw per-cow DataFrame or the expanded per-image DataFrame —
    the split key is always the sku column.
    """
    unique_skus = df[sku_col].unique()
    test_size = round(1.0 - train, 10)
    val_ratio = val / test_size

    train_skus, temp_skus = train_test_split(unique_skus, test_size=test_size, random_state=seed)
    val_skus, test_skus = train_test_split(temp_skus, test_size=1 - val_ratio, random_state=seed)

    train_df = df[df[sku_col].isin(train_skus)].reset_index(drop=True)
    val_df = df[df[sku_col].isin(val_skus)].reset_index(drop=True)
    test_df = df[df[sku_col].isin(test_skus)].reset_index(drop=True)

    return train_df, val_df, test_df
