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


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Load the labels CSV and perform basic validation."""
    df = pd.read_csv(csv_path)
    required = {"image_path", "weight_kg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing columns: {missing}")
    return df


def split_dataframe(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / val / test sets."""
    test_size = round(1.0 - train, 10)
    val_ratio = val / test_size
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_ratio, random_state=seed)
    return train_df, val_df, test_df
