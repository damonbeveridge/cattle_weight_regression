"""Extract the raw .tar.gz archive, validate labels, and write split CSVs.

Produces three files in data/processed/:
  labels_train.csv, labels_val.csv, labels_test.csv

Each file has one row per IMAGE (4 views × N cows), with columns:
  sku, weight_kg, image_path, view_idx

The split is performed at the cow level so no cow's views span train and test.
"""

import argparse
from pathlib import Path

from cattle_weight_regression.config import load_config
from cattle_weight_regression.data.loader import (
    expand_to_images,
    extract_archive,
    load_labels,
    split_by_cow,
)
from cattle_weight_regression.utils.logging import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")


def run(archive: Path | None = None) -> None:
    cfg = load_config("data")
    archive = archive or Path(cfg["raw_dir"]) / "images.tar.gz"
    raw_dir = Path(cfg["raw_dir"])
    n_views: int = cfg.get("n_views", 4)
    sku_col: str = cfg.get("sku_col", "sku")
    weight_col: str = cfg.get("weight_col", "weight_kg")
    seed: int = cfg.get("seed", 42)
    train_split: float = cfg.get("train_split", 0.70)
    val_split: float = cfg.get("val_split", 0.15)

    # 1. Extract archive
    logger.info("Extracting %s → %s", archive, raw_dir)
    extract_archive(archive, raw_dir)

    # 2. Load and validate labels
    labels_path = Path(cfg["labels_file"])
    df = load_labels(labels_path, sku_col=sku_col, weight_col=weight_col)
    logger.info("Loaded %d cows", len(df))
    logger.info(
        "Weight range: %.1f – %.1f kg  |  mean: %.1f kg",
        df[cfg["weight_col"]].min(),
        df[cfg["weight_col"]].max(),
        df[cfg["weight_col"]].mean(),
    )

    # 3. Split by cow (prevents leakage), then expand to per-image rows
    train_cows, val_cows, test_cows = split_by_cow(df, sku_col=sku_col, train=train_split, val=val_split, seed=seed)
    logger.info("Split: %d train / %d val / %d test cows", len(train_cows), len(val_cows), len(test_cows))

    train_df = expand_to_images(train_cows, n_views=n_views, sku_col=sku_col)
    val_df   = expand_to_images(val_cows,   n_views=n_views, sku_col=sku_col)
    test_df  = expand_to_images(test_cows,  n_views=n_views, sku_col=sku_col)
    logger.info(
        "Expanded: %d train / %d val / %d test images",
        len(train_df), len(val_df), len(test_df),
    )

    # 4. Save split CSVs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = PROCESSED_DIR / f"labels_{name}.csv"
        split_df.to_csv(out, index=False)
        logger.info("Saved %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Path to images.tar.gz")
    args = parser.parse_args()
    run(args.archive)
