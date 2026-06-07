"""Extract the raw .tar.gz archive and validate the labels CSV."""

import argparse
from pathlib import Path

from cattle_weight_regression.data.loader import extract_archive, load_labels
from cattle_weight_regression.utils.logging import get_logger

logger = get_logger(__name__)


def run(archive: Path | None = None, dest: Path | None = None) -> None:
    archive = archive or Path("data/raw/images.tar.gz")
    dest = dest or Path("data/raw")

    logger.info("Extracting %s → %s", archive, dest)
    extract_archive(archive, dest)

    labels_path = dest / "labels.csv"
    df = load_labels(labels_path)
    logger.info("Loaded %d labelled samples", len(df))
    logger.info("Weight range: %.1f – %.1f kg", df["weight_kg"].min(), df["weight_kg"].max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--dest", type=Path)
    args = parser.parse_args()
    run(args.archive, args.dest)
