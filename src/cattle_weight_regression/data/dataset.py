"""PyTorch Datasets for cattle weight regression."""

from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CattleWeightDataset(Dataset):
    """Single-image dataset — one row per image view.

    Expects df to have an image_path column (relative to image_dir) and a weight
    column whose name matches weight_col. Use loader.expand_to_images() and
    loader.split_by_cow() to produce the split DataFrames.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform=None,
        weight_col: str = "weight_kg",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.weight_col = weight_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["image_path"]).convert("RGB")
        weight = float(row[self.weight_col])
        if self.transform:
            image = self.transform(image)
        return image, weight


class MultiViewCattleDataset(Dataset):
    """Multi-view dataset — one sample is all n views of a single cow.

    Expects df to be the raw per-cow DataFrame (one row per cow, not expanded).
    Returns (views, weight) where views is a list of n_views tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        n_views: int = 4,
        transform=None,
        sku_col: str = "sku",
        weight_col: str = "weight_kg",
    ):
        # Deduplicate to one row per cow in case an expanded df is accidentally passed
        self.df = df.drop_duplicates(subset=[sku_col]).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.n_views = n_views
        self.transform = transform
        self.sku_col = sku_col
        self.weight_col = weight_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], float]:
        row = self.df.iloc[idx]
        sku = row[self.sku_col]
        weight = float(row[self.weight_col])

        views = []
        for view_idx in range(self.n_views):
            img_path = self.image_dir / sku / f"{sku}_{view_idx}.jpg"
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            views.append(img)

        return views, weight
