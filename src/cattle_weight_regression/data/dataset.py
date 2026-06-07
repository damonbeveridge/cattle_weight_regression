"""PyTorch Dataset for cattle weight regression."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CattleWeightDataset(Dataset):
    def __init__(self, labels_csv: Path, image_dir: Path, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["image_path"]).convert("RGB")
        weight = float(row["weight_kg"])
        if self.transform:
            image = self.transform(image)
        return image, weight
