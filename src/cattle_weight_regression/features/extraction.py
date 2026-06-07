"""Feature extraction utilities (body measurements, keypoint-derived features)."""

from pathlib import Path

import numpy as np


def extract_bbox_features(bbox: list[float]) -> dict:
    """Derive simple geometric features from a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return {
        "width_px": width,
        "height_px": height,
        "area_px": width * height,
        "aspect_ratio": width / height if height > 0 else np.nan,
    }
