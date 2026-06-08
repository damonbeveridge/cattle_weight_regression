"""Image transforms and augmentation pipelines."""

from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_hw(image_size) -> tuple[int, int]:
    """Return (H, W) from either an int (square) or a [W, H] list/tuple from config."""
    if isinstance(image_size, int):
        return (image_size, image_size)
    w, h = image_size   # config convention is [width, height]
    return (h, w)


def get_train_transforms(
    image_size: int | list | tuple = 640,
    mean: list[float] = _IMAGENET_MEAN,
    std: list[float] = _IMAGENET_STD,
    horizontal_flip: float = 0.5,
    rotation_degrees: int = 0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
) -> transforms.Compose:
    h, w = _to_hw(image_size)
    pipeline = [transforms.Resize((h, w))]

    if horizontal_flip > 0:
        pipeline.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
    if rotation_degrees > 0:
        pipeline.append(transforms.RandomRotation(degrees=rotation_degrees))
    if brightness_jitter > 0 or contrast_jitter > 0:
        pipeline.append(transforms.ColorJitter(brightness=brightness_jitter, contrast=contrast_jitter))

    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(pipeline)


def get_val_transforms(
    image_size: int | list | tuple = 640,
    mean: list[float] = _IMAGENET_MEAN,
    std: list[float] = _IMAGENET_STD,
) -> transforms.Compose:
    h, w = _to_hw(image_size)
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transforms_from_config(cfg: dict, mode: str) -> transforms.Compose:
    """Build a transform pipeline from the contents of features.yaml.

    Args:
        cfg:  parsed features.yaml dict
        mode: 'train' or 'val'
    """
    image_size = cfg.get("image_size", 640)
    mean: list[float] = cfg.get("mean", _IMAGENET_MEAN)
    std: list[float] = cfg.get("std", _IMAGENET_STD)

    if mode == "val":
        return get_val_transforms(image_size=image_size, mean=mean, std=std)

    aug = cfg.get("augmentation", {})
    aug_enabled: bool = aug.get("enabled", True)

    return get_train_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        horizontal_flip=aug.get("horizontal_flip", 0.0) if aug_enabled else 0.0,
        rotation_degrees=aug.get("rotation_degrees", 0) if aug_enabled else 0,
        brightness_jitter=aug.get("brightness_jitter", 0.0) if aug_enabled else 0.0,
        contrast_jitter=aug.get("contrast_jitter", 0.0) if aug_enabled else 0.0,
    )