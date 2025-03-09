from pathlib import Path
from typing import Any

import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from RHEED_semantic_segmentation import utils


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        deg: str,
        image_type: str = "before",
        transform: BasicTransform | None = None,
    ) -> None:
        super().__init__()
        root_dir = Path(root_dir)

        self.image_root_dir = root_dir / image_type / deg
        self.mask_root_dir = root_dir / "masks" / deg
        self.image_paths = sorted(self.image_root_dir.glob("*.tiff"))
        self.mask_paths = sorted(self.mask_root_dir.glob("*.png"))

        if len(self.image_paths) != len(self.mask_paths):
            msg = "The number of images and masks must be the same."
            raise ValueError(msg)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image_path, mask_path = self.get_data_paths(index)

        image = utils.auto_scale(Image.open(image_path))
        mask = np.array(Image.open(mask_path), dtype=np.long)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask

    def get_data_paths(self, index: int) -> tuple[Path, Path]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        if image_path.stem != mask_path.stem:
            msg = "Image name do not match mask name."
            raise ValueError(msg)

        return image_path, mask_path


def make_datasets(
    root_dir: str | Path,
    image_type: str = "before",
    train_rate: float = 0.8,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
) -> Dataset:
    datasets_par_deg = []
    degs = ["0", "45", "90", "135"]
    for deg in degs:
        dataset = SegmentationDataset(root_dir, deg, image_type)
        datasets_par_deg.append(dataset)

    train_dataset, val_dataset = utils.split_dataset(
        ConcatDataset(datasets_par_deg), train_rate, train_transform, val_transform
    )
    return train_dataset, val_dataset


def make_dataloaders(
    root_dir: str | Path,
    image_type: str = "before",
    train_rate: float = 0.8,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
    loader_params: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = make_datasets(
        root_dir, image_type, train_rate, train_transform, val_transform
    )
    train_loader = DataLoader(train_dataset, drop_last=True, **loader_params)
    val_loader = DataLoader(val_dataset, drop_last=True, **loader_params)

    return train_loader, val_loader
