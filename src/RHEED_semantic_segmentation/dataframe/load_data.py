from pathlib import Path
from typing import Any

from albumentations.core.transforms_interface import BasicTransform
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from RHEED_semantic_segmentation.dataframe.rheed_data import RHEEDData
from RHEED_semantic_segmentation.utils import dataset as utils_dataset
from RHEED_semantic_segmentation.utils import image as utils_image


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_type: str = "raw",
        transform: BasicTransform | None = None,
    ) -> None:
        super().__init__()
        root_dir = Path(root_dir)

        mask_root_dir = root_dir / "label"

        label_paths = sorted(mask_root_dir.glob("**/*.json"))
        image_paths = []
        for label_path in label_paths:
            relative = label_path.relative_to(root_dir)
            parts = list(relative.parts)
            parts[0] = image_type
            image_path = Path(root_dir, *parts).with_suffix(".tiff")

            if not image_path.exists():
                msg = f"Image not found: {image_path}"
                raise FileNotFoundError(msg)

            image_paths.append(image_path)

        if len(image_paths) != len(label_paths):
            msg = "The number of images and masks must be the same."
            raise ValueError(msg)

        self.data_list: list[RHEEDData] = []
        for image_path, label_path in zip(image_paths, label_paths, strict=False):
            self.data_list.append(RHEEDData(image_path, label_path))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, mask = self.data_list[index].obtain_images()

        image = utils_image.auto_scale(image)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask

    def get_data_paths(self, index: int) -> tuple[Path, Path]:
        image_path, mask_path = self.data_list[index].get_paths()

        if image_path.stem != mask_path.stem:
            msg = "Image name do not match mask name."
            raise ValueError(msg)

        return image_path, mask_path


def make_dataset(
    root_dirs: list[str | Path],
    image_type: str = "raw",
) -> Dataset:
    datasets = []
    for root_dir in root_dirs:
        root_dir_path = Path(root_dir)
        if not root_dir_path.exists():
            msg = f"Root directory does not exist: {root_dir_path}"
            raise FileNotFoundError(msg)

        if not root_dir_path.is_dir():
            msg = f"Root path is not a directory: {root_dir_path}"
            raise NotADirectoryError(msg)

        dataset = SegmentationDataset(root_dir_path, image_type)
        datasets.append(dataset)

    return ConcatDataset(datasets)


def make_dataloaders(
    root_dirs: list[str | Path],
    image_type: str = "raw",
    train_rate: float = 0.8,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
    loader_params: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    dataset = make_dataset(root_dirs, image_type)

    train_dataset, val_dataset = utils_dataset.split_dataset(
        dataset, train_rate, train_transform, val_transform
    )
    train_loader = DataLoader(train_dataset, drop_last=True, **loader_params)
    val_loader = DataLoader(val_dataset, drop_last=True, **loader_params)

    return train_loader, val_loader
