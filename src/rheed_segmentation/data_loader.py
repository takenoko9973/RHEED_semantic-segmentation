import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config.experiment_config import ExperimentConfig
from .utils import labelme


@dataclass(frozen=True)
class LabelPairPath:
    image_path: Path
    json_path: Path
    filename: str | None = None

    def __post_init__(self) -> None:
        if self.filename is None:
            object.__setattr__(self, "filename", self.image_path.name)


class ImageLabelLoader:
    """画像とラベルの読み込み用クラス"""

    def __init__(self, label_map: dict[str, int], generate_per_labels: bool = False) -> None:
        self.label_map = label_map
        self.generate_per_labels = generate_per_labels

    def load(self, lp: LabelPairPath) -> tuple[np.ndarray, np.ndarray | dict[str, np.ndarray]]:
        image = Image.open(lp.image_path)
        image = np.array(image).astype(np.uint16)

        with lp.json_path.open() as f:
            json_data = json.load(f)

        if self.generate_per_labels:
            mask = labelme.create_masks_per_labels(json_data, self.label_map)
        else:
            mask = labelme.create_mask(json_data, self.label_map)

        return image, mask


class SegmentationDataset(Dataset):
    def __init__(
        self,
        label_pair_paths: list[LabelPairPath],
        image_label_loader: ImageLabelLoader,
        transform: BasicTransform | None = None,
    ) -> None:
        # 0.0 は検証用にするため除外
        self.label_pair_paths = list(
            filter(lambda pair_path: pair_path.image_path.stem != "0.0", label_pair_paths)
        )
        self.image_label_loader = image_label_loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.label_pair_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, np.ndarray | dict[str, np.ndarray]]:
        image, mask = self.image_label_loader.load(self.label_pair_paths[idx])

        if self.transform:
            if isinstance(mask, np.ndarray):
                transformed = self.transform(image=image, mask=mask)
                image: Image = transformed["image"]
                mask: np.ndarray = transformed["mask"]
            elif isinstance(mask, dict):
                self.transform.add_targets(dict.fromkeys(mask, "mask"))
                transformed = self.transform(image=image, **mask)
                image: Image = transformed["image"]
                mask: dict[str, np.ndarray] = {k: transformed[k] for k in mask}

        return image, mask


def split_data(
    label_path_pair: list[LabelPairPath], val_ratio: float = 0.2
) -> tuple[list[LabelPairPath], list[LabelPairPath]]:
    random.shuffle(label_path_pair)

    val_size = int(len(label_path_pair) * val_ratio)
    val_paths = label_path_pair[:val_size]
    train_paths = label_path_pair[val_size:]
    return train_paths, val_paths


def load_paths(root_dir: Path, mask_paths: list[Path], image_type: str) -> list[LabelPairPath]:
    img_paths = []

    for mask_path in mask_paths:
        relative = mask_path.relative_to(root_dir)
        parts = list(relative.parts)
        parts[0] = image_type

        image_path = Path(root_dir, *parts).with_suffix(".tiff")
        if not image_path.exists():
            msg = f"Image not found: {image_path}"
            raise FileNotFoundError(msg)

        img_paths.append(image_path)

    return [LabelPairPath(img, mask) for img, mask in zip(img_paths, mask_paths, strict=True)]


def make_dataloaders(
    config: ExperimentConfig,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
) -> tuple[DataLoader, DataLoader]:
    # 必要な情報を config から取得
    data_dirs = config.data_dirs
    batch_size: int = config.training.batch_size
    num_workers: int = config.training.num_workers
    per_labels: bool = config.per_label

    # ファイル一覧取得
    label_pair_paths = []
    for data_dir in data_dirs:
        label_dir = Path("data", "label", data_dir)
        label_paths = sorted(label_dir.glob("**/*.json"))
        label_pair_paths.append(load_paths("data", label_paths, "raw"))

    label_pair_paths = list(itertools.chain.from_iterable(label_pair_paths))

    # ランダム分割
    train_paths, val_paths = split_data(label_pair_paths, val_ratio=0.2)

    # Dataset作成
    imageloader = ImageLabelLoader(config.labels, generate_per_labels=per_labels)
    train_dataset = SegmentationDataset(train_paths, imageloader, transform=train_transform)
    val_dataset = SegmentationDataset(val_paths, imageloader, transform=val_transform)

    # Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
