import json
from pathlib import Path

import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rheed_segmentation.config.experiment_config import ExperimentConfig
from rheed_segmentation.utils import labelme

from .label_pair_path import LabelPairPath
from .utils import collect_dataset_paths, split_data


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


def make_dataloaders(
    config: ExperimentConfig,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
) -> tuple[DataLoader, DataLoader]:
    # 必要な情報を config から取得
    batch_size: int = config.training.batch_size
    num_workers: int = config.training.num_workers
    per_labels: bool = config.per_label

    # ファイル一覧取得
    label_pair_paths = collect_dataset_paths(Path("data"), config.data_dirs)
    if not label_pair_paths:
        msg = "指定されたディレクトリにデータが見つかりませんでした。"
        raise FileNotFoundError(msg)

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
