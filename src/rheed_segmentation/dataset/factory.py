from pathlib import Path

from albumentations.core.transforms_interface import BasicTransform
from torch.utils.data import DataLoader

from rheed_segmentation.config.experiment_config import ExperimentConfig

from .dataset import SegmentationDataset
from .finder import collect_dataset_paths
from .loader import ImageLabelLoader
from .splitter import split_data


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
