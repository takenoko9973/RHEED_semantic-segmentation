import random
from typing import Any

import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform
from PIL import Image
from torch.utils.data import Dataset


def auto_scale(image: Image.Image) -> np.ndarray:
    """PIL画像を自動的にビット深度に基づいて[0, 1]にスケーリング

    Args:
        image (PIL.Image.Image): PIL画像

    Returns:
        np.ndarray: スケーリングされた画像データ ([0, 1] 範囲のNumPy配列)

    """
    bit_depth = get_image_bit_depth(image)
    max_pixel_value: int = 2**bit_depth

    # 正規化
    image_array = np.array(image, dtype=np.float32)
    return image_array / max_pixel_value


def get_image_bit_depth(image: Image.Image) -> int:
    # モードに基づいてビット深度を推定
    mode_to_bit_depth = {
        "L": 8,  # グレースケール
        "P": 8,  # インデックスドカラー
        "I;16": 16,  # 16ビット整数
        "I;16B": 16,  # 16ビット整数
        "I": 32,  # 32ビット整数
        "F": 32,  # 32ビット浮動小数点
    }

    # ビット深度を取得
    return mode_to_bit_depth.get(image.mode, 8)


class SegmentationSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        transform: BasicTransform | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, mask = self.dataset[self.indices[index]]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask

    def __len__(self) -> int:
        return len(self.indices)


def split_dataset(
    dataset: Dataset,
    train_rate: float,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
    suffle_seed: int = 0,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < train_rate < 1.0:
        msg = "The split ratio must be in the range (0, 1)."
        raise ValueError(msg)

    train_size = int(train_rate * len(dataset))

    rgn = np.random.default_rng(suffle_seed)
    indices = rgn.permutation(np.arange(len(dataset)))

    train_dataset = SegmentationSubset(dataset, indices[:train_size], train_transform)
    val_dataset = SegmentationSubset(dataset, indices[train_size:], val_transform)

    return train_dataset, val_dataset


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_f1_from_confusion_matrix(cm: np.ndarray) -> float:
    sum_over_row: int = cm.sum(axis=0)  # TP + FP
    sum_over_col: int = cm.sum(axis=1)  # TP + FN
    true_positives = np.diag(cm)  # TP

    denominator = sum_over_row + sum_over_col  # 2TP + FP + FN

    f1 = 2 * true_positives / denominator

    return f1, np.nanmean(f1)
