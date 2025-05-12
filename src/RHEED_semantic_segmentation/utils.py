import math
import random
import uuid
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageDraw
import torch
from albumentations.core.transforms_interface import BasicTransform
from loguru import logger
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


def shape_to_mask(
    img_shape: tuple[int, ...],
    points: list[list[float]],
    shape_type: str | None = None,
    line_width: int = 10,
    point_size: int = 5,
) -> npt.NDArray[np.bool_]:
    mask = PIL.Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    elif shape_type in [None, "polygon"]:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    else:
        raise ValueError(f"shape_type={shape_type!r} is not supported.")
    return np.array(mask, dtype=bool)


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask: npt.NDArray[np.bool_]
        if shape_type == "mask":
            if not isinstance(shape["mask"], np.ndarray):
                raise ValueError("shape['mask'] must be numpy.ndarray")
            mask = np.zeros(img_shape[:2], dtype=bool)
            (x1, y1), (x2, y2) = np.asarray(points).astype(int)
            mask[y1 : y2 + 1, x1 : x2 + 1] = shape["mask"]
        else:
            mask = shape_to_mask(img_shape[:2], points, shape_type)

        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


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
