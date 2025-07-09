import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from PIL import Image
from torch.utils.data import Dataset

from .loader import ImageLabelLoader
from .path import LabelPairPath


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
            # マスクの形式 (単一 or 辞書) に応じてデータ拡張の適用方法を切り替える
            if isinstance(mask, np.ndarray):
                transformed = self.transform(image=image, mask=mask)
                image: Image = transformed["image"]
                mask: np.ndarray = transformed["mask"]
            elif isinstance(mask, dict):
                # albumentationsが複数のマスクを扱えるようにターゲットを追加
                self.transform.add_targets(dict.fromkeys(mask, "mask"))
                transformed = self.transform(image=image, **mask)
                image: Image = transformed["image"]
                mask: dict[str, np.ndarray] = {k: transformed[k] for k in mask}

        return image, mask
