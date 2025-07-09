import json

import numpy as np
from PIL import Image

from rheed_segmentation.utils import labelme

from .path import LabelPairPath


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
