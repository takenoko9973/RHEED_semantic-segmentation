from typing import Any

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image


class AutoScaleTransform(ImageOnlyTransform):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **_: Any) -> np.ndarray:  # noqa: ANN401
        pil_image = Image.fromarray(img)

        bit_depth = self.get_image_bit_depth(pil_image)
        max_pixel_value = 2**bit_depth

        # [0, 1] に正規化
        return img.astype(np.float32) / max_pixel_value

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()

    @staticmethod
    def get_image_bit_depth(image: Image.Image) -> int:
        mode_to_bit_depth = {
            "L": 8,  # グレースケール
            "P": 8,  # インデックスカラー
            "I;16": 16,  # 16ビット整数
            "I;16B": 16,
            "I": 32,  # 32ビット整数
            "F": 32,  # 32ビット浮動小数点
        }
        return mode_to_bit_depth.get(image.mode, 8)
