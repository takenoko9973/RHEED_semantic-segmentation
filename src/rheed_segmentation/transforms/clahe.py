from typing import Any

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class CLAHE(ImageOnlyTransform):
    def __init__(
        self, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (4, 4), p: float = 1.0
    ) -> None:
        super().__init__(p=p)

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, img: np.ndarray, **_: Any) -> np.ndarray:  # noqa: ANN401
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(img)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
