from typing import Any

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class GaussianBlur(ImageOnlyTransform):
    def __init__(self, ksize: float = (3, 3), sigma: float = -1, p: float = 1.0) -> None:
        super().__init__(p=p)

        self.ksize = ksize
        self.sigma = sigma

    def apply(self, img: np.ndarray, **_: Any) -> np.ndarray:  # noqa: ANN401
        return cv2.GaussianBlur(img, ksize=self.ksize, sigmaX=self.sigma)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
