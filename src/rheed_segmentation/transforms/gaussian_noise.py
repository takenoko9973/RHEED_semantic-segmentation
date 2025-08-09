from typing import Any

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class GaussianNoise(ImageOnlyTransform):
    def __init__(
        self,
        std_range: tuple[float, float] = (0.2, 0.44),  # sqrt(10 / 255), sqrt(50 / 255)
        mean_range: tuple[float, float] = (0.0, 0.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        self.std_range = std_range
        self.mean_range = mean_range

    def apply(self, img: np.ndarray, **_: Any) -> np.ndarray:  # noqa: ANN401
        img_max = np.max(img)

        mean = (
            self.mean_range[0]
            if self.mean_range[0] == self.mean_range[1]
            else np.random.uniform(*self.mean_range)  # noqa: NPY002
        )
        std = (
            self.std_range[0]
            if self.std_range[0] == self.std_range[1]
            else np.random.uniform(*self.std_range)  # noqa: NPY002
        )

        noise = np.random.normal(mean, std, img.shape)  # noqa: NPY002
        noisy_image = img + noise
        return np.clip(noisy_image, 0, img_max)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
