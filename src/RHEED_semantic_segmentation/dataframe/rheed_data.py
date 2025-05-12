import json as JSON  # noqa: N812
from pathlib import Path

import numpy as np
from PIL import Image

from RHEED_semantic_segmentation import utils


class RHEEDData:
    def __init__(self, image_path: str | Path, label_path: str | Path) -> None:
        if image_path.stem != label_path.stem:
            msg = "The name of image and label must be the same."
            raise ValueError(msg)

        self.data_image = RHEEDDataImage(image_path)
        self.data_label = RHEEDDataLabel(label_path)

    def __repr__(self) -> str:
        return f"RHEEDData(image={self.data_image}, label={self.data_label})"

    def get_paths(self) -> tuple[Path, Path]:
        return self.data_image.image_path, self.data_label.label_path

    def obtain_images(self) -> tuple[Image.Image, np.ndarray]:
        image = self.data_image.open_image()
        label = self.data_label.open_image()

        return image, label


class RHEEDDataImage:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def __repr__(self) -> str:
        return f"RHEEDDataImage(image_path={self.image_path})"

    def open_image(self) -> Image.Image:
        return Image.open(self.image_path)


class RHEEDDataLabel:
    def __init__(self, label_path: Path) -> None:
        self.label_path = label_path
        with label_path.open() as f:
            self.data = JSON.load(f)

        self.label_name_to_value = {
            "__ignore__": -1,
            "_background_": 0,
            "spot": 1,
            "streak": 2,
            "kikuchi": 3,
        }

    def __repr__(self) -> str:
        return f"RHEEDDataLabel(label_path={self.label_path})"

    def open_image(self) -> np.ndarray:
        img_shape = self.data["imageHeight"], self.data["imageWidth"]
        lbl, _ = utils.shapes_to_label(
            img_shape, reversed(self.data["shapes"]), self.label_name_to_value
        )

        return lbl.astype(np.long)
