import json as JSON  # noqa: N812
from pathlib import Path

import numpy as np
from PIL import Image

from RHEED_semantic_segmentation.utils import label as utils_label


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

    def obtain_images(self) -> tuple[Image.Image, np.ndarray, dict[np.ndarray]]:
        image = self.data_image.open_image()
        full_mask, masks = self.data_label.create_masks()

        return image, full_mask, masks


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

    def get_label_id(self, label_name: str) -> int:
        return self.label_name_to_value[label_name]

    def create_masks(self) -> dict[str, np.ndarray]:
        img_shape = self.data["imageHeight"], self.data["imageWidth"]

        full_mask, _ = utils_label.shapes_to_label(
            img_shape,
            sorted(self.data["shapes"], key=lambda shape: self.get_label_id(shape["label"])),
            self.label_name_to_value,
        )

        masks = {}
        for label in self.label_name_to_value:
            if self.get_label_id(label) <= 0:
                continue

            shapes = filter(lambda shape: shape["label"] == label, self.data["shapes"])
            mask, _ = utils_label.shapes_to_label(
                img_shape,
                shapes,
                {"_background_": 0, label: 1},
            )
            masks[label] = mask

        return full_mask, masks
