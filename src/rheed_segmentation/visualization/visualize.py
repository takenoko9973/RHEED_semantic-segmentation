from pathlib import Path

import numpy as np

from rheed_segmentation.utils.labelme import convert_color_lbl


def save_prediction(pred: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pred_image = convert_color_lbl(pred)
    pred_image.save(save_path.with_suffix(".png"))
