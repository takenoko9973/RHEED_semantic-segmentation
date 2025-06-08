from pathlib import Path

import numpy as np
from PIL import Image

from rheed_segmentation.config.experiment_config import load_config
from rheed_segmentation.visualization.model import load_model, predict
from rheed_segmentation.visualization.preprocess import preprocess
from rheed_segmentation.visualization.visualize import save_prediction

model_name = "best.pth"

result_root = Path("results")
data_root = Path("data")
sample_image_paths = [
    (
        "SC-STO-250422/expo50_gain60/raw/250425_900_fil-6_O2-0/0/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-0/0.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250425_900_fil-6_O2-0/45/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-0/45.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250424_900_fil-6_O2-10/0/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-10/0.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250424_900_fil-6_O2-10/45/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-10/45.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250425_900_fil-6_O2-20/0/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-20/0.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250425_900_fil-6_O2-20/45/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-20/45.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250430_900_fil-7_O2-30/0/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-30/0.png",
    ),
    (
        "SC-STO-250422/expo50_gain60/raw/250430_900_fil-7_O2-30/45/0.0.tiff",
        "SC-STO-250422/expo50_gain60/O2-30/45.png",
    ),
]


def main() -> None:
    for experiment_path in result_root.glob("*/"):
        config_path = experiment_path / "config.yaml"
        model_path = experiment_path / model_name
        pred_path = experiment_path / "preds"

        if pred_path.exists():
            continue

        config = load_config(config_path)
        model = load_model(model_path, config.training)

        for image_path, save_path in sample_image_paths:
            image = np.array(Image.open(data_root / image_path)).astype(np.uint16)
            image_tensor = preprocess(image, config.transforms)
            pred = predict(model, image_tensor, config.per_label)[0]
            save_prediction(pred.numpy(), pred_path / save_path)


if __name__ == "__main__":
    main()
