import re
from pathlib import Path

import numpy as np
from PIL import Image

from rheed_segmentation.config.experiment_config import load_config
from rheed_segmentation.utils.result_manager import ResultDirManager
from rheed_segmentation.visualization.model import load_model, predict
from rheed_segmentation.visualization.preprocess import preprocess
from rheed_segmentation.visualization.visualize import save_prediction

model_name = "best.pth"

result_root = Path("results")
data_root = Path("data")
sample_image_paths = [
    "SC-STO-250422/expo50_gain60/250425_900_fil-6_O2-0",
    "SC-STO-250422/expo50_gain60/250424_900_fil-6_O2-10",
    "SC-STO-250422/expo50_gain60/250425_900_fil-6_O2-20",
    "SC-STO-250422/expo50_gain60/250430_900_fil-7_O2-30",
    "SC-MgO-250430/expo50_gain60/250508_900_fil-7_O2-0",
    "SC-MgO-250430/expo50_gain60/250508_900_fil-7_O2-10",
    "SC-MgO-250430/expo50_gain60/250513_900_fil-8_O2-20",
]


def inference(experiment_path: Path, data_dir: Path) -> None:
    data_name = re.match(r".+(O2\-\d+)", data_dir.name)[1]

    config_path = experiment_path / "config.yaml"
    model_path = experiment_path / model_name
    pred_path = experiment_path / "preds" / data_dir.with_name(data_name)

    if pred_path.exists():
        return

    config = load_config(config_path)
    model = load_model(model_path, config.training)

    for rot in ["0", "45"]:
        image_path = data_root / "raw" / data_dir / rot / "0.0.tiff"

        if not image_path.exists():
            continue

        image = np.array(Image.open(image_path)).astype(np.uint16)
        image_tensor = preprocess(image, config.transforms)
        pred = predict(model, image_tensor, config.per_label)[0]
        save_prediction(pred.numpy(), pred_path / f"{rot}.tiff")


def main() -> None:
    result_dir_manager = ResultDirManager()
    result_date_dirs = result_dir_manager.get_result_dirs()

    for result_date_dir in result_date_dirs:
        for result_protocol_dir in result_date_dir.fetch_protocol_dirs():
            print(result_protocol_dir)
            for data_dir in sample_image_paths:
                inference(result_protocol_dir.path, Path(data_dir))


if __name__ == "__main__":
    main()
