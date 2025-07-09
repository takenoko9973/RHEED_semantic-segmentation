from pathlib import Path

import numpy as np
from PIL import Image
from torch import nn

from rheed_segmentation.config.experiment_config import ExperimentConfig
from rheed_segmentation.visualization.model import predict

from .preprocess import preprocess


def inference(config: ExperimentConfig, model: nn.Module, image_path: Path) -> np.ndarray:
    image = np.array(Image.open(image_path)).astype(np.uint16)
    image_tensor = preprocess(image, config.transforms)
    pred = predict(model, image_tensor, config.per_label)[0]

    return pred.numpy()
