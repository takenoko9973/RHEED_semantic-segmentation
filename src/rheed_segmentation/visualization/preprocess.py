import numpy as np
import torch

from rheed_segmentation.config import TargetMode, TransformPipelineConfig


def preprocess(image: np.ndarray, transform: TransformPipelineConfig) -> torch.Tensor:
    compose = transform.to_transform_compose(target=TargetMode.VAL)
    transformed = compose(image=image)
    image_tensor: torch.Tensor = transformed["image"][0]
    return image_tensor.unsqueeze(0).unsqueeze(0)
