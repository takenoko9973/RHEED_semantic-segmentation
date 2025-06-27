from pathlib import Path

import torch

from rheed_segmentation.config import TrainingConfig
from rheed_segmentation.utils.postprocessing import merge_predictions_by_priority


def load_model(model_path: Path, training_config: TrainingConfig) -> torch.nn.Module:
    model = training_config.model
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, per_label: bool) -> torch.Tensor:
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.softmax(output, dim=1)
        if per_label:
            return merge_predictions_by_priority(output).cpu()
        return torch.argmax(output, dim=1).cpu()
