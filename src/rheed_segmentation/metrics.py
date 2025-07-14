import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor

from .utils.postprocessing import merge_masks_by_priority, merge_predictions_by_priority


def compute_confusion_matrix(
    outputs: Tensor, masks: Tensor | dict[str, Tensor], num_classes: int
) -> np.ndarray:
    probs = torch.softmax(outputs, dim=1)

    if isinstance(masks, dict):
        true_labels = merge_masks_by_priority(masks)
        preds = merge_predictions_by_priority(probs)
    else:
        true_labels = masks
        preds = torch.argmax(probs, dim=1)

    return confusion_matrix(
        true_labels.view(-1).cpu().numpy(),
        preds.view(-1).cpu().numpy(),
        labels=list(range(num_classes)),
    )


def compute_f1_from_confusion_matrix(cm: np.ndarray) -> tuple[np.ndarray, float]:
    sum_over_row: int = cm.sum(axis=0)  # TP + FP
    sum_over_col: int = cm.sum(axis=1)  # TP + FN
    true_positives = np.diag(cm)  # TP

    denominator = sum_over_row + sum_over_col  # 2TP + FP + FN

    f1 = 2 * true_positives / denominator

    return f1, np.nanmean(f1)
