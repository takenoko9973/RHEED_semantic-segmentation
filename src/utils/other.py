import random

import numpy as np
import torch


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_f1_from_confusion_matrix(cm: np.ndarray) -> float:
    sum_over_row: int = cm.sum(axis=0)  # TP + FP
    sum_over_col: int = cm.sum(axis=1)  # TP + FN
    true_positives = np.diag(cm)  # TP

    denominator = sum_over_row + sum_over_col  # 2TP + FP + FN

    f1 = 2 * true_positives / denominator

    return f1, np.nanmean(f1)
