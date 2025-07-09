import random

from .path import LabelPairPath


def split_data(
    label_path_pair: list[LabelPairPath], val_ratio: float = 0.2
) -> tuple[list[LabelPairPath], list[LabelPairPath]]:
    random.shuffle(label_path_pair)

    val_size = int(len(label_path_pair) * val_ratio)
    val_paths = label_path_pair[:val_size]
    train_paths = label_path_pair[val_size:]
    return train_paths, val_paths
