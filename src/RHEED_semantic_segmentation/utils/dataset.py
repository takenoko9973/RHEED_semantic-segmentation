from typing import Any

import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from torch.utils.data import Dataset


class SegmentationSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        transform: BasicTransform | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, Any, dict]:
        image, full_mask, label_masks = self.dataset[self.indices[index]]

        if self.transform:
            self.transform.add_targets(dict.fromkeys(label_masks, "mask"))

            transformed = self.transform(image=image, mask=full_mask, **label_masks)
            image, full_mask, label_masks = (
                transformed["image"],
                transformed["mask"],
                {k: transformed[k] for k in label_masks},
            )

        return image, full_mask, label_masks

    def __len__(self) -> int:
        return len(self.indices)


def split_dataset(
    dataset: Dataset,
    train_rate: float,
    train_transform: BasicTransform | None = None,
    val_transform: BasicTransform | None = None,
    suffle_seed: int = 0,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < train_rate < 1.0:
        msg = "The split ratio must be in the range (0, 1)."
        raise ValueError(msg)

    train_size = int(train_rate * len(dataset))

    rgn = np.random.default_rng(suffle_seed)
    indices = rgn.permutation(np.arange(len(dataset)))

    train_dataset = SegmentationSubset(dataset, indices[:train_size], train_transform)
    val_dataset = SegmentationSubset(dataset, indices[train_size:], val_transform)

    return train_dataset, val_dataset
