import json
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from rheed_segmentation.utils import labelme
from rheed_segmentation.utils.postprocessing import merge_masks_by_priority

label_path = Path("data/label/SC-STO-250422/expo50_gain60/250425_900_fil-6_O2-0/0/0.0.json")
# label_path = Path("data/label/SC-STO-250422/expo50_gain60/250424_900_fil-6_O2-10/0/0.0.json")
label_map = {"_background_": 0, "spot": 1, "streak": 2, "kikuchi": 3}

with label_path.open("r") as f:
    json_data = json.load(f)

masks = labelme.create_masks_per_labels(json_data, label_map)
masks = {label: torch.Tensor(masks[label]).unsqueeze(0) for label in masks}

merged_mask = merge_masks_by_priority(masks)
merged_mask = labelme.convert_color_lbl(merged_mask.numpy()[0])
plt.imshow(merged_mask)
plt.show()
