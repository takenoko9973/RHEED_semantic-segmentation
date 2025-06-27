import json
import pprint
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str((Path(__file__).parent / "../src").resolve()))


from rheed_segmentation.utils import labelme

path = Path("data/SC-STO-250422/expo50_gain60/label/250425_900_fil-6_O2-0/0/0.0.json")
with path.open() as f:
    lbl_dict = json.load(f)


labels = {"_background_": 0, "spot": 1, "streak": 2, "kikuchi": 3}

mask = labelme.create_mask(lbl_dict, labels)
mask_dict = labelme.create_masks_per_labels(lbl_dict, labels)

labelme.convert_color_lbl(mask).save("mask.png")

for label in mask_dict:
    labelme.convert_color_lbl(mask_dict[label]).save(f"mask_{label}.png")
