import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, str((Path(__file__).parent / "../src").resolve()))

from rheed_segmentation.config import load_config
from rheed_segmentation.config.transform_config import TargetMode

config_file = Path(__file__).parent / "test_config.yaml"
# img_path = Path("data/raw/SC-STO-250422/expo50_gain60/250425_900_fil-6_O2-0/0/0.0.tiff")
img_path = Path("data/raw/SC-STO-250422/expo50_gain60/250424_900_fil-6_O2-10/0/0.0.tiff")


def main() -> None:
    config = load_config(config_file)

    img = np.array(Image.open(img_path)).astype(np.uint16)

    compose = config.transforms.to_transform_compose(target=TargetMode.VAL)
    img2 = compose(image=img)["image"][0]

    plt.imshow(img)
    plt.show()
    plt.imshow(img2)
    plt.show()


if __name__ == "__main__":
    main()
