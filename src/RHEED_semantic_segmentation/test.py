import json
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, optim

from RHEED_semantic_segmentation import load_data, utils
from RHEED_semantic_segmentation.model import UNet
from RHEED_semantic_segmentation.train import SegmentationTrainer


def segmentation2mask(mask: np.ndarray, color_palette: list) -> np.ndarray:
    # onehotベクトルからカラーインデックスを取り出して、該当するRGB3次元を付与
    H, W = mask.shape[:2]
    img = np.zeros(mask.shape).tolist()

    for height in range(H):
        for width in range(W):
            index = mask[height, width]
            rgb = color_palette[index]
            img[height][width] = rgb

    return np.asarray(img)


image_type = "after"
deg = "45"

n_classes = 4
use_data_path = Path("models", image_type, "best.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

# モデル類
model = UNet(n_channels=1, n_classes=n_classes).to(device)
model_state = torch.load(use_data_path, weights_only=False)
model.load_state_dict(model_state)

# 画像読み込み
test_transform = albu.Compose(
    [
        albu.Resize(135, 180),  # (540, 720) -> (135, 180)
        albu.ToTensorV2(),
    ]
)

image_path = Path("data", image_type, deg, "0.0.tiff")
mask_path = Path("data", "masks", deg, "0.0.png")
test_image = Image.open(image_path)
test_mask = Image.open(mask_path)
color_palette = np.array(test_mask.getpalette()).reshape(-1, 3)

IMAGE_SIZE = 256


test_image = utils.auto_scale(test_image)
test_image = test_transform(image=test_image)["image"].to(device)

# 予測
model.eval()

outputs = model(test_image.unsqueeze(0))
probs = torch.softmax(outputs, dim=1)
preds = torch.argmax(probs, dim=1)

pred_array = preds[0].cpu().detach().numpy()
pred_array = segmentation2mask(pred_array, color_palette).astype(np.uint8)
pred_image = Image.fromarray(pred_array)

save_path = Path("data", "preds", image_type, deg + ".png")
save_path.parent.mkdir(exist_ok=True, parents=True)
pred_image.save(Path("data", "preds", image_type, deg + ".png"))
