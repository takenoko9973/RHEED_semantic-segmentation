from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from PIL import Image

from RHEED_semantic_segmentation.model import UNet
from RHEED_semantic_segmentation.utils import image as utils_image
from RHEED_semantic_segmentation.utils import label as utils_label


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


image_type = "raw"
root_dir = Path("downloads/SC-STO-250422/expo50_gain60")
model_id = "250512"

n_classes = 4
use_data_path = Path("models", image_type, model_id, "best.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

# モデル類
model = UNet(n_channels=1, n_classes=n_classes).to(device)
model_state = torch.load(use_data_path, weights_only=False)
model.load_state_dict(model_state)
model.eval()

# 画像読み込み
test_transform = albu.Compose(
    [
        albu.Resize(135, 180),  # (540, 720) -> (135, 180)
        albu.ToTensorV2(),
    ]
)

for image_path in root_dir.glob("**/*.tiff"):
    test_image = Image.open(image_path)
    test_image = utils_image.auto_scale(test_image)
    test_image = test_transform(image=test_image)["image"].to(device)

    outputs = model(test_image.unsqueeze(0))
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

    pred_array = preds[0].cpu().detach().numpy()
    pred_image = utils_label.convert_color_lbl(pred_array)

    parts = list(image_path.parts)
    parts[0] = "preds"
    save_path = Path(*parts).with_suffix(".png")

    save_path.parent.mkdir(exist_ok=True, parents=True)
    pred_image.save(save_path)
