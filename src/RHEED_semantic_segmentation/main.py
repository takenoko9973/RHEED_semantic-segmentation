from pathlib import Path

import albumentations as albu
import toml
import torch
from torch import Tensor, nn, optim

import RHEED_semantic_segmentation.utils.other as utils_other
from RHEED_semantic_segmentation.criterion import FocalLoss
from RHEED_semantic_segmentation.dataframe import load_data
from RHEED_semantic_segmentation.model import UNet
from RHEED_semantic_segmentation.train import SegmentationTrainer

# 再現性のため、シード値固定
utils_other.init_random_seed(0)

# 条件定義
name = "250522"
image_types = ["raw", "CLAHE"]
train_rate = 0.8
n_classes = 4

lr = 1e-3
epochs = 300

comment = "CrossEntropyLoss(), Adam(weight_decay=1e-4)"

# モデル類
model = UNet(n_channels=1, n_classes=n_classes)
# criterion = FocalLoss(alpha=Tensor([0.1, 0.4, 0.25, 0.25]), gamma=2, label_smoothing=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

trainer = SegmentationTrainer(model, n_classes, criterion, optimizer, scheduler)

# データ定義
train_transform = albu.Compose(
    [
        albu.Resize(135, 180),  # (540, 720) -> (135, 180)
        albu.HorizontalFlip(p=0.5),
        albu.ToTensorV2(),
    ]
)
val_transform = albu.Compose(
    [
        albu.Resize(135, 180),  # (540, 720) -> (135, 180)
        albu.ToTensorV2(),
    ]
)
loader_param = {"batch_size": 4, "num_workers": 4, "pin_memory": True, "shuffle": True}

for image_type in image_types:
    save_dir = Path("models") / image_type / name

    train_loader, val_loader = load_data.make_dataloaders(
        ["downloads/SC-STO-250422/expo50_gain60"],
        image_type,
        train_rate,
        train_transform,
        val_transform,
        loader_param,
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "name": name,
        "lr": lr,
        "epochs": epochs,
        "trainer": str(trainer),
        "comment": comment,
    }
    save_history = save_dir / "config.toml"
    with save_history.open(mode="w", encoding="utf-8") as f:
        toml.dump(experiment_config, f)

    # 学習開始
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.train_process(train_loader, val_loader, epochs, device, save_dir)
    trainer.save_model(save_dir / "last.pth")
