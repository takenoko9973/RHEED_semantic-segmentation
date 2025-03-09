# %%
import json
from pathlib import Path

import albumentations as albu
import torch
from torch import nn, optim

from RHEED_semantic_segmentation import load_data
from RHEED_semantic_segmentation.model import UNet
from RHEED_semantic_segmentation.train import SegmentationTrainer

# %%
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
image_type = "after"
train_rate = 0.8
loader_param = {"batch_size": 8, "num_workers": 4, "pin_memory": True, "shuffle": True}

# %%
train_loader, val_loader = load_data.make_dataloaders(
    "data",
    image_type,
    train_rate,
    train_transform,
    val_transform,
    loader_param,
)

# %%
n_classes = 4

model = UNet(n_channels=1, n_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

trainer = SegmentationTrainer(model, n_classes, criterion, optimizer, scheduler)

save_dir = Path("models")

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
history = trainer.train_process(train_loader, val_loader, 105, device, save_dir)

# %%
save_history = Path("history.json")
with save_history.open(mode="w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=4)

trainer.save_model(save_dir / "last.pth")
