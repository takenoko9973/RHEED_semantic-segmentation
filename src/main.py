import albumentations as albu
from torch import nn, optim

from rheed_segmentation.data_loader import make_dataloaders
from rheed_segmentation.model import UNet
from rheed_segmentation.train import LossComputer, Trainer
from rheed_segmentation.transforms import AutoScaleTransform
from rheed_segmentation.utils.result_manager import ResultDirManager


def main() -> None:
    labels = {
        "_background_": 0,
        "spot": 1,
        "streak": 2,
        "kikuchi": 3,
    }

    config = {
        "data_dir": "data/SC-STO-250422/expo50_gain60",
        "label_map": labels,
        "per_label": True,
    }

    train_transform = albu.Compose(
        [
            AutoScaleTransform(),
            albu.Resize(135, 180),  # (540, 720) -> (135, 180)
            albu.HorizontalFlip(p=0.5),
            albu.ToTensorV2(),
        ]
    )
    val_transform = albu.Compose(
        [
            AutoScaleTransform(),
            albu.Resize(135, 180),  # (540, 720) -> (135, 180)
            albu.ToTensorV2(),
        ]
    )
    result_manager = ResultDirManager()

    train_loader, val_loader = make_dataloaders(config, train_transform, val_transform)
    model = UNet(n_channels=1, n_classes=len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    loss_computer = LossComputer(criterion, len(labels))
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_computer=loss_computer,
        optimizer=optimizer,
        save_dir=result_manager.create_log_file("raw"),
        device="cuda",
        scheduler=scheduler,
    )
    trainer.train(10)


if __name__ == "__main__":
    main()
