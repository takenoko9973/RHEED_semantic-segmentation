import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn, optim
from torch._prims_common import DeviceLikeType
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm


class LossComputer:
    def __init__(
        self, criterion: _Loss, num_classes: int, class_weights: list[float] | None = None
    ) -> None:
        self.criterion = criterion
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights else None

    def compute(
        self, preds: Tensor, targets: Tensor | dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        B, _, H, W = preds.size()  # noqa: N806

        if isinstance(targets, Tensor):
            loss: Tensor = self.criterion(preds, targets)
            return loss, {"all": loss.item()}

        if isinstance(targets, dict):
            total_loss = torch.zeros(1, device=preds.device)
            per_label_losses = {}
            for c, (label, target) in enumerate(targets.items()):
                label_preds = torch.stack(
                    [preds[:, 0, :, :], preds[:, c + 1, :, :]],
                    dim=1,
                )

                loss = self.criterion(label_preds, target.long())  # target: (B, H, W)
                per_label_losses[label] = loss.item()
                total_loss += loss

            return total_loss, per_label_losses

        msg = f"Unsupported target type: {type(targets)}. Expected Tensor or dict[str, Tensor]."
        raise ValueError(msg)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_computer: LossComputer,
        optimizer: optim.Optimizer,
        device: DeviceLikeType,
        save_dir: str | Path,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)

        self.best_val_loss = float("inf")

    def train(self, num_epochs: int) -> None:
        epoch_loop = tqdm(range(num_epochs))
        for epoch in epoch_loop:
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate_one_epoch(epoch)

            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validate_loss": val_loss,
            }
            epoch_loop.set_postfix(epoch_info)

            if self.scheduler:
                self.scheduler.step(epoch)

            if (epoch + 1) % 20 == 0:
                self._save_checkpoint(self.save_dir / f"epoch_{epoch + 1}.pth")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(self.save_dir / "best.pth")

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Train Epoch {epoch + 1}",
            leave=False,
        )
        for _, data in loop:
            images: Tensor = data[0].to(self.device)
            if isinstance(data[1], dict):
                masks: dict[str, Tensor] = {
                    label: mask.to(self.device) for label, mask in data[1].items()
                }
            else:
                masks: Tensor = data[1].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss, per_labels_losses = self.loss_computer.compute(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self, epoch: int) -> float:
        self.model.eval()

        total_loss = 0.0
        loop = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=f"Varidation Epoch {epoch + 1}",
            leave=False,
        )
        with torch.no_grad():
            for _, data in loop:
                images: Tensor = data[0].to(self.device)
                if isinstance(data[1], dict):
                    masks: dict[str, Tensor] = {
                        label: mask.to(self.device) for label, mask in data[1].items()
                    }
                else:
                    masks: Tensor = data[1].to(self.device)

                outputs = self.model(images)
                loss, per_labels_losses = self.loss_computer.compute(outputs, masks)

                total_loss += loss.item()
                loop.set_postfix({"loss": loss.item()})

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, path: Path) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
