from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn, optim
from torch._prims_common import DeviceLikeType
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.postprocessing import merge_predictions_by_priority
from .utils.result_manager import ResultDir


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
        save_dir: ResultDir,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

        self.best_val_loss = float("inf")

    def train(self, num_epochs: int) -> None:
        epoch_loop = tqdm(range(1, num_epochs + 1))
        for epoch in epoch_loop:
            train_loss = self._train_one_epoch(epoch)
            val_loss, cm = self._validate_one_epoch(epoch)

            epoch_info = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validate_loss": val_loss,
            }
            epoch_loop.set_postfix(epoch_info)

            self._save_metrics(epoch, train_loss, val_loss, cm.tolist())

            if epoch % 20 == 0:
                self._save_checkpoint(self.save_dir.path / f"epoch_{epoch}.pth")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(self.save_dir.path / "best.pth")

            if self.scheduler:
                self.scheduler.step()

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        loop = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), desc="Train", leave=False
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
            loss, _ = self.loss_computer.compute(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self, epoch: int) -> tuple[float, np.ndarray]:
        self.model.eval()

        num_classes = self.loss_computer.num_classes
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        total_loss = 0.0
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Val", leave=False)
        with torch.no_grad():
            for _, data in loop:
                images: Tensor = data[0].to(self.device)
                if isinstance(data[1], dict):
                    masks: dict = {label: mask.to(self.device) for label, mask in data[1].items()}
                else:
                    masks: Tensor = data[1].to(self.device)

                outputs = self.model(images)
                loss, _ = self.loss_computer.compute(outputs, masks)

                cm += self._compute_confusion_matrix(outputs, masks, num_classes)

                total_loss += loss.item()
                loop.set_postfix({"loss": loss.item()})

        return total_loss / len(self.val_loader), cm

    def _merge_target_dict(self, masks: dict[str, Tensor]) -> Tensor:
        label_map = torch.zeros_like(next(iter(masks.values())), dtype=torch.long)  # (B, H, W)

        for class_idx, label in enumerate(masks):
            label_mask = masks[label] > 0  # ラベルあり部分
            label_map[label_mask] = class_idx + 1  # 背景=0と仮定

        return label_map

    def _compute_confusion_matrix(
        self, outputs: Tensor, masks: Tensor | dict[str, Tensor], num_classes: int
    ) -> np.ndarray:
        probs = torch.softmax(outputs, dim=1)
        if isinstance(masks, dict):
            true_labels = self._merge_target_dict(masks)
            preds = merge_predictions_by_priority(probs)
        else:
            true_labels = masks
            preds = torch.argmax(probs, dim=1)

        return confusion_matrix(
            true_labels.view(-1).cpu().numpy(),
            preds.view(-1).cpu().numpy(),
            labels=list(range(num_classes)),
        )

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def _save_metrics(self, epoch: int, train_loss: float, val_loss: float, cm: list) -> None:
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validate_loss": val_loss,
            "confusion_matrix": cm,
        }
        self.save_dir.write_history_file(record)
