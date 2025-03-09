from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn, optim
from torch._prims_common import DeviceLikeType
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        criterion: _Loss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.n_classes = n_classes

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_process(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: DeviceLikeType = "CPU",
        save_model_dir: str | Path | None = None,
    ) -> list[dict]:
        if save_model_dir:
            save_model_dir = Path(save_model_dir)

        self.model.to(device)

        history = []
        best_iou = 0.0
        with tqdm(range(epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                pbar_epoch.set_description(f"[Epoch {epoch + 1}]")

                train_loss = self.train_epoch(train_loader, device)
                val_loss, class_iou, mean_iou = self.validate_epoch(val_loader, device)

                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validate_loss": val_loss,
                    "mean_iou": mean_iou,
                }
                pbar_epoch.set_postfix(epoch_info)

                epoch_info["class_iou"] = class_iou
                history.append(epoch_info)

                if save_model_dir:
                    if (epoch + 1) % 20 == 0:
                        self.save_model(save_model_dir / f"epoch_{epoch + 1}.pth")

                    if mean_iou > best_iou:
                        best_iou = mean_iou
                        self.save_model(save_model_dir / "best.pth")

                if self.scheduler is not None:
                    self.scheduler.step()

        return history

    def train_epoch(self, loader: DataLoader, device: DeviceLikeType) -> float:
        self.model.train()

        epoch_loss = 0.0
        with tqdm(enumerate(loader), total=len(loader), leave=False) as pbar_loss:
            pbar_loss.set_description("train")

            for _, data in pbar_loss:
                images: Tensor = data[0].to(device)
                true_masks: Tensor = data[1].to(device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss: Tensor = self.criterion(outputs, true_masks.long())

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar_loss.set_postfix({"loss": loss.item()})

        return epoch_loss / len(loader)

    def validate_epoch(self, loader: DataLoader, device: DeviceLikeType) -> float:
        self.model.eval()

        # IoU 評価用
        labels = np.arange(self.n_classes)
        cm = np.zeros((self.n_classes, self.n_classes))

        epoch_loss = 0.0
        with (
            tqdm(enumerate(loader), total=len(loader), leave=False) as pbar_loss,
            torch.no_grad(),
        ):
            pbar_loss.set_description("validate")

            for _, data in pbar_loss:
                images: Tensor = data[0].to(device)
                true_masks: Tensor = data[1].to(device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                # ピクセルごとの混合行列を計算
                for j in range(len(true_masks)):
                    true_mask = true_masks[j].cpu().detach().numpy().flatten()
                    pred_mask = preds[j].cpu().detach().numpy().flatten()
                    cm += confusion_matrix(true_mask, pred_mask, labels=labels)

                loss: Tensor = self.criterion(outputs, true_masks.long())

                epoch_loss += loss.item()
                pbar_loss.set_postfix({"loss": loss.item()})

        class_iou, mean_iou = calc_IoU(cm)

        return epoch_loss / len(loader), class_iou, mean_iou

    def save_model(self, model_file_path: str | Path) -> None:
        model_file_path = Path(model_file_path)

        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_file_path)


def calc_IoU(cm: np.ndarray) -> float:  # noqa: N802
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    denominator = sum_over_row + sum_over_col - true_positives

    iou = true_positives / denominator

    return iou, np.nanmean(iou)
