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

from RHEED_semantic_segmentation.utils import other as utils_other


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

    def __str__(self):
        return ",\n".join(
            [
                f"model: {self.model}",
                f"n_classes: {self.n_classes}",
                f"criterion: {self.criterion}",
                f"optimizer: {self.optimizer}",
                f"scheduler: {self.scheduler}",
            ]
        )

    def train_process(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: DeviceLikeType = "CPU",
        save_model_dir: str | Path | None = None,
    ) -> None:
        if save_model_dir:
            save_model_dir = Path(save_model_dir)

        history_path = save_model_dir / "history.jsonl"

        self.model.to(device)

        history = []
        best_f1 = 0.0
        with tqdm(range(epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                pbar_epoch.set_description(f"[Epoch {epoch + 1}]")

                train_loss = self.train_epoch(train_loader, device)
                val_loss, val_cm = self.validate_epoch(val_loader, device)

                _, macro_f1 = utils_other.compute_f1_from_confusion_matrix(val_cm)

                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validate_loss": val_loss,
                    "macro_f1": macro_f1,
                }
                pbar_epoch.set_postfix(epoch_info)

                history_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validate_loss": val_loss,
                    "confusion_matrix": val_cm.tolist(),
                }
                with history_path.open(mode="a", encoding="utf-8") as f:
                    json.dump(history_info, f)
                    f.write("\n")

                if save_model_dir:
                    if (epoch + 1) % 20 == 0:
                        self.save_model(save_model_dir / f"epoch_{epoch + 1}.pth")

                    if macro_f1 > best_f1:
                        best_f1 = macro_f1
                        self.save_model(save_model_dir / "best.pth")

                if self.scheduler is not None:
                    self.scheduler.step()

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

    def validate_epoch(
        self, loader: DataLoader, device: DeviceLikeType
    ) -> tuple[float, np.ndarray]:
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

        return epoch_loss / len(loader), cm

    def save_model(self, model_file_path: str | Path) -> None:
        model_file_path = Path(model_file_path)

        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_file_path)
