import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss


def one_hot_soft_label(hard_mask: Tensor, num_classes: int, smoothing: float = 0.1) -> Tensor:
    # hard_mask: (B, H, W) 整数ラベル
    one_hot = F.one_hot(hard_mask, num_classes).permute(0, 3, 1, 2).float()  # → (B, C, H, W)
    return one_hot * (1 - smoothing) + smoothing / num_classes


class FocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        alpha: Tensor | float | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        B, C, H, W = inputs.shape  # noqa: N806

        # softmax: p ∈ [0,1], sum_c p_c = 1
        probs = F.softmax(inputs, dim=1).clamp(min=1e-8, max=1.0)
        log_probs = torch.log(probs)

        # focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma  # shape: (B, C, H, W)

        # apply alpha weighting if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device).view(1, C, 1, 1)  # shape: (1, C, 1, 1)
            focal_weight = focal_weight * alpha  # クラスごとの重み付け

        # focal loss
        soft_targets = one_hot_soft_label(targets, C, smoothing=self.label_smoothing)
        loss = -soft_targets * focal_weight * log_probs  # shape: (B, C, H, W)
        loss = loss.sum(dim=1)  # → shape: (B, H, W)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
