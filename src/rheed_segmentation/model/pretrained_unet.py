import segmentation_models_pytorch as smp
from torch import Tensor, nn


class PretrainedUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        n_channels: int = 1,
        n_classes: int = 4,
        encoder_freeze: bool = False,
    ) -> None:
        super().__init__()

        # smp.Unetモデルを初期化
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
        )

        if encoder_freeze:
            # エンコーダーのパラメータを凍結
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
