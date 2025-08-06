import segmentation_models_pytorch as smp
from torch import Tensor, nn


class PretrainedUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        n_channels: int = 1,
        n_classes: int = 4,
    ) -> None:
        super().__init__()

        # smp.Unetモデルを初期化
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
        )
        # from segmentation_models_pytorch.encoders import get_preprocessing_fn

        # preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
