import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        x = self.dropout(x)
        return x  # noqa: RET504


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x  # noqa: RET504


class UNetDropout(nn.Module):
    def __init__(self, n_channels: int, n_classes: int) -> None:
        super().__init__()
        self.TCB1 = DoubleConv(n_channels, 64, 64)
        self.TCB2 = DoubleConv(64, 128, 128)
        self.TCB3 = DoubleConv(128, 256, 256)
        self.TCB4 = DoubleConv(256, 512, 512)
        self.TCB5 = DoubleConv(512, 1024, 1024, dropout=0.5)
        self.TCB6 = DoubleConv(1024, 512, 512, dropout=0.3)
        self.TCB7 = DoubleConv(512, 256, 256, dropout=0.3)
        self.TCB8 = DoubleConv(256, 128, 128)
        self.TCB9 = DoubleConv(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2)

        self.UC1 = UpConv(1024, 512)
        self.UC2 = UpConv(512, 256)
        self.UC3 = UpConv(256, 128)
        self.UC4 = UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = self._crop_and_concat(x4, x)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = self._crop_and_concat(x3, x)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = self._crop_and_concat(x2, x)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = self._crop_and_concat(x1, x)
        x = self.TCB9(x)

        x = self.conv1(x)

        return x  # noqa: RET504

    def _crop_and_concat(self, enc: Tensor, dec: Tensor) -> Tensor:
        """エンコーダとデコーダのテンソルサイズを調整して結合"""
        diff_y = enc.size()[2] - dec.size()[2]
        diff_x = enc.size()[3] - dec.size()[3]

        dec = F.pad(dec, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

        return torch.cat([enc, dec], dim=1)
