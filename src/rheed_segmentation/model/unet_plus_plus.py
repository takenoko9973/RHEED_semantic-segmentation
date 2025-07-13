import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class VGGBlock(nn.Module):
    """UNet++の各ノードで使用される基本的な畳み込みブロック。VGGにインスパイアされた、Conv-ReLU-Conv-ReLUの構成。"""

    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out  # noqa: RET504


class UNetPlusPlus(nn.Module):
    """UNet++モデル本体。入れ子構造と密なスキップ接続が特徴。"""

    def __init__(self, n_channels: int, n_classes: int, deep_supervision: bool = False) -> None:
        super().__init__()

        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        # プーリング層とアップサンプリング層
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 入れ子構造の各ノードを定義
        # x_00, x_10, ... というグリッド状の配置をイメージ
        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # Deep Supervisionが有効な場合、各レベルの出力を生成する層
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # エンコーダーパス (グリッドの左端)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # 入れ子状のスキップ接続
        x0_1 = self.conv0_1(self._crop_and_concat(x0_0, self.up(x1_0)))
        x1_1 = self.conv1_1(self._crop_and_concat(x1_0, self.up(x2_0)))
        x2_1 = self.conv2_1(self._crop_and_concat(x2_0, self.up(x3_0)))
        x3_1 = self.conv3_1(self._crop_and_concat(x3_0, self.up(x4_0)))

        x0_2 = self.conv0_2(self._crop_and_concat(torch.cat([x0_0, x0_1], 1), self.up(x1_1)))
        x1_2 = self.conv1_2(self._crop_and_concat(torch.cat([x1_0, x1_1], 1), self.up(x2_1)))
        x2_2 = self.conv2_2(self._crop_and_concat(torch.cat([x2_0, x2_1], 1), self.up(x3_1)))

        x0_3 = self.conv0_3(self._crop_and_concat(torch.cat([x0_0, x0_1, x0_2], 1), self.up(x1_2)))
        x1_3 = self.conv1_3(self._crop_and_concat(torch.cat([x1_0, x1_1, x1_2], 1), self.up(x2_2)))

        x0_4 = self.conv0_4(
            self._crop_and_concat(torch.cat([x0_0, x0_1, x0_2, x0_3], 1), self.up(x1_3))
        )

        # Deep Supervisionの有無で出力を分岐
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        output = self.final(x0_4)
        return output  # noqa: RET504

    def _crop_and_concat(self, enc: Tensor, dec: Tensor) -> Tensor:
        """エンコーダとデコーダのテンソルサイズを調整して結合"""
        diff_y = enc.size()[2] - dec.size()[2]
        diff_x = enc.size()[3] - dec.size()[3]

        dec = F.pad(dec, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

        return torch.cat([enc, dec], dim=1)
