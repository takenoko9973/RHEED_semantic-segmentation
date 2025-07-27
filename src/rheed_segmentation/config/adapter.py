from abc import ABC, abstractmethod
from typing import Any

from rheed_segmentation.config.schema import TargetMode


class IConfigAdapter(ABC):
    """古いバージョンの設定辞書を、次のバージョンの辞書形式に変換するためのインターフェース。"""

    @abstractmethod
    def adapt(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """設定辞書を次のバージョンへ変換する。"""


class V1Adapter(IConfigAdapter):
    """バージョン1形式を、バージョン2形式に変換するアダプター。"""

    def adapt(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        print("V1Adapter: v1形式をv2形式へ変換します...")

        # 変換ロジック
        adapted_dict = config_dict.copy()

        # v1の'transform'キーは、実質的に学習時のデータ拡張に相当
        old_transforms: list[dict] = adapted_dict.pop("transforms", [])

        # v2のスキーマ構造に合わせて辞書を再構築
        adapted_dict["transforms"] = {
            "train": {
                "base": [],
                "augmentations": [
                    transform
                    for transform in old_transforms
                    if transform.get("target", TargetMode._BOTH)  # noqa: SLF001
                    in (TargetMode._BOTH, TargetMode.TRAIN)  # noqa: SLF001
                ],
                "final": [],
            },
            "val": {
                "base": [],
                "augmentations": [
                    transform
                    for transform in old_transforms
                    if transform.get("target", TargetMode._BOTH)  # noqa: SLF001
                    in (TargetMode._BOTH, TargetMode.VAL)  # noqa: SLF001
                ],
                "final": [],
            },
        }
        return adapted_dict
